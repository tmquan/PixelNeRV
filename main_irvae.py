import os
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import torch
import torch.nn as nn
import torchvision
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler, 
)

from pytorch3d.renderer.cameras import (
    CamerasBase,
    FoVPerspectiveCameras, 
    look_at_view_transform
)

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
from argparse import ArgumentParser
from typing import Optional, Sequence

from monai.networks.nets import VarAutoEncoder
from monai.networks.layers.factories import Act, Norm

from datamodule import UnpairedDataModule
from nerv.inverse_renderer import NeRVFrontToBackInverseRenderer
from dvr.renderer import DirectVolumeFrontToBackRenderer

class IRVAELightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.logsdir = hparams.logsdir
        self.lr = hparams.lr
        self.shape = hparams.shape
        self.filter = hparams.filter

        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.save_hyperparameters()

        self.renderer = DirectVolumeFrontToBackRenderer(
            image_width=512, 
            image_height=512, 
            n_pts_per_ray=400, 
            min_depth=2.0, 
            max_depth=6.0
        )

        self.vae3d = VarAutoEncoder(
            spatial_dims=3, 
            in_shape=(1, 256, 256, 256),
            out_channels=1,
            latent_size=64,
            channels=(16, 32, 48, 64),
            strides=(1, 2, 2, 2), 
            dropout=0.5,
            norm=Norm.BATCH,
            use_sigmoid=True
        )

        # self.vae2d = VarAutoEncoder(
        #     spatial_dims=2, 
        #     in_shape=(1, 512, 512),
        #     out_channels=1,
        #     latent_size=64,
        #     channels=(16, 32, 64, 80),
        #     strides=(1, 2, 2, 2), 
        #     dropout=0.5,
        #     norm=Norm.BATCH,
        #     use_sigmoid=True
        # )

        self.loss_sl1 = nn.SmoothL1Loss(reduction="mean", beta=0.02)
        self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        
    def forward(self, figures):
        return self.inv_renderer(figures)    


    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]

        if stage=='train':
            if (batch_idx % 2) == 1:
                masked = image3d>0
                noises = torch.rand_like(image3d) * masked.to(image3d.dtype)
                alpha_ = torch.rand(self.batch_size, 1, 1, 1, 1, device=_device)
                alpha_ = alpha_.expand_as(image3d)
                image3d = alpha_ * image3d + (1 - alpha_) * noises

        # Construct the locked camera
        dist = 4.0 * torch.ones(self.batch_size, device=_device)
        elev = torch.ones(self.batch_size, device=_device) * 0
        azim = torch.ones(self.batch_size, device=_device) * 0
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=45, aspect_ratio=1).to(_device)

        # # Construct the random camera
        # dist_random = 4.0 * torch.ones(self.batch_size, device=_device)
        # elev_random = torch.rand(self.batch_size, device=_device) * 180 - 90
        # azim_random = torch.rand(self.batch_size, device=_device) * 360
        # R_random, T_random = look_at_view_transform(dist=dist_random, elev=elev_random, azim=azim_random)
        # camera_random = FoVPerspectiveCameras(R=R_random, T=T_random, fov=45, aspect_ratio=1).to(_device)

        # CT pathway
        image3d_alp = torch.ones_like(image3d)
        image3d_prj = self.renderer.forward(
            image3d=image3d, 
            opacity=image3d_alp, 
            cameras=cameras
        )

        image3d_rec, mu3d, logvar3d, z3d = self.vae3d(image3d)
        # image3d_rep, mu2d, logvar2d, z2d = self.vae2d(image3d_prj)
        # image2d_rec, mu2x, logvar2x, z2x = self.vae2d(image2d)

        # Decode the Xray
        # image2d_rep = self.vae3d.decode_forward(z2x, use_sigmoid=True)
        mu2d, logvar2d = torch.std(image2d, dim=[1,2,3], keepdim=False)
        z2d = self.vae3d.reparameterize(mu2d, logvar2d)
        image2d_rep = self.vae3d.decode_forward(z2d, use_sigmoid=True)

        if batch_idx == 0:
            viz2d = torch.cat([
                        torch.cat([image2d,
                                   image3d_prj, 
                                   z3d
                                   ], dim=-2).transpose(2, 3),
                        torch.cat([image3d[..., self.shape//2, :],
                                   image3d[..., :, self.shape//2],
                                   image3d_rec[..., self.shape//2, :],
                                   image3d_rec[..., :, self.shape//2],
                                   image2d_rep[..., self.shape//2, :],
                                   image2d_rep[..., :, self.shape//2],
                                   ], dim=-2).transpose(2, 3)
                    ], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        

        # loss_bce = self.loss_bce(image3d, image3d_rec) \
        #          + self.loss_bce(image3d_rep, image3d_prj) \
        #          + self.loss_bce(image2d_rec, image2d) 
        # beta = 100
        # loss_kld = torch.mean(-0.5 * beta * torch.mean(1 + logvar3d - mu3d.pow(2) - logvar3d.exp(), dim=1), dim=0) \
        #          + torch.mean(-0.5 * beta * torch.mean(1 + logvar2d - mu2d.pow(2) - logvar2d.exp(), dim=1), dim=0) \
        #          + torch.mean(-0.5 * beta * torch.mean(1 + logvar2x - mu2x.pow(2) - logvar2x.exp(), dim=1), dim=0)

        loss_bce = self.loss_bce(image3d, image3d_rec)
        beta = 100
        loss_kld = torch.mean(-0.5 * beta * torch.mean(1 + logvar3d - mu3d.pow(2) - logvar3d.exp(), dim=1), dim=0) 
        loss_sl1 = self.loss_sl1(mu3d, mu2d.view(mu3d.shape)) + self.loss_sl1(logvar3d, logvar2d.view(logvar3d.shape))
        loss = loss_bce + loss_kld + 3*loss_sl1
        # loss = self.loss_sl1(z3d, image2d.view(z3d.shape))
        info = {f'loss': loss}
        return info

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str] = 'common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=self.lr / 10)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="NeRV")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")

    # Model arguments
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logsfrecaling', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--filter", type=str, default='sobel', help="None, sobel, laplacian, canny")

    parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5,
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
        ],
        # accumulate_grad_batches=5,
        # strategy="ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        strategy="fsdp",  # "fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,  # if hparams.use_amp else 32,
        # stochastic_weight_avg=True,
        # deterministic=False,
        # profiler="simple",
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    train_label3d_folders = [
    ]

    train_image2d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    val_image2d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        img_shape=hparams.shape * 2,
        vol_shape=hparams.shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = IRVAELightningModule(
        hparams=hparams
    )
    model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

    trainer.fit(
        model,
        datamodule,
    )

    # test

    # serve
