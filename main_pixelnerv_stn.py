import os
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras, 
    look_at_view_transform
)


from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
from argparse import ArgumentParser
from typing import Optional
from monai.networks.nets import Unet, EfficientNetBN
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers import Reshape

from positional_encodings.torch_encodings import PositionalEncodingPermute3D

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer, minimized, normalized, standardized


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    

class PixelNeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, shape=256, sh=0, pe=8):
        super().__init__()
        self.sh = sh
        self.pe = pe
        self.shape = shape

        if self.pe>0:
            self.encoder_net = PositionalEncodingPermute3D(self.pe) # 8
            pe_channels = self.pe
        else:
            pe_channels = 0

        if self.sh > 0:
            from rsh import rsh_cart_2, rsh_cart_3
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.shape)
            ys = torch.linspace(-1, 1, steps=self.shape)
            xs = torch.linspace(-1, 1, steps=self.shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1) # torch.Size([100, 100, 100, 3])
            if self.sh==2: 
                shw = rsh_cart_2(zyx) 
                assert out_channels == 9
            elif self.sh==3: 
                shw = rsh_cart_3(zyx)
                assert out_channels == 16
            else:
                ValueError("Spherical Harmonics only support 2 and 3 degree")
            # torch.Size([100, 100, 100, 9 or 16])
            self.register_buffer('shbasis', shw.unsqueeze(0).permute(0, 4, 1, 2, 3))

        self.clarity_net = nn.Sequential(
            Unet(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=shape,
                channels=(32, 48, 80, 224, 640, 800),
                strides=(2, 2, 2, 2, 2),
                num_res_units=4,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.4,
            ),
            Reshape(*[1, shape, shape, shape]),
        )
        
        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1+pe_channels,
                out_channels=1,
                channels=(32, 48, 80, 224, 640, 800),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.4,
            ),
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2+pe_channels,
                out_channels=1,
                channels=(32, 48, 80, 224, 640, 800),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.4,
            ),
        )

        self.refiner_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=3+pe_channels,
                out_channels=out_channels,
                channels=(32, 48, 80, 224, 640, 800),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.4,
            ), 
        )
             
    def forward(self, figures, norm_type="standardized"):
        clarity = self.clarity_net(figures)
        if self.pe > 0:
            pos_enc = torch.zeros((clarity.shape[0], self.pe, clarity.shape[2], clarity.shape[3], clarity.shape[3]), device=clarity.device)
            encoded = self.encoder_net(pos_enc)
            density = self.density_net(torch.cat([encoded, clarity], dim=1))
            mixture = self.mixture_net(torch.cat([encoded, clarity, density], dim=1))
            results = self.refiner_net(torch.cat([encoded, clarity, density, mixture], dim=1))
        else:
            density = self.density_net(torch.cat([clarity], dim=1))
            mixture = self.mixture_net(torch.cat([clarity, density], dim=1))
            results = self.refiner_net(torch.cat([clarity, density, mixture], dim=1))
        
        if self.sh > 0:
            volumes = F.relu( results*self.shbasis.repeat(figures.shape[0], 1, 1, 1, 1) )
        else:
            volumes = F.relu( results )

        return volumes  
        
class PixelNeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.logsdir = hparams.logsdir
        self.lr = hparams.lr
        self.shape = hparams.shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.st = hparams.st
        self.sh = hparams.sh
        self.pe = hparams.pe
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices

        self.n_pts_per_ray = hparams.n_pts_per_ray

        self.save_hyperparameters()

        if self.st>0:
            self.stn_modifier = EfficientNetBN(
                model_name="efficientnet-b7", #(32, 48, 80, 224, 640, 800)
                in_channels=1,
                num_classes=6,
            )
            # Initialize the weights/bias with identity transformation
            self.stn_modifier._fc.weight.data.zero_()
            self.stn_modifier._fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            
        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.shape, 
            image_height=self.shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=2.0, 
            max_depth=6.0
        )
        
        self.inv_renderer = PixelNeRVFrontToBackInverseRenderer(
            in_channels=3, 
            out_channels=9 if self.sh==2 else 16 if self.sh==3 else 1, 
            shape=self.shape, 
            sh=self.sh, 
            pe=self.pe,
        )

        self.loss_smoothl1 = nn.SmoothL1Loss(reduction="mean", beta=0.02)

    # Spatial transformer network forward function
    def stn_forward(self, x):
        theta = self.stn_modifier(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        xs = F.grid_sample(x, grid)
        return xs

    def forward(self, figures, elev, azim):      
        return self.inv_renderer(torch.cat([figures, 
                                            elev.view(-1, 1, 1, 1).repeat(1, 1, self.shape, self.shape) * 0.5 + 0.5, # -1 1 to 0 1
                                            azim.view(-1, 1, 1, 1).repeat(1, 1, self.shape, self.shape),
                                            ], dim=1))

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]

        # Construct the locked camera
        dist_locked = 4.0 * torch.ones(self.batch_size, device=_device)
        elev_locked = torch.zeros(self.batch_size, device=_device)
        azim_locked = torch.zeros(self.batch_size, device=_device) 
        R_locked, T_locked = look_at_view_transform(
            dist=dist_locked, 
            elev=elev_locked, # * 0, 
            azim=azim_locked, # * 0
        )
        camera_locked = FoVPerspectiveCameras(R=R_locked, T=T_locked, fov=45, aspect_ratio=1).to(_device)

        # Construct the random camera
        dist_random = 4.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.clamp(
                            torch.randn(self.batch_size, device=_device), 
                            min=-0.5, max=0.5) # -0.5 0.5 -> -45 45 ;   -1 1 -> -90 90
        azim_random = torch.rand(self.batch_size, device=_device) # 0 1 -> 0 360
        R_random, T_random = look_at_view_transform(
            dist=dist_random, 
            elev=elev_random * 90, 
            azim=azim_random * 360
        )
        camera_random = FoVPerspectiveCameras(R=R_random, T=T_random, fov=45, aspect_ratio=1).to(_device)

        # CT pathway
        src_volume_ct_locked = image3d
        est_figure_ct_locked = self.fwd_renderer.forward(image3d=src_volume_ct_locked, opacity=None, cameras=camera_locked)
        est_figure_ct_random = self.fwd_renderer.forward(image3d=src_volume_ct_locked, opacity=None, cameras=camera_random)
        
        # XR pathway
        if self.st == 1:
            src_figure_xr_hidden = self.stn_forward(image2d)
        else:
            src_figure_xr_hidden = image2d
        est_volume_ct_locked, est_volume_ct_random, est_volume_xr_locked = \
            torch.split(
                self.forward(
                    torch.cat([est_figure_ct_locked, est_figure_ct_random, src_figure_xr_hidden]), 
                    torch.cat([elev_locked, elev_random, elev_locked]), 
                    torch.cat([azim_locked, azim_random, azim_locked]), 
                ),
                self.batch_size
            )

        rec_figure_ct_locked_locked = self.fwd_renderer.forward(image3d=est_volume_ct_locked, opacity=None, cameras=camera_locked)
        rec_figure_ct_locked_random = self.fwd_renderer.forward(image3d=est_volume_ct_locked, opacity=None, cameras=camera_random)

        rec_figure_ct_random_locked = self.fwd_renderer.forward(image3d=est_volume_ct_random, opacity=None, cameras=camera_locked)
        rec_figure_ct_random_random = self.fwd_renderer.forward(image3d=est_volume_ct_random, opacity=None, cameras=camera_random)
        
        est_figure_xr_locked_locked = self.fwd_renderer.forward(image3d=est_volume_xr_locked, opacity=None, cameras=camera_locked)
        
        # Compute the loss
        im3d_loss = self.loss_smoothl1(src_volume_ct_locked, est_volume_ct_random.sum(dim=1, keepdim=True)) \
                  + self.loss_smoothl1(src_volume_ct_locked, est_volume_ct_locked.sum(dim=1, keepdim=True)) 

        im2d_loss = self.loss_smoothl1(est_figure_ct_locked, rec_figure_ct_locked_locked) \
                  + self.loss_smoothl1(est_figure_ct_random, rec_figure_ct_locked_random) \
                  + self.loss_smoothl1(est_figure_ct_locked, rec_figure_ct_random_locked) \
                  + self.loss_smoothl1(est_figure_ct_random, rec_figure_ct_random_random) \
                  + self.loss_smoothl1(src_figure_xr_hidden, est_figure_xr_locked_locked) 
                            
        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage == 'train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage == 'train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)

        loss = self.alpha*im3d_loss + self.gamma*im2d_loss 

        if batch_idx == 0:
            viz2d = torch.cat([
                        torch.cat([src_volume_ct_locked[..., self.shape//2, :], 
                                   est_figure_ct_locked,
                                   est_figure_ct_random,
                                   est_volume_ct_locked.sum(dim=1, keepdim=True)[..., self.shape//2, :],
                                   ], dim=-2).transpose(2, 3),
                        torch.cat([rec_figure_ct_locked_locked,
                                   rec_figure_ct_locked_random,
                                   rec_figure_ct_random_locked,
                                   rec_figure_ct_random_random,
                                   ], dim=-2).transpose(2, 3),
                        torch.cat([image2d, 
                                   src_figure_xr_hidden,
                                   est_volume_xr_locked.sum(dim=1, keepdim=True)[..., self.shape//2, :],
                                   est_figure_xr_locked_locked,
                                   ], dim=-2).transpose(2, 3),

                        # torch.cat([est_figure_ct_locked,
                        #            est_figure_ct_random,
                        #            rec_figure_ct_locked_locked,
                        #            rec_figure_ct_locked_random,
                        #            rec_figure_ct_random_locked,
                        #            rec_figure_ct_random_random,
                        #            ], dim=-2).transpose(2, 3),
                        # torch.cat([src_volume_ct_locked[..., self.shape//2, :], 
                        #            image2d, 
                        #            src_figure_xr_hidden,
                        #            est_volume_xr_locked.sum(dim=1, keepdim=True)[..., self.shape//2, :],
                        #            est_figure_xr_locked_locked,
                        #            est_volume_ct_locked.sum(dim=1, keepdim=True)[..., self.shape//2, :],
                        #            ], dim=-2).transpose(2, 3)
                    ], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)

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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")

    # Model arguments
    parser.add_argument("--n_pts_per_ray", type=int, default=512, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--st", type=int, default=1, help="with spatial transformer network")
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    
    parser.add_argument("--alpha", type=float, default=1., help="im3d loss")
    parser.add_argument("--gamma", type=float, default=1., help="im2d loss")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
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
        accumulate_grad_batches=4,
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
        img_shape=hparams.shape,
        vol_shape=hparams.shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = PixelNeRVLightningModule(
        hparams=hparams
    )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

    trainer.fit(
        model,
        datamodule,
        ckpt_path=hparams.ckpt if hparams.ckpt is not None else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve
