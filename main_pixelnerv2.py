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
torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras, 
    look_at_view_transform
)

from diffusers import UNet2DModel
# from pytorch_lightning.utilities.seed import seed_everything
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
from argparse import ArgumentParser
from typing import Optional
from monai.networks.nets import Unet, EfficientNetBN, DenseNet121
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers import Reshape

from positional_encodings.torch_encodings import PositionalEncodingPermute3D
from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer, minimized, normalized, standardized

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

class PixelNeRVFrontToBackFrustumFeaturer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, backbone="efficientnet-b7"):
        super().__init__()
        assert backbone in backbones.keys()
        self.model = EfficientNetBN(
            model_name=backbone, #(24, 32, 56, 160, 448)
            pretrained=True, 
            spatial_dims=2,
            in_channels=in_channels,
            num_classes=out_channels,
            adv_prop=True,
        )
        # self.model._fc.weight.data.zero_()
        # self.model._fc.bias.data.zero_()

    def forward(self, figures):
        camfeat = self.model.forward(figures)
        return camfeat

class PixelNeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, shape=256, sh=0, pe=8, backbone="efficientnet-b7"):
        super().__init__()
        self.sh = sh
        self.pe = pe
        self.shape = shape
        assert backbone in backbones.keys()
        if self.pe>0:
            encoder_net = PositionalEncodingPermute3D(self.pe) # 8
            pe_channels = self.pe
            pos_enc = torch.ones([1, self.pe, self.shape, self.shape, self.shape])
            encoded = encoder_net(pos_enc)
            self.register_buffer('encoded', encoded)
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
                shw = rsh_cart_2(zyx.view(-1, 3)) 
                assert out_channels == 9
            elif self.sh==3: 
                shw = rsh_cart_3(zyx.view(-1, 3))
                assert out_channels == 16
            else:
                ValueError("Spherical Harmonics only support 2 and 3 degree")
            # self.register_buffer('shbasis', shw.unsqueeze(0).permute(0, 4, 1, 2, 3))
            self.register_buffer('shbasis', shw.view(out_channels, self.shape, self.shape, self.shape))
     
        self.clarity_net = UNet2DModel(
            sample_size=shape,  
            in_channels=1,  
            out_channels=shape,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 48, 80, 224, 640), #(32, 48, 80, 224, 640),  # More channels -> more parameters
            norm_num_groups=8,
            down_block_types=(
                "DownBlock2D",  
                "DownBlock2D",  
                "DownBlock2D",
                "AttnDownBlock2D",  
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",    
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",    
            ),
            class_embed_type="timestep",
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1+pe_channels,
                out_channels=1,
                channels=(32, 48, 80, 224, 640), #(32, 48, 80, 224, 640),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.2,
            ),
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2+pe_channels,
                out_channels=1,
                channels=(32, 48, 80, 224, 640), #(32, 48, 80, 224, 640),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.2,
            ),
        )

        self.refiner_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=3+pe_channels,
                out_channels=out_channels,
                channels=(32, 48, 80, 224, 640), #(32, 48, 80, 224, 640),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.2,
            ), 
        )
             
    def forward(self, figures, azim, elev, n_views=2):
        samples = figures.shape[0] # 3
        clarity = self.clarity_net(figures, azim*1000, elev*2000)[0].view(-1, 1, self.shape, self.shape, self.shape)
        
        # Multiview can stack along batch dimension, last dimension is for X-ray
        clarity_ct, clarity_xr = torch.split(clarity, n_views)
        clarity_ct = clarity_ct.mean(dim=0, keepdim=True)
        clarity = torch.cat([clarity_ct, clarity_xr])

        if self.pe > 0:
            density = self.density_net(torch.cat([self.encoded.repeat(clarity.shape[0], 1, 1, 1, 1), clarity], dim=1))
            mixture = self.mixture_net(torch.cat([self.encoded.repeat(clarity.shape[0], 1, 1, 1, 1), clarity, density], dim=1))
            shcoeff = self.refiner_net(torch.cat([self.encoded.repeat(clarity.shape[0], 1, 1, 1, 1), clarity, density, mixture], dim=1))
        else:
            density = self.density_net(torch.cat([clarity], dim=1))
            mixture = self.mixture_net(torch.cat([clarity, density], dim=1))
            shcoeff = self.refiner_net(torch.cat([clarity, density, mixture], dim=1))

        if self.sh > 0:
            # shcomps = shcoeff*self.shbasis.repeat(clarity.shape[0], 1, 1, 1, 1) 
            sh_comps_raw = torch.einsum('abcde,bcde->abcde', shcoeff, self.shbasis)
            # Take the absolute value of the spherical harmonic components
            sh_comps_abs = torch.abs(sh_comps_raw)
            sh_comps_max = sh_comps_abs.max()
            sh_comps_min = sh_comps_abs.min()
            # Normalize the spherical harmonic components
            shcomps = (sh_comps_abs - sh_comps_min) / (sh_comps_max - sh_comps_min + 1e-8)
        else:
            shcomps = shcoeff 

        volumes = torch.cat([clarity, shcomps], dim=1)
        volumes_ct, volumes_xr = torch.split(volumes, 1)
        volumes_ct = volumes_ct.repeat(n_views, 1, 1, 1, 1)
        volumes = torch.cat([volumes_ct, volumes_xr])

        return volumes 

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
def mean_and_tanh(x, eps=1e-8): return ( F.tanh(x.mean(dim=1, keepdim=True)) * 0.5 + 0.5 )  
def mean_and_relu(x, eps=1e-8): return ( F.relu(x.mean(dim=1, keepdim=True)) )  
def make_cameras(dist, elev, azim):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(
        dist=dist.float(), 
        elev=elev.float() * 90, 
        azim=azim.float() * 180
    )
    return FoVPerspectiveCameras(R=R, T=T, fov=45, aspect_ratio=1).to(_device)

class PixelNeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.gan = hparams.gan
        self.cam = hparams.cam
        self.shape = hparams.shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.lambda_gp = hparams.lambda_gp
       
        self.logsdir = hparams.logsdir
       
        self.st = hparams.st
        self.sh = hparams.sh
        self.pe = hparams.pe
        
        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.backbone = hparams.backbone
        self.devices = hparams.devices
        
        self.save_hyperparameters()
    
        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.shape, 
            image_height=self.shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=2.0, 
            max_depth=6.0
        )
        
        self.inv_renderer = PixelNeRVFrontToBackInverseRenderer(
            in_channels=2, 
            out_channels=9 if self.sh==2 else 16 if self.sh==3 else 1, 
            shape=self.shape, 
            sh=self.sh, 
            pe=self.pe,
            backbone=self.backbone,
        )

        self.cam_settings = PixelNeRVFrontToBackFrustumFeaturer(
            in_channels=1, 
            out_channels=3, # azim + elev + prob
            backbone=self.backbone,
        )
        self.cam_settings.model._fc.weight.data[:2].zero_()
        self.cam_settings.model._fc.bias.data[:2].zero_()
        self.loss = nn.L1Loss(reduction="mean")

    def forward_screen(self, image3d, cameras):      
        return self.fwd_renderer(image3d, cameras) 

    def forward_volume(self, image2d, azim, elev, n_views=2):      
        return self.inv_renderer(image2d * 2.0 - 1.0, azim.squeeze(), elev.squeeze(), n_views) 

    def forward_camera(self, image2d):
        return self.cam_settings(image2d * 2.0 - 1.0)[:,:2]

    def forward_critic(self, image2d):
        return self.cam_settings(image2d * 2.0 - 1.0)[:,2:]

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]
            
        # Construct the random cameras
        src_azim_random = torch.randn(self.batch_size, device=_device).clamp_(-0.9, 0.9)
        src_elev_random = torch.randn(self.batch_size, device=_device).clamp_(-0.9, 0.9)
        src_dist_random = 4.0 * torch.ones(self.batch_size, device=_device)
        camera_random = make_cameras(src_dist_random, src_elev_random, src_azim_random)

        with torch.no_grad():
            est_figure_ct_random = self.forward_screen(image3d=image3d, cameras=camera_random)
            src_figure_xr_hidden = image2d

        est_dist_random = 4.0 * torch.ones(self.batch_size, device=_device)
        est_dist_hidden = 4.0 * torch.ones(self.batch_size, device=_device)
        est_dist_hidden = 4.0 * torch.ones(self.batch_size, device=_device)
        
        # Reconstruct the cameras
        est_feat_random, \
        est_feat_hidden = torch.split(
            self.forward_camera(
                image2d=torch.cat([est_figure_ct_random, src_figure_xr_hidden])
            ), self.batch_size
        )

        est_azim_random, est_elev_random = torch.split(est_feat_random, 1, dim=1)
        est_azim_hidden, est_elev_hidden = torch.split(est_feat_hidden, 1, dim=1)

        camera_random = make_cameras(est_dist_random, est_elev_random, est_azim_random)
        camera_hidden = make_cameras(est_dist_hidden, est_elev_hidden, est_azim_hidden)

        with torch.no_grad():
            est_figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=camera_hidden)

        cam_view = [self.batch_size, 1]       
        # Jointly estimate the volumes, single view, random view and multiple views
        rng_figure = torch.randint(low=0, high=3, size=(1, 1))
        if stage=='train' and rng_figure==1:
            est_volume_ct_random, \
            est_volume_xr_hidden = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_random, src_figure_xr_hidden]),
                    azim=torch.cat([est_azim_random.view(cam_view), est_azim_hidden.view(cam_view)]),
                    elev=torch.cat([est_elev_random.view(cam_view), est_elev_hidden.view(cam_view)]),
                    n_views=1
                ), self.batch_size
            )
            est_volume_ct_hidden = est_volume_ct_random
        elif stage=='train' and rng_figure==2:
            est_volume_ct_hidden, \
            est_volume_xr_hidden = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_hidden, src_figure_xr_hidden]),
                    azim=torch.cat([est_azim_hidden.view(cam_view), est_azim_hidden.view(cam_view)]),
                    elev=torch.cat([est_elev_hidden.view(cam_view), est_elev_hidden.view(cam_view)]),
                    n_views=1
                ), self.batch_size
            )
            est_volume_ct_random = est_volume_ct_hidden
        else:
            est_volume_ct_random, \
            est_volume_ct_hidden, \
            est_volume_xr_hidden = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_random, est_figure_ct_hidden, src_figure_xr_hidden]),
                    azim=torch.cat([est_azim_random.view(cam_view), est_azim_hidden.view(cam_view), est_azim_hidden.view(cam_view)]),
                    elev=torch.cat([est_elev_random.view(cam_view), est_elev_hidden.view(cam_view), est_elev_hidden.view(cam_view)]),
                    n_views=2,
                ), self.batch_size
            )    
           
        # Reconstruct the appropriate XR
        rec_figure_ct_random = self.forward_screen(image3d=est_volume_ct_random[:,1:], cameras=camera_random)
        rec_figure_ct_hidden = self.forward_screen(image3d=est_volume_ct_hidden[:,1:], cameras=camera_hidden)
        est_figure_xr_hidden = self.forward_screen(image3d=est_volume_xr_hidden[:,1:], cameras=camera_hidden)

        # Perform Post activation like DVGO      
        mid_volume_ct_random = est_volume_ct_random[:,:1]
        mid_volume_ct_hidden = est_volume_ct_hidden[:,:1]
        mid_volume_xr_hidden = est_volume_xr_hidden[:,:1]

        est_volume_ct_random = est_volume_ct_random[:,1:].sum(dim=1, keepdim=True)
        est_volume_ct_hidden = est_volume_ct_hidden[:,1:].sum(dim=1, keepdim=True)
        est_volume_xr_hidden = est_volume_xr_hidden[:,1:].sum(dim=1, keepdim=True)

        # Compute the loss
        # Per-pixel_loss
        im2d_loss_ct_random = self.loss(est_figure_ct_random, rec_figure_ct_random) 
        im2d_loss_ct_hidden = self.loss(est_figure_ct_hidden, rec_figure_ct_hidden) 
        im2d_loss_xr_hidden = self.loss(src_figure_xr_hidden, est_figure_xr_hidden) 

        im3d_loss_ct_random = self.loss(image3d, est_volume_ct_random) + self.loss(image3d, mid_volume_ct_random) 
        im3d_loss_ct_hidden = self.loss(image3d, est_volume_ct_hidden) + self.loss(image3d, mid_volume_ct_hidden) 
    
        # view_loss_ct_random = self.loss(src_azim_random, est_azim_random) \
        #                     + self.loss(src_elev_random, est_elev_random)
        view_loss_ct_random = self.loss(torch.cat([src_azim_random, src_elev_random]), 
                                        torch.cat([est_azim_random, est_elev_random]))

        im2d_loss_ct = im2d_loss_ct_random + im2d_loss_ct_hidden 
        im2d_loss_xr = im2d_loss_xr_hidden
        im3d_loss_ct = im3d_loss_ct_random + im3d_loss_ct_hidden
        view_loss_ct = view_loss_ct_random 

        view_cond_xr = self.loss(est_azim_hidden, torch.zeros_like(est_azim_hidden)) \
                     + self.loss(est_elev_hidden, torch.zeros_like(est_elev_hidden))

        im2d_loss = im2d_loss_ct + im2d_loss_xr
        im3d_loss = im3d_loss_ct
        
        view_loss = view_loss_ct 
        view_cond = view_cond_xr

        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_view_loss', view_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        p_loss = self.alpha*im3d_loss + self.gamma*im2d_loss 
        c_loss = self.theta*view_loss + self.omega*view_cond

        if self.gan:
            if optimizer_idx==0:
                # Compute generator loss
                fake_images = torch.cat([rec_figure_ct_random, rec_figure_ct_hidden, est_figure_xr_hidden])
                fake_scores = self.forward_critic(fake_images)
                # g_loss = -torch.mean(fake_scores)
                g_loss = F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))
                loss = p_loss + g_loss
                self.log(f'{stage}_g_loss', g_loss, on_step=(stage=='train'), prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            elif optimizer_idx==1:
                # Compute discriminator loss
                real_images = torch.cat([est_figure_ct_random, est_figure_ct_hidden, src_figure_xr_hidden])
                real_scores = self.forward_critic(real_images)
                fake_images = torch.cat([rec_figure_ct_random, rec_figure_ct_hidden, est_figure_xr_hidden])
                fake_scores = self.forward_critic(fake_images.detach())

                # d_loss = -torch.mean(real_scores) + torch.mean(fake_scores) # + gradient_penalty
                d_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores)) \
                       + F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores)) 
                loss = c_loss + d_loss
                self.log(f'{stage}_d_loss', d_loss, on_step=(stage=='train'), prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            else:
                loss = p_loss + c_loss
        else:
            loss = p_loss + c_loss

        if batch_idx==0:
            viz2d = torch.cat([
                        torch.cat([image3d[..., self.shape//2, :], 
                                   est_figure_ct_random,
                                   est_figure_ct_hidden,
                                   ], dim=-2).transpose(2, 3),
                        torch.cat([est_volume_ct_hidden[..., self.shape//2, :],
                                   rec_figure_ct_random,
                                   rec_figure_ct_hidden,
                                   ], dim=-2).transpose(2, 3),
                        torch.cat([image2d, 
                                   est_volume_xr_hidden[..., self.shape//2, :],
                                   est_figure_xr_hidden,
                                   ], dim=-2).transpose(2, 3),
                    ], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)

        info = {f'loss': loss}
        return info

    # def training_step(self, batch, batch_idx):
    #     return self._common_step(batch, batch_idx, optimizer_idx=0, stage='train')
    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._common_step(batch, batch_idx, optimizer_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=-1, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=-1, stage='test')

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
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        # return [optimizer], [scheduler]
        # opt_inv = torch.optim.AdamW(self.inv_renderer.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # opt_cam = torch.optim.AdamW(self.cam_settings.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # sch_inv = torch.optim.lr_scheduler.MultiStepLR(opt_inv, milestones=[100, 200], gamma=0.1)
        # sch_cam = torch.optim.lr_scheduler.MultiStepLR(opt_cam, milestones=[100, 200], gamma=0.1)
        # return [opt_inv, opt_cam], [sch_inv, sch_cam]
        if self.gan:
            opt_gen = torch.optim.AdamW([
                {'params': self.inv_renderer.parameters()},
            ], lr=self.lr, betas=(0.5, 0.999))
            opt_dis = torch.optim.AdamW([
                {'params': self.cam_settings.parameters()},
            ], lr=self.lr, betas=(0.5, 0.999))
            sch_gen = torch.optim.lr_scheduler.MultiStepLR(opt_gen, milestones=[100, 200], gamma=0.1)
            sch_dis = torch.optim.lr_scheduler.MultiStepLR(opt_dis, milestones=[100, 200], gamma=0.1)
            return [opt_gen, opt_dis], [sch_gen, sch_dis]
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
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
    
    parser.add_argument("--gan", action="store_true", help="whether to train with GAN")
    parser.add_argument("--cam", action="store_true", help="train cam locked or hidden")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    
    parser.add_argument("--alpha", type=float, default=1., help="vol loss")
    parser.add_argument("--gamma", type=float, default=1., help="img loss")
    parser.add_argument("--theta", type=float, default=1., help="cam loss")
    parser.add_argument("--omega", type=float, default=1., help="cam cond")
    parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty")
    
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logsfrecaling', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--backbone", type=str, default='efficientnet-b7', help="Backbone for network")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        # filename='epoch={epoch}-validation_loss={validation_loss_epoch:.2f}',
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True, 
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
            # swa_callback
        ],
        accumulate_grad_batches=4,
        # strategy="ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        strategy="ddp_sharded",  # "colossalai", "fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        # plugins=DDPStrategy(find_unused_parameters=False),
        precision=16 if hparams.amp else 32,
        # gradient_clip_val=0.01, 
        # gradient_clip_algorithm="value"
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