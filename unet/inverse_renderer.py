import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import Norm, Reshape, AffineTransform
from monai.networks.nets import Unet, EfficientNetBN
from monai.networks.nets.flexible_unet import encoder_feature_channel
# from pytorch3d.renderer import NDCMultinomialRaysampler, VolumeRenderer
# from pytorch3d.structures import Volumes
from kornia.geometry.transform.imgwarp import warp_affine3d
# from .rsh import rsh_cart_2, rsh_cart_3

class UnetFrontToBackInverseRenderer(nn.Module):
    def __init__(self, shape=256, in_channels=1, mid_channels=10, out_channels=1, with_stn=True):
        super().__init__()
        self.shape = shape
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.with_stn = with_stn
        self.clarity_net = nn.Sequential(
            Unet(
                spatial_dims=2,
                in_channels=self.in_channels,
                out_channels=shape,
                channels=encoder_feature_channel["efficientnet-l2"],
                strides=(2, 2, 2, 2),
                num_res_units=4,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.5,
            ),
            Reshape(*[1, shape, shape, shape]),
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=encoder_feature_channel["efficientnet-l2"],
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.5,
            ),
        )

        # self.mixture_net = nn.Sequential(
        #     Unet(
        #         spatial_dims=3,
        #         in_channels=2,
        #         out_channels=self.mid_channels,
        #         channels=encoder_feature_channel["efficientnet-b8"],
        #         strides=(2, 2, 2, 2),
        #         num_res_units=2,
        #         kernel_size=3,
        #         up_kernel_size=3,
        #         act=("LeakyReLU", {"inplace": True}),
        #         norm=Norm.BATCH,
        #         dropout=0.5,
        #     ),
        # )

        if self.with_stn:
            self.affine_theta = nn.Sequential(
                EfficientNetBN("efficientnet-b8", 
                    spatial_dims=2, 
                    in_channels=1, 
                    num_classes=3*4,
                    pretrained=True
                ),
                nn.Tanh()
            )
            self.affine_tform = AffineTransform(
                normalized=False
            )

        
    def forward(self, figures):
        clarity = self.clarity_net(figures)
        volumes_opacits = self.density_net(clarity)
        #Call the spatial transformer network to correct the pose
        if self.with_stn:
            theta = self.affine_theta(figures).view(figures.shape[0], 3, 4).float()
            # theta = torch.cat([
            #     theta, 
            #     torch.zeros(self.batch_size, 3, 3).to(figures.device)
            # ], dim=-1)
            volumes_opacits = self.affine_tform(volumes_opacits, theta)
        volumes,opacits = torch.split(volumes_opacits, [self.mid_channels-1, 1], dim=1)
        return volumes, F.softplus(opacits)
        #Call the spatial transformer network to correct the pose
        if self.with_stn:
            theta = self.affine_theta(figures).view(figures.shape[0], 3, 4).float()
            # theta = torch.cat([
            #     theta, 
            #     torch.zeros(self.batch_size, 3, 3).to(figures.device)
            # ], dim=-1)
            density = self.affine_tform(density, theta)
        
        return density, torch.ones_like(density)