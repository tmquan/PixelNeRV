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
    def __init__(self, shape=256, in_channels=1, out_channels=1):
        super().__init__()
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.clarity_net = nn.Sequential(
            Unet(
                spatial_dims=2,
                in_channels=self.in_channels,
                out_channels=shape,
                channels=encoder_feature_channel["efficientnet-b8"],
                strides=(2, 2, 2, 2),
                num_res_units=4,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                dropout=0.4,
                norm=Norm.BATCH,
            ),
            Reshape(*[1, shape, shape, shape]),
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=encoder_feature_channel["efficientnet-b8"],
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                dropout=0.4,
                norm=Norm.BATCH,
            ),
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2,
                out_channels=1,
                channels=encoder_feature_channel["efficientnet-b8"],
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                dropout=0.4,
                norm=Norm.BATCH,
            ),
            Unet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.out_channels+1,
                channels=encoder_feature_channel["efficientnet-b8"],
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                dropout=0.4,
                norm=Norm.BATCH,
            ),
        )
        
    def forward(self, figures):
        clarity = self.clarity_net(figures)
        density = self.density_net(clarity)
        volumes_opacits = self.mixture_net(torch.cat([clarity, density], dim=1))
        volumes,opacits = torch.split(volumes_opacits, [self.out_channels, 1], dim=1)
        return volumes, F.softplus(opacits)
        