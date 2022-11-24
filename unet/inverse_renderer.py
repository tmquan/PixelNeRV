import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import Norm, Reshape
from monai.networks.nets import Unet
from monai.networks.nets.flexible_unet import encoder_feature_channel
# from pytorch3d.renderer import NDCMultinomialRaysampler, VolumeRenderer
# from pytorch3d.structures import Volumes

# from .rsh import rsh_cart_2, rsh_cart_3

class UnetFrontToBackInverseRenderer(nn.Module):
    def __init__(self, shape=256, in_channels=1, mid_channels=10, out_channels=1):
        super().__init__()
        self.shape = shape
        self.in_channels = in_channels
        self.mid_channels = mid_channels
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
                norm=Norm.BATCH,
                dropout=0.5,
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
                norm=Norm.BATCH,
                dropout=0.5,
            ),
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2,
                out_channels=self.mid_channels,
                channels=encoder_feature_channel["efficientnet-b8"],
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.5,
            ),
        )

        # # Generate grid
        # zs = torch.linspace(-1, 1, steps=self.shape)
        # ys = torch.linspace(-1, 1, steps=self.shape)
        # xs = torch.linspace(-1, 1, steps=self.shape)
        # z, y, x = torch.meshgrid(zs, ys, xs)
        # zyx = torch.stack([z, y, x], dim=-1) # torch.Size([100, 100, 100, 3])
        # shw = rsh_cart_3(zyx) if self.mid_channels==17 else rsh_cart_2(zyx) 
        # # torch.Size([100, 100, 100, 9 or 16])
        # self.register_buffer('shbasis', shw.unsqueeze(0).permute(0, 4, 1, 2, 3))


    # def forward(self, figures):
    #     clarity = self.clarity_net(figures)
    #     density = self.density_net(clarity)
    #     shcodes_opacits = self.mixture_net(torch.cat([clarity, density], dim=1))
    #     shcodes,opacits = torch.split(shcodes_opacits, [self.mid_channels-1, 1], dim=1)
    #     decomps = (shcodes.to(figures.device)*self.shbasis.repeat(figures.shape[0], 1, 1, 1, 1))
    #     # volumes = decomps.mean(dim=1, keepdim=True)
    #     # return volumes, F.softplus(opacits)
    #     return decomps, F.softplus(opacits)
    
    def forward(self, figures):
        clarity = self.clarity_net(figures)
        density = self.density_net(clarity)
        volumes_opacits = self.mixture_net(torch.cat([clarity, density], dim=1))
        volumes,opacits = torch.split(volumes_opacits, [self.mid_channels-1, 1], dim=1)
        return volumes, F.softplus(opacits)
        # shcodes_opacits = self.mixture_net(torch.cat([clarity, density], dim=1))
        # shcodes,opacits = torch.split(shcodes_opacits, [self.mid_channels-1, 1], dim=1)
        # decomps = (shcodes.to(figures.device)*self.shbasis.repeat(figures.shape[0], 1, 1, 1, 1))
        # # volumes = decomps.mean(dim=1, keepdim=True)
        # # return volumes, F.softplus(opacits)
        # return decomps, F.softplus(opacits)