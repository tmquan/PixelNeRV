import torch
import torch.nn as nn
from pytorch3d.structures import Volumes

from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
)
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from dvr.raymarcher import EmissionAbsorptionFrontToBackRaymarcher

def minimized(x, eps=1e-8): return (x + eps)/(x.max() + eps)
def normalized(x, eps=1e-8): return (x - x.min() + eps) / (x.max() - x.min() + eps)
def standardized(x, eps=1e-8): return (x - x.mean()) / (x.std() + eps)  # 1e-6 to avoid zero division

class DirectVolumeRenderer(nn.Module):
    def __init__(
        self, 
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320, 
        min_depth: float = 2.0,
        max_depth: float = 6.0, 
        stratified_sampling: bool = False,
    ):
        super().__init__()
        self.raymarcher = EmissionAbsorptionRaymarcher()  # BackToFront Raymarcher
        self.raysampler = NDCMultinomialRaysampler(  
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,  
            min_depth=min_depth,
            max_depth=max_depth,
            stratified_sampling=stratified_sampling, 
        )
        self.renderer = VolumeRenderer(
            raysampler=self.raysampler,
            raymarcher=self.raymarcher,
        )

    def forward(
        self, 
        image3d, 
        cameras, 
        opacity=None, 
        norm_type="standardized", 
        scaling_factor=0.1, 
        is_grayscale=True,
        return_bundle=False, 
    ) -> torch.Tensor:

        features = image3d.repeat(1, 3, 1, 1, 1) if image3d.shape[1] == 1 else image3d
        if opacity is None:
            if image3d.shape[1] != 1:
                densities = torch.ones_like(image3d[:,[0]]) * scaling_factor  
            else:
                densities = torch.ones_like(image3d) * scaling_factor
        else:
            densities = opacity * scaling_factor
        # print(image3d.shape, densities.shape)
        shape = max(image3d.shape[1], image3d.shape[2])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=3.0 / shape,
        )
        # screen_RGBA, ray_bundles = self.renderer(cameras=cameras, volumes=volumes) #[...,:3]
        # rays_points = ray_bundle_to_ray_points(ray_bundles)
        screen_RGBA, bundle = self.renderer(cameras=cameras, volumes=volumes) # [...,:3]

        screen_RGBA = screen_RGBA.permute(0,3,2,1)  # 3 for NeRF
        if is_grayscale:
            screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        else:
            screen_RGB = screen_RGBA[:, :3]

        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        
        if return_bundle:
            return screen_RGB, bundle
        return screen_RGB


class DirectVolumeFrontToBackRenderer(DirectVolumeRenderer):
    def __init__(
        self, 
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320, 
        min_depth: float = 2.0,
        max_depth: float = 6.0, 
        stratified_sampling: bool = False,
    ):
        super().__init__()
        self.raymarcher = EmissionAbsorptionFrontToBackRaymarcher()  # FrontToBack
        self.raysampler = NDCMultinomialRaysampler(
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified_sampling=stratified_sampling,
        )
        self.renderer = VolumeRenderer(
            raysampler=self.raysampler,
            raymarcher=self.raymarcher,
        )

class DirectVolumeStratifiedFrontToBackRenderer(DirectVolumeRenderer):
    def __init__(
        self, 
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320, 
        min_depth: float = 2.0,
        max_depth: float = 6.0, 
    ):
        super().__init__()
        self.raymarcher_randn = EmissionAbsorptionFrontToBackRaymarcher()  # FrontToBack
        self.raymarcher_equal = EmissionAbsorptionFrontToBackRaymarcher()  # FrontToBack
        self.raysampler_randn = NDCMultinomialRaysampler(
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified_sampling=True,
        )
        self.raysampler_equal = NDCMultinomialRaysampler(
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified_sampling=False,
        )
        self.renderer_randn = VolumeRenderer(
            raysampler=self.raysampler_randn,
            raymarcher=self.raymarcher_randn,
        )
        self.renderer_equal = VolumeRenderer(
            raysampler=self.raysampler_equal,
            raymarcher=self.raymarcher_equal,
        )

    def forward(
        self, 
        image3d, 
        cameras, 
        opacity=None, 
        norm_type="standardized", 
        scaling_factor=0.1, 
        is_grayscale=True,
        return_bundle=False, 
        stratified_sampling: bool = False,
    ) -> torch.Tensor:

        features = image3d.repeat(1, 3, 1, 1, 1) if image3d.shape[1] == 1 else image3d
        if opacity is None:
            if image3d.shape[1] != 1:
                densities = torch.ones_like(image3d[:,[0]]) * scaling_factor  
            else:
                densities = torch.ones_like(image3d) * scaling_factor
        else:
            densities = opacity * scaling_factor
        # print(image3d.shape, densities.shape)
        shape = max(image3d.shape[1], image3d.shape[2])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=3.0 / shape,
        )
        if stratified_sampling:
            screen_RGBA, bundle = self.renderer_randn(cameras=cameras, volumes=volumes) # [...,:3]
        else:
            screen_RGBA, bundle = self.renderer_equal(cameras=cameras, volumes=volumes) # [...,:3]
            
        screen_RGBA = screen_RGBA.permute(0,3,2,1)  # 3 for NeRF
        if is_grayscale:
            screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        else:
            screen_RGB = screen_RGBA[:, :3]

        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        
        if return_bundle:
            return screen_RGB, bundle
        return screen_RGB