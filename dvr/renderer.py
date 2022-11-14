import torch
import torch.nn as nn
from pytorch3d.structures import Volumes

from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
)

from .raymarcher import EmissionAbsorptionRaymarcherFrontToBack

class DirectVolumeRenderer(nn.Module):
    def __init__(self, image_width=256, image_height=256, n_pts_per_ray=320, min_depth=2.0, max_depth=6.0):
        super().__init__()
        raysampler = NDCMultinomialRaysampler(  
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,  
            min_depth=min_depth,
            max_depth=max_depth,
        )
        raymarcher = EmissionAbsorptionRaymarcherFrontToBack()  # X-Ray Raymarcher
        renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        )
        self._renderer = renderer

    def forward(self, cameras, image3d, opacity=None, norm_type="standardized", eps=1e-8):
        features = image3d.repeat(
            1, 3, 1, 1, 1) if image3d.shape[1] == 1 else image3d
        if opacity is None:
            densities = torch.ones_like(image3d.mean(
                dim=1))*0.1 if image3d.shape[1] != 1 else torch.ones_like(image3d)*0.1
        else:
            densities = opacity*0.1

        shape = max(image3d.shape[1], image3d.shape[2])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=3.0 / shape,
        )
        # screen_RGBA, ray_bundles = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]
        # rays_points = ray_bundle_to_ray_points(ray_bundles)
        screen_RGBA, _ = self._renderer(
            cameras=cameras, volumes=volumes)  # [...,:3]

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1)  # 3 for NeRF
        screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        def minimized(x): return (x + eps)/(x.max() + eps)

        def normalized(x): return (x - x.min() + eps) / \
            (x.max() - x.min() + eps)

        def standardized(x): return (x - x.mean()) / \
            (x.std() + eps)  # 1e-6 to avoid zero division
        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        return screen_RGB
