import torch
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)


class EmissionAbsorptionFrontToBackRaymarcher(EmissionAbsorptionRaymarcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
         
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities.flip(dims=(-1,)), shift=-self.surface_thickness
        ).flip(dims=(-1,))  # Reverse the direction of the absorption to match X-ray detector
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - \
            torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        return torch.cat((features, opacities), dim=-1)
