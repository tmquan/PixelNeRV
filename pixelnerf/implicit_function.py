# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import List, Tuple
from pytorch3d.renderer.cameras import CamerasBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import Rotate
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from nerf.implicit_function import NeRF, MLPWithInputSkips


# from .utils import repeat_interleave
def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in=3,
        d_out=4,
        n_blocks=5,
        d_latent=0,
        d_hidden=128,
        beta=0.0,
        combine_layer=3,  # TODO: try varying this?
        combine_type="average",
        use_spade=False,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]
        )

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            # Initialize the weights and biases of the linear layers
            # use before each residual connection
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(
                    self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
                )
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(
                        self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, x, z, multiple_inputs=False):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """

        # assert zx.size(-1) == self.d_latent + self.d_in
        if self.d_latent > 0:
            assert z.shape[-1] == self.d_latent
        else:
            raise ValueError("unsupported")
        if self.d_in > 0:
            x = self.lin_in(x)
        else:
            raise ValueError("unsupported")

        for blkid in range(self.n_blocks):
            if blkid == self.combine_layer:
                # The following implements camera frustum culling, requires torch_scatter
                #  if combine_index is not None:
                #      combine_type = (
                #          "mean"
                #          if self.combine_type == "average"
                #          else self.combine_type
                #      )
                #      if dim_size is not None:
                #          assert isinstance(dim_size, int)
                #      x = torch_scatter.scatter(
                #          x,
                #          combine_index,
                #          dim=0,
                #          dim_size=dim_size,
                #          reduce=combine_type,
                #      )
                #  else:
                # x = utils.combine_interleaved(
                #     x, combine_inner_dims, self.combine_type
                # )
                if multiple_inputs:
                    if self.combine_type == "average":
                        x = torch.mean(x, dim=1)
                    elif self.combine_type == "max":
                        x = torch.max(x, dim=1)[0]

            # Pass the skip input through the linear layer and then add to the oriignal input x
            if self.d_latent > 0 and blkid < self.combine_layer:
                tz = self.lin_z[blkid](z)
                if self.use_spade:
                    sz = self.scale_z[blkid](z)
                    x = sz * x + tz
                else:
                    x = x + tz

        # Pass through the resnet block
        x = self.blocks[blkid](x)
        out = self.lin_out(self.activation(x))
        return out


class PixelNeRF(NeRF):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: List[int] = (5,),
        # Settings for PixelNeRF
        transform_to_source_view: bool = True, 
        use_image_feats: bool = True,
        image_feature_dim: int = 128,
        resnetfc: bool = True, # Set to true for PixelNeRF MLP
        use_view_dirs: bool = True, # use camera view directions as input
        use_depth: bool = False,
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
        """
        super().__init__(
            n_harmonic_functions_xyz,
            n_harmonic_functions_dir,
            n_hidden_neurons_xyz,
            n_hidden_neurons_dir,
            n_layers_xyz,
            append_xyz,  # type: ignore
        )

        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3 + 3 #???
        if use_image_feats and not resnetfc:
            embedding_dim_xyz += image_feature_dim
    
        if resnetfc:
            # PixelNeRF MLP
            assert use_image_feats

            if use_view_dirs:
                # Add 3 for ray dirs without harmonic embedding
                # 63 -> 66
                embedding_dim_xyz += 3 

            # self.mlp_xyz = SemanticResnetFC(
            #     d_latent=image_feature_dim, 
            #     d_in=embedding_dim_xyz, 
            #     n_classes=n_classes, 
            #     use_depth=use_depth
            # )
            self.mlp_xyz = ResnetFC(
                d_latent=image_feature_dim, 
                d_in=embedding_dim_xyz, 
            )

        else:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers_xyz,
                embedding_dim_xyz,
                n_hidden_neurons_xyz,
                embedding_dim_xyz,
                n_hidden_neurons_xyz,
                input_skips=append_xyz,  # type: ignore
            )

            # self.semantic_layer = torch.nn.Sequential(
            #     torch.nn.Linear(n_hidden_neurons_xyz, n_hidden_neurons_dir),
            #     torch.nn.ReLU(True),
            #     torch.nn.Linear(n_hidden_neurons_dir, n_classes),
            #     # we're aggregating logits, no softmax.
            #     #torch.nn.Softmax(dim=3),
            # )

        self.resnetfc = resnetfc
        self.transform_to_source_view = transform_to_source_view
        self.use_image_feats = use_image_feats
        self.use_depth = use_depth
        self.use_view_dirs = use_view_dirs

    # def _get_semantics(self, features: torch.Tensor):
    #     semantic_pred = self.semantic_layer(features)
    #     return semantic_pred

    def _get_densities(
        self,
        raw_densities: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
    ):
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]
        if density_noise_std > 0.0:
            raw_densities = (
                raw_densities + torch.randn_like(raw_densities) * density_noise_std
            )
        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()
        return densities

    def forward(
        self,
        ray_bundle: RayBundle,
        density_noise_std: float = 0.0,
        cameras: CamerasBase = None,
        source_cameras: CamerasBase = None,
        source_image_feats: torch.Tensor = None,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        rays_dirs_world = ray_bundle.directions
        # rays_points_world.shape = [minibatch x ... x 3]
        #import pdb; pdb.set_trace()
        # print(rays_points_world.shape)
        N, num_rays, num_points_per_ray, _ = rays_points_world.shape    
        # N, num_width, num_height, num_points_per_ray, _ = rays_points_world.shape    
        # num_rays = num_width*num_height
        
        # source camera transforms
        RT_transform, projection_transform = get_camera_transforms(source_cameras)

        # source_cameras_N is actually N*NS so we can extract the num source views
        NS = int(source_cameras._N / N)

        if self.transform_to_source_view:
            # Transform points and directions to be in terms of the source view
            # instead of world coords.
            rays_points, rays_dirs = rays_world_to_view(
                rays_points=rays_points_world,
                rays_directions=rays_dirs_world,
                transform=RT_transform,
                num_views=NS
            )
        else:
            # Original NeRF, keep everything in world coordinates
            rays_points = rays_points_world
            rays_dirs = rays_dirs_world

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = torch.cat(
            (self.harmonic_embedding_xyz(rays_points), rays_points),
            dim=-1,
        )
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        if self.use_image_feats:
            # Sample source image aligned features for each ray point
            image_feats = sample_features(source_image_feats, rays_points, projection_transform)

            if not self.resnetfc:
                # Concat with ray embeddings
                embeds_xyz = torch.cat((embeds_xyz, image_feats), dim=-1) 

        if self.resnetfc:
            # PixelNeRF model
            # Concat ray dirs to ray point encoding without applying the harmonic embedding
            # Need to expand the shape to match the rays
            rays_dirs = rays_dirs[:, :, None, :].expand(-1, -1, num_points_per_ray, -1)
            if self.use_view_dirs:
                embeds_xyz = torch.cat((embeds_xyz, rays_dirs), dim=-1)
            if NS > 1:
                embeds_xyz = embeds_xyz.reshape(N, NS, *embeds_xyz.shape[1:])
                image_feats = image_feats.reshape(N, NS, *image_feats.shape[1:])
            
            # features, semantic_features = self.mlp_xyz(
            #     x=embeds_xyz, 
            #     z=image_feats, 
            #     multiple_inputs=NS > 1
            # )
            features = self.mlp_xyz(
                x=embeds_xyz, 
                z=image_feats, 
                multiple_inputs=NS > 1
            )
        
            # The output should be averaged across all source views 
            features = features.reshape(N, num_rays, num_points_per_ray, features.shape[-1])

            # Get ray colors and raw densities 
            rays_colors = features[..., :3] # rays_colors.shape = [minibatch x ... x 3] in [0-1]
            raw_rays_densities = features[..., 3:4] 

            rays_colors = torch.sigmoid(rays_colors)
            raw_rays_densities = torch.relu(raw_rays_densities)

            # NeRF equation (3)
            # convert density sigma to
            # 1 - exp(- delta * sigma) for volume rendering 
            rays_densities = self._get_densities(
                raw_rays_densities, ray_bundle.lengths, density_noise_std
            )
            # rays_densities = raw_rays_densities

            # # semantics
            # rays_semantics = semantic_features

            # depth
            rays_depths = None
            if self.use_depth:
                rays_depths = ray_bundle.lengths.unsqueeze(-1)

        else:
            # self.mlp maps each harmonic embedding to a latent feature space.
            features = self.mlp_xyz(embeds_xyz, embeds_xyz)
            # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

            raw_densities = self.density_layer(features)
            rays_densities = self._get_densities(
                raw_densities, ray_bundle.lengths, density_noise_std
            )
            # rays_densities.shape = [minibatch x ... x 1] in [0-1]

            rays_colors = self._get_colors(features, rays_dirs)
            # rays_colors.shape = [minibatch x ... x 3] in [0-1]

            # rays_semantics = self._get_semantics(features)

        # shape: [minibatch x ... x (3 + num_classes)]
        # if rays_depths is None:
        #     rays_features = torch.cat([rays_colors, rays_semantics], dim=3)
        # else:
        #     rays_features = torch.cat([rays_colors, rays_semantics, rays_depths], dim=3)
        if rays_depths is None:
            rays_features = torch.cat([rays_colors], dim=3)
        else:
            rays_features = torch.cat([rays_colors, rays_depths], dim=3)

        return rays_densities, rays_colors
        # return rays_densities, rays_features

    def forward_mesh(
        self,
        rays_points: torch.Tensor = None,
        rays_dirs: torch.Tensor = None,
        source_cameras: CamerasBase = None,
        source_image_feats: torch.Tensor = None,
        **kwargs,
    ):
        """
        we use query_pts to generate mesh directly.
        """
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = torch.cat(
            (self.harmonic_embedding_xyz(rays_points), rays_points),
            dim=-1,
        )
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]
        RT_transform, projection_transform = get_camera_transforms(source_cameras)

        if self.use_image_feats:
            # Sample source image aligned features for each ray point
            image_feats = sample_features(source_image_feats, rays_points, projection_transform)

            if not self.resnetfc:
                # Concat with ray embeddings
                embeds_xyz = torch.cat((embeds_xyz, image_feats), dim=-1) 

        if self.resnetfc:
            # PixelNeRF model
            # Concat ray dirs to ray point encoding without applying the harmonic embedding
            embeds_xyz = torch.cat((embeds_xyz, rays_dirs), dim=-1)

            # model
            # features, semantic_features = self.mlp_xyz(x=embeds_xyz, z=image_feats, multiple_inputs=False)
            features = self.mlp_xyz(x=embeds_xyz, z=image_feats, multiple_inputs=False)
        
            # The output should be averaged across all source views 
            # features = features.reshape(N, num_rays, num_points_per_ray, features.shape[-1])

            # Get ray colors and raw densities 
            rays_colors = features[..., :3] # rays_colors.shape = [minibatch x ... x 3] in [0-1]
            raw_rays_densities = features[..., 3:4] 

            rays_colors = torch.sigmoid(rays_colors)
            raw_rays_densities = torch.relu(raw_rays_densities)

            # rays_densities = self._get_densities(
            #    raw_rays_densities, 6000, 0.0
            # )

            rays_densities = raw_rays_densities
            # rays_semantics = semantic_features

        else:
            # self.mlp maps each harmonic embedding to a latent feature space.
            features = self.mlp_xyz(embeds_xyz, embeds_xyz)
            # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

            raw_densities = self.density_layer(features)
            # rays_densities = self._get_densities(
            #    raw_densities, ray_bundle.lengths, density_noise_std
            # )
            # rays_densities.shape = [minibatch x ... x 1] in [0-1]
            rays_densities = raw_densities

            rays_colors = self._get_colors(features, rays_dirs)
            # rays_colors.shape = [minibatch x ... x 3] in [0-1]

            # rays_semantics = self._get_semantics(features)

        # shape: [minibatch x ... x (3 + 102)]
        # rays_features = torch.cat([rays_colors, rays_semantics], dim=3)

        return rays_densities, rays_colors


def get_camera_transforms(camera: CamerasBase):
    RT = camera.get_world_to_view_transform()
    P = camera.get_projection_transform()
    return RT, P


def sample_features(image_feats,  rays_points,  projection_transform) -> torch.Tensor:
    """
    Get pixel-aligned image features at 2D image coordinates
    Args:
        image_feats: (N, D, H, W) tensor of features per pixel
        rays_points: (N, num_rays, num_points_per_ray, 3) points in 
            view coords of the source image
        projection_transform: Transforms3D object with the transform to 
            project the ray points onto the image plane. 
        image_size image size, either (width, height) or single int.
    
    Returns:
        latent_features: (N, num_rays, num_points_per_ray, D) image
            aligned features per ray point
    """
    # (B, C, H, W)
    latent_dim = image_feats.shape[1]
    # latent_scaling = torch.empty(2, dtype=torch.float32, device=image_feats.device)
    # latent_scaling[0] = image_feats.shape[-1]
    # latent_scaling[1] = image_feats.shape[-2]
    # latent_scaling = latent_scaling / (latent_scaling - 1) * 2.0

    rays_points_shape = rays_points.shape
    N = rays_points_shape[0]

    # (N, num_rays, num_points_per_ray, 3) -> (N, P, 3)
    # where P: num_rays * num_points_per_ray
    rays_points_reshape = rays_points.reshape(N, -1, 3)

    # Project into image
    projected_ray_points = projection_transform.transform_points(rays_points_reshape, eps=1e-2)

    pmin, pmax = projected_ray_points.min(), projected_ray_points.max()
    
    # Only need the uv, not the z
    uv = projected_ray_points[...,:2]

    # flip the sign to convert from pytorch3d to grid_sample convention
    uv = -uv
    
    # # If projected points are not in the [-1, 1] range.
    # if pmin < -1.0 or pmax > 1.0:
    #     # latent scaling to ensure uvs are in the range of the latent values 
    #     scale = latent_scaling / torch.tensor(image_size).type_as(image_feats)
    #     uv = uv * scale - 1.0

    #  (N, P, 2) -> (N, P, 1, 2)
    uv = uv.unsqueeze(2)

    ray_points_features = F.grid_sample(
        image_feats,
        uv,
        align_corners=True,
        mode="bilinear",
        # padding_mode="border",
    )
    
    # (N, latent_dim, P, 1) -> (N, latent_dim, P) 
    ray_points_features = ray_points_features.reshape(ray_points_features.shape[:-1])
    # (N, latent_dim, P) -> (N, P, latent_dim) 
    ray_points_features = ray_points_features.permute(0, 2, 1)
    # (N, P, latent_dim) -> (B, num_rays, num_points_per_ray, D)
    ray_points_features = ray_points_features.reshape(rays_points_shape[:-1] + (latent_dim,))

    return ray_points_features


def rays_world_to_view(
    rays_points, rays_directions, transform, num_views
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform the ray points and directions from world coordinates
    to be in terms of the source view coordinates. 
    Args:
        rays_points: (N, num_rays, num_points_per_ray, 3) tensor
            of ray points in world coords
        rays_directions: (N, num_rays, 3), tensor of ray directions
        transform: Transforms3D object with the transform to
            apply to the ray points/directions
    Returns:
        rays_points_view: (N*NS, num_rays, num_points_per_ray, 3)
        rays_directions_view: (N*NS, num_rays, 3)
    """
    rays_points_shape = tuple(rays_points.shape)
    N = rays_points_shape[0]
    NS = num_views
    rays_dirs_shape = rays_directions.shape

    # Reshape if there are multiple source views:
    if NS > 1:
        # (N, num_rays, num_points_per_ray, 3) -> (N*NS, num_rays, num_points_per_ray, 3)
        rays_points = repeat_interleave(rays_points, NS)
        # (N, num_rays, 3) -> (N*NS, num_rays, 3)
        rays_directions = repeat_interleave(rays_directions, NS)

    # Reshape from (N*NS, num_rays, num_points_per_ray, 3) -> (N, P, 3)
    # where P = num_rays * num_points_per_ray
    rays_points_reshape = rays_points.reshape(N*NS, -1, 3)

    # Transform the points, output shape -> (N*NS, P, 3)
    rays_points_view = transform.transform_points(rays_points_reshape)

    # Reshape back from (N*NS, P, 3) -> (N*NS, num_rays, num_points_per_ray, 3)
    rays_points_view = rays_points_view.reshape(N*NS, *rays_points_shape[1:])

    # Transform the viewing direction by rotating only
    R = transform.get_matrix()[:,:3,:3]
    rotation = Rotate(R=R, device=transform.device)

    # Reshape from (N*NS, num_rays, num_points_per_ray, 3) -> (N*NS, P, 3)
    rays_dirs_reshape = rays_directions.reshape(N*NS, -1, 3)

    # Transform to view coords, output shape (N*NS, P, 3)
    rays_dirs_view = rotation.transform_normals(rays_dirs_reshape)

    # Reshape back from (N*NS, P, 3) -> (N*NS, num_rays, num_points_per_ray, 3)
    rays_dirs_view = rays_dirs_view.reshape(N*NS, *rays_dirs_shape[1:])

    return rays_points_view, rays_dirs_view

