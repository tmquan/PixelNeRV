import math
from typing import List

import torch
from pytorch3d.renderer import MonteCarloRaysampler, NDCGridRaysampler, NDCMultinomialRaysampler, RayBundle
from pytorch3d.renderer.cameras import CamerasBase

from pytorch3d.renderer.implicit.raysampling import _xy_to_ray_bundle
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf

from nerf.raysampler import NeRFRaysampler


def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


class SphereRaysampler(torch.nn.Module):
    """
    Samples rays from a full 3D sphere, so that we can visualize the whole 3D scene.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
        stratified: bool = False,
        stratified_test: bool = False,
    ):
        super().__init__()
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._image_width = image_width
        self._image_height = image_height
        pass

    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.
        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.
        Returns:
            n_chunks: The total number of chunks.
        """

        assert batch_size == 1
        n_chunks = math.ceil(
            self._image_height * self._image_width / chunksize
        )
        return n_chunks

        return 1

    def forward(
        self,
        cameras: CamerasBase,
        chunksize: int = None,
        chunk_idx: int = 0,
        camera_hash: str = None,
        caching: bool = False,
        target_sem_label=None,
        **kwargs,
    ) -> RayBundle:
        # we uniformly sample points on a unit sphere as directions
        # shape: (1, chunk_size, 3)
        theta = np.linspace(0, 2 * np.pi, self._image_width)
        phi = np.linspace(0, np.pi, self._image_height)
        x = np.outer(np.cos(theta), np.sin(phi))[..., None]
        y = np.outer(np.sin(theta), np.sin(phi))[..., None]
        z = np.outer(np.ones_like(theta), np.cos(phi))[..., None]
        dirs = np.concatenate((x, y, z), axis=-1)
        dirs = np.transpose(dirs, (1, 0, 2))  # (width, height, 3)
        dirs = dirs.reshape(-1, 3)[None]
        rays_directions_world_full = torch.cuda.FloatTensor(dirs)
        rays_directions_world = rays_directions_world_full[:, (
            chunksize * chunk_idx):(chunksize * (chunk_idx + 1))]
        real_chunk_size = rays_directions_world.shape[1]

        # origins are the center of the sphere, which is fixed
        rays_origins_world = cameras.get_camera_center()
        rays_origins_world = rays_origins_world[None].expand(
            1, real_chunk_size, 3)

        # zs uniformly sample points from z_near to z_far along the ray direction
        depths = torch.linspace(
            self._min_depth,
            self._max_depth,
            self._n_pts_per_ray,  # dtype=xy_grid.dtype, device=xy_grid.device
        )
        rays_zs = depths[None, None].expand(1, real_chunk_size, 64).cuda()
        #rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray)

        # xy_grid: it does not matter in our case
        xy_grid = torch.cuda.FloatTensor(
            [0.5, 0.5])[None, None].expand(1, real_chunk_size, 2)

        full_ray_bundle = RayBundle(
            rays_origins_world,  # .view(batch_size, *spatial_size, 3),
            rays_directions_world,  # .view(batch_size, *spatial_size, 3),
            rays_zs,  # .view(batch_size, *spatial_size, n_pts_per_ray),
            xy_grid,
        )

        # ray_bundle = RayBundle(
        #     *[
        #         v.view(n_pixels, -1)[sel_rays]
        #         .view(batch_size, sel_rays.numel() // batch_size, -1)
        #         .to(device)
        #         for v in full_ray_bundle
        #     ]
        # )

        return full_ray_bundle


class WeightedMonteCarloRaysampler(torch.nn.Module):
    """
    Samples a fixed number of pixels within denoted xy bounds uniformly at random.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        n_rays_per_image: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel.
            max_x: The largest x-coordinate of each ray's source pixel.
            min_y: The smallest y-coordinate of each ray's source pixel.
            max_y: The largest y-coordinate of each ray's source pixel.
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point.
        """
        super().__init__()
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        rays_xy = torch.cat(
            [
                torch.rand(
                    size=(batch_size, self._n_rays_per_image, 1),
                    dtype=torch.float32,
                    device=device,
                )
                * (high - low)
                + low
                for low, high in (
                    (self._min_x, self._max_x),
                    (self._min_y, self._max_y),
                )
            ],
            dim=2,
        )

        return _xy_to_ray_bundle(
            cameras, rays_xy, self._min_depth, self._max_depth, self._n_pts_per_ray, unit_directions=False,
        )


class PixelNeRFRaysampler(NeRFRaysampler):
    """
    Implements the raysampler of NeRF.
    Depending on the `self.training` flag, the raysampler either samples
    a chunk of random rays (`self.training==True`), or returns a subset of rays
    of the full image grid (`self.training==False`).
    The chunking of rays allows for efficient evaluation of the NeRF implicit
    surface function without encountering out-of-GPU-memory errors.
    Additionally, this raysampler supports pre-caching of the ray bundles
    for a set of input cameras (`self.precache_rays`).
    Pre-caching the rays before training greatly speeds-up the ensuing
    raysampling step of the training NeRF iterations.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
        stratified: bool = False,
        stratified_test: bool = False,
    ):
        """
        Args:
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
        """

        super().__init__(
            n_pts_per_ray,
            min_depth,
            max_depth,
            n_rays_per_image,
            image_width,
            image_height,
            stratified,
            stratified_test,
        )

        self._stratified = stratified
        self._stratified_test = stratified_test

        # Initialize the grid ray sampler.
        self._grid_raysampler = NDCMultinomialRaysampler(
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # Initialize the Monte Carlo ray sampler.
        self._mc_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # create empty ray cache
        self._ray_cache = {}

    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.
        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.
        Returns:
            n_chunks: The total number of chunks.
        """
        return int(
            math.ceil(
                (self._grid_raysampler._xy_grid.numel()  # type: ignore
                 * 0.5 * batch_size) / chunksize
            )
        )

    def _print_precaching_progress(self, i, total, bar_len=30):
        """
        Print a progress bar for ray precaching.
        """
        position = round((i + 1) / total * bar_len)
        pbar = "[" + "█" * position + " " * (bar_len - position) + "]"
        print(pbar, end="\r")

    def precache_rays(self, cameras: List[CamerasBase], camera_hashes: List):
        """
        Precaches the rays emitted from the list of cameras `cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `camera_hashes`.
        The cached rays are moved to cpu and stored in `self._ray_cache`.
        Raises `ValueError` when caching two cameras with the same hash.
        Args:
            cameras: A list of `N` cameras for which the rays are pre-cached.
            camera_hashes: A list of `N` unique identifiers of each
                camera from `cameras`.
        """
        print(f"Precaching {len(cameras)} ray bundles ...")
        full_chunksize = (
            self._grid_raysampler._xy_grid.numel()  # type: ignore
            // 2
            * self._grid_raysampler._n_pts_per_ray
        )
        if self.get_n_chunks(full_chunksize, 1) != 1:
            raise ValueError("There has to be one chunk for precaching rays!")
        for camera_i, (camera, camera_hash) in enumerate(zip(cameras, camera_hashes)):
            ray_bundle = self.forward(
                camera,
                caching=True,
                chunksize=full_chunksize,
            )
            if camera_hash in self._ray_cache:
                raise ValueError("There are redundant cameras!")
            self._ray_cache[camera_hash] = RayBundle(
                *[v.to("cpu").detach() for v in ray_bundle]
            )
            self._print_precaching_progress(camera_i, len(cameras))
        print("")

    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.
        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.
        Args:
            `ray_bundle`: The input `RayBundle`.
        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        z_vals = lower + (upper - lower) * torch.rand_like(lower)
        return ray_bundle._replace(lengths=z_vals)

    def _normalize_raybundle(self, ray_bundle: RayBundle):
        """
        Normalizes the ray directions of the input `RayBundle` to unit norm.
        """
        ray_bundle = ray_bundle._replace(
            directions=torch.nn.functional.normalize(
                ray_bundle.directions, dim=-1)
        )
        return ray_bundle

    def forward(
        self,
        cameras: CamerasBase,
        chunksize: int = None,
        chunk_idx: int = 0,
        camera_hash: str = None,
        caching: bool = False,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            chunksize: The number of rays per chunk.
                Active only when `self.training==False`.
            chunk_idx: The index of the ray chunk. The number has to be in
                `[0, self.get_n_chunks(chunksize, batch_size)-1]`.
                Active only when `self.training==False`.
            camera_hash: A unique identifier of a pre-cached camera. If `None`,
                the cache is not searched and the rays are calculated from scratch.
            caching: If `True`, activates the caching mode that returns the `RayBundle`
                that should be stored into the cache.
        Returns:
            A named tuple `RayBundle` with the following fields:
                origins: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the locations of ray origins in the world coordinates.
                directions: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the directions of each ray in the world coordinates.
                lengths: A tensor of shape
                    `(batch_size, n_rays_per_image, n_pts_per_ray)`
                    containing the z-coordinate (=depth) of each ray in world units.
                xys: A tensor of shape
                    `(batch_size, n_rays_per_image, 2)`
                    containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        if (camera_hash is None) and (not caching) and self.training:
            # Sample random rays from scratch.
            ray_bundle = self._mc_raysampler(cameras)
            ray_bundle = self._normalize_raybundle(ray_bundle)
        else:
            if camera_hash is not None:
                # The case where we retrieve a camera from cache.
                if batch_size != 1:
                    raise NotImplementedError(
                        "Ray caching works only for batches with a single camera!"
                    )
                full_ray_bundle = self._ray_cache[camera_hash]
            else:
                # We generate a full ray grid from scratch.
                full_ray_bundle = self._grid_raysampler(cameras)
                full_ray_bundle = self._normalize_raybundle(full_ray_bundle)

            n_pixels = full_ray_bundle.directions.shape[:-1].numel()

            if self.training:
                # During training we randomly subsample rays.
                sel_rays = torch.randperm(n_pixels, device=device)[
                    : self._mc_raysampler._n_rays_per_image
                ]
            else:
                # In case we test, we take only the requested chunk.
                if chunksize is None:
                    chunksize = n_pixels * batch_size
                start = chunk_idx * chunksize * batch_size
                end = min(start + chunksize, n_pixels)
                sel_rays = torch.arange(
                    start,
                    end,
                    dtype=torch.long,
                    device=full_ray_bundle.lengths.device,
                )
            # print(
            #     full_ray_bundle.directions.shape[:-1],
            #     n_pixels
            # )
            # print(
            #     len(full_ray_bundle),
            #     [v.shape for v in full_ray_bundle]
            # )
            # Take the "sel_rays" rays from the full ray bundle.
            ray_bundle = RayBundle(
                *[
                    v.view(n_pixels, -1)[sel_rays]
                    .view(batch_size, sel_rays.numel() // batch_size, -1)
                    .to(device)
                    for v in full_ray_bundle
                ]
            )
            # print(
            #     len(ray_bundle),
            #     [v.shape for v in full_ray_bundle]
            # )

        if (
            (self._stratified and self.training)
            or (self._stratified_test and not self.training)
        ) and not caching:  # Make sure not to stratify when caching!
            ray_bundle = self._stratify_ray_bundle(ray_bundle)

        return ray_bundle
