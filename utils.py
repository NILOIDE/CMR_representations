from typing import Union, Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F

MEAN_SAX_LV_VALUE = 222.7909
MAX_SAX_VALUE = 487.0
MEAN_4CH_LV_VALUE = 224.8285
MAX_4CH_LV_VALUE = 473.0


def normalize_image(im: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize array to range [0, 1] """
    min_, max_ = 0.0, im.max()
    im_ = (im - min_) / (max_ - min_)
    return im_


def normalize_image_with_percentile(im: Union[np.ndarray, torch.Tensor], percentile=99.9) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize array to range [0, Nth percentile] """
    min_, max_ = 0.0, np.percentile(im, percentile)
    im_ = (im - min_) / (max_ - min_)
    return im_.clip(0.0, 1.0)


def normalize_image_with_mean_lv_value(im: Union[np.ndarray, torch.Tensor], mean_value=MEAN_SAX_LV_VALUE, target_value=0.5) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize such that LV pool has value of 0.5. Assumes min value is 0.0. """
    return im / (mean_value / target_value)


def fast_trilinear_interpolation(input_array: torch.Tensor,
                                 y_indices: torch.Tensor,
                                 x_indices: torch.Tensor,
                                 z_indices: torch.Tensor) -> torch.Tensor:
    """ Trilinear interpolation of a batch of 3D volumes.
     :param input_array: Images used as source for the sampling.                Shape: (batch, height, width, depth)
     :param y_indices: Indices of the 1st spatial dimension of a given image.   Shape: (batch, num_points)
     :param x_indices: Input image of shape (batch, height, width, depth)       Shape: (batch, num_points)
     :param z_indices: Input image of shape (batch, height, width, depth)       Shape: (batch, num_points)
     """
    x0 = torch.floor(y_indices.detach()).to(torch.long)
    y0 = torch.floor(x_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)

    x = y_indices - x0
    y = x_indices - y0
    z = z_indices - z0

    b, _ = torch.meshgrid(torch.arange(0, x.shape[0], device=x.device),
                          torch.arange(0, x.shape[1], device=x.device))
    b_ = b.reshape(-1)
    x0_ = x0.reshape(-1)
    x1_ = x1.reshape(-1)
    y0_ = y0.reshape(-1)
    y1_ = y1.reshape(-1)
    z0_ = z0.reshape(-1)
    z1_ = z1.reshape(-1)
    x_ = x.reshape(-1)
    y_ = y.reshape(-1)
    z_ = z.reshape(-1)
    output_ = (
        input_array[b_, x0_, y0_, z0_] * (1 - x_) * (1 - y_) * (1 - z_) +
        input_array[b_, x1_, y0_, z0_] * x_ * (1 - y_) * (1 - z_) +
        input_array[b_, x0_, y1_, z0_] * (1 - x_) * y_ * (1 - z_) +
        input_array[b_, x0_, y0_, z1_] * (1 - x_) * (1 - y_) * z_ +
        input_array[b_, x1_, y0_, z1_] * x_ * (1 - y_) * z_ +
        input_array[b_, x0_, y1_, z1_] * (1 - x_) * y_ * z_ +
        input_array[b_, x1_, y1_, z0_] * x_ * y_ * (1 - z_) +
        input_array[b_, x1_, y1_, z1_] * x_ * y_ * z_
    )
    output = output_.reshape(x0.shape)
    return output


def flip_affine(affines, needs_flip):
    # If the original affine had a determinant is <= 0, it is an umproper affine matrix and it needs to be flipped
    needs_flip = needs_flip[:, None, None].repeat((1, 4, 4))
    flip = torch.eye(3, dtype=affines.dtype, device=affines.device)
    flip[0, 0] = -1
    flip = flip.repeat((affines.shape[0], 1, 1))
    affines_flipped = affines.clone()
    affines_flipped[:, :3, :3] = torch.bmm(flip, affines[:, :3, :3])
    affines_flipped[:, :3, 3:] = torch.bmm(flip, affines[:, :3, 3:])
    affines = torch.where(needs_flip, affines_flipped, affines)
    return affines


def mat_to_params(affines, spacings, needs_flip, cy_thresh=1e-3):
    affines = flip_affine(affines, needs_flip)
    affines[:, :3, :3] = torch.bmm(affines[:, :3, :3], torch.diag_embed(1 / spacings))
    translation = affines[:, :3, 3]

    # The rotation euler params are extracted following nibabel's mat2euler
    cy = torch.sqrt(affines[:, 2, 2] * affines[:, 2, 2] + affines[:, 1, 2] * affines[:, 1, 2])  # math.sqrt(r33 * r33 + r23 * r23)

    z = torch.atan2(-affines[:, 0, 1], affines[:, 0, 0])
    y = torch.atan2(affines[:, 0, 2], cy)
    x = torch.atan2(-affines[:, 1, 2], affines[:, 2, 2])

    z_eps = torch.atan2(-affines[:, 1, 0], affines[:, 1, 1])
    x_eps = torch.zeros_like(x)

    z = torch.where(cy > cy_thresh, z, z_eps)
    x = torch.where(cy > cy_thresh, x, x_eps)
    rotation = torch.stack((z, y, x), dim=1)

    params = torch.cat((rotation, translation), 1)
    return params


def params_to_mat(params, spacings, needs_flip):
    assert params.shape[1] == 6
    rotation, translation = params[:, :3], params[:, 3:]
    cos = torch.cos(rotation)
    sin = torch.sin(rotation)
    rotation_z = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_z[:, 0, 0] = cos[:, 0]
    rotation_z[:, 0, 1] = -sin[:, 0]
    rotation_z[:, 1, 0] = sin[:, 0]
    rotation_z[:, 1, 1] = cos[:, 0]

    rotation_y = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_y[:, 0, 0] = cos[:, 1]
    rotation_y[:, 0, 2] = sin[:, 1]
    rotation_y[:, 2, 0] = -sin[:, 1]
    rotation_y[:, 2, 2] = cos[:, 1]

    rotation_x = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_x[:, 1, 1] = cos[:, 2]
    rotation_x[:, 1, 2] = -sin[:, 2]
    rotation_x[:, 2, 1] = sin[:, 2]
    rotation_x[:, 2, 2] = cos[:, 2]

    rotation = torch.bmm(rotation_x, rotation_y)
    rotation = torch.bmm(rotation, rotation_z)
    rotation = torch.bmm(rotation, torch.diag_embed(spacings))

    affines = torch.eye(4, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    affines[:, :3, :3] = rotation
    affines[:, :3, 3] = translation
    affines = flip_affine(affines, needs_flip)
    return affines


def make_coordinate_tensor(shape):
    """Make a coordinate tensor."""
    coordinate_tensor = [torch.arange(0, i) for i in shape]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=len(shape))
    return coordinate_tensor


def make_masked_coordinate_tensor(mask):
    """Make a coordinate tensor."""
    coordinate_tensor = make_coordinate_tensor(mask.shape)
    coordinate_tensor = coordinate_tensor.reshape([np.prod(mask.shape), len(mask.shape)])
    coordinate_tensor = coordinate_tensor[mask.flatten(), :]
    return coordinate_tensor


def to_1hot(class_indices: torch.Tensor, num_class) -> torch.Tensor:
    """ Converts index array to 1-hot structure. """
    assert len(class_indices.shape) in {1, 2},  "Shape should be either (batch, 1) or (batch,)"
    if len(class_indices.shape) == 2:
        class_indices = class_indices.squeeze(1)
    N = class_indices.shape[0]
    seg = class_indices.to(torch.long).reshape((-1,))
    seg_1hot = torch.zeros((N, num_class), dtype=torch.float32, device=class_indices.device)
    seg_1hot[torch.arange(0, seg.shape[0], dtype=torch.long), seg] = 1
    return seg_1hot


def compute_neighbourhood_matrix(coords: torch.Tensor, ref_coords: torch.Tensor, dist_thresh: float = 0.15):
    assert len(coords.shape) == 2
    assert len(ref_coords.shape) == 2
    if dist_thresh < 0:
        return torch.zeros((coords.shape[0], coords.shape[0]), dtype=torch.uint8)
    dists = ref_coords[None].tile((coords.shape[0], 1, 1)) - coords[:, None].tile((1, ref_coords.shape[0], 1))
    dists = dists.norm(dim=-1)
    neigh_matrix = dists <= dist_thresh
    within_range_mask = neigh_matrix.any(dim=-1)
    return within_range_mask


def compute_3d_image_gradients(volume):
    def get_3d_central_diff_kernel(device, dtype):
        kernel = torch.zeros((3, 1, 3, 3, 3), device=device, dtype=dtype)
        # Derivative along x (axis 2 -> W dimension)
        kernel[2, 0, 1, 1, 0] = -0.5
        kernel[2, 0, 1, 1, 2] = +0.5
        # Derivative along y (axis 1 -> H dimension)
        kernel[1, 0, 1, 0, 1] = -0.5
        kernel[1, 0, 1, 2, 1] = +0.5
        # Derivative along z (axis 0 -> D dimension)
        kernel[0, 0, 0, 1, 1] = -0.5
        kernel[0, 0, 2, 1, 1] = +0.5

        return kernel
    # volume: (B, 1, D, H, W)
    device, dtype = volume.device, volume.dtype
    kernel = get_3d_central_diff_kernel(device, dtype)
    gradients = F.conv3d(volume, kernel, padding=1)  # Pad to keep size
    return gradients # (B, 3, D, H, W)


def compute_3d_image_gradients_dim_wise(volume):
    def get_3d_central_difference_kernels(device, dtype):
        # x derivative
        kernel_x = torch.zeros((1, 1, 3, 1, 1), device=device, dtype=dtype)
        kernel_x[0, 0, :, 0, 0] = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype)

        # y derivative
        kernel_y = torch.zeros((1, 1, 1, 3, 1), device=device, dtype=dtype)
        kernel_y[0, 0, 0, :, 0] = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype)

        # z derivative
        kernel_z = torch.zeros((1, 1, 1, 1, 3), device=device, dtype=dtype)
        kernel_z[0, 0, 0, 0, :] = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype)

        return kernel_x, kernel_y, kernel_z
    # volume: (B, 1, D, H, W)
    device, dtype = volume.device, volume.dtype
    kernel_x, kernel_y, kernel_z = get_3d_central_difference_kernels(device, dtype)

    grad_x = F.conv3d(volume, kernel_x, padding=(1, 0, 0))
    grad_y = F.conv3d(volume, kernel_y, padding=(0, 1, 0))
    grad_z = F.conv3d(volume, kernel_z, padding=(0, 0, 1))

    return grad_x, grad_y, grad_z



def compute_3d_image_hessian(volume):
    device, dtype = volume.device, volume.dtype
    # Define second derivative kernels
    kernel = torch.zeros((6, 1, 3, 3, 3), device=device, dtype=dtype)
    # f_xx
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    kernel[0, 0, 1, 1, 1] = -2.0
    # f_yy
    kernel[1, 0, 1, 0, 1] = 1.0
    kernel[1, 0, 1, 2, 1] = 1.0
    kernel[1, 0, 1, 1, 1] = -2.0
    # f_zz
    kernel[2, 0, 0, 1, 1] = 1.0
    kernel[2, 0, 2, 1, 1] = 1.0
    kernel[2, 0, 1, 1, 1] = -2.0
    # f_xy
    kernel[3, 0, 1, 0, 0] =  0.25
    kernel[3, 0, 1, 2, 2] =  0.25
    kernel[3, 0, 1, 0, 2] = -0.25
    kernel[3, 0, 1, 2, 0] = -0.25
    # f_xz
    kernel[4, 0, 0, 1, 0] =  0.25
    kernel[4, 0, 2, 1, 2] =  0.25
    kernel[4, 0, 0, 1, 2] = -0.25
    kernel[4, 0, 2, 1, 0] = -0.25
    # f_yz
    kernel[5, 0, 0, 0, 1] =  0.25
    kernel[5, 0, 2, 2, 1] =  0.25
    kernel[5, 0, 0, 2, 1] = -0.25
    kernel[5, 0, 2, 0, 1] = -0.25
    # Apply convolution
    hessian = F.conv3d(volume, kernel, padding=1)
    return hessian  # (B, 6, D, H, W)


def temporal_median_filter(video, kernel_size=3):
    """Apply temporal median filter along time dimension with sliding window."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    # video: (B, C, H, W, T)
    B, C, H, W, T = video.shape
    # Loop-around padding
    video = torch.cat((video[...,-(kernel_size//2):], video, video[...,:kernel_size//2]), -1)
    # Unfold time axis
    unfolded = video.unfold(dimension=-1, size=kernel_size, step=1)  # (B, C, H, W, T_new, kernel_size)
    # Take median along kernel_size dimension
    median = unfolded.median(dim=-1).values  # (B, C, H, W, T_new)
    return median

def spatial_median_filter(video, kernel_size=(3, 3, 1)):
    """Apply temporal median filter along time dimension with sliding window."""
    # video: (B, C, H, W, T)
    B, C, H, W, T = video.shape
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*len(video.shape[2:])
    if any([k % 2 == 0 for k in kernel_size]):
        raise ValueError(f"kernel_size must be odd: {kernel_size}")
    # Loop-around padding
    video_pad = F.pad(video, (0,0, kernel_size[0]//2, kernel_size[0]//2, kernel_size[1]//2, kernel_size[1]//2, 0,0,0,0), mode='constant', value=0.0)
    video_pad = torch.cat((video_pad[...,-(kernel_size[2]//2):], video_pad, video_pad[...,:kernel_size[2]//2]), -1)
    # Unfold time axis
    unfolded = video_pad.unfold(dimension=2, size=kernel_size[0], step=1)  # (B, C, H_new, W, T, k)
    unfolded = unfolded.unfold(dimension=3, size=kernel_size[1], step=1)  # (B, C, H_new, W_new, T, k, k)
    unfolded = unfolded.unfold(dimension=4, size=kernel_size[2], step=1)  # (B, C, H,_new W_new, T_new, k, k, k)
    # Take median along kernel_size dimension
    unfolded_ = unfolded.reshape(B, C, H, W, T, -1)
    median = unfolded_.median(dim=-1).values  # (B, C, H, W, T)
    return median