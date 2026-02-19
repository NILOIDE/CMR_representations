from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict

import cv2
import numpy as np
import scipy
import skimage
import torch
import torch.nn.functional as F
import meshplot as mp
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

MEAN_SAX_LV_VALUE = 222.7909
MAX_SAX_VALUE = 487.0
MEAN_4CH_LV_VALUE = 224.8285
MAX_4CH_LV_VALUE = 473.0


def get_center_coord(segmentation_map):
    # Get the largest contour
    contours, _ = cv2.findContours(segmentation_map.numpy().astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return torch.tensor((cy, cx))
        raise ValueError('Invalid contour.')
    raise ValueError('No foreground found.')


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

@torch.jit.script
def fast_trilinear_interpolation(input_array: torch.Tensor,
                                 y_indices: torch.Tensor,
                                 x_indices: torch.Tensor,
                                 z_indices: torch.Tensor) -> torch.Tensor:
    """
        Optimized Trilinear interpolation using Flat Indexing.
        Shape assumptions:
          input_array: (Batch, Height, Width, Depth, Channels)
          indices:     (Batch, Num_Points)
        """
    # 1. Get dimensions
    B, H, W, D, C = input_array.shape

    # 2. Compute floor coordinates (integers)
    # We detach to ensure no gradients flow into the indices
    x0 = torch.floor(y_indices.detach()).to(torch.long)
    y0 = torch.floor(x_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)

    # 3. Clamp to volume boundaries
    # (Using clamp is faster than modulo or manual checks)
    x0 = torch.clamp(x0, 0, H - 1)
    y0 = torch.clamp(y0, 0, W - 1)
    z0 = torch.clamp(z0, 0, D - 1)

    x1 = torch.clamp(x0 + 1, 0, H - 1)
    y1 = torch.clamp(y0 + 1, 0, W - 1)
    z1 = torch.clamp(z0 + 1, 0, D - 1)

    # 4. Compute fractional weights
    # Gradients MUST flow through these
    wa = y_indices - x0
    wb = x_indices - y0
    wc = z_indices - z0

    # Expand weights for broadcasting with channels later: (Batch*Points, 1)
    # We flatten them here to match the flat index shape
    wa = wa.reshape(-1, 1)
    wb = wb.reshape(-1, 1)
    wc = wc.reshape(-1, 1)

    # 5. Compute Linear Indices
    # Instead of vol[b, x, y, z], we use vol_flat[idx]
    # Stride calculation:
    stride_b = H * W * D
    stride_h = W * D
    stride_w = D

    # Create batch offsets
    # num_points = x0.shape[1]
    # b_idx = torch.arange(B, device=input_array.device).view(B, 1).expand(-1, num_points).reshape(-1)

    # Efficient Batch Index generation:
    # We assume indices are (Batch, N). We can infer N from x0.
    N = x0.shape[1]
    b_idx = torch.arange(B, device=input_array.device).repeat_interleave(N)

    # Flatten spatial indices
    x0f = x0.reshape(-1)
    y0f = y0.reshape(-1)
    z0f = z0.reshape(-1)
    x1f = x1.reshape(-1)
    y1f = y1.reshape(-1)
    z1f = z1.reshape(-1)

    # Pre-compute base offsets for the corners to save adds
    base_00 = b_idx * stride_b + x0f * stride_h + y0f * stride_w
    base_01 = b_idx * stride_b + x0f * stride_h + y1f * stride_w
    base_10 = b_idx * stride_b + x1f * stride_h + y0f * stride_w
    base_11 = b_idx * stride_b + x1f * stride_h + y1f * stride_w

    # Calculate final 1D indices for all 8 corners
    i000 = base_00 + z0f
    i001 = base_00 + z1f
    i010 = base_01 + z0f
    i011 = base_01 + z1f
    i100 = base_10 + z0f
    i101 = base_10 + z1f
    i110 = base_11 + z0f
    i111 = base_11 + z1f

    # 6. Gather values
    # Flatten input to (Batch * H * W * D, Channels)
    vol_flat = input_array.reshape(-1, C)

    v000 = vol_flat[i000]
    v001 = vol_flat[i001]
    v010 = vol_flat[i010]
    v011 = vol_flat[i011]
    v100 = vol_flat[i100]
    v101 = vol_flat[i101]
    v110 = vol_flat[i110]
    v111 = vol_flat[i111]

    # 7. Interpolate
    # (1-w) * v0 + w * v1
    c00 = v000 * (1 - wa) + v100 * wa
    c01 = v010 * (1 - wa) + v110 * wa
    c10 = v001 * (1 - wa) + v101 * wa
    c11 = v011 * (1 - wa) + v111 * wa

    c0 = c00 * (1 - wb) + c01 * wb
    c1 = c10 * (1 - wb) + c11 * wb

    c = c0 * (1 - wc) + c1 * wc

    # 8. Reshape to output
    # Output shape: (Batch, Num_Points, Channels)
    return c.reshape(B, N, C)


def fast_4Dlinear_interpolation(input_array: torch.Tensor,
                                 y_indices: torch.Tensor,
                                 x_indices: torch.Tensor,
                                 z_indices: torch.Tensor,
                                 t_indices: torch.Tensor) -> torch.Tensor:
    """ 4D-linear interpolation of a batch of 4D volumes.
     :param input_array: Images used as source for the sampling.                Shape: (batch, height, width, depth)
     :param y_indices:                                                          Shape: (batch, num_points)
     :param x_indices:                                                          Shape: (batch, num_points)
     :param z_indices:                                                          Shape: (batch, num_points)
     :param t_indices:                                                          Shape: (batch, num_points)
     """
    x0 = torch.floor(y_indices.detach()).to(torch.long)
    y0 = torch.floor(x_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    t0 = torch.floor(t_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    t1 = t0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    t0 = torch.clamp(t0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)
    t1 = torch.clamp(t1, 0, input_array.shape[3] - 1)

    x = y_indices - x0
    y = x_indices - y0
    z = z_indices - z0
    t = t_indices - t0

    b, _ = torch.meshgrid(torch.arange(0, x.shape[0], device=x.device),
                          torch.arange(0, x.shape[1], device=x.device))
    b_ = b.reshape(-1)
    x0_ = x0.reshape(-1)
    x1_ = x1.reshape(-1)
    y0_ = y0.reshape(-1)
    y1_ = y1.reshape(-1)
    z0_ = z0.reshape(-1)
    z1_ = z1.reshape(-1)
    t0_ = t0.reshape(-1)
    t1_ = t1.reshape(-1)
    x_ = x.reshape(-1, 1)
    y_ = y.reshape(-1, 1)
    z_ = z.reshape(-1, 1)
    t_ = t.reshape(-1, 1)
    output_ = (
        # 4
        input_array[b_, x0_, y0_, z0_, t0_] * (1 - x_) * (1 - y_) * (1 - z_) * (1 - t_) +
        # 3
        input_array[b_, x1_, y0_, z0_, t0_] * x_ * (1 - y_) * (1 - z_) * (1 - t_) +
        input_array[b_, x0_, y1_, z0_, t0_] * (1 - x_) * y_ * (1 - z_) * (1 - t_) +
        input_array[b_, x0_, y0_, z1_, t0_] * (1 - x_) * (1 - y_) * z_ * (1 - t_) +
        input_array[b_, x0_, y0_, z0_, t1_] * (1 - x_) * (1 - y_) * (1 - z_) * t_ +
        # 2
        input_array[b_, x1_, y1_, z0_, t0_] * x_ * y_ * (1 - z_) * (1 - t_) +
        input_array[b_, x1_, y0_, z1_, t0_] * x_ * (1 - y_) * z_ * (1 - t_) +
        input_array[b_, x1_, y0_, z0_, t1_] * x_ * (1 - y_) * (1 - z_) * t_ +
        input_array[b_, x0_, y1_, z1_, t0_] * (1 - x_) * y_ * z_ * (1 - t_) +
        input_array[b_, x0_, y1_, z0_, t1_] * (1 - x_) * y_ * (1 - z_) * t_ +
        input_array[b_, x0_, y0_, z1_, t1_] * (1 - x_) * (1 - y_) * z_ * t_ +
        # 1
        input_array[b_, x0_, y1_, z1_, t1_] * (1 - x_) * y_ * z_ * t_ +
        input_array[b_, x1_, y0_, z1_, t1_] * x_ * (1 - y_) * z_ * t_ +
        input_array[b_, x1_, y1_, z0_, t1_] * x_ * y_ * (1 - z_) * t_ +
        input_array[b_, x1_, y1_, z1_, t0_] * x_ * y_ * z_ * (1 - t_) +
        # 0
        input_array[b_, x1_, y1_, z1_, t1_] * x_ * y_ * z_ * t_
    )
    output = output_.reshape(*x0.shape, input_array.shape[-1])
    return output


def fast_nearest_neighbor_interpolation(input_array: torch.Tensor,
                                        y_indices: torch.Tensor,
                                        x_indices: torch.Tensor,
                                        z_indices: torch.Tensor) -> torch.Tensor:
    """ Nearest neighbor interpolation of a batch of 3D volumes.
     :param input_array: Images used as source for the sampling.                Shape: (batch, height, width, depth)
     :param y_indices: Indices of the 1st spatial dimension of a given image.   Shape: (batch, num_points)
     :param x_indices: Input image of shape (batch, height, width, depth)       Shape: (batch, num_points)
     :param z_indices: Input image of shape (batch, height, width, depth)       Shape: (batch, num_points)
     """
    # Round to nearest integer instead of floor
    x_nearest = torch.round(y_indices).to(torch.long)
    y_nearest = torch.round(x_indices).to(torch.long)
    z_nearest = torch.round(z_indices).to(torch.long)

    # Clamp to valid range
    x_nearest = torch.clamp(x_nearest, 0, input_array.shape[1] - 1)
    y_nearest = torch.clamp(y_nearest, 0, input_array.shape[2] - 1)
    z_nearest = torch.clamp(z_nearest, 0, input_array.shape[3] - 1)

    # Create batch indices
    b, _ = torch.meshgrid(torch.arange(0, x_nearest.shape[0], device=x_nearest.device),
                          torch.arange(0, x_nearest.shape[1], device=x_nearest.device),
                          indexing='ij')
    b_ = b.reshape(-1)
    x_ = x_nearest.reshape(-1)
    y_ = y_nearest.reshape(-1)
    z_ = z_nearest.reshape(-1)

    # Simple indexing - no interpolation weights needed
    output_ = input_array[b_, x_, y_, z_]
    output = output_.reshape(x_nearest.shape)

    return output


def flip_affine(affines, needs_flip):
    # If the original affine had a determinant is <= 0, it is an improper affine matrix and it needs to be flipped
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


def make_coordinate_tensor(shape, device):
    """Make a coordinate tensor."""
    coordinate_tensor = [torch.arange(0, i, device=device) for i in shape]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=len(shape))
    return coordinate_tensor


def make_masked_coordinate_tensor(mask):
    """Make a coordinate tensor."""
    coordinate_tensor = make_coordinate_tensor(mask.shape, device=mask.device)
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


def compute_3d_image_gradients(volume):
    def get_3d_central_diff_kernel(device, dtype):
        kernel = torch.zeros((3, 1, 5, 5, 5), device=device, dtype=dtype)
        # Derivative along x (axis 2 -> W)
        kernel[2, 0, 2, 2, 0] = -1 / 12
        kernel[2, 0, 2, 2, 1] = +8 / 12
        kernel[2, 0, 2, 2, 3] = -8 / 12
        kernel[2, 0, 2, 2, 4] = +1 / 12
        # Derivative along y (axis 1 -> H)
        kernel[1, 0, 2, 0, 2] = -1 / 12
        kernel[1, 0, 2, 1, 2] = +8 / 12
        kernel[1, 0, 2, 3, 2] = -8 / 12
        kernel[1, 0, 2, 4, 2] = +1 / 12
        # Derivative along z (axis 0 -> D)
        kernel[0, 0, 0, 2, 2] = -1 / 12
        kernel[0, 0, 1, 2, 2] = +8 / 12
        kernel[0, 0, 3, 2, 2] = -8 / 12
        kernel[0, 0, 4, 2, 2] = +1 / 12

        return kernel * -1
    # volume: (B, 1, D, H, W)

    assert volume.dtype == torch.float, "Must be a [0,1] float tensor"
    assert not (volume[...,0] > 1.0).any(), "Must be a [0,1] float tensor"
    device, dtype = volume.device, volume.dtype
    volume_pad = torch.cat((volume[..., -2:], volume, volume[..., :2]), -1)
    volume_pad = F.pad(volume_pad, (0,0,2,2,2,2))  # F.pad takes padding sequence backwards
    kernel = get_3d_central_diff_kernel(device, dtype)
    gradients = F.conv3d(volume_pad, kernel, padding=0)  # Pad to keep size
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


def _denoise_single_frame(args: Tuple[List[np.ndarray], int, int, float, Optional[int], Optional[int]]) -> np.ndarray:
    frames_uint8_pad, frame_index, temporal_window_size, h, template_window_size, search_window_size = args
    denoised_frame = cv2.fastNlMeansDenoisingMulti(frames_uint8_pad, frame_index, temporal_window_size,
        h, template_window_size, search_window_size)
    return denoised_frame


def nlm_denoise_multi_parallel(
        frames: torch.Tensor,
        temporal_window_size: int,
        h: float,
        template_window_size: Optional[int] = None,
        search_window_size: Optional[int] = None,
        num_processes: Optional[int] = None
) -> torch.Tensor:
    """
    Parallel version of NLM denoising using multiprocessing

    Args:
        frames: Input tensor of shape (H, W, T)
        temporal_window_size: Number of frames to use for denoising
        h: Filter strength
        template_window_size: Template patch size
        search_window_size: Search window size
        num_processes: Number of processes to use (default: cpu_count())

    Returns:
        Denoised frames tensor of shape (H, W, T)
    """
    assert frames.dtype == torch.float, "Must be a [0,1] float tensor"
    assert not (frames[...,0] > 1.0).any(), "Must be a [0,1] float tensor"
    from multiprocessing import Pool, cpu_count
    if num_processes is None:
        num_processes = cpu_count()
    H, W, T = frames.shape
    # Convert to uint8 numpy arrays
    frames_uint8 = (frames * 255.).round().clip(0., 255.).to(torch.uint8)
    frames_uint8 = frames_uint8.numpy()
    frames_uint8 = [frames_uint8[..., i] for i in range(T)]
    # Pad frames for temporal window
    frames_uint8_pad = (frames_uint8[-(temporal_window_size // 2):] +
                        frames_uint8 +
                        frames_uint8[:(temporal_window_size // 2)])
    # Prepare arguments for each frame
    frame_args = [
        (frames_uint8_pad, i, temporal_window_size, h, template_window_size, search_window_size)
        for i in range(temporal_window_size // 2, T + (temporal_window_size // 2))
    ]
    # Process frames in parallel
    with Pool(processes=num_processes) as pool:
        frames_dn = pool.map(_denoise_single_frame, frame_args)
    # Stack results and convert back to torch tensor
    frames_dn = np.stack(frames_dn, -1)
    frames_dn = torch.from_numpy(frames_dn).to(torch.float32) / 255.
    assert frames_dn.shape == (H, W, T)
    return frames_dn


def to_gif(imgs, dir_name, name="arr"):
    from PIL import Image
    name = Path(dir_name) / name
    name.parent.mkdir(exist_ok=True)
    assert isinstance(imgs, torch.Tensor)
    assert imgs.dtype == torch.float32
    # assert not (imgs > 1.9).any()
    imgs = imgs.moveaxis(-1, 0)
    imgs = (imgs*255).round().clip(0.0,255.0).to(torch.uint8).numpy()
    # imgs = np.stack([imgs]*3, 1)
    imgs = [Image.fromarray(img) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(f"{str(name)}.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


def process_segmentation_with_marching_cubes(segmentation, spacing=(1.0, 1.0, 1.0),
                                             level=0.5, step_size=1, allow_degenerate=False):
    """
    Process 3D segmentation using marching cubes for each foreground class.

    Args:
        segmentation: numpy uint8 array of shape (D, H, W) with class labels
        spacing: tuple of voxel spacing in each dimension
        level: isosurface level for marching cubes
        step_size: step size for marching cubes (larger = faster, less detailed)
        allow_degenerate: whether to allow degenerate triangles

    Returns:
        dict: Dictionary containing mesh data for each class
    """
    H, W, D = segmentation.shape
    # Padding
    segmentation_pad = np.zeros((H + 2, W + 2, D + 2), np.uint8)
    segmentation_pad[1:-1, 1:-1, 1:-1] = segmentation
    # Get unique classes (excluding background class 0)
    unique_classes = np.unique(segmentation_pad)
    foreground_classes = unique_classes[unique_classes > 0]
    print(f"Found {len(foreground_classes)} foreground classes: {foreground_classes}")
    meshes = {}

    for class_id in foreground_classes:
        # Create binary mask for current class
        binary_mask = (segmentation_pad == class_id).astype(np.uint8)
        # Skip if class has too few voxels
        if np.sum(binary_mask) < 10:
            print(f"Skipping class {class_id}: too few voxels ({np.sum(binary_mask)})")
            continue
        try:
            # Apply marching cubes
            vertices, faces, normals, values = skimage.measure.marching_cubes(
                binary_mask,
                level=level,
                spacing=spacing,
                step_size=step_size,
                allow_degenerate=allow_degenerate
            )
            # Store mesh data
            meshes[class_id] = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'values': values,
                'n_vertices': len(vertices),
                'n_faces': len(faces),
                'volume': np.sum(binary_mask) * np.prod(spacing)
            }
            print(f"Class {class_id}: {len(vertices)} vertices, {len(faces)} faces")
        except Exception as e:
            print(f"Error processing class {class_id}: {e}")
            continue
    return meshes


def create_meshplot_visualization(meshes, file_name, colors=None):
    """
    Create meshplot visualization for multiple meshes.

    Args:
        meshes: dict of mesh data from process_segmentation_with_marching_cubes
        file_name:
        colors: optional list of colors for each class
    Returns:
        meshplot viewer object
    """
    mp.website()  # Initializing as website won't save plot to working dir
    # Default colors if not provided
    if colors is None:
        colors = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)))
    plot = None
    for idx, (class_id, mesh_data) in enumerate(meshes.items()):
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        # Select color
        color = colors[idx] if idx < len(colors) else np.random.rand(3)
        # Create or add to plot
        if plot is None:
            plot = mp.plot(vertices, faces, c=color[:3], return_plot=True)
        else:
            plot.add_mesh(vertices, faces, c=color[:3])
    plot.save(file_name)
    return plot


def video_array_to_file(array: Union[np.ndarray, torch.Tensor],
                        file_path: Union[str, Path], video_duration: float = 2.):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    if "." in file_path[-4:]:
        assert file_path[-4:] == ".mp4"
    else:
        file_path = file_path + ".mp4"
    if isinstance(array, torch.Tensor):
        array = array.numpy()

    # Convert to uint8 range [0, 255]
    if any([array.dtype == i for i in {np.float32, np.float64, float}]):
        assert not np.any(array > 1.0)
        array = (array * 255).astype(np.uint8)
    T, C, H, W = array.shape
    assert C in {1,3}, 'Channel dim should be of size 1 or 3'

    # Define video codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1', 'H264'
    fps = T // video_duration  # frames per second
    out = cv2.VideoWriter(file_path, fourcc, fps, (W, H))
    for t in range(T):
        frame = array[t]
        frame = np.transpose(frame, (1, 2, 0))
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def data_frame_to_line_plot(df, x, metric_name, subj_idx, save_path):
    assert df.shape[0] == len(x), f"Metric: {metric_name} appears to have a dataframe issue"
    plt.figure(figsize=(10, 6))
    # Plot each column
    for column in df.columns:
        plt.plot(list(x[column]), list(df[column]), label=column, marker='o')
    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(f'{str(subj_idx)}_{metric_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


def draw_3d_vectors_on_image(
        image,
        vectors,
        vector_downsample=9,
        upscale_factor=10,
        arrow_thickness=0.5,
        arrow_head_length=0.1):
    """
    Draw 3D vectors on an image using OpenCV, encoding out-of-plane direction with red-blue color.

    Parameters
    ----------
    image : np.ndarray
        The base image (H, W, 3), (H, W, 1) or (H, W) numpy array.
    vectors : np.ndarray
        Array of shape (H, W, 3) numpy array giving vector components (vx, vy, vz).
    vector_downsample : int
        Downsampling of vector array to have vectors be more spaced out.
    Returns
    -------
    overlay : np.ndarray
        The image with arrows drawn (uint8 BGR) (H, W, 3).
    """

    # Ensure 3-channel BGR
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        overlay = cv2.cvtColor(image[...,0], cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    H, W, C = overlay.shape
    if overlay.dtype != np.uint8:
        if overlay.dtype == np.float32 or overlay.dtype == float or overlay.dtype == np.float64:
            if np.max(image) <= 1.0:
                overlay *= 255
            overlay = overlay.astype(np.uint8)
        else:
            overlay = overlay.astype(np.uint8)
    points = np.stack(np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1])), axis=-1)
    points_ = points[::vector_downsample, ::vector_downsample].reshape(-1, 2)
    vectors_ = vectors[::vector_downsample, ::vector_downsample].reshape(-1, 3)
    points_ *= upscale_factor
    vectors_ *= upscale_factor

    overlay = cv2.resize(overlay, (H*upscale_factor, W*upscale_factor), interpolation=cv2.INTER_LINEAR)
    # vectors_ *= -1  # Plotting is upside down I guess?
    # Draw arrows
    for (x, y), (vx, vy, vz) in zip(points_, vectors_):
        # if abs(vx) < min_deform_thresh and abs(vy) < min_deform_thresh and abs(vz) < min_deform_thresh:
        #     continue
        # Compute endpoint
        x2 = int(round(x + vx))
        y2 = int(round(y + vy))
        # Map z to color: blue (negative), red (positive)
        vz = round(vz)*20
        # Interpolate color: blue (-1) → green (0) → red (+1)
        if vz >= 0:
            # between green and red
            r = int(vz)
            g = int(255 - vz)
            b = 0
        else:
            # between blue and green
            vz_abs = abs(vz)
            r = 0
            g = int(255 - vz_abs)
            b = int(vz_abs)

        b, g, r = max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r))
        color = (b, g, r)  # BGR for OpenCV
        # r = int(min(255, max(0, vz)))
        # b = int(min(255, max(0, -vz)))
        # color = (b, 255, r)
        cv2.arrowedLine(overlay, (int(x), int(y)), (x2, y2), color,
                        thickness=int(round(arrow_thickness*upscale_factor)), tipLength=arrow_head_length*upscale_factor)
    overlay = cv2.resize(overlay, (overlay.shape[0]//upscale_factor, overlay.shape[1]//upscale_factor), interpolation=cv2.INTER_LINEAR)
    return overlay


def is_LV_bloodpool_enclosed(segmentation_map):
    """
    Detects if the LV blood pool class is fully surrounded by LV myocardium class by checking if Class 1 touches Class 2.
    """
    # Create masks
    inner_mask = (segmentation_map == 1)
    background_mask = np.bitwise_or(segmentation_map == 0, segmentation_map > 2)

    # Dilate the inner object by 1 pixel to find its immediate neighbors
    # Using a 3x3 structure ensures we check 8-connected neighbors (diagonals included)
    dilation_structure = np.ones((3, 3), dtype=bool)
    dilated_inner = scipy.ndimage.binary_dilation(inner_mask, structure=dilation_structure)

    # The 'boundary' is the area added by dilation (excluding the original inner object)
    boundary_region = dilated_inner & ~inner_mask

    # Check if any pixel in this boundary region is Background (0)
    # If the overlap is not empty, the background is touching the inner object
    is_touching_background = np.any(background_mask & boundary_region)

    return ~is_touching_background


def extract_2dt_contours(segmentation_slices, class_ids=(1, 2, 3), min_area=10, resolution_factor=5, debug_img_slices=None):
    """
    Extracts smoothed, hierarchical contours from a 2D+t segmentation volume.

    Parameters:
    - segmentation_slices: List of 3D numpy arrays (Height, Width, Time).
    - class_ids: List of integers (e.g., [1, 2, 3]).
                 Hierarchy: Class N includes all classes <= N (excluding 0).
    - min_area: Minimum pixel area to keep a component.
    - resolution_factor: Multiplier for the number of points in the smoothed contour.
                         Higher = smoother, more high-res line.

    Returns:
    - results: results[slice_idx][frame_idx][class_id] = List of 2D(N, 2) numpy array of len 0, 1, or 2 contours.
    """
    results = []
    slice_type = 'LA'
    for i, seg in enumerate(segmentation_slices):
        seg = seg.numpy()
        # Determine slice type
        if i == 3:
            slice_type = 'basal'
        elif slice_type == 'basal':
            prev_seg = segmentation_slices[i-1][..., 25].numpy()
            if np.any(prev_seg) and is_LV_bloodpool_enclosed(prev_seg):
                slice_type = 'midventricular'
        elif slice_type == 'midventricular':
            if i >= (len(segmentation_slices) - 1) - 3:  # If within the last 3 slices
                slice_type = 'apical'

        # Process segmentation slice into contour(s)
        results_slice = []
        H, W, T = seg.shape
        for t in range(T):
            frame = seg[:, :, t]
            results_slice.append({k: [] for k in class_ids})
            if slice_type in {'midventricular', 'apical'}:
                if np.any(seg[..., t] == 1) and not is_LV_bloodpool_enclosed(seg[..., t]):
                    continue  # If LV BP is not fully enclosed, assume incorrect annotation and take no training samples
            for c_id in class_ids:
                # --- 1. Hierarchical Mask Creation ---
                # "Class 2 is union of 1+2", etc.
                # We select all pixels > 0 and <= current_class_id
                binary_mask = (frame > 0) & (frame <= c_id)

                # --- 2. Keep Largest Component(s) ---
                labeled = skimage.measure.label(binary_mask)
                if labeled.max() == 0:
                    continue
                regions = skimage.measure.regionprops(labeled)
                # Make sure there are regions larger than the threshold
                regions = [r for r in regions if r.area > min_area]
                if len(regions) == 0:
                    continue
                # Pick largest regions
                regions = sorted(regions, key=lambda r: r.area, reverse=True)
                if slice_type == 'LA':
                    if len(regions) > 1:
                        # Should only ever have 1 region. If multiple, assume incorrect annotation and take no training samples
                        continue
                    selected_regions = regions[:1]  # Select only largest
                elif slice_type == 'basal':
                    if c_id < 3:
                        selected_regions = regions[:1]  # Select only largest
                    else:
                        selected_regions = regions[:2]  # Only allow RV to have multiple regions
                elif slice_type == 'midventricular':
                    if len(regions) > 1:
                        continue  # Should only ever have 1 region. If multiple, assume incorrect annotation and take no training samples
                    selected_regions = regions[:1]  # Select only largest
                elif slice_type == 'apical':
                    if c_id < 3:
                        selected_regions = regions[:1]  # Select only largest
                    else:
                        selected_regions = regions[:2]  # Only allow RV to have multiple regions
                else:
                    raise NotImplementedError

                # There will always be 1 or 2 regions
                for c in selected_regions:
                    # Isolate the component
                    component_mask = (labeled == c.label).astype(float)

                    # --- 3. Marching Squares ---
                    # Returns list of (row, col) coordinates
                    contours = skimage.measure.find_contours(component_mask, level=0.5)
                    # Handle topological anomalies (holes/islands) by taking the longest
                    if not contours:
                        continue
                    raw_contour = max(contours, key=len)

                    # --- 4. B-Spline Smoothing & Upsampling ---
                    # We use splprep (parametric B-spline) for curve smoothing
                    # Check if we have enough points to fit a spline (need > k=3)
                    if len(raw_contour) > 3:
                        # Transpose to shape (2, N) for splprep
                        y_coords = raw_contour[:, 0]
                        x_coords = raw_contour[:, 1]

                        # tck: tuple (knots, coefficients, degree)
                        # u: parameter values
                        # s: smoothing factor. Higher s = smoother. s=0 = strict interpolation.
                        # per=True: Closed curve
                        try:
                            tck, u = scipy.interpolate.splprep([y_coords, x_coords], s=2.0, per=True)

                            # Generate new points
                            # We create a new linspace with MORE points than original
                            u_new = np.linspace(u.min(), u.max(), len(raw_contour) * resolution_factor)
                            y_new, x_new = scipy.interpolate.splev(u_new, tck, der=0)

                            # Stack back to (N, 2)
                            smooth_contour = np.column_stack((y_new, x_new))
                            results_slice[t][c_id].append(smooth_contour)
                        except (KeyError, IndexError) as e:
                            raise e
                        except Exception as e:
                            # Fallback if spline fitting fails (e.g. self-intersecting or too small)
                            results_slice[t][c_id].append(raw_contour)
                    else:
                        results_slice[t][c_id].append(raw_contour)
        results.append(results_slice)
    if len(results) != len(segmentation_slices):
        raise ValueError('There should be as many results as slices')
    if any(len(r) != 50 for r in results):
        raise ValueError('There should be 50 frames')
    if any(any(len(c) != len(class_ids) for c in r) for r in results):
        raise ValueError('There should as many contour classes as number of classes')
    return results


def equalize_sdf_hierarchies(sdf: torch.Tensor) -> torch.Tensor:
    num_classes = sdf.shape[-1]
    new_sdf = sdf.clone()
    for c in range(1, num_classes):
        new_sdf[..., c] = torch.amin(new_sdf[..., c-1:c], dim=-1)
    return new_sdf


def sdf_hierarchy_to_seg(sdf):
    """ Given a tensor of SDF grids, convert them to segmentations. We assume these SDFs are defined as a hierarchy.
    Ie. the segmentation class #2 is defined as ((SDF_2 <= 0) & ~(SDF_1 <= 0)) """
    *spatial_dims, num_classes = sdf.shape
    sdf = equalize_sdf_hierarchies(sdf)
    seg = torch.zeros((*spatial_dims, num_classes+1), dtype=torch.bool, device=sdf.device)
    for c in range(0, num_classes+1):
        if c == 0:
            seg[..., c] = sdf[..., -1] > 0
        elif c == 1:
            seg[..., c] = sdf[..., 0] <= 0
        else:
            seg[..., c] = (sdf[..., c-1] <= 0) & (sdf[..., c-2] > 0)
    return seg


def keep_last_true(segmentations):
    """
    Given a boolean tensor of shape (*spatial_dims, num_classes),
    keep only the last True value along the class dimension for each spatial location.

    Args:
        segmentations: torch.Tensor of shape (*spatial_dims, num_classes) with dtype bool

    Returns:
        torch.Tensor of same shape where only the last True per spatial location remains True
    """
    # Flip along the class dimension (last axis)
    reversed_seg = torch.flip(segmentations, dims=[-1])
    last_true_idx_reversed = torch.argmax(reversed_seg.to(torch.long), dim=-1)
    has_true = torch.any(reversed_seg, dim=-1)

    # Convert back to original indexing
    num_classes = segmentations.shape[-1]
    last_true_idx = num_classes - 1 - last_true_idx_reversed

    # Create output tensor
    result = torch.zeros_like(segmentations)
    spatial_shape = segmentations.shape[:-1]
    spatial_indices = torch.meshgrid(
        *[torch.arange(s, device=segmentations.device) for s in spatial_shape],
        indexing='ij'
    )
    # Set only the last True for each spatial location
    mask_indices = tuple(idx[has_true] for idx in spatial_indices)
    result[mask_indices + (last_true_idx[has_true],)] = True

    return result


def burn_contours(image, contours_list, color=(1.0, 1.0, 1.0)):
    """
    image: 2D tensor (H, W, 3)
    contours_list: A list of tensors/arrays for the specific slice/time
                   (e.g., contour_per_class[class_idx][slice_idx])
    """
    # 1. Work on a copy to avoid corrupting the original data
    debug_img = image.clone()

    # Determine "White" value (max possible value for this datatype)
    # If float 0-1, use 1.0. If uint8 0-255, use 255.

    height, width, C = debug_img.shape

    for contour in contours_list:
        # contour shape is (N, 4) -> [slice, x, y, time] or similar
        # Extract columns 1 and 2.
        # NOTE: You previously mentioned swapping X/Y fixed orientation.
        # Standard Matrix indexing is img[ROW, COL], which is img[y, x].

        if contour.shape[-1] == 4:
            coords = contour[:,1:3]
        else:
            coords = contour
        if isinstance(coords, torch.Tensor):
            coords = coords

        # Round to nearest integer to get valid array indices
        # We assume column 0 is X (col) and column 1 is Y (row) based on your plot
        # But for numpy indexing, we need [row, col] -> [y, x]

        # If your plot required swapping, ensure you extract them as:
        # col_indices (x) = coords[:, 0]
        # row_indices (y) = coords[:, 1]

        x_indices = torch.round(coords[:, 0]).long()
        y_indices = torch.round(coords[:, 1]).long()

        # 2. Boundary Check (Crucial!)
        # Filter out points that fall outside the image dimensions
        valid_mask = (
                (x_indices >= 0) & (x_indices < width) &
                (y_indices >= 0) & (y_indices < height)
        )

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]

        # 3. "Burn" the pixels
        # Numpy indexing is [row, col] -> [y, x]
        debug_img[x_indices, y_indices] = torch.tensor(color, dtype=torch.float32, device=debug_img.device)

    return debug_img