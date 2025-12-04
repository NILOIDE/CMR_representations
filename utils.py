from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict

import cv2
import numpy as np
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