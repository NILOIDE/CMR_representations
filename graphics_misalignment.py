import copy
from itertools import combinations, product

import pyvista as pv
import time

import h5py
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from skimage.measure import marching_cubes
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
import nibabel as nib
import trimesh
from tqdm import tqdm
from trimesh.smoothing import filter_taubin
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
from skimage import measure

from optimize_affines import compute_pairwise_loss
from sa_la_interp import interpolate_seg_to_other_view
from utils import params_to_mat

# Rendering parameters --------------------------------------------------
colors_vol = {
    1: "rgb(150,0,0)",
    2: "rgb(0,150,0)",
    3: "rgb(200, 200, 0)",
}
# colors_slice = {
#     1: "rgb(150,50,100)",
#     2: "rgb(40,100,125)",
#     3: "rgb(150,80,8)",
# }
colors_slice = {
    1: "rgb(150,0,0)",
    2: "rgb(0,150,0)",
    3: "rgb(200, 200, 0)",
}
label_names = {
    1: "LV BloodPool",
    2: "LV Myocardium",
    3: "RV BloodPool",
}
light_pos = {'x': 10.,
             'y': 10.,
             'z': 0.}
lighting = dict(
    ambient=0.2,     # less fill light
    diffuse=1.0,     # strong directional shading
    specular=0.5,    # highlight reflections
    roughness=0.9,   # slightly glossy
    fresnel=0.1
)


def create_slice_mesh(seg_path, aff, coord_max, coord_min):
    seg_file = nib.load(seg_path)
    H, W, D, T = seg_file.shape
    seg = seg_file.dataobj[...,int(T*FRAME)]


    labels = [3, 2, 1]
    meshes = []
    for label in labels:
        mask = seg == label
        if not np.any(mask):
            continue
        mask = np.concatenate((np.zeros_like(mask), mask, np.zeros_like(mask)), 2)
        a = mask[...,1].astype(np.uint8)
        verts, faces, normals, _ = marching_cubes(mask, level=0.5)
        verts = verts - np.array([0, 0, 1])  # Only adjust z for padding
        verts *= np.array((1.0, 1.0, 0.20))
        verts_aug = np.concatenate((verts, np.ones_like(verts[:,:1])), -1)
        verts = (aff @ verts_aug.T).T[:, :3]
        verts = (verts - coord_min[:3]) / (coord_max[:3] - coord_min[:3])
        verts = verts * 2 - 1
        m = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=colors_slice[label],
            opacity=1.0,  # <--- fully opaque
            # flatshading=True,
            lighting=lighting,
            name=label_names[label],
            lightposition=light_pos,
        )
        meshes.append(m)
    return meshes


def create_volume_mesh(seg_path, flip=False):
    seg_file = nib.load(seg_path)
    H, W, D, T = seg_file.shape
    seg = seg_file.dataobj[...,int(T*FRAME)]
    aff_file = nib.load(seg_path)
    labels = [3, 2, 1]
    meshes = []
    for label in labels:
        mask = seg == label
        if not np.any(mask):
            continue
        verts, faces, normals, _ = marching_cubes(mask, level=0.5)
        # verts_aug = np.concatenate((verts, np.ones_like(verts[:,:1])), -1)
        # aff = copy.deepcopy(aff_file.affine)
        # if flip:
        #     # Only flip the x-coordinate direction (left-right)
        #     flip_mat = np.diag([-1, 1, 1, 1])
        #     aff = aff @ flip_mat
        # verts = (aff @ verts_aug.T).T[:, :3]
        verts = verts / 300
        verts = verts * 2 - 1.0
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh = filter_taubin(mesh, lamb=0.5, nu=-0.5, iterations=15)
        v, f = mesh.vertices, mesh.faces
        f = np.hstack([np.full((f.shape[0], 1), 3), f]).ravel()
        clipped = pv.PolyData(v, f,)# clip the mesh
        # if label == 1:
        #     clipped = clipped.clip(
        #         normal=(0, 1, 0),     # diagonal plane normal
        #         origin=(0,-0.0,0),      # a point on the plane
        #         invert=False           # flip if you want the other half
        #     )
        # else:
        #     clipped = clipped.clip(
        #         normal=(0, 1, 0),     # diagonal plane normal
        #         origin=(0,-0.0,0),      # a point on the plane
        #         invert=False           # flip if you want the other half
        #     )
        v = clipped.points
        f = clipped.faces.reshape(-1, 4)[:, 1:]  # remove leading 3
        m = go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            color=colors_vol[label],
            opacity=0.2,  # <--- fully opaque
            # flatshading=True,
            lighting=lighting,
            name=label_names[label],
            lightposition=light_pos,
        )
        meshes.append(m)
    return meshes


def create_plot(volume_pred_path, opt_paths, affs, coords_max, coords_min):
    fig = go.Figure()
    meshes = create_volume_mesh(str(volume_pred_path))
    for m in meshes:
        fig.add_trace(m)
    for i, (seg_path, aff_path) in enumerate(zip(opt_paths, og_seg_paths)):
        objs = create_slice_mesh(seg_path, affs[i], coords_max, coords_min)
        for o in objs:
            fig.add_trace(o)

    zoom = 0.7
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1, 1]),
            yaxis=dict(visible=False, range=[-1, 1]),
            zaxis=dict(visible=False, range=[-1, 1]),
            aspectmode='cube',
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=-1 * zoom, y=-2 * zoom, z=0.5 * zoom),  # Camera position (viewing from back)
                center=dict(x=0, y=0, z=0),  # Look-at point
                up=dict(x=0, y=0, z=1)  # Up direction
            ),
        ),
        dragmode='orbit',
        margin=dict(l=0, r=0, t=0, b=0),
        scene_aspectmode='data',
    )

    fig.show()


def get_h5_affs(h5_path):
    with h5py.File(h5_path, 'r') as f:
        aff_params = torch.from_numpy(f['aff_params_padded'][:])
        spacings_padded = torch.from_numpy(f['spacings_padded'][:])
        flippings_padded = torch.from_numpy(f['flippings_padded'][:])
        affs = params_to_mat(aff_params, spacings_padded, flippings_padded).numpy()
        affs = [affs[i] for i in range(affs.shape[0])]
        coord_max = f['coord_max'][:]
        coord_min = f['coord_min'][:]
    return affs, coord_max, coord_min


from scipy.ndimage import map_coordinates
def project_slice_with_interpolation(seg_volume, seg_slice, affine_matrix, coord_max, coord_min, order=0):
    """
    Alternative: Sample volume values at slice positions (inverse operation).
    Use this if you want slice voxels to take values from the volume.

    Parameters:
    -----------
    seg_volume : ndarray
        3D segmentation volume
    seg_slice : ndarray
        2D slice that will be filled with volume values
    affine_matrix : ndarray
        4x4 affine transformation matrix
    order : int
        Interpolation order (0=nearest, 1=linear, etc.)

    Returns:
    --------
    filled_slice : ndarray
        Slice with values sampled from volume
    """
    if seg_slice.ndim == 3:
        seg_slice = seg_slice.squeeze()

    h, w = seg_slice.shape

    # Create coordinate grid
    i_coords, j_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    slice_coords_homog = np.stack([
        i_coords.ravel(),
        j_coords.ravel(),
        np.zeros(h * w),
        np.ones(h * w)
    ], axis=1)

    # Transform to volume coordinates
    volume_coords_homog = slice_coords_homog @ affine_matrix.T
    volume_coords = volume_coords_homog[:, :3]
    volume_coords = (volume_coords - coord_min[:3]) / (coord_max[:3] - coord_min[:3])
    volume_coords = volume_coords * 300

    # Transpose for map_coordinates (expects [z, y, x] format)
    volume_coords_transposed = volume_coords.T

    # Sample from volume using interpolation
    sampled_values = map_coordinates(
        seg_volume,
        volume_coords_transposed,
        order=order,
        mode='constant',
        cval=0
    )

    filled_slice = sampled_values.reshape(h, w)

    return filled_slice


def seg_to_rgb(seg_slice, intensity_slice, color_map=None):
    """
    Convert segmentation slice to RGB image with intensity background.

    Parameters:
    -----------
    seg_slice : ndarray
        2D segmentation array with integer labels (H, W)
    intensity_slice : ndarray
        2D intensity/grayscale image for background (H, W)
    color_map : dict, optional
        Dictionary mapping label values to RGB tuples or strings
        Default colors for labels 1, 2, 3

    Returns:
    --------
    rgb_slice : ndarray
        RGB image of shape (H, W, 3) with uint8 values
    """
    if color_map is None:
        color_map = {
            1: (150, 0, 0),
            2: (0, 150, 0),
            3: (200, 200, 0),
        }

    # Convert string colors to tuples if needed
    processed_color_map = {}
    for label, color in color_map.items():
        if isinstance(color, str):
            # Parse "rgb(r,g,b)" format
            color = color.replace('rgb(', '').replace(')', '')
            color = tuple(map(int, color.split(',')))
        processed_color_map[label] = color

    # Create RGB image starting with intensity as background
    h, w = seg_slice.shape

    # Normalize intensity to 0-255 range if needed
    intensity_norm = intensity_slice.astype(np.float32)
    if intensity_norm.max() > 0:
        intensity_norm = (intensity_norm - intensity_norm.min()) / (intensity_norm.max() - intensity_norm.min())
        intensity_norm = (intensity_norm * 255).astype(np.uint8)
    else:
        intensity_norm = intensity_norm.astype(np.uint8)

    # Create RGB by replicating grayscale across 3 channels
    rgb_slice = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=-1)

    # Overlay segmentation colors for foreground labels (non-zero)
    for label, color in processed_color_map.items():
        mask = seg_slice == label
        rgb_slice[mask] = color

    return rgb_slice


def seg_to_contours(seg_slice, thickness=1):
    """
    Convert segmentation to contours for each label.

    Parameters:
    -----------
    seg_slice : ndarray
        2D segmentation array with integer labels
    thickness : int
        Contour thickness in pixels

    Returns:
    --------
    contour_mask : ndarray
        2D boolean array where True indicates contour pixels
    label_contours : dict
        Dictionary mapping each label to its contour mask
    """
    contour_mask = np.zeros_like(seg_slice, dtype=bool)
    label_contours = {}

    # Get unique labels (excluding background 0)
    labels = np.unique(seg_slice)
    labels = labels[labels != 0]

    for label in labels:
        # Create binary mask for this label
        mask = (seg_slice == label)

        # Erode the mask
        eroded = binary_erosion(mask, iterations=thickness)

        # Contour is the difference between original and eroded
        contour = mask & ~eroded

        label_contours[label] = contour
        contour_mask |= contour

    return contour_mask, label_contours


def overlay_contours_on_image(intensity_slice, seg_slice, color_map=None,
                              thickness=1, contour_only=False, smooth=True,
                              smooth_sigma=1.0):
    """
    Overlay segmentation contours on intensity image.

    Parameters:
    -----------
    intensity_slice : ndarray
        2D grayscale image (H, W)
    seg_slice : ndarray
        2D segmentation array (H, W)
    color_map : dict
        Mapping from label to RGB color
    thickness : int
        Contour thickness in pixels
    contour_only : bool
        If True, only draw contours. If False, fill regions with semi-transparent overlay
    smooth : bool
        If True, smooth the contours using Gaussian filter
    smooth_sigma : float
        Sigma for Gaussian smoothing (higher = smoother)

    Returns:
    --------
    rgb_image : ndarray
        RGB image with contours overlaid (H, W, 3)
    """
    if color_map is None:
        color_map = {
            1: (150, 0, 0),
            2: (0, 150, 0),
            3: (200, 200, 0),
        }

    # Normalize intensity to RGB
    intensity_norm = intensity_slice.astype(np.float32)
    if intensity_norm.max() > 0:
        intensity_norm = (intensity_norm - intensity_norm.min()) / (intensity_norm.max() - intensity_norm.min())
        intensity_norm = (intensity_norm * 255).astype(np.uint8)
    else:
        intensity_norm = intensity_norm.astype(np.uint8)

    rgb_image = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=-1)
    all_contours = np.zeros_like(seg_slice, dtype=bool)

    for label in [1,3,2]:
        if label not in color_map:
            continue

        color = color_map[label]
        if isinstance(color, str):
            color = color.replace('rgb(', '').replace(')', '')
            color = tuple(map(int, color.split(',')))

        # Create binary mask for this label
        mask = (seg_slice == label).astype(float)

        if smooth:
            # Smooth the mask before extracting contours
            mask_smooth = gaussian_filter(mask, sigma=smooth_sigma)
            mask = (mask_smooth > 0.5).astype(float)

        # Find contours using marching squares
        contours = measure.find_contours(mask, 0.5)

        # Draw contours with specified thickness
        for contour in contours:
            # Round to integer coordinates
            contour = np.round(contour).astype(int)

            # Clip to image bounds
            contour[:, 0] = np.clip(contour[:, 0], 0, rgb_image.shape[0] - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, rgb_image.shape[1] - 1)

            # Draw contour with thickness
            for i in range(len(contour)):
                y, x = contour[i]
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    for dx in range(-thickness // 2, thickness // 2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rgb_image.shape[0] and 0 <= nx < rgb_image.shape[1]:
                            rgb_image[ny, nx] = color
                            all_contours[ny, nx] = True

    # Optional: add semi-transparent fill
    if not contour_only:
        for label, color in color_map.items():
            if isinstance(color, str):
                color = color.replace('rgb(', '').replace(')', '')
                color = tuple(map(int, color.split(',')))

            mask = (seg_slice == label) & ~all_contours
            if mask.any():
                # Blend: 70% intensity, 30% color
                rgb_image[mask] = (0.7 * rgb_image[mask] + 0.3 * np.array(color)).astype(np.uint8)

    return rgb_image

def inter_vol_to_slices(vol_path, affs, seg_paths, im_paths):
    vol_seg = nib.load(str(vol_path))
    vold_seg = vol_seg.dataobj[..., int(vol_seg.shape[-1] * FRAME)]
    interps = []
    for i, (s, im) in enumerate(zip(seg_paths, im_paths)):
        slice_seg = nib.load(str(s))
        slice_seg = slice_seg.dataobj[..., int(slice_seg.shape[-1] * FRAME)].astype(np.uint8)
        filled_slice = project_slice_with_interpolation(vold_seg, slice_seg, affs[i], og_max, og_min)
        filled_slice = zoom(filled_slice, (4, 4,), order=0)
        slice_int = nib.load(str(im))
        slice_int = slice_int.dataobj[..., 0, int(slice_int.shape[-1] * FRAME)].astype(np.uint8)
        slice_int = zoom(slice_int, (4, 4), order=0)
        rgb_slice = overlay_contours_on_image(slice_int, filled_slice, colors_slice,
                                              thickness=2, contour_only=True, smooth=True, smooth_sigma=1.5)
        interps.append(rgb_slice)
    return interps


def compute_intersection_metrics(image_paths, seg_paths, affs, sampling_step_mm=2.0):
    affs = torch.tensor(affs)
    la_la_product = list(combinations([0,1,2], 2))
    # Long-axis to Short-axis image pairs combinations (if 3 LA images and N SA images, that will be 3*N pairs)
    sa_la_product = list(product([0,1,2], list(range(3, len(image_paths)))))
    idx_pairs = torch.tensor(la_la_product + sa_la_product)
    images = [nib.load(i).dataobj[...,0,:] for i in image_paths]
    shapes = torch.tensor([i.shape for i in images])
    images = torch.tensor(images)
    int_loss = compute_pairwise_loss(images.float(), affs, idx_pairs, shapes,
                                     metric='ncc', interp_type='linear', sampling_step_mm=sampling_step_mm)
    int_metric = 1 - int_loss

    sa_idx_range = (3, images.shape[0])
    la_la_product = list(combinations([0,1,2], 2))
    # Long-axis to Short-axis image pairs combinations (if 3 LA images and N SA images, that will be 3*N pairs)
    sa_la_product = list(product([0,1,2], list(range(sa_idx_range[0], sa_idx_range[1]))))
    idx_pairs = torch.tensor(la_la_product + sa_la_product)
    segs = [nib.load(i).dataobj[...,0,:] for i in seg_paths]
    segs = torch.tensor(segs)
    segs = torch.cat((segs[:3], segs[sa_idx_range[0]:sa_idx_range[-1]]), 0)
    affs_segs = torch.cat((affs[:3], affs[sa_idx_range[0]:sa_idx_range[-1]]), 0)
    shapes_segs = torch.cat((shapes[:3], shapes[sa_idx_range[0]:sa_idx_range[-1]]), 0)
    seg_loss = compute_pairwise_loss(segs.float(), affs_segs, idx_pairs, shapes_segs,
                                     metric='dice', interp_type='nn', sampling_step_mm=sampling_step_mm)
    seg_metric = 1 - seg_loss
    seg_t_indices = [0, 16, 32]
    seg_metric = seg_metric[:, seg_t_indices]
    return int_metric.mean(), seg_metric[1:].mean()


FRAME = 0.0
results_dir = Path(r"D:\logs\20251208-194802-inference_trainSet_labelOnlyModel_noFT\logs\test_opt0000_ft2500_slices")
train_data_dir = Path(r'D:\logs\train_data')
metrics_int_before = []
metrics_seg_before = []
metrics_int_after = []
metrics_seg_after = []
for subj in tqdm(list(results_dir.iterdir())[:], desc='Subjects'):
    nifti_pred_dir = subj / "epoch_000000" / 'niftis'
    train_data_path = train_data_dir / subj.name / 'prep_data.h5'
    nifti_og_dir = subj / "epoch_000000" / 'niftis' / 'original'

    volume_pred_path = results_dir.parent / (results_dir.name[:-6] + "volumes") / "epoch_0" / subj.name / 'pred' / 'full_seg.nii.gz'

    opt_seg_paths = [i for i in nifti_pred_dir.iterdir() if "seg_" in i.name]
    opt_seg_paths = sorted(opt_seg_paths)
    opt_im_paths = [i for i in nifti_pred_dir.iterdir() if "seg_" in i.name]
    opt_im_paths = sorted(opt_im_paths)
    og_seg_paths = [i for i in nifti_og_dir.iterdir() if "seg_" in i.name]
    og_seg_paths = sorted(og_seg_paths)
    og_im_paths = [i for i in nifti_og_dir.iterdir() if "seg_" not in i.name]
    og_im_paths = sorted(og_im_paths)
    og_affs, og_max, og_min = get_h5_affs(train_data_path)
    opt_affs = [nib.load(p).affine for p in opt_seg_paths]
    # create_plot(volume_pred_path, og_seg_paths, og_affs, og_max, og_min)
    # time.sleep(1)
    # create_plot(volume_pred_path, og_seg_paths, opt_affs, og_max, og_min)

    int_metric_before, seg_metric_before = compute_intersection_metrics(og_im_paths, og_seg_paths, og_affs)
    metrics_int_before.append(int_metric_before)
    metrics_seg_before.append(seg_metric_before)
    int_metric_after, seg_metric_after = compute_intersection_metrics(og_im_paths, og_seg_paths, opt_affs)
    metrics_int_after.append(int_metric_after)
    metrics_seg_after.append(seg_metric_after)

    # ims_before = inter_vol_to_slices(volume_pred_path, og_affs, og_seg_paths, og_im_paths)
    # ims_after = inter_vol_to_slices(volume_pred_path, opt_affs, opt_seg_paths, og_im_paths)
    # render_save_dir = Path('alignment') / subj.name
    # render_save_dir.mkdir(exist_ok=True, parents=True)
    # for i, (b, a) in enumerate(zip(ims_before, ims_after)):
    #     # Create filename with zero-padded index
    #     filename = render_save_dir / f"{i:02d}_before.png"
    #     img = Image.fromarray(b)
    #     img.save(str(filename))
    #     filename = render_save_dir / f"{i:02d}_after.png"
    #     img = Image.fromarray(a)
    #     img.save(str(filename))
print('Int metric before:', np.mean(metrics_int_before), np.std(metrics_int_before))
print('Seg metric before:', np.mean(metrics_seg_before), np.std(metrics_seg_before))
print('Int metric after:', np.mean(metrics_int_after), np.std(metrics_int_after))
print('Seg metric after:', np.mean(metrics_seg_after), np.std(metrics_seg_after))

