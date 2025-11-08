import glob
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import nibabel as nib
import torch.cuda
import tqdm


def calculate_normal_vector(plane):
    # Create the normal vector
    normal_vector = np.array(plane[:3])
    # Normalize the normal vector
    normalized_vector = normal_vector / np.linalg.norm(normal_vector)
    return normalized_vector


def calculate_angle_between_planes(plane1, plane2):
    # Calculate the normal vectors of the planes
    normal_vector1 = calculate_normal_vector(plane1)
    normal_vector2 = calculate_normal_vector(plane2)
    # Calculate the dot product of the normal vectors
    dot_product = np.dot(normal_vector1, normal_vector2)
    dot_product_norm = dot_product / (np.linalg.norm(normal_vector1) * np.linalg.norm(normal_vector2))
    if dot_product_norm > 1.0:
        return 0.0
    if dot_product_norm < -1.0:
        return 180.0
    # Calculate the angle between the planes using the dot product
    angle_rad = np.arccos(dot_product_norm)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def get_nifti_image_plane(nifti, z):
    points = get_3_points_from_nifti_slice(nifti, z)
    # Create a matrix A from the coordinates
    A = np.vstack(points)
    # Create a vector B with ones
    B = np.ones(3)
    # Solve the linear equation Ax = B
    x = np.linalg.solve(A, B)
    # Extract the coefficients of the plane equation
    a, b, c = x
    # Compute the constant term d
    d = -np.dot(x, points[0])
    # Return the plane equation coefficients
    return np.array([a, b, c, d])


def get_3_points_from_nifti_slice(nifti, z):
    center = np.array([nifti.shape[0]//2, nifti.shape[1]//2, z, 1])
    corner0 = np.array([0., 0., z, 1])
    corner1 = np.array([0., nifti.shape[1]//2, z, 1])
    center_ = nifti.affine.dot(center)
    corner0_ = nifti.affine.dot(corner0)
    corner1_ = nifti.affine.dot(corner1)
    return center_[:3], corner0_[:3], corner1_[:3]


def get_center_point_from_nifti_slice(nifti, z):
    center = np.array([nifti.shape[0]//2, nifti.shape[1]//2, z, 1])
    center_ = nifti.affine.dot(center)
    return center_[:3]


def calculate_distance_point_to_plane(point, plane):
    # Extend point in order to perform vector dot product
    point = np.array([*point, 1])
    # Calculate the distance between the point and the plane
    # dist = np.abs(a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    distance = point.dot(plane) / np.sqrt(plane[:3].dot(plane[:3]))
    return distance


def calculate_image_plane_angles_between_niftis(nifti1, z1, nifti2, z2):
    nifti1_plane = get_nifti_image_plane(nifti1, z1)
    nifti2_plane = get_nifti_image_plane(nifti2, z2)
    angle = calculate_angle_between_planes(nifti1_plane, nifti2_plane)
    return angle


def calculate_distance_between_nifti_plane_and_center(nifti1, z1, nifti2, z2):
    nifti1_plane = get_nifti_image_plane(nifti1, z1)  # Mid-ventricular slice == 4 (5th slice from base)
    center = get_center_point_from_nifti_slice(nifti2, z2)
    dist = calculate_distance_point_to_plane(center, nifti1_plane)
    return dist


def calculate_distances(p1, p2, sqrt=True):
    assert len(p1.shape) == len(p2.shape)
    assert p1.shape[-1] == p2.shape[-1]
    dists = (p1[..., None, :]-p2[..., None, :, :])
    dists = dists * dists
    dists = dists.sum(-1)
    if sqrt:
        dists = dists.sqrt()
    return dists


def interpolate_sa_seg_to_la(sa_seg_file: str, sa_img_file: str, la_file: str, frames: Tuple[int, ...] = (0, )):
    """ Single SA volume. Interpolate to LA image  """
    sa_seg_nii = nib.load(sa_seg_file)
    sa_seg = sa_seg_nii.dataobj[:].astype(np.uint8)
    sa_nii = nib.load(sa_img_file)
    sa_img = sa_nii.dataobj[:].astype(int)
    sa_aff = sa_seg_nii.affine
    la_nii = nib.load(la_file)
    la_shape = la_nii.shape
    la_shape = la_shape if len(la_shape) == 3 else (*la_shape[:2], 1)
    la_aff = la_nii.affine

    interp_ims = []
    interp_segs = []
    for f in frames:
        interp_im, interp_seg = interpolate_seg_to_other_view(sa_seg[..., f], sa_img[..., f], sa_aff, la_shape, la_aff,
                                                   oob_dist_thresh=sa_seg_nii.header.get_zooms()[2])
        interp_ims.append(interp_im)
        interp_segs.append(interp_seg)
    return interp_ims, interp_segs


def interpolate_seg_to_other_view(seg: np.ndarray,
                                  img: np.ndarray,
                                  seg_aff: np.ndarray,
                                  target_shape: Optional[Tuple[int, int, int]],
                                  target_aff: Optional[np.ndarray],
                                  target_w_coords: Optional[np.ndarray], oob_dist_thresh=10.):
    assert len(seg.shape) == 3
    seg_coords = np.meshgrid(*[np.arange(i, dtype=float) for i in seg.shape], indexing="ij")
    seg_coords_ = np.stack(seg_coords, -1).reshape(-1, 3)
    seg_coords_aug_ = np.concatenate((seg_coords_, np.ones_like(seg_coords_[..., :1])), axis=-1)
    seg_w_coords_aug_ = (seg_aff @ seg_coords_aug_.T).T
    seg_w_coords_ = seg_w_coords_aug_[..., :3]
    seg_ = seg.reshape(-1, 1)
    img_ = img.reshape(-1, 1)

    assert len(target_shape) == 3
    if target_w_coords is None:
        target_coords = np.meshgrid(*[np.arange(i, dtype=float) for i in target_shape], indexing="ij")
        target_coords_ = np.stack(target_coords, -1).reshape(-1, 3)
        target_coords_aug_ = np.concatenate((target_coords_, np.ones_like(target_coords_[..., :1])), axis=-1)
        target_w_coords_aug_ = (target_aff @ target_coords_aug_.T).T
        target_w_coords_ = target_w_coords_aug_[..., :3]
    else:
        target_w_coords_ = target_w_coords.reshape(-1, 3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_w_coords_ = torch.tensor(seg_w_coords_, device=device)
    target_w_coords_ = torch.tensor(target_w_coords_, device=device)

    closest = []
    is_out_of_bounds = []
    chunk_size = 100
    for i in tqdm.tqdm(range(0, target_w_coords_.shape[0], chunk_size), desc=f"Interpolating target image {chunk_size} points at a time."):
        dists = calculate_distances(target_w_coords_[i:i+chunk_size], seg_w_coords_)
        closest_idx = dists.argmin(-1)
        closest.append(closest_idx.cpu().numpy())
        oob = dists[np.arange(closest_idx.shape[0]), closest_idx] > oob_dist_thresh
        is_out_of_bounds.append(oob.cpu().numpy())
    closest = np.concatenate(closest, 0)
    is_out_of_bounds = np.concatenate(is_out_of_bounds, 0)

    target_seg_ = seg_[closest]
    target_seg_[is_out_of_bounds] = 0
    target_seg = target_seg_.reshape(target_shape)
    target_val_ = img_[closest]
    target_val_[is_out_of_bounds] = 0
    target_val = target_val_.reshape(target_shape)
    return target_val, target_seg


def interpolate_sa_segs_to_la_from_nifti(sa_seg_dir: str, sa_img_dir: str, target_file: str, 
                                         frames: Optional[Tuple[int, ...]] = None,
                                         sa_seg_prefix: str = 'seg_sa_', sa_img_prefix: str = 'sa_',
                                         oob_dist_thresh = 9999.):
    """ Directory of SA slices. Assumes segmentation and image files are in the same directory. Interpolate to LA image  """
    # Load intensities of SA slices and get their affines
    sa_img_dir = Path(sa_img_dir)
    assert sa_img_dir.exists()
    sa_img_files = [str(i) for i in sa_img_dir.glob('*.nii.gz') if sa_img_prefix in str(i) and not sa_seg_prefix in str(i)]
    sa_img_files.sort(key=lambda x: int(Path(x).name[len(sa_img_prefix):len(sa_img_prefix)+2]))
    sa_img_niis = [nib.load(i) for i in sa_img_files]
    sa_img_arrays = [i.dataobj[:].astype(int) for i in sa_img_niis]
    sa_img_affs = [i.affine for i in sa_img_niis]
    sa_img_shape = sa_img_arrays[0].squeeze().shape
    assert len(sa_img_shape) == 3
    assert sa_img_shape[-1] == 50

    # Load segmentations of SA slices
    sa_seg_dir = Path(sa_seg_dir)
    sa_seg_files = [str(i) for i in sa_seg_dir.glob('*.nii.gz') if sa_seg_prefix in str(i)]
    sa_seg_files.sort(key=lambda x: int(Path(x).name[len(sa_seg_prefix):len(sa_seg_prefix)+2]))
    sa_seg_niis = [nib.load(i) for i in sa_seg_files]
    sa_seg_arrays = [i.dataobj[:].astype(np.uint8) for i in sa_seg_niis]

    # Load target slice and its affine
    target_nii = nib.load(target_file)
    target_aff = target_nii.affine
    target_shape = tuple(i for i in target_nii.shape if i > 1)
    assert len(target_shape) == 3
    assert target_shape[-1] == 50

    target_seg, target_img = interpolate_sa_segs_to_la(sa_seg_arrays, sa_img_arrays, sa_img_affs,
                                                       target_shape, target_aff,
                                                       frames=frames, oob_dist_thresh=oob_dist_thresh)
    return target_seg, target_img

def interpolate_sa_segs_to_la(sa_seg_arrays: List[np.ndarray],
                              sa_img_arrays: List[np.ndarray],
                              sa_affs: List[np.ndarray],
                              target_shape: Optional[Tuple[int, int, int]] = None,
                              target_aff: Optional[np.ndarray] = None,
                              target_w_coords: Optional[np.ndarray] = None,
                              frames: Optional[Tuple[int, ...]] = None,
                              oob_dist_thresh = 9999.):
    sa_img_shapes = [(*i.shape[:2], i.shape[-1]) for i in sa_img_arrays]
    H, W, T = sa_img_shapes[0]
    if target_shape is None:
        target_shape = target_w_coords.shape[:-1]
    assert all([len(i) == 3 for i in sa_img_shapes])
    assert all([i[-1] == T for i in sa_img_shapes])
    assert len(target_shape) == 3
    assert target_shape[-1] == T

    # Compute world coords of all SA slices
    sa_coords = [np.stack(np.meshgrid(np.arange(shape[0], dtype=float),
                                      np.arange(shape[1], dtype=float),
                                      [0], indexing="ij"),
                         axis=-1).squeeze(-2)
                 for shape in sa_img_shapes]
    sa_coords_aug = [np.concatenate((i, np.ones_like(i[..., :1])), axis=-1) for i in sa_coords]
    sa_coords_aug_ = [i.reshape(-1, 4) for i in sa_coords_aug]
    sa_w_coords_aug_slices_ = [(aff @ c_.T).T for aff, c_ in zip(sa_affs, sa_coords_aug_)]
    sa_w_coords_slices_ = [i[...,:3] for i in sa_w_coords_aug_slices_]
    sa_img_arrays_slices_ = [i.reshape(-1, T) for i in sa_img_arrays]
    sa_seg_arrays_slices_ = [i.reshape(-1, T) for i in sa_seg_arrays]

    if target_w_coords is None:
        # Compute world coords of target slice
        target_coords = np.stack(np.meshgrid(np.arange(target_shape[0], dtype=float),
                                         np.arange(target_shape[1], dtype=float),
                                         [0], indexing="ij"), axis=-1)
        target_coords = target_coords.squeeze(-2)
        target_coords_aug = np.concatenate((target_coords, np.ones_like(target_coords[..., :1])), axis=-1)
        target_coords_aug_ = target_coords_aug.reshape(-1, 4)
        target_w_coords_aug_ = (target_aff @ target_coords_aug_.T).T
        target_w_coords_ = target_w_coords_aug_[..., :-1]
    else:
        target_w_coords_ = target_w_coords.reshape(-1, 3)

    # Bring world coords into GPU and flatten SA coords
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_w_coords_ = torch.tensor(target_w_coords_, device=device)
    sa_w_coords_slices_ = torch.cat([torch.tensor(i, device=device) for i in sa_w_coords_slices_], dim=0)
    assert len(target_w_coords_.shape) == 2
    assert len(sa_w_coords_slices_.shape) == 2

    # For each target slice point, compute closest SA point in world space.
    # Also track if target point is further than the out-of-bounds threshold from any SA point.
    closest = []
    is_out_of_bounds = []
    chunk_size = 200
    for i in tqdm.tqdm(range(0, target_w_coords_.shape[0], chunk_size),
                       desc=f"Interpolating target image {chunk_size} points at a time."):
        dists = calculate_distances(target_w_coords_[i:i + chunk_size], sa_w_coords_slices_)
        closest_idx = dists.argmin(-1)
        closest.append(closest_idx.cpu().numpy())
        oob = dists[np.arange(closest_idx.shape[0]), closest_idx] > oob_dist_thresh
        is_out_of_bounds.append(oob.cpu().numpy())
    closest = np.concatenate(closest, 0)
    is_out_of_bounds = np.concatenate(is_out_of_bounds, 0)

    # Given each target point's closest SA neighbor, retrieve intensities and segmentations of said neighbor.
    # Do so for each selected frame. If target point is marked as out-of-bounds, default to intensity/seg value of 0.0.
    sa_img_arrays_slices_ = np.concatenate(sa_img_arrays_slices_, axis=0)
    sa_seg_array_slices_ = np.concatenate(sa_seg_arrays_slices_, axis=0)
    assert len(sa_img_arrays_slices_.shape) == 2
    assert len(sa_seg_array_slices_.shape) == 2
    img_frames = []
    seg_frames = []
    if frames is None:
        frames = list(range(target_shape[-1]))
    for frame in frames:
        target_seg_ = sa_seg_array_slices_[closest, frame]
        target_seg_[is_out_of_bounds] = 0
        target_seg = target_seg_.reshape(target_shape[:2])
        seg_frames.append(target_seg)
        target_val_ = sa_img_arrays_slices_[closest, frame]
        target_val_[is_out_of_bounds] = 0
        target_val = target_val_.reshape(target_shape[:2])
        img_frames.append(target_val)
    img_frames = np.stack(img_frames, -1)
    seg_frames = np.stack(seg_frames, -1)
    return seg_frames, img_frames


if __name__ == '__main__':
    la_im = r"/home/nil/data/ukbb/cardiac/unaligned_subjects/1000456/la_4ch.nii.gz"
    sa_seg = r"/home/nil/data/ukbb/cardiac/unaligned_subjects/1000456/sa_slices/"
    sa_im = r"/home/nil/data/ukbb/cardiac/unaligned_subjects/1000456/sa_slices/"
    la_seg_interp, la_im_interp = interpolate_sa_segs_to_la_from_nifti(sa_seg, sa_im, la_im, frames=None, oob_dist_thresh=10.)
    print("Finished")
