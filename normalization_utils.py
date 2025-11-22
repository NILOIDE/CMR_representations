from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from typing import List, Optional, Tuple

from data_utils import array_to_nifti
from geo_utils import get_image_plane_from_affine, plane_intersection, plane_line_intersection, rotation_matrix, \
    get_image_plane_from_array, angle_between_vectors
from sa_la_interp import interpolate_sa_segs_to_la
from utils import get_center_coord, to_gif


def scanner_to_image_coords(world_points: torch.Tensor, affines: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable way to convert a point from scanner/world coordinates to image coordinates.

    Args:
        world_points: 3D point in scanner coordinates [x, y, z]
        affines: 4x4 affine transformation matrix

    Returns:
        3D point in image coordinates [i, j, k]
    """
    # Convert to homogeneous coordinates
    if len(world_points.shape) == 1:
        world_points = world_points[None]
    if len(affines.shape) == 2:
        affines = affines[None]
    if world_points.shape[-1] == 3:
        world_points = torch.cat([world_points, torch.ones(1, device=world_points.device)])
    assert world_points.shape[-1] == 4
    # Apply inverse affine transformation
    world_points = torch.linalg.solve(affines, world_points)
    # Return the spatial coordinates (drop homogeneous coordinate)
    return world_points[:, :3]

def crop_around_heart(affines: List[torch.Tensor],
                      segs: List[torch.Tensor],
                      arrays: List[List[torch.Tensor]],
                      crop_size_2ch: Tuple[int, int] = (80, 80),
                      crop_size_3ch: Tuple[int, int] = (80, 80),
                      crop_size_4ch: Tuple[int, int] = (80, 80),
                      crop_size_sa: Tuple[int, int] = (80, 80),
                      debug=False) \
        -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]]]:
    centers = find_heart_center_on_slices(affines, segs)
    centers = torch.stack(centers)
    centers = centers.round().long()
    crop_sizes = torch.tensor((crop_size_2ch, crop_size_3ch, crop_size_4ch, *[crop_size_sa]*len(affines[3:])),
                              dtype=torch.long)
    # New origin based on center
    new_origins = (centers - crop_sizes//2).clip(min=0)
    crop_ends = new_origins + crop_sizes
    # Crop the segmentations
    new_segs = [i[o[0]:e[0], o[1]:e[1]] for i, o, e in zip(segs, new_origins, crop_ends)]
    new_arrays = []
    for i, arr in enumerate(arrays):
        new_arr = [i[o[0]:e[0], o[1]:e[1]] for i, o, e in zip(arr, new_origins, crop_ends)]
        new_arrays.append(new_arr)
    new_affines = [update_affine_after_crop(aff, o) for aff, o, in zip(affines, new_origins)]
    # for i, affine in enumerate(affines):
    #     new_affine = affine.clone()
    #     # The translation in image coordinates due to cropping
    #     offset = torch.tensor([new_origins[i][0], new_origins[i][1], 0.0], dtype=affine.dtype, device=affine.device)
    #     # Convert the pixel offset to scanner space offset
    #     scanner_offset = affine[:3, :3] @ offset
    #     # Update the translation component (last column, first 3 rows)
    #     new_affine[:3, 3] += scanner_offset
    #     new_affines.append(new_affine)
    if debug:
        [to_gif(torch.cat((s.float()/4, im), dim=1), 'debug_crop', str(i)) for i, (s, im) in enumerate(zip(new_segs, new_arrays[0]))]
    return new_affines, new_segs, new_arrays


def update_affine_after_crop(affine_matrix, crop_start_xy):
    """
    Update affine matrix after cropping an image.
    Args:
        affine_matrix: 4x4 affine transformation matrix
        crop_start_xy: tuple of (x_start, y_start) crop coordinates
    Returns:
        Updated 4x4 affine matrix
    """
    affine_new = affine_matrix.clone()
    rotation = affine_matrix[:3, :3]
    # Calculate offset in scanner coordinates
    crop_start = torch.ones((3,))
    crop_start[:2] = crop_start_xy
    offset = rotation @ crop_start
    # Update translation
    affine_new[:3, 3] += offset
    return affine_new




def find_heart_center_on_slices(affines: List[torch.Tensor],
                                segs: Optional[List[torch.Tensor]] = None,
                                debug=False) -> List[torch.Tensor]:
    # Find heart center coord in mid-ventricular slice
    lv_basal_slice, lv_apex_slice = find_basal_apical_from_sa_segmentation(segs[3:])
    lv_basal_slice, lv_apex_slice = 3+lv_basal_slice, 3+lv_apex_slice
    lv_midventr_slice = (lv_apex_slice + lv_basal_slice) // 2
    midventr_lv_center = get_center_coord(segs[lv_midventr_slice][..., 0] == 1)
    midventr_rv_center = get_center_coord(segs[lv_midventr_slice][..., 0] == 3)
    midventr_heart_center = (midventr_lv_center + midventr_rv_center) / 2

    # Project center coordinate to LA slices
    midventr_heart_center_aug = torch.cat((midventr_heart_center, torch.tensor((0,)), torch.tensor((1,))), dim=0)
    la2ch_center = scanner_to_image_coords(affines[lv_midventr_slice] @ midventr_heart_center_aug, affines[0])[0, :2]
    la3ch_center = scanner_to_image_coords(affines[lv_midventr_slice] @ midventr_heart_center_aug, affines[1])[0, :2]
    la4ch_center = scanner_to_image_coords(affines[lv_midventr_slice] @ midventr_heart_center_aug, affines[2])[0, :2]
    # Assume SA slices are parallel. heart center is other SA slices is same as mid-ventricular slice
    sa_centers = [midventr_heart_center] * len(affines[3:])
    centers = [la2ch_center, la3ch_center, la4ch_center, *sa_centers]
    return centers


def find_LV_center_on_slices_from_intersections(affines: List[torch.Tensor],
                                                segs: Optional[List[torch.Tensor]] = None,
                                                debug=False) -> List[torch.Tensor]:
    # Get LA vector from 4ch and 2ch intersection
    plane_equations = get_image_plane_from_affine(torch.stack(affines))
    la_vector_points = plane_intersection(plane_equations[2][None], plane_equations[0][None])[0]

    # Find LV center coordinate for each SA slice
    sa_lv_centers_scanner = []
    sa_lv_centers_image_xy = []

    for i, (sa_slice_eq, sa_aff) in enumerate(zip(plane_equations[3:], affines[3:])):
        lv_center_scanner = plane_line_intersection(la_vector_points[0], la_vector_points[1], sa_slice_eq)
        sa_lv_centers_scanner.append(lv_center_scanner)
        # Convert to image coordinates
        # TODO: Currently Z image-space coordinate is nowhere close to 0.0. Why??
        lv_center_image = scanner_to_image_coords(lv_center_scanner, sa_aff)
        lv_center_image_xy = lv_center_image[:2]
        sa_lv_centers_image_xy.append(lv_center_image_xy)

    # Handle long-axis views normally
    num_sa_slices = len(affines[3:])
    mid_sa_relative_idx = num_sa_slices // 2
    mid_sa_absolute_idx = 3 + mid_sa_relative_idx

    sa_2ch_vector_points = plane_intersection(plane_equations[mid_sa_absolute_idx][None], plane_equations[0][None])[0]
    sa_4ch_vector_points = plane_intersection(plane_equations[mid_sa_absolute_idx][None], plane_equations[2][None])[0]

    lv_center_2ch_scanner = plane_line_intersection(sa_4ch_vector_points[0], sa_4ch_vector_points[1],
                                                    plane_equations[0])
    lv_center_3ch_scanner = plane_line_intersection(sa_4ch_vector_points[0], sa_4ch_vector_points[1],
                                                    plane_equations[1])
    lv_center_4ch_scanner = plane_line_intersection(sa_2ch_vector_points[0], sa_2ch_vector_points[1],
                                                    plane_equations[2])

    # Convert LA centers to image coordinates
    lv_center_2ch_image = scanner_to_image_coords(lv_center_2ch_scanner, affines[0])
    lv_center_3ch_image = scanner_to_image_coords(lv_center_3ch_scanner, affines[1])
    lv_center_4ch_image = scanner_to_image_coords(lv_center_4ch_scanner, affines[2])
    lv_center_2ch_image_xy = lv_center_2ch_image[:2]
    lv_center_3ch_image_xy = lv_center_3ch_image[:2]
    lv_center_4ch_image_xy = lv_center_4ch_image[:2]
    la_lv_centers_image = [lv_center_2ch_image_xy, lv_center_3ch_image_xy, lv_center_4ch_image_xy] + sa_lv_centers_image_xy

    return la_lv_centers_image


def find_basal_apical_from_sa_segmentation(segs: List[torch.Tensor],
                                           basal_seg_thresh=200,  # TODO: Check thresh param is ok
                                           apical_seg_thresh=10):
    if segs is not None:
        # If no segmentations provided, we work with top-most and bottom-most SA slices
        lv_basal_slice = 0
        lv_apex_slice = len(segs) - 1
    else:
        # Iterate from base to apex until we find a LV_pool segmentation
        lv_basal_slice = None
        for i in range(len(segs)):
            slice_lv_sum_per_frame = (segs[i]==1).sum(0,1)  # How much blood pool is present?
            slice_lv_presence_per_frame = slice_lv_sum_per_frame > basal_seg_thresh
            # If 50% of frames are above seg size thresh, this is basal slice
            if slice_lv_presence_per_frame.sum() / slice_lv_presence_per_frame.shape[-1] > 0.5:
                lv_basal_slice = i
                break
            if i > 3:
                raise ValueError("Heart base is weirdly low!")
        # Iterate from apex to base until we find a LV_pool segmentation
        lv_apex_slice = None
        for i in range(len(segs) - 1, 0, -1):
            slice_lv_sum_per_frame = (segs[i]==2).sum(0,1)  # How much myocardium is present?
            slice_lv_presence_per_frame = slice_lv_sum_per_frame > apical_seg_thresh
            # If 50% of frames are above seg size thresh, this is apical slice
            if slice_lv_presence_per_frame.sum() / slice_lv_presence_per_frame.shape[-1] > 0.5:
                lv_apex_slice = i
                break
            if i < len(segs) - 3:
                raise ValueError("Heart apex is weirdly high!")
    return lv_basal_slice, lv_apex_slice


def normalize_slice_orientation(affines: List[torch.Tensor],
                                segs: Optional[List[torch.Tensor]] = None,
                                myo_search_step_size: int = 1,
                                myo_search_num_samples: int = 100,
                                use_sa_normal_as_la: bool = True,
                                debug=False) -> List[torch.Tensor]:
    """ We find landmarks in ED frame, first 3 slices are LA slices """
    lv_basal_slice, lv_apex_slice = find_basal_apical_from_sa_segmentation(segs[3:])
    lv_basal_slice, lv_apex_slice = 3+lv_basal_slice, 3+lv_apex_slice
    # Find long-axis (la) vector and where it intersects with middle short-axis (sa) slice. LV=Left ventricle, RV=Right ventricle
    lv_midventr_slice = (lv_apex_slice + lv_basal_slice) // 2

    affines = [a.double() for a in affines]
    plane_equations = get_image_plane_from_affine(torch.stack(affines)).double()

    lv_rv_vector_points = plane_intersection(plane_equations[2][None], plane_equations[lv_midventr_slice][None])[0].double()
    if use_sa_normal_as_la:
        # Use SA normal as long axis vector
        la_vector = plane_equations[lv_midventr_slice][:3]
        la_2ch4ch_intersection = plane_intersection(plane_equations[2][None], plane_equations[0][None])[0].double()
        la_point = plane_line_intersection(la_2ch4ch_intersection[1], la_2ch4ch_intersection[0], plane_equations[lv_midventr_slice])
        la_vector_points = torch.stack((la_point, la_point + la_vector))
    else:
        # Use  2ch and 4ch intersection as long axis vector
        la_vector_points = plane_intersection(plane_equations[2][None], plane_equations[0][None])[0].double()
        la_vector = la_vector_points[1] - la_vector_points[0]

    # Calculate affine matrix that aligns LA vector with z axis
    z_vector = torch.tensor((0, 0, 1), dtype=la_vector.dtype)
    la_to_z_angle = angle_between_vectors(z_vector, la_vector)  # the angle to z axis
    # torch.arccos(torch.dot(la_vector / la_vector.norm(), z_vector).clip(-1., 1.))
    la_rot_axis = torch.cross(z_vector, la_vector)
    la_rot_axis = la_rot_axis / la_rot_axis.norm()
    rot_la = rotation_matrix(la_to_z_angle, la_rot_axis)

    # Point where LA vector crosses SA mid-ventr plane is LV center
    w_lv_center = plane_line_intersection(la_vector_points[1], la_vector_points[0],
                                                 plane_equations[lv_midventr_slice])
    w_lv_center_aug = torch.cat((w_lv_center, torch.ones_like(w_lv_center[:1])), dim=-1)[None]
    i_lv_center_aug = (torch.linalg.inv(affines[lv_midventr_slice]) @ w_lv_center_aug.T).T
    i_lv_center = i_lv_center_aug[:, :2]
    lv_rv_vector_points_aug = torch.cat((lv_rv_vector_points, torch.ones_like(lv_rv_vector_points[:,:1])), dim=-1)
    lv_rv_vector_points_image_aug = (torch.linalg.inv(affines[lv_midventr_slice]) @ lv_rv_vector_points_aug.T).T
    assert lv_rv_vector_points_image_aug[:,2].abs().sum() < 1e-4,  "z coord in image coords should be near 0.0"
    assert (lv_rv_vector_points_image_aug[:,3] - 1.).abs().sum() < 1e-5,  "Aug corner should be near 1.0"
    # In order to find the heart center, we sample segmentatin along the 4ch intersection line on the SA mid-ventr slice
    # Then find which direction leads towards the RV, and find the center of the MYO seg class between LV and RV blobs
    lv_rv_vector_points_image = lv_rv_vector_points_image_aug[:, :2]
    lv_rv_vector_image = lv_rv_vector_points_image[1] - lv_rv_vector_points_image[0]
    if segs is None:
        raise ValueError
    else:
        lv_rv_vector_image_step_size = lv_rv_vector_image / lv_rv_vector_image.norm() * myo_search_step_size
        lv_rv_vector_image_steps = lv_rv_vector_image_step_size[None].tile(myo_search_num_samples, 1)
        lv_rv_vector_image_steps *= torch.arange(-myo_search_num_samples // 2, myo_search_num_samples // 2)[:, None]
        lv_myo_sample_points = i_lv_center.tile(myo_search_num_samples, 1) + lv_rv_vector_image_steps
        lv_myo_sample_points_norm = (lv_myo_sample_points / torch.tensor(segs[lv_midventr_slice].shape[:2]) * 2 - 1)
        lv_sample_segs = torch.nn.functional.grid_sample(segs[lv_midventr_slice][None,None,...,0].double(),
                                                             lv_myo_sample_points_norm[None,None].flip(-1),
                                                             mode='nearest', align_corners=True).to(torch.uint8).squeeze()
        if not (lv_sample_segs==3).any():
            raise ValueError( f"Slice {lv_midventr_slice}, {lv_sample_segs}")
        rv_center = torch.where(lv_sample_segs==3)[0].median()
        if rv_center >= myo_search_num_samples//2:
            rv_to_lv_samples = lv_sample_segs[myo_search_num_samples//2:]
            myo_center_along_line = torch.where(rv_to_lv_samples==2)[0].median() + myo_search_num_samples//2
        else:
            rv_to_lv_samples = lv_sample_segs[:myo_search_num_samples//2]
            myo_center_along_line = torch.where(rv_to_lv_samples==2)[0].median()
        i_myo_center = lv_myo_sample_points[myo_center_along_line]
    i_myo_center_aug = torch.cat((i_myo_center, torch.zeros_like(i_myo_center[:1]), torch.ones_like(i_myo_center[:1])), -1)
    w_myo_center_aug = (affines[lv_midventr_slice] @ i_myo_center_aug.T).T
    w_myo_center = w_myo_center_aug[:-1]

    # First translate to put rotation center at origin
    translate = torch.eye(4, dtype=la_vector.dtype)
    translate[:3, 3] = -w_myo_center
    # Then rotate (inverse rotation)
    rotation_inv = torch.eye(4, dtype=la_vector.dtype)
    rotation_inv[:3, :3] = rot_la.T  # Transpose for inverse
    normalization_aff = rotation_inv @ translate
    oriented_affines = [normalization_aff @ aff for aff in affines]

    plane_equations = get_image_plane_from_array(torch.stack(oriented_affines))
    lv_rv_vector_points = plane_intersection(plane_equations[2][None], plane_equations[lv_midventr_slice][None])[0].double()
    lv_rv_vector = lv_rv_vector_points[1] - lv_rv_vector_points[0]
    if use_sa_normal_as_la:
        la_vector = plane_equations[lv_midventr_slice][:3]
    else:
        la_vector_points = plane_intersection(plane_equations[2][None], plane_equations[0][None])[0].double()
        la_vector = la_vector_points[1] - la_vector_points[0]
    # Calculate affine matrix that aligns LV-RV vector with y axis
    y_vector = torch.tensor((0, 1, 0), dtype=lv_rv_vector.dtype)
    rv_la_cross_vector = torch.cross(la_vector, lv_rv_vector)
    rv_la_cross_vector = rv_la_cross_vector / rv_la_cross_vector.norm()
    rv_to_y_angle = angle_between_vectors(rv_la_cross_vector, y_vector) # the angle to y axis
    rot_rv = rotation_matrix(rv_to_y_angle, la_vector)
    normalization_aff = torch.eye(4, dtype=la_vector.dtype)
    # Rotate rv vector to y axis (inverse rotation)
    normalization_aff[:3, :3] = rot_rv.T
    final_affines = [normalization_aff @ aff for aff in oriented_affines]

    final_affines = [a.float() for a in final_affines]

    if debug:
        # Intermediate rotation
        planes = get_image_plane_from_array(torch.stack(oriented_affines))
        v = planes[lv_midventr_slice][:3]
        ang_sacross_la_z_1 = torch.rad2deg(angle_between_vectors(v, torch.tensor([0, 0., 1.])))
        ang_sacross_la_y_1 = torch.rad2deg(angle_between_vectors(v, torch.tensor([0., 1., 0.])))
        ang_sacross_la_x_1 = torch.rad2deg(angle_between_vectors(v, torch.tensor([1., 0., 0.])))
        v = plane_intersection(planes[0:1], planes[2:3])
        ang_24ch_la_z_1 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([0, 0., 1.])))
        ang_24ch_la_y_1 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([0., 1., 0.])))
        ang_24ch_la_x_1 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([1., 0., 0.])))
        w_lv_center_1 = plane_line_intersection(v[0, 1], v[0, 0],
                                         planes[lv_midventr_slice])
        w_lv_center_1_aug = torch.cat((w_lv_center_1, torch.ones_like(w_lv_center_1[:1])))
        w_myo_center_1 = (oriented_affines[lv_midventr_slice] @ i_myo_center_aug.T).T

        v = plane_intersection(planes[lv_midventr_slice][None], planes[2:3])
        ang_lvrv_z_2 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([0, 0., 1.]))).abs().item()
        ang_lvrv_y_2 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([0., 1., 0.]))).abs().item()
        ang_lvrv_x_2 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([1., 0., 0.]))).abs().item()

        # final normalization
        planes = get_image_plane_from_array(torch.stack(final_affines))
        v = planes[lv_midventr_slice][:3]
        ang_sacross_la_z_2 = torch.rad2deg(angle_between_vectors(v, torch.tensor([0, 0., 1.]))).abs().item()
        ang_sacross_la_z_2 = min(ang_sacross_la_z_2, abs(180-ang_sacross_la_z_2))
        ang_sacross_la_y_2 = torch.rad2deg(angle_between_vectors(v, torch.tensor([0., 1., 0.]))).abs().item()
        ang_sacross_la_x_2 = torch.rad2deg(angle_between_vectors(v, torch.tensor([1., 0., 0.]))).abs().item()
        thresh = 1.0 if use_sa_normal_as_la else 10.
        if ang_sacross_la_z_2 > thresh:
            raise ValueError
        if abs(ang_sacross_la_y_2 - 90) > thresh or abs(ang_sacross_la_x_2 - 90) > thresh:
            raise ValueError
        v = plane_intersection(planes[0:1], planes[2:3])
        ang_24ch_la_z_2 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([0, 0., 1.]))).abs().item()
        ang_24ch_la_z_2 = min(ang_24ch_la_z_2, abs(180-ang_24ch_la_z_2))
        ang_24ch_la_y_2 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([0., 1., 0.]))).abs().item()
        ang_24ch_la_x_2 = torch.rad2deg(angle_between_vectors(v[0, 1] - v[0, 0], torch.tensor([1., 0., 0.]))).abs().item()
        thresh = 10.0 if use_sa_normal_as_la else 1.
        if ang_24ch_la_z_2 > thresh:
            raise ValueError
        if abs(ang_24ch_la_y_2 - 90) > thresh or abs(ang_24ch_la_x_2 - 90) > thresh:
            raise ValueError
        w_lv_center_2 = plane_line_intersection(v[0, 1], v[0, 0],
                                            planes[lv_midventr_slice])
        w_myo_center_2 = (oriented_affines[lv_midventr_slice] @ i_myo_center_aug.T).T
        if  w_myo_center_2[:3].norm() > 1e-4:
            raise ValueError
        # -------- Logging segmentations as niftis ----------------------------
        assert segs is not None
        la2ch_seg = segs[0]
        if not la2ch_seg.any():
            la2ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in affines[3:]],
                                                     target_shape=(segs[0].shape[0], segs[0].shape[1], segs[0].shape[-1]),
                                                     target_aff=affines[0].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        la3ch_seg = segs[0]
        if not la3ch_seg.any():
            la3ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in affines[3:]],
                                                     target_shape=(segs[1].shape[0], segs[1].shape[1], segs[1].shape[-1]),
                                                     target_aff=affines[1].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        la4ch_seg = segs[0]
        if not la4ch_seg.any():
            la4ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in affines[3:]],
                                                     target_shape=(segs[2].shape[0], segs[2].shape[1], segs[2].shape[-1]),
                                                     target_aff=affines[2].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        path = Path('debug_alignment')
        fold = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        path = path / fold
        path.mkdir(exist_ok=True, parents=True)
        array_to_nifti(str(path/f"pre_opt_la2ch.nii.gz"), la2ch_seg[:, :, None], affines[0].numpy())
        array_to_nifti(str(path/f"pre_opt_la3ch.nii.gz"), la3ch_seg[:, :, None], affines[1].numpy())
        array_to_nifti(str(path/f"pre_opt_la4ch.nii.gz"), la4ch_seg[:, :, None], affines[2].numpy())
        array_to_nifti(str(path/f"pre_opt_sa3.nii.gz"), segs[6][:, :, None].numpy(), affines[6].numpy())
        array_to_nifti(str(path/f"pre_opt_sa4.nii.gz"), segs[7][:, :, None].numpy(), affines[7].numpy())
        array_to_nifti(str(path/f"pre_opt_sa5.nii.gz"), segs[8][:, :, None].numpy(), affines[8].numpy())
        array_to_nifti(str(path/f"post_opt_la2ch.nii.gz"), la2ch_seg[:, :, None], final_affines[0].numpy())
        array_to_nifti(str(path/f"post_opt_la3ch.nii.gz"), la3ch_seg[:, :, None], final_affines[1].numpy())
        array_to_nifti(str(path/f"post_opt_la4ch.nii.gz"), la4ch_seg[:, :, None], final_affines[2].numpy())
        array_to_nifti(str(path/f"post_opt_sa3.nii.gz"), segs[6][:, :, None].numpy(), final_affines[6].numpy())
        array_to_nifti(str(path/f"post_opt_sa4.nii.gz"), segs[7][:, :, None].numpy(), final_affines[7].numpy())
        array_to_nifti(str(path/f"post_opt_sa5.nii.gz"), segs[8][:, :, None].numpy(), final_affines[8].numpy())
    return final_affines
