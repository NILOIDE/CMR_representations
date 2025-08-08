from pathlib import Path

import numpy as np
import torch
from typing import List, Optional, Tuple

from data_utils import array_to_nifti
from geo_utils import get_image_plane_from_affine, plane_intersection, plane_line_intersection, rotation_matrix
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
    new_origins = (centers - crop_sizes//2).clip(min=0)
    crop_ends = new_origins + crop_sizes
    new_segs = [i[o[0]:e[0], o[1]:e[1]] for i, o, e in zip(segs, new_origins, crop_ends)]
    new_arrays = []
    for i, arr in enumerate(arrays):
        new_arr = [i[o[0]:e[0], o[1]:e[1]] for i, o, e in zip(arr, new_origins, crop_ends)]
        new_arrays.append(new_arr)
    if debug:
        [to_gif(torch.cat((s.float()/4, im), dim=1), 'debug_crop', str(i)) for i, (s, im) in enumerate(zip(new_segs, new_arrays[0]))]
    return affines, new_segs, new_arrays



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


def normalize_slice_orientation(affines: List[torch.Tensor], segs: Optional[List[torch.Tensor]] = None, debug=False) -> List[torch.Tensor]:
    """ We find landmarks in ED frame, first 3 slices are LA slices """
    lv_basal_slice, lv_apex_slice = find_basal_apical_from_sa_segmentation(segs[3:])
    lv_basal_slice, lv_apex_slice = 3+lv_basal_slice, 3+lv_apex_slice
    # Find long-axis (la) vector and where it intersects with middle short-axis (sa) slice. LV=Left ventricle, RV=Right ventricle
    lv_midventr_slice = (lv_apex_slice + lv_basal_slice) // 2

    # Find LV center on slices above and below the mid-ventricular slice to define LA vector
    above_midventr_lv_center = get_center_coord(segs[lv_midventr_slice-1][..., 0] == 1)
    above_midventr_lv_center_aug = torch.cat((above_midventr_lv_center, torch.tensor((0,)), torch.tensor((1,))), dim=0)
    w_above_midventr_center_aug = affines[lv_midventr_slice-1] @ above_midventr_lv_center_aug
    below_midventr_lv_center = get_center_coord(segs[lv_midventr_slice+1][..., 0] == 1)
    below_midventr_lv_center_aug = torch.cat((below_midventr_lv_center, torch.tensor((0,)), torch.tensor((1,))), dim=0)
    w_below_midventr_lv_center_aug = affines[lv_midventr_slice+1] @ below_midventr_lv_center_aug
    la_vector_points = (w_above_midventr_center_aug[:3], w_below_midventr_lv_center_aug[:3])

    # Find LV center and RV center on the mid-ventricular slice to define LV-RV vector
    w_lv_midway_center = (w_above_midventr_center_aug[:3] + w_below_midventr_lv_center_aug[:3]) / 2
    rv_midway_center = get_center_coord(segs[lv_midventr_slice][..., 0] == 3)
    rv_midway_center_aug = torch.cat((rv_midway_center, torch.tensor((0,)), torch.tensor((1,))), dim=0)
    w_rv_midway_center_aug = affines[lv_midventr_slice] @ rv_midway_center_aug
    lv_rv_vector_points = (w_lv_midway_center[:3], w_rv_midway_center_aug[:3])

    w_hear_center = (lv_rv_vector_points[0] + lv_rv_vector_points[1]) / 2

    # plane_equations = get_image_plane_from_affine(torch.stack(affines))
    # la_vector_points = plane_intersection(plane_equations[2][None], plane_equations[0][None])[0]
    la_vector = la_vector_points[1] - la_vector_points[0]
    # lv_rv_vector_points = plane_intersection(plane_equations[2][None], plane_equations[lv_midventr_slice][None])[0]
    lv_rv_vector = lv_rv_vector_points[1] - lv_rv_vector_points[0]
    # w_lv_midway_center = plane_line_intersection(la_vector_points[1], la_vector_points[0],
    #                                              plane_equations[lv_midventr_slice])
    # Calculate affine matrix that aligns LA vector with z axis
    z_vector = torch.tensor((0, 0, 1), dtype=la_vector.dtype)
    la_to_z_angle = torch.arccos(torch.dot(la_vector / la_vector.norm(), z_vector).clip(-1., 1.))  # the angle to z axis
    la_rot_axis = torch.cross(z_vector, la_vector)
    la_rot_axis = la_rot_axis / la_rot_axis.norm()
    rot_la = rotation_matrix(la_to_z_angle, la_rot_axis)
    # Calculate affine matrix that aligns LV-RV vector with y axis
    y_vector = torch.tensor((0, 1, 0), dtype=lv_rv_vector.dtype)
    rv_la_cross_vector = torch.cross(la_vector, lv_rv_vector) @ rot_la
    rv_la_cross_vector = rv_la_cross_vector / rv_la_cross_vector.norm()
    la_to_y_angle = torch.arccos(
        torch.dot(rv_la_cross_vector / rv_la_cross_vector.norm(), y_vector).clip(-1., 1.))  # the angle to y axis
    rot_rv = rotation_matrix(la_to_y_angle, z_vector)

    # First translate to put rotation center at origin
    translate = torch.eye(4, dtype=la_vector.dtype)
    translate[:3, 3] = -w_hear_center
    # Then rotate (inverse rotation)
    rotation_inv = torch.eye(4, dtype=la_vector.dtype)
    rotation_inv[:3, :3] = (rot_rv @ rot_la).T  # Transpose for inverse
    # Then translate back
    # post_translate = torch.eye(4, dtype=la_vector.dtype)
    # post_translate[:3, 3] = w_lv_midway_center
    # Combined transformation: post_translate @ rotation_inv @ pre_translate
    normalization_aff = rotation_inv @ translate
    oriented_affines = [normalization_aff @ aff for aff in affines]

    if debug:
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
        path.mkdir(exist_ok=True)
        array_to_nifti(str(path/f"pre_opt_la2ch.nii.gz"), la2ch_seg[:, :, None], affines[0].numpy())
        array_to_nifti(str(path/f"pre_opt_la3ch.nii.gz"), la3ch_seg[:, :, None], affines[1].numpy())
        array_to_nifti(str(path/f"pre_opt_la4ch.nii.gz"), la4ch_seg[:, :, None], affines[2].numpy())
        array_to_nifti(str(path/f"pre_opt_sa3.nii.gz"), segs[5][:, :, None].numpy(), affines[5].numpy())
        array_to_nifti(str(path/f"pre_opt_sa4.nii.gz"), segs[6][:, :, None].numpy(), affines[6].numpy())
        array_to_nifti(str(path/f"pre_opt_sa5.nii.gz"), segs[7][:, :, None].numpy(), affines[7].numpy())
        array_to_nifti(str(path/f"post_opt_la2ch.nii.gz"), la2ch_seg[:, :, None], oriented_affines[0].numpy())
        array_to_nifti(str(path/f"post_opt_la3ch.nii.gz"), la3ch_seg[:, :, None], oriented_affines[1].numpy())
        array_to_nifti(str(path/f"post_opt_la4ch.nii.gz"), la4ch_seg[:, :, None], oriented_affines[2].numpy())
        array_to_nifti(str(path/f"post_opt_sa3.nii.gz"), segs[5][:, :, None].numpy(), oriented_affines[5].numpy())
        array_to_nifti(str(path/f"post_opt_sa4.nii.gz"), segs[6][:, :, None].numpy(), oriented_affines[6].numpy())
        array_to_nifti(str(path/f"post_opt_sa5.nii.gz"), segs[7][:, :, None].numpy(), oriented_affines[7].numpy())
    return oriented_affines


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
