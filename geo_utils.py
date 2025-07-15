import math
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch

from data_utils import array_to_nifti
from sa_la_interp import interpolate_sa_segs_to_la


def batch_normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    assert len(vec.shape) == 2
    vec_n = vec / vec.norm(dim=-1)[:, None].tile((1, 3))
    return vec_n


def get_image_plane_from_array(affines):
    points_voxel_space = torch.tensor([[0., 0., 0., 1.],
                                       [1., 0., 0., 1.],
                                       [0., 1., 0., 1.]
                                       ], dtype=torch.float32, device=affines.device)
    points_voxel_space = torch.tile(points_voxel_space, (affines.shape[0], 1))
    affines_ = torch.repeat_interleave(affines, 3, dim=0)
    points_scanner_space = torch.einsum("ijk,ik->ij", [affines_, points_voxel_space]).reshape(affines.shape[0], 3, -1)
    return get_image_plane(points_scanner_space)


def get_image_plane(points: torch.Tensor) -> torch.Tensor:
    assert points.shape[1] == 3 and points.shape[2] >= 3
    v1 = points[:, 0, :3] - points[:, 2, :3]  # Vector 1
    v2 = points[:, 1, :3] - points[:, 2, :3]  # Vector 2
    normal = torch.cross(v1, v2)  # Normal to plane
    # https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
    # evaluates a * x3 + b * y3 + c * z3 which equals d
    d = torch.einsum("ij,ij->i", [normal, points[:, 0, :3]])  # dot(normal, point)
    # Return the plane equation coefficients
    plane_eq = torch.cat((normal, d[:, None]), dim=1)
    return plane_eq


def plane_intersection(a: torch.Tensor, b: torch.Tensor):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order
    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_normal, b_normal = a[:, :3], b[:, :3]
    dir_inter = torch.cross(a_normal, b_normal)  # Line direction
    A = torch.stack([a_normal, b_normal, dir_inter], dim=1)
    solution = torch.zeros((a.shape[0],), dtype=a.dtype, device=a.device)
    d = torch.stack([a[:, 3], b[:, 3], solution], dim=1)
    p_inter = torch.linalg.solve(A, d)  # TODO
    line = torch.stack((p_inter, p_inter + dir_inter), dim=1)
    return line


def plane_line_intersection(p0, p1, plane, epsilon=1e-6):
    """ Adapted from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection """
    u = (p1 - p0)
    tdenom = plane[:3] @ u
    if abs(tdenom) < epsilon:
        raise ValueError("Line is parallel to plane.")

    tnumer = plane[:3] @ p0 + plane[3]
    t = -tnumer / tdenom
    out = p0 + t * u
    return out


def closest_point_on_line(line: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    # https://blender.stackexchange.com/questions/94464/finding-the-closest-point-on-a-line-defined-by-two-points
    assert len(line.shape) == 3
    assert line.shape[1] == 2
    assert line.shape[2] == 3
    assert len(point.shape) == 2
    assert point.shape[-1] == 3
    direction_line = line[:, 1] - line[:, 0]
    direction_line_n = batch_normalize_vector(direction_line)
    direction_point = point - line[:, 0]
    # Dot product gives us distance to projected point along line
    dist_along_line = torch.einsum("ij,ij->i", [direction_point, direction_line_n])  # Batch-wise dot product
    # Projected point is start of line plus (distance * direction)
    projected_point = line[:, 0] + dist_along_line[:, None].tile((1, 3)) * direction_line_n
    return projected_point


def get_image_plane_from_affine(affines):
    points_voxel_space = torch.tensor([[0., 0., 0., 1.],
                                       [1., 0., 0., 1.],
                                       [0., 1., 0., 1.]
                                       ], dtype=affines.dtype, device=affines.device)
    points_voxel_space = torch.tile(points_voxel_space, (affines.shape[0], 1))
    affines_ = torch.repeat_interleave(affines, 3, dim=0)
    points_scanner_space = torch.einsum("ijk,ik->ij", [affines_, points_voxel_space]).reshape(affines.shape[0], 3, -1)
    return get_image_plane(points_scanner_space)


def rotation_matrix(theta, axis):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.linalg.norm(axis)
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], dtype=torch.float32)


def normalize_slice_orientation(affines: List[torch.Tensor], segs: Optional[List[torch.Tensor]] = None, debug=False) -> List[torch.Tensor]:
    """ We find landmarks in ED frame, first 3 slices are LA slices """
    ED_frame = 0
    if segs is not None:
        # If no segmentations provided, we work with top-most and bottom-most SA slices
        lv_basal_slice = 3
        lv_apex_slice = len(affines) - 1
    else:
        # Iterate from base to apex until we find a LV_pool segmentation
        lv_basal_slice = None
        for i in range(len(segs)):
            s = segs[3 + i][ED_frame]
            if (s == 1).any():
                lv_basal_slice = 3 + i
                break
            if i > 3:
                raise ValueError("Heart base is weirdly low!")
        # Iterate from apex to base until we find a LV_pool segmentation
        lv_apex_slice = None
        for i in range(len(segs) - 1, 3, -1):
            s = segs[i][ED_frame]
            if (s == 1).any():
                lv_apex_slice = i
                break
            if i < len(segs) - 3:
                raise ValueError("Heart apex is weirdly high!")
        # Define landmark coordinates
        base_seg_thresh = 200  # TODO: Check thresh param is ok
        if torch.sum(segs[lv_basal_slice][ED_frame] == 1) < base_seg_thresh:
            # We do +1 because basal slice is not always properly segmented
            lv_basal_slice = lv_basal_slice + 1
    # Find long-axis (la) vector and where it intersects with middle short-axis (sa) slice. LV=Left ventricle, RV=Right ventricle
    lv_midventr_slice = (lv_apex_slice + lv_basal_slice) // 2
    plane_equations = get_image_plane_from_affine(torch.stack(affines))
    la_vector_points = plane_intersection(plane_equations[2][None], plane_equations[0][None])[0]
    la_vector = la_vector_points[1] - la_vector_points[0]
    lv_rv_vector_points = plane_intersection(plane_equations[2][None], plane_equations[lv_midventr_slice][None])[0]
    lv_rv_vector = lv_rv_vector_points[1] - lv_rv_vector_points[0]
    w_lv_midway_center = plane_line_intersection(la_vector_points[1], la_vector_points[0],
                                                 plane_equations[lv_midventr_slice])
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
    translate[:3, 3] = -w_lv_midway_center
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