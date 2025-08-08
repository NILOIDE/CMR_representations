import math
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch

from data_utils import array_to_nifti
from sa_la_interp import interpolate_sa_segs_to_la
from utils import flip_affine


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
