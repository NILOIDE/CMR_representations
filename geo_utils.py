from typing import Tuple

import numpy as np
import torch

Line = Tuple[np.ndarray, np.ndarray]
PlaneEq = np.ndarray


def batch_normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    assert len(vec.shape) == 2
    vec_n = vec / vec.norm(dim=-1)[:, None].tile((1, 3))
    return vec_n


def calculate_angle_between_planes(plane1, plane2):
    # Calculate the normal vectors of the planes
    normal_vector1 = calculate_normal_vector(plane1)
    normal_vector2 = calculate_normal_vector(plane2)
    # Calculate the dot product of the normal vectors
    dot_product = np.dot(normal_vector1, normal_vector2)
    # Calculate the angle between the planes using the dot product
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def get_image_plane_from_array(affines):
    points_voxel_space = torch.tensor([[0., 0., 0., 1.],
                                       [1., 0., 0., 1.],
                                       [0., 1., 0., 1.]
                                       ], dtype=torch.float32)
    points_voxel_space = torch.tile(points_voxel_space, (affines.shape[0], 1))
    affines_ = torch.repeat_interleave(affines, 3, dim=0)
    points_scanner_space = torch.einsum("ijk,ik->ij", [affines_, points_voxel_space]).reshape(affines.shape[0], 3, -1)
    return get_image_plane(points_scanner_space)


def get_image_plane(points: torch.Tensor) -> torch.Tensor:
    assert points.shape[1] == 3 and points.shape[2] >= 3
    # # Create a matrix A from the coordinates
    # A = points[..., :3]
    # # Create a vector B with ones
    # B = torch.ones((points.shape[0], 3))
    # # Solve the linear equation Ax = B
    # x = torch.linalg.solve(A, B)  # TODO:
    # # Extract the coefficients of the plane equation
    # a, b, c = x
    # # Compute the constant term d
    # d = -np.dot(x, points[0])

    v1 = points[:, 0, :3] - points[:, 2, :3]  # Vector 1
    v2 = points[:, 1, :3] - points[:, 2, :3]  # Vector 2
    normal = torch.cross(v1, v2)  # Normal to plane
    # https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
    # evaluates a * x3 + b * y3 + c * z3 which equals d
    d = torch.einsum("ij,ij->i", [normal, points[:, 0, :3]])  # dot(normal, point)
    # Return the plane equation coefficients
    plane_eq = torch.cat((normal, d[:, None]), dim=1)  #TODO
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
    # aXb_vec = intersec_dir / np.linalg.norm(intersec_dir) * 10

    # x = (a[1] * b[3] - b[1] * a[3]) / (a[0] * b[1] - b[0] * a[1])
    # y = (b[0] * a[3] - a[0] * b[3]) / (a[0] * b[1] - b[0] * a[1])
    # point_on_line = np.array([x, y, 0])
    A = torch.stack([a_normal, b_normal, dir_inter], dim=1)
    d = torch.stack([a[:, 3], b[:, 3], torch.zeros((a.shape[0],))], dim=1)
    p_inter = torch.linalg.solve(A, d)  # TODO
    line = torch.stack((p_inter, p_inter + dir_inter), dim=1)
    return line


def plane_line_intersection(plane_eq, line: Line):
    line_direction = line[1] - line[0]
    plane_eq = np.array(plane_eq)
    dot_prod = np.dot(plane_eq[:3], line_direction)

    # Check if the dot product is zero, which means the line is parallel to the plane
    if dot_prod == 0:
        raise ValueError("Line does not intersect plane")
    # Compute the parameter t that gives the intersection point
    line_direction_ = np.array([*line_direction, 1])
    t = - np.dot(plane_eq, line_direction_) / dot_prod
    # Compute the intersection point by plugging t into the line equation
    inter_pt = line[0] + t * line_direction
    return inter_pt


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


def get_image_edge_planes(im, affine) -> Tuple[PlaneEq, PlaneEq, PlaneEq, PlaneEq]:
    corner1 = affine @ np.array([0, 0, 0, 1])
    corner2 = affine @ np.array([0, im.shape[1]-1, 0, 1])
    corner3 = affine @ np.array([im.shape[0]-1, im.shape[1]-1, 0, 1])
    corner4 = affine @ np.array([im.shape[0]-1, 0, 0, 1])
    plane = get_image_plane([corner1[:3], corner2[:3], corner3[:3]])
    plane_normal = plane[:3]
    plane12 = get_image_plane([corner1[:3], corner2[:3], corner1[:3] + plane_normal])
    plane23 = get_image_plane([corner2[:3], corner3[:3], corner2[:3] + plane_normal])
    plane34 = get_image_plane([corner3[:3], corner4[:3], corner3[:3] + plane_normal])
    plane41 = get_image_plane([corner4[:3], corner1[:3], corner4[:3] + plane_normal])
    return plane12, plane23, plane34, plane41


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
    nifti1_plane = get_image_plane_from_array(nifti1, z1)
    nifti2_plane = get_image_plane_from_array(nifti2, z2)
    angle = calculate_angle_between_planes(nifti1_plane, nifti2_plane)
    return angle


def calculate_distance_between_nifti_plane_and_center(nifti1, z1, nifti2, z2):
    nifti1_plane = get_image_plane_from_array(nifti1, z1)  # Mid-ventricular slice == 4 (5th slice from base)
    center = get_center_point_from_nifti_slice(nifti2, z2)
    dist = calculate_distance_point_to_plane(center, nifti1_plane)
    return dist
