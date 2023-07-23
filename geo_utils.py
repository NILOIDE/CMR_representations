from typing import Tuple

import numpy as np

Line = Tuple[np.ndarray, np.ndarray]
PlaneEq = np.ndarray


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
    # Calculate the angle between the planes using the dot product
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def get_image_plane_from_array(im, affine):
    points = get_3_points_from_slice(im, affine)
    return get_image_plane(points)


def get_image_plane(points) -> PlaneEq:
    assert len(points) == 3
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


def get_3_points_from_slice(im, affine):
    center = np.array([im.shape[0]//2, im.shape[1]//2, 0, 1])
    corner0 = np.array([0., 0., 0, 1])
    corner1 = np.array([0., im.shape[1]//2, 0, 1])
    center_ = affine @ center
    corner0_ = affine @ corner0
    corner1_ = affine @ corner1
    return center_[:3], corner0_[:3], corner1_[:3]


def plane_intersection(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order
    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
    aXb_vec = np.cross(a_vec, b_vec)
    aXb_vec = aXb_vec / np.linalg.norm(aXb_vec) * 10

    x = (a[1] * b[3] - b[1] * a[3]) / (a[0] * b[1] - b[0] * a[1])
    y = (b[0] * a[3] - a[0] * b[3]) / (a[0] * b[1] - b[0] * a[1])
    point_on_line = np.array([x, y, 0])

    return point_on_line, point_on_line + aXb_vec


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


def closest_point_on_line(line: Line, point: np.ndarray) -> Line:
    gradient = line[1] - line[0]
    det = np.sum(gradient * gradient)
    a = np.sum(gradient * (point - line[0])) / det
    return line[0] + a * gradient


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
