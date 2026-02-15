import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
                      images: List[torch.Tensor],
                      gt_avail_array: List[torch.Tensor],
                      crop_size_2ch: Tuple[int, int] = (80, 80),
                      crop_size_3ch: Tuple[int, int] = (80, 80),
                      crop_size_4ch: Tuple[int, int] = (80, 80),
                      crop_size_sa: Tuple[int, int] = (80, 80),
                      debug=False,
                      rotate=True) \
        -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # 1. Get centers in ORIGINAL coordinates
    centers, angles = find_heart_center_and_rotations(affines, segs, rotate=rotate)

    crop_sizes = torch.tensor((crop_size_2ch, crop_size_3ch, crop_size_4ch,
                               *[crop_size_sa] * len(affines[3:])), dtype=torch.long)

    new_segs = []
    new_images = []
    new_gt_avails = []
    new_affines = []

    for i, (affine, seg, im, gt_avail, center, size, angle) in enumerate(zip(
            affines, segs, images, gt_avail_array, centers, crop_sizes, angles
    )):
        new_origin = (center.round().long() - size // 2).clip(min=0)
        end = new_origin + size
        # Segs
        seg_rot = rotate_slice(seg, angle, mode='nearest', center_rc=center)
        seg_crop = seg_rot[new_origin[0]:end[0], new_origin[1]:end[1]]
        new_segs.append(seg_crop)

        # Images
        im_rot = rotate_slice(im, angle, mode='bilinear', center_rc=center)
        im_crop = im_rot[new_origin[0]:end[0], new_origin[1]:end[1]]
        new_images.append(im_crop)

        # GT Avail
        gt_rot = rotate_slice(gt_avail, angle, mode='nearest', center_rc=center)
        gt_crop = gt_rot[new_origin[0]:end[0], new_origin[1]:end[1]]
        new_gt_avails.append(gt_crop)

        # Update Affine
        # Update logic: Map [Crop Index] -> [Rotated Image Index] -> [Original Image Index] -> [World]
        new_aff = update_affine_arbitrary_rotation_crop(affine, angle, center, new_origin)
        new_affines.append(new_aff)
    if debug:
        path = Path('debug_crop')
        fold = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        path = path / fold
        path.mkdir(exist_ok=True, parents=True)
        # -------- Logging segmentations gifs  ----------------------------
        [to_gif(torch.cat((s.float()/4, im), dim=1), str(path / "gifs"), str(i)) for i, (s, im) in enumerate(zip(new_segs, new_images))]
        # -------- Logging segmentations as niftis ----------------------------
        assert segs is not None
        la2ch_seg = new_segs[0]
        if not la2ch_seg.any():
            la2ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in new_affines[3:]],
                                                     target_shape=(segs[0].shape[0], segs[0].shape[1], segs[0].shape[-1]),
                                                     target_aff=new_affines[0].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        else:
            la2ch_seg = la2ch_seg.numpy()
        la3ch_seg = new_segs[1]
        if not la3ch_seg.any():
            la3ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in new_affines[3:]],
                                                     target_shape=(segs[1].shape[0], segs[1].shape[1], segs[1].shape[-1]),
                                                     target_aff=new_affines[1].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        else:
            la3ch_seg = la3ch_seg.numpy()
        la4ch_seg = new_segs[2]
        if not la4ch_seg.any():
            la4ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in new_affines[3:]],
                                                     target_shape=(segs[2].shape[0], segs[2].shape[1], segs[2].shape[-1]),
                                                     target_aff=new_affines[2].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        else:
            la4ch_seg = la4ch_seg.numpy()
        la2ch_seg = deepcopy(la2ch_seg)
        la3ch_seg = deepcopy(la3ch_seg)
        la4ch_seg = deepcopy(la4ch_seg)
        new_segs_2 = deepcopy(new_segs)
        la2ch_seg[0] = 4
        la2ch_seg[:, 0] = 4
        la3ch_seg[0] = 4
        la3ch_seg[:, 0] = 4
        la4ch_seg[0] = 4
        la4ch_seg[:, 0] = 4
        la2ch_seg[-1] = 4
        la2ch_seg[:, -1] = 4
        la3ch_seg[-1] = 4
        la3ch_seg[:, -1] = 4
        la4ch_seg[-1] = 4
        la4ch_seg[:, -1] = 4
        for s in new_segs_2:
            s[0] = 4
            s[-1] = 4
            s[:, 0] = 4
            s[:, -1] = 4
        array_to_nifti(str(path/f"pre_opt_la2ch.nii.gz"), segs[0][:, :, None].numpy().astype(int), affines[0].numpy())
        array_to_nifti(str(path/f"pre_opt_la3ch.nii.gz"), segs[1][:, :, None].numpy().astype(int), affines[1].numpy())
        array_to_nifti(str(path/f"pre_opt_la4ch.nii.gz"), segs[2][:, :, None].numpy().astype(int), affines[2].numpy())
        array_to_nifti(str(path/f"pre_opt_sa3.nii.gz"), segs[6][:, :, None].numpy().astype(int), affines[6].numpy())
        array_to_nifti(str(path/f"pre_opt_sa4.nii.gz"), segs[7][:, :, None].numpy().astype(int), affines[7].numpy())
        array_to_nifti(str(path/f"pre_opt_sa5.nii.gz"), segs[8][:, :, None].numpy().astype(int), affines[8].numpy())
        array_to_nifti(str(path/f"post_opt_la2ch.nii.gz"), la2ch_seg[:, :, None].astype(int), new_affines[0].numpy())
        array_to_nifti(str(path/f"post_opt_la3ch.nii.gz"), la3ch_seg[:, :, None].astype(int), new_affines[1].numpy())
        array_to_nifti(str(path/f"post_opt_la4ch.nii.gz"), la4ch_seg[:, :, None].astype(int), new_affines[2].numpy())
        array_to_nifti(str(path/f"post_opt_sa3.nii.gz"), new_segs_2[6][:, :, None].numpy().astype(int), new_affines[6].numpy())
        array_to_nifti(str(path/f"post_opt_sa4.nii.gz"), new_segs_2[7][:, :, None].numpy().astype(int), new_affines[7].numpy())
        array_to_nifti(str(path/f"post_opt_sa5.nii.gz"), new_segs_2[8][:, :, None].numpy().astype(int), new_affines[8].numpy())

    return new_affines, new_segs, new_images, new_gt_avails


def update_affine_arbitrary_rotation_crop(original_affine, angle_degrees, pivot_point, crop_offset):
    """
    Computes the new affine matrix after in-plane rotation and cropping.

    Parameters:
    - original_affine: 4x4 numpy array (Pixel -> World)
    - crop_offset: tuple (x, y, z). The top-left corner of the crop
                   *relative to the rotated image grid*.
    - angle_degrees: Rotation angle (counter-clockwise).
    - pivot_point: tuple (x, y, z). The point in the *original image* around which to rotate.

    Returns:
    - new_affine: The 4x4 matrix mapping the new cropped/rotated pixel space
                  directly to world space.
    """

    # --- 1. Create the Crop Matrix (Translation) ---
    # Maps "Cropped Pixel" -> "Rotated (Full) Pixel"
    # This simply adds the offset to the pixel index.
    cx, cy, cz = crop_offset[0], crop_offset[1], 0.0
    M_crop = np.array([
        [1, 0, 0, cx],
        [0, 1, 0, cy],
        [0, 0, 1, cz],
        [0, 0, 0, 1]
    ])

    # --- 2. Create the Rotation Matrix (with Pivot) ---
    # Maps "Rotated Pixel" -> "Original Source Pixel"
    px, py, pz = pivot_point[0], pivot_point[1], 0.0
    theta = math.radians(angle_degrees)
    c, s = math.cos(theta), math.sin(theta)

    # Translate Pivot to Origin
    T_to_origin = np.array([
        [1, 0, 0, -px],
        [0, 1, 0, -py],
        [0, 0, 1, -pz],
        [0, 0, 0, 1]
    ])

    # Rotate around Z (In-plane)
    R_z = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Translate Pivot back
    T_back = np.array([
        [1, 0, 0, px],
        [0, 1, 0, py],
        [0, 0, 1, pz],
        [0, 0, 0, 1]
    ])

    # Compose Rotation: Move -> Rotate -> Move Back
    M_rot = T_back @ R_z @ T_to_origin

    # --- 3. Combine All ---
    # Order: World <- Original <- Rotated <- Cropped
    # We multiply the original affine by the modifiers on the right.
    new_affine = original_affine @ M_rot @ M_crop

    return new_affine


def rotate_point_around_center(point: torch.Tensor, center: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotates a 2D coordinate point around a center point.
    """
    angle_rad = math.radians(angle_deg)
    c, s = math.cos(angle_rad), math.sin(angle_rad)

    # 2D Rotation Matrix (Counter-Clockwise)
    # [ x' ] = [ cos -sin ] [ x - cx ]   [ cx ]
    # [ y' ]   [ sin  cos ] [ y - cy ] + [ cy ]

    # Note: point is usually (row, col) i.e. (y, x).
    # Be careful with x/y vs row/col.
    # Assuming point is (row, col) -> (y, x):
    # If we rotate image CCW, the point (row, col) also moves CCW.

    rel = point - center
    # Matrix mult manually
    # new_row = rel_row * cos - rel_col * sin
    # new_col = rel_row * sin + rel_col * cos
    new_row = rel[0] * c - rel[1] * s
    new_col = rel[0] * s + rel[1] * c

    return center + torch.stack([new_row, new_col])


def rotate_slice(tensor: torch.Tensor,
                 angle_deg: float,
                 center_rc: Optional[torch.Tensor] = None,
                 mode: str = 'bilinear') -> torch.Tensor:
    """
    Rotates a (H, W, T) tensor around a specific center point.

    Args:
        tensor: Input tensor of shape (H, W, T) or (H, W, C).
        angle_deg: Rotation angle in degrees (Counter-Clockwise).
        center_rc: Optional (Row, Col) of the rotation center in pixel coordinates.
                   If None, rotates around the image geometric center.
        mode: Interpolation mode ('bilinear' or 'nearest').
    """
    # 1. Shape Handling: (H, W, T) -> (T, 1, H, W)
    orig_shape = tensor.shape
    if len(orig_shape) == 3:
        # User convention: (H, W, T) -> (T, 1, H, W)
        tensor = tensor.permute(2, 0, 1).unsqueeze(1)
    elif len(orig_shape) == 2:
        # (H, W) -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    N, C, H, W = tensor.shape
    device = tensor.device
    dtype = torch.float32  # grid requires float

    # 2. Define Rotation Angle (Inverse logic for grid_sample)
    # To rotate image CCW by X, we sample from grid rotated CW by X (-X).
    angle_rad = -math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot_matrix = torch.tensor([[c, -s],
                               [s, c]], device=device, dtype=dtype)

    # 3. Calculate Translation for Center
    if center_rc is not None:
        if isinstance(center_rc, (tuple, list)):
            center_rc = torch.tensor(center_rc, device=device, dtype=dtype)

        # Convert pixel (Row, Col) -> normalized (x, y) range [-1, 1]
        # Note: grid_sample (x,y) corresponds to (Col, Row)
        # align_corners=False convention:
        #   -1 maps to -0.5 index
        #    1 maps to size-0.5 index
        # Formula: norm = (2 * (pix + 0.5) / size) - 1

        center_row, center_col = center_rc[0], center_rc[1]

        norm_x = (2.0 * (center_col + 0.5) / W) - 1.0
        norm_y = (2.0 * (center_row + 0.5) / H) - 1.0

        center_norm = torch.tensor([norm_x, norm_y], device=device, dtype=dtype)

        # The affine grid transformation is: P_in = R * P_out + T
        # We want P_out = Center -> P_in = Center (Fixed Point)
        # Center = R * Center + T  =>  T = Center - R * Center

        translation = center_norm - (rot_matrix @ center_norm)
    else:
        translation = torch.zeros(2, device=device, dtype=dtype)

    # 4. Construct Full Affine Matrix (2x3)
    # [ c  -s  tx ]
    # [ s   c  ty ]
    theta = torch.zeros((1, 2, 3), device=device, dtype=dtype)
    theta[0, :2, :2] = rot_matrix
    theta[0, :2, 2] = translation

    # Tile for batch size
    theta = theta.repeat(N, 1, 1)

    # 5. Grid Sample
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    rotated = F.grid_sample(tensor.float(), grid, mode=mode, padding_mode='zeros', align_corners=False)

    # 6. Restore Shape
    if len(orig_shape) == 3:
        # (T, 1, H, W) -> (H, W, T)
        return rotated.squeeze(1).permute(1, 2, 0).to(tensor.dtype)
    elif len(orig_shape) == 2:
        return rotated.squeeze(0).squeeze(0).to(tensor.dtype)


def find_heart_center_and_rotations(affines: List[torch.Tensor],
                                    segs: Optional[List[torch.Tensor]] = None,
                                    rotate=False) \
        -> Tuple[List[torch.Tensor], List[float]]:
    """
    Returns center coordinates for cropping AND rotation angles to align
    LA slices with the SA plane.
    """
    # --- 1. Original Center Logic ---
    lv_basal_slice, lv_apex_slice = find_basal_apical_from_sa_segmentation(segs[3:])
    # Adjust indices (SA starts at index 3 in the lists)
    lv_basal_slice, lv_apex_slice = 3 + lv_basal_slice, 3 + lv_apex_slice

    lv_midventr_slice = (lv_apex_slice + lv_basal_slice) // 2 - 1

    # Get centers from mid-ventricular slice
    midventr_lv_center = get_center_coord(segs[lv_midventr_slice][..., 0] == 1)
    midventr_rv_center = get_center_coord(segs[lv_midventr_slice][..., 0] == 3)
    midventr_heart_center = (midventr_lv_center + midventr_rv_center) / 2

    # Project center coordinate to LA slices
    midventr_heart_center_aug = torch.cat((midventr_heart_center, torch.tensor((0,)), torch.tensor((1,))), dim=0)

    la2ch_center = scanner_to_image_coords(affines[lv_midventr_slice] @ midventr_heart_center_aug, affines[0])[0, :2]
    la3ch_center = scanner_to_image_coords(affines[lv_midventr_slice] @ midventr_heart_center_aug, affines[1])[0, :2]
    la4ch_center = scanner_to_image_coords(affines[lv_midventr_slice] @ midventr_heart_center_aug, affines[2])[0, :2]

    sa_centers = [midventr_heart_center] * len(affines[3:])
    centers = [la2ch_center, la3ch_center, la4ch_center, *sa_centers]
    if not rotate:
        return centers, [0.0 for _ in centers]

    # --- 2. New Rotation Logic ---
    # We use the FIRST SA slice (index 3) as the reference plane.
    ref_sa_affine = affines[3]

    # Calculate angles for LA slices (indices 0, 1, 2)
    angle_2ch = get_intersection_angle(affines[0], ref_sa_affine)
    angle_3ch = get_intersection_angle(affines[1], ref_sa_affine)
    angle_4ch = get_intersection_angle(affines[2], ref_sa_affine)

    # SA slices do not need rotation relative to themselves (0.0)
    sa_angles = [0.0] * len(affines[3:])

    angles = [-angle_2ch, -angle_3ch, -angle_4ch, *sa_angles]

    return centers, angles


def get_intersection_angle(la_affine: torch.Tensor, sa_affine: torch.Tensor) -> float:
    """
    Calculates the rotation angle required to make the intersection of the
    SA plane with the LA image horizontal.
    """
    device = la_affine.device

    # 1. Get Plane Normals in World Space
    # The normal is the cross product of the first two columns (X and Y vectors)
    # or roughly the 3rd column (slice direction)

    # LA Normal
    vec_u_la = la_affine[:3, 0]
    vec_v_la = la_affine[:3, 1]
    normal_la = torch.cross(vec_u_la, vec_v_la)

    # SA Normal (Reference Plane)
    vec_u_sa = sa_affine[:3, 0]
    vec_v_sa = sa_affine[:3, 1]
    normal_sa = torch.cross(vec_u_sa, vec_v_sa)

    # 2. Calculate Intersection Vector in World Space
    # The intersection of two planes is perpendicular to both normals
    intersect_vec_world = torch.cross(normal_la, normal_sa)

    # Handle parallel planes edge case (norm is near 0)
    if torch.linalg.norm(intersect_vec_world) < 1e-4:
        return 0.0

    # 3. Project Intersection Vector into LA Image Space
    # We need to express intersect_vec_world as (u, v) components in the LA image.
    # We can solve the linear system: V_world = u * U_la + v * V_la + w * W_la
    # Or simply multiply by inverse of rotational part of LA affine

    la_rotation_matrix = la_affine[:3, :3]
    # V_img = inv(R) @ V_world
    intersect_vec_img = torch.linalg.solve(la_rotation_matrix, intersect_vec_world)

    # 4. Calculate Angle in Image Plane (u, v)
    u, v = intersect_vec_img[0], intersect_vec_img[1]
    angle_rad = math.atan2(v, u)
    angle_deg = math.degrees(angle_rad)

    # 5. Determine Correction Angle
    # If the line is at +30 deg, we want to rotate the image by -30 deg to make it 0 (horizontal).
    return -angle_deg


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
        if not (lv_sample_segs==3).any() or not (lv_sample_segs==2).any():
            raise ValueError( f"Slice {lv_midventr_slice} has no RV, {lv_sample_segs}")
        rv_center = torch.where(lv_sample_segs==3)[0].median()
        if rv_center.item() >= myo_search_num_samples//2:
            rv_to_lv_samples = lv_sample_segs[myo_search_num_samples//2:]
            if not (rv_to_lv_samples==2).any():
                raise ValueError( f"Slice {lv_midventr_slice} has no LV MYO, {lv_sample_segs}")
            myo_center_along_line = torch.where(rv_to_lv_samples==2)[0].median() + myo_search_num_samples//2
        else:
            rv_to_lv_samples = lv_sample_segs[:myo_search_num_samples//2]
            if not (rv_to_lv_samples==2).any():
                raise ValueError( f"Slice {lv_midventr_slice} has no LV MYO, {lv_sample_segs}")
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
        thresh = 12.0 if use_sa_normal_as_la else 1.
        if ang_24ch_la_z_2 > thresh:
            raise ValueError
        if abs(ang_24ch_la_y_2 - 90) > thresh or abs(ang_24ch_la_x_2 - 90) > thresh:
            raise ValueError
        w_lv_center_2 = plane_line_intersection(v[0, 1], v[0, 0],
                                            planes[lv_midventr_slice])
        w_myo_center_2 = (oriented_affines[lv_midventr_slice] @ i_myo_center_aug.T).T
        if w_myo_center_2[:3].norm() > 1e-4:
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
        else:
            la2ch_seg = la2ch_seg.numpy()
        la3ch_seg = segs[0]
        if not la3ch_seg.any():
            la3ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in affines[3:]],
                                                     target_shape=(segs[1].shape[0], segs[1].shape[1], segs[1].shape[-1]),
                                                     target_aff=affines[1].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        else:
            la3ch_seg = la3ch_seg.numpy()
        la4ch_seg = segs[0]
        if not la4ch_seg.any():
            la4ch_seg, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in segs[3:]],
                                                     [i.numpy() for i in affines[3:]],
                                                     target_shape=(segs[2].shape[0], segs[2].shape[1], segs[2].shape[-1]),
                                                     target_aff=affines[2].numpy(),
                                                     frames=None,
                                                     oob_dist_thresh=10.)
        else:
            la4ch_seg = la4ch_seg.numpy()
        path = Path('debug_alignment')
        fold = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        path = path / fold
        path.mkdir(exist_ok=True, parents=True)
        la2ch_seg = deepcopy(la2ch_seg)
        la3ch_seg = deepcopy(la3ch_seg)
        la4ch_seg = deepcopy(la4ch_seg)
        segs = deepcopy(segs)
        la2ch_seg[0] = 4
        la3ch_seg[0] = 4
        la4ch_seg[0] = 4
        la2ch_seg[-1] = 4
        la3ch_seg[-1] = 4
        la4ch_seg[-1] = 4
        for s in segs:
            s[0] = 4
            s[-1] = 4
        array_to_nifti(str(path/f"pre_opt_la2ch.nii.gz"), la2ch_seg[:, :, None].astype(int), affines[0].numpy())
        array_to_nifti(str(path/f"pre_opt_la3ch.nii.gz"), la3ch_seg[:, :, None].astype(int), affines[1].numpy())
        array_to_nifti(str(path/f"pre_opt_la4ch.nii.gz"), la4ch_seg[:, :, None].astype(int), affines[2].numpy())
        array_to_nifti(str(path/f"pre_opt_sa3.nii.gz"), segs[6][:, :, None].numpy().astype(int), affines[6].numpy())
        array_to_nifti(str(path/f"pre_opt_sa4.nii.gz"), segs[7][:, :, None].numpy().astype(int), affines[7].numpy())
        array_to_nifti(str(path/f"pre_opt_sa5.nii.gz"), segs[8][:, :, None].numpy().astype(int), affines[8].numpy())
        array_to_nifti(str(path/f"post_opt_la2ch.nii.gz"), la2ch_seg[:, :, None].astype(int), final_affines[0].numpy())
        array_to_nifti(str(path/f"post_opt_la3ch.nii.gz"), la3ch_seg[:, :, None].astype(int), final_affines[1].numpy())
        array_to_nifti(str(path/f"post_opt_la4ch.nii.gz"), la4ch_seg[:, :, None].astype(int), final_affines[2].numpy())
        array_to_nifti(str(path/f"post_opt_sa3.nii.gz"), segs[6][:, :, None].numpy().astype(int), final_affines[6].numpy())
        array_to_nifti(str(path/f"post_opt_sa4.nii.gz"), segs[7][:, :, None].numpy().astype(int), final_affines[7].numpy())
        array_to_nifti(str(path/f"post_opt_sa5.nii.gz"), segs[8][:, :, None].numpy().astype(int), final_affines[8].numpy())
    return final_affines
