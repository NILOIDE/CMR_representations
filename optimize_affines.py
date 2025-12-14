from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Tuple, Union, Iterable
import nibabel as nib
import numpy as np
import torch
import tqdm
from dataclasses import dataclass
from itertools import combinations, product
import cv2
import matplotlib.pyplot as plt


from geo_utils import get_image_plane_from_array, plane_intersection, closest_point_on_line, batch_normalize_vector
from utils import normalize_image, normalize_image_with_mean_lv_value, fast_trilinear_interpolation, mat_to_params, \
    params_to_mat, fast_nearest_neighbor_interpolation
from data_utils import SubjectFiles, find_subjects
from metrics import L2, L1, NCC, Dice

DEVICE = "cuda"

# BGR colors
c_red = (0, 0, 255)
c_green = (0, 255, 0)
c_mint = (175, 255, 127)
c_orange = (0, 127, 255)


class OptimizableImage:
    def __init__(self, image, affine, spacing, name, index, seg=None):
        self.image = image
        self.seg = seg
        self.affine = affine
        assert len(spacing) == 3
        self.spacing = np.array(spacing)
        self.needs_flip = np.linalg.det(affine[:3, :3]) <= 0
        self.parameters = self.get_params_from_affine()  # r1, r2, r3, t1, t2, t3
        self.name = name
        self.index = index

    def flip_rot_matrix(self, affine):
        """ Maik is a magician"""
        flip = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        affine[:3, :3] = flip @ affine[:3, :3]
        affine[:3, -1:] = flip @ affine[:3, -1:]
        return affine

    def get_params_from_affine(self) -> np.ndarray:
        # Using quaternions instead of Euler angles yields <x2 better precision: 3.45e-07 vs 2.52e-07
        affine = self.affine.copy()
        if self.needs_flip:  # TODO: This may not always have to be done? What happens when determinant flips?
            affine = self.flip_rot_matrix(affine)
        affine[:3, :3] = affine[:3, :3] @ np.diag(1 / self.spacing)
        euler_angles = nib.eulerangles.mat2euler(affine[:3, :3])# returns z,y,x
        assert np.abs(affine[:3, :3] - nib.eulerangles.euler2mat(*euler_angles)).sum() < 1e-6, np.abs(affine[:3, :3] - nib.eulerangles.euler2mat(*euler_angles)).sum()
        translations = affine[:3, 3]
        return np.array([*euler_angles, *translations])

    def update_affine(self, rotation_params: np.ndarray, translation_params: np.ndarray):
        affine = np.eye(4, dtype=float)
        rot_ = nib.eulerangles.euler2mat(*rotation_params)
        rot = rot_ @ np.diag(self.spacing)
        affine[:3, :3] = rot
        affine[:3, 3] = translation_params
        if self.needs_flip:
            affine = self.flip_rot_matrix(affine)
        return affine


class SubjectData:
    def __init__(self, subject: SubjectFiles):
        self.planes = []
        self.seg_planes = []
        index = 0
        if subject.la4ch is not None:
            try:
                nii_4ch = nib.load(subject.la4ch)
                nii_4ch_seg = nib.load(subject.la4ch_seg)
                self.planes.append(OptimizableImage(nii_4ch.dataobj[:].squeeze(2), nii_4ch.affine, nii_4ch.header.get_zooms()[:3], "la_4ch", index=index, seg=nii_4ch_seg.dataobj[:]))
                index += 1
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 4ch image not found")
        if subject.la3ch is not None:
            try:
                nii_3ch = nib.load(subject.la3ch)
                # nii_3ch_seg = nib.load(subject.la3ch_seg)
                self.planes.append(OptimizableImage(nii_3ch.dataobj[:].squeeze(2), nii_3ch.affine, nii_3ch.header.get_zooms()[:3], "la_3ch", index=index))
                index += 1
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 3ch image not found")
        if subject.la2ch is not None:
            try:
                nii_2ch = nib.load(subject.la2ch)
                nii_2ch_seg = nib.load(subject.la2ch_seg)
                self.planes.append(OptimizableImage(nii_2ch.dataobj[:].squeeze(2), nii_2ch.affine, nii_2ch.header.get_zooms()[:3], "la_2ch", index=index, seg=nii_2ch_seg.dataobj[:]))
                index += 1
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 2ch image not found")
        # Index pairs for all Long-axis to Long-axis image pairs (if 2ch, 3ch, 4ch are present, that will be 3 pairs)
        la_la_product = [*combinations(list(range(len(self.planes))), 2)]
        # Long-axis to Short-axis image pairs combinations (if 3 LA images and N SA images, that will be 3*N pairs)
        sa_la_product = list(product(list(range(len(self.planes))), list(range(len(self.planes), len(self.planes) + len(subject.sax)))))
        self.idx_pairs = torch.tensor(la_la_product + sa_la_product)
        # Load Short-axis planes
        for i, (sa_lice, sa_lice_seg) in enumerate(zip(subject.sax, subject.sax_seg)):
            im_nii = nib.load(sa_lice)
            im_nii_seg = nib.load(sa_lice_seg)
            self.planes.append(OptimizableImage(im_nii.dataobj[:].squeeze(2), im_nii.affine, im_nii.header.get_zooms()[:3], f"sa{i}-{len(subject.sax)}", index=index, seg=im_nii_seg.dataobj[:]))
            index += 1
        assert min([min(i) for i in self.idx_pairs]) == 0
        assert max([max(i) for i in self.idx_pairs]) == len(self.planes) - 1

        self.images = [torch.tensor(i.image, dtype=torch.float32, device=DEVICE) for i in self.planes]
        self.shapes = torch.stack([torch.tensor(i.shape, dtype=torch.int64, device=DEVICE) for i in self.images])
        self.affines = torch.stack([torch.tensor(i.affine, dtype=torch.float32, device=DEVICE) for i in self.planes], dim=0)
        self.spacings = torch.stack([torch.tensor(i.spacing, dtype=torch.float32, device=DEVICE) for i in self.planes], dim=0)
        self.max_im_shape = self.shapes.amax(dim=0)
        self.images_pad = torch.zeros((len(self.images), *self.max_im_shape), dtype=torch.float32, device=DEVICE)
        for i, (im, sh) in enumerate(zip(self.images, self.shapes)):
            self.images_pad[i, :sh[0], :sh[1]] = im
            assert im.any(1).any(0).all()

    def update_affines(self, affines: torch.Tensor):
        affines = affines.detach().cpu().numpy()
        assert affines.shape == (len(self.planes), 4, 4)
        for i in range(affines.shape[0]):
            self.planes[i].affine = affines[i]
        self.affines = torch.stack([torch.tensor(i.affine, dtype=torch.float32, device=DEVICE) for i in self.planes], dim=0)

    def save_niftis(self, directory: str, affines: torch.Tensor = None):
        if affines is not None:
            self.update_affines(affines)
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        for plane in self.planes:
            nii = nib.Nifti1Image(plane.image[:, :, None], affine=plane.affine)
            if plane.name[:2] == "sa":
                path = Path(directory) / "sa_slices"
                path.mkdir(exist_ok=True)
                path = path / (plane.name + ".nii.gz")
            else:
                path = Path(directory) / (plane.name + ".nii.gz")
            nib.save(nii, path)

            if plane.seg is not None:
                nii = nib.Nifti1Image(plane.seg[:, :, None], affine=plane.affine)
                if plane.name[:2] == "sa":
                    path = Path(directory) / "sa_slices"
                    path.mkdir(exist_ok=True)
                    path = path / ("seg_" + plane.name + ".nii.gz")
                else:
                    path = Path(directory) / ("seg_" + plane.name + ".nii.gz")
                nib.save(nii, path)
        return


def visualize_intersections(images, affines, pair_indices, shapes, names=None):
    metric = L1()
    sample_coords_scanner_space = compute_intersection_sampling_line(affines, pair_indices, shapes)
    images1, images2 = images[pair_indices[:, 0]], images[pair_indices[:, 1]]
    affines1, affines2 = affines[pair_indices[:, 0]], affines[pair_indices[:, 1]]
    shapes1, shapes2 = shapes[pair_indices[:, 0]], shapes[pair_indices[:, 1]]
    sampled_im1, sample_mask1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space, images1, affines1, shapes1)
    sampled_im2, sample_mask2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space, images2, affines2, shapes2)
    loss = metric(sampled_im1, sampled_im2, mask1=sample_mask1, mask2=sample_mask2)

    images = images.cpu().detach().numpy()
    pair_indices = pair_indices.cpu().detach().numpy()
    sampled_im1 = sampled_im1.cpu().detach().numpy()
    sampled_im2 = sampled_im2.cpu().detach().numpy()
    sample_mask1 = sample_mask1.cpu().detach().numpy()
    sample_mask2 = sample_mask2.cpu().detach().numpy()
    line_im1 = line_im1.cpu().detach().numpy()
    line_im2 = line_im2.cpu().detach().numpy()
    loss = loss.cpu().detach().numpy()

    # Visualize
    for i in range(pair_indices.shape[0]):
        idx1, idx2 = pair_indices[i, 0], pair_indices[i, 1]
        name1, name2 = "im1", "im2"
        if names is not None:
            name1, name2 = names[idx1], names[idx2]
        scaling = 10
        im1_vis = normalize_image_with_mean_lv_value(sampled_im1[i, ..., 0]) * 255
        im1_vis_rgb = np.stack([im1_vis] * 3, axis=-1)
        im2_vis = normalize_image_with_mean_lv_value(sampled_im2[i, ..., 0]) * 255
        im2_vis_rgb = np.stack([im2_vis] * 3, axis=-1)
        mask = np.logical_and(sample_mask1[i, ..., 0], sample_mask2[i, ..., 0]).astype(np.float32)
        mask_rgb = np.stack([mask] * 3, -1)
        mask_rgb = np.where(mask_rgb, np.array([c_green] * mask.shape[0]), np.array([c_red] * mask.shape[0]))

        cat_line = np.stack((*[im1_vis_rgb] * 2,
                             np.zeros_like(mask_rgb), mask_rgb, np.zeros_like(mask_rgb),
                             *[im2_vis_rgb] * 2,), axis=0)
        cat_line_ = cv2.UMat(cat_line.astype(np.uint8))
        cat_line_ = cv2.resize(cat_line_, (im1_vis.shape[0] * scaling, cat_line.shape[0] * scaling))
        cv2.imshow(f"{name1}-{name2} sampling line.   "
                   f"Loss (top):  {float('%.5f' % loss[i])}", cat_line_)

        # Plot images and draw sampling lines along images
        scaling = 2
        im1_vis = images[idx1, ..., 0]
        im2_vis = images[idx2, ..., 0]
        im_vis = np.concatenate((im1_vis, im2_vis), axis=1)
        im_vis = normalize_image(im_vis) * 255
        im_vis_ = cv2.UMat(np.stack([im_vis.astype(np.uint8)] * 3, axis=-1))

        # Plot lines of old affines
        for j in range(line_im1.shape[1] - 1):
            p1 = line_im1[i, j, :2].round().astype(int)
            p2 = line_im1[i, j + 1, :2].round().astype(int)
            c = c_green if sample_mask1[i, j, 0] else c_red
            cv2.line(im_vis_,
                     (p1[1], p1[0],),
                     (p2[1], p2[0],),
                     c)
        for j in range(line_im2.shape[1] - 1):
            p1 = line_im2[i, j, :2].round().astype(int)
            p2 = line_im2[i, j + 1, :2].round().astype(int)
            c = c_green if sample_mask1[i, j, 0] else c_red
            cv2.line(im_vis_,
                     (p1[1] + images.shape[2], p1[0],),
                     (p2[1] + images.shape[2], p2[0],),
                     c)
        im_vis_ = cv2.resize(im_vis_, (im2_vis.shape[1] * scaling * 2, im2_vis.shape[0] * scaling * 2))
        cv2.imshow(f"{name1}-{name2} intersection.  "
                   f"Loss:  {float('%.5f' % loss[i])}", im_vis_)

        cv2.waitKey()


def visualize_intersection_differences(images, affines, affines_new, pair_indices, shapes, names=None):
    metric = L1()
    images1, images2 = images[pair_indices[:, 0]], images[pair_indices[:, 1]]
    shapes1, shapes2 = shapes[pair_indices[:, 0]], shapes[pair_indices[:, 1]]

    sample_coords_scanner_space = compute_intersection_sampling_line(affines, pair_indices, shapes)
    affines1, affines2 = affines[pair_indices[:, 0]], affines[pair_indices[:, 1]]
    sampled_im1, sample_mask1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space, images1, affines1, shapes1)
    sampled_im2, sample_mask2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space, images2, affines2, shapes2)
    loss = metric(sampled_im1, sampled_im2, mask1=sample_mask1, mask2=sample_mask2)

    sample_coords_scanner_space = compute_intersection_sampling_line(affines_new, pair_indices, shapes)
    affines1_new, affines2_new = affines_new[pair_indices[:, 0]], affines_new[pair_indices[:, 1]]
    sampled_im1_new, sample_mask1_new, line_im1_new = sample_image_along_scanner_line(sample_coords_scanner_space, images1, affines1_new, shapes1)
    sampled_im2_new, sample_mask2_new, line_im2_new = sample_image_along_scanner_line(sample_coords_scanner_space, images2, affines2_new, shapes2)
    loss_new = metric(sampled_im1_new, sampled_im2_new, mask1=sample_mask1_new, mask2=sample_mask2_new)

    images = images.cpu().detach().numpy()
    pair_indices = pair_indices.cpu().detach().numpy()
    sampled_im1 = sampled_im1.cpu().detach().numpy()
    sampled_im2 = sampled_im2.cpu().detach().numpy()
    sampled_im1_new = sampled_im1_new.cpu().detach().numpy()
    sampled_im2_new = sampled_im2_new.cpu().detach().numpy()
    sample_mask1 = sample_mask1.cpu().detach().numpy()
    sample_mask2 = sample_mask2.cpu().detach().numpy()
    sample_mask1_new = sample_mask1_new.cpu().detach().numpy()
    sample_mask2_new = sample_mask2_new.cpu().detach().numpy()
    line_im1 = line_im1.cpu().detach().numpy()
    line_im2 = line_im2.cpu().detach().numpy()
    line_im1_new = line_im1_new.cpu().detach().numpy()
    line_im2_new = line_im2_new.cpu().detach().numpy()
    loss = loss.cpu().detach().numpy()
    loss_new = loss_new.cpu().detach().numpy()
    # Visualize
    for i in range(pair_indices.shape[0]):
        idx1, idx2 = pair_indices[i, 0], pair_indices[i, 1]
        name1, name2 = "im1", "im2"
        if names is not None:
            name1, name2 = names[idx1], names[idx2]
        scaling = 10
        # Original
        im1_vis = normalize_image_with_mean_lv_value(sampled_im1[i, ..., 0]) * 255
        im1_vis_rgb = np.stack([im1_vis]*3, axis=-1)
        im2_vis = normalize_image_with_mean_lv_value(sampled_im2[i, ..., 0]) * 255
        im2_vis_rgb = np.stack([im2_vis]*3, axis=-1)
        mask = np.logical_and(sample_mask1[i, ..., 0], sample_mask2[i, ..., 0]).astype(np.float32)
        mask_rgb = np.stack([mask]*3, -1)
        mask_rgb = np.where(mask_rgb, np.array([c_mint]*mask.shape[0]), np.array([c_orange]*mask.shape[0]))
        # New
        im1_vis_new = normalize_image_with_mean_lv_value(sampled_im1_new[i, ..., 0]) * 255
        im1_vis_new_rgb = np.stack([im1_vis_new] * 3, axis=-1)
        im2_vis_new = normalize_image_with_mean_lv_value(sampled_im2_new[i, ..., 0]) * 255
        im2_vis_new_rgb = np.stack([im2_vis_new] * 3, axis=-1)
        mask_new = np.logical_and(sample_mask1_new[i, ..., 0], sample_mask2_new[i, ..., 0]).astype(np.float32)
        mask_new_rgb = np.stack([mask_new]*3, -1)
        mask_new_rgb = np.where(mask_new_rgb, np.array([c_green]*mask.shape[0]), np.array([c_red]*mask.shape[0]))

        cat_line = np.stack((*[im1_vis_rgb]*2,
                             np.zeros_like(mask_rgb), mask_rgb, np.zeros_like(mask_rgb),
                             *[im2_vis_rgb]*2,
                             *[np.zeros_like(mask_rgb)]*5,
                             *[im1_vis_new_rgb]*2,
                             np.zeros_like(mask_new_rgb), mask_new_rgb, np.zeros_like(mask_new_rgb),
                             *[im2_vis_new_rgb]*2,), axis=0)
        cat_line_ = cv2.UMat(cat_line.astype(np.uint8))
        cat_line_ = cv2.resize(cat_line_, (im1_vis.shape[0] * scaling, cat_line.shape[0] * scaling))
        cv2.imshow(f"{name1}-{name2} sampling line.   "
                   f"Original loss (top):  {float('%.5f' % loss[i])},   "
                   f"New loss (bottom):  {float('%.5f' % loss_new[i])}    "
                   f"Loss diff:  {float('%.5f' % (loss_new[i] - loss[i]))}", cat_line_)

        # Plot images and draw sampling lines along images
        scaling = 2
        im1_vis = images[idx1, ..., 0]
        im2_vis = images[idx2, ..., 0]
        im_vis = np.concatenate((im1_vis, im2_vis), axis=1)
        im_vis = normalize_image(im_vis) * 255
        im_vis = np.concatenate((im_vis, im_vis), axis=0)
        im_vis_ = cv2.UMat(np.stack([im_vis.astype(np.uint8)]*3, axis=-1))

        # Plot lines of old affines
        for j in range(line_im1.shape[1]-1):
            p1 = line_im1[i, j, :2].round().astype(int)
            p2 = line_im1[i, j+1, :2].round().astype(int)
            c = c_mint if sample_mask1[i, j, 0] else c_orange
            cv2.line(im_vis_,
                     (p1[1], p1[0],),
                     (p2[1], p2[0],),
                     c)
        for j in range(line_im2.shape[1]-1):
            p1 = line_im2[i, j, :2].round().astype(int)
            p2 = line_im2[i, j+1, :2].round().astype(int)
            c = c_mint if sample_mask1[i, j, 0] else c_orange
            cv2.line(im_vis_,
                     (p1[1] + images.shape[2], p1[0],),
                     (p2[1] + images.shape[2], p2[0],),
                     c)
        # Plot lines of new affines
        for j in range(line_im1_new.shape[1]-1):
            p1 = line_im1_new[i, j, :2].round().astype(int)
            p2 = line_im1_new[i, j+1, :2].round().astype(int)
            c = c_green if sample_mask1[i, j, 0] else c_red
            cv2.line(im_vis_,
                     (p1[1], p1[0] + images.shape[1],),
                     (p2[1], p2[0] + images.shape[1],),
                     c)
        for j in range(line_im2_new.shape[1]-1):
            p1 = line_im2_new[i, j, :2].round().astype(int)
            p2 = line_im2_new[i, j+1, :2].round().astype(int)
            c = c_green if sample_mask1[i, j, 0] else c_red
            cv2.line(im_vis_,
                     (p1[1] + images.shape[2], p1[0] + images.shape[1],),
                     (p2[1] + images.shape[2], p2[0] + images.shape[1],),
                     c)
        im_vis_ = cv2.resize(im_vis_, (im2_vis.shape[1] * scaling * 2, im2_vis.shape[0] * scaling * 2))
        cv2.imshow(f"{name1}-{name2} intersection.  "
                   f"Original loss (top):  {float('%.5f' % loss[i])},   "
                   f"New loss (bottom):  {float('%.5f' % loss_new[i])}", im_vis_)

        cv2.waitKey()


def find_image_border_intercepts(intersec_lines, affines, shapes) -> torch.Tensor:
    #https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    # (Batch, num_corners, (xyz1))
    corners = torch.zeros((intersec_lines.shape[0], 4, 4), dtype=torch.float32, device=DEVICE)
    corners[..., -1] = 1.0
    corners[:, 1, 1] = shapes[:, 0]
    corners[:, 3, 1] = shapes[:, 0]
    corners[:, 2, 0] = shapes[:, 1]
    corners[:, 3, 1] = shapes[:, 1]

    affines_ = torch.tile(affines[:, None], (1, 4, 1, 1))
    # Don't know if doing this is slower but it avoids me having to reshape a bunch of times and pray the ordering is correct
    corners_scanner_space = torch.einsum("ijkl,ijl->ijk", [affines_, corners])
    border1 = corners_scanner_space[:, 0, :3] - corners_scanner_space[:, 1, :3]
    border2 = corners_scanner_space[:, 1, :3] - corners_scanner_space[:, 2, :3]
    border3 = corners_scanner_space[:, 2, :3] - corners_scanner_space[:, 3, :3]
    border4 = corners_scanner_space[:, 3, :3] - corners_scanner_space[:, 0, :3]
    # borders = torch.stack([border1, border2, border3, border4], dim=1)

    # Calculate denomitator
    A = intersec_lines[:, 0] - intersec_lines[:, 1]
    magA = A.norm(dim=-1)
    magB1 = border1.norm(dim=-1)
    magB2 = border2.norm(dim=-1)
    magB3 = border3.norm(dim=-1)
    magB4 = border4.norm(dim=-1)

    _A = A / magA[:, None].tile((1, 3))
    _B1 = border1 / magB1[:, None].tile((1, 3))
    _B2 = border2 / magB2[:, None].tile((1, 3))
    _B3 = border3 / magB3[:, None].tile((1, 3))
    _B4 = border4 / magB4[:, None].tile((1, 3))

    cross1 = torch.cross(_A, _B1)
    cross2 = torch.cross(_A, _B2)
    cross3 = torch.cross(_A, _B3)
    cross4 = torch.cross(_A, _B4)
    raise NotImplementedError


def image_center_to_scanner_space(affines: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
    centers = torch.zeros((shapes.shape[0], 4), dtype=affines.dtype, device=affines.device)
    centers[:, -1] = 1.0
    centers[:, :2] = shapes[:, :2] / 2

    centers_scanner_space = torch.einsum("ijk,ik->ij", [affines, centers])
    return centers_scanner_space[..., :3]


def find_sampling_center(image_centers1, image_centers2, intersec_lines) -> torch.Tensor:
    centers1_on_line = closest_point_on_line(intersec_lines, image_centers1)
    centers2_on_line = closest_point_on_line(intersec_lines, image_centers2)
    sampling_centers = (centers1_on_line + centers2_on_line) / 2
    return sampling_centers


def compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_mm=None, num_samples=100) -> torch.Tensor:
    """ Compute a sampling line of num_samples points along the intersection lane between all plane pairs.
     Each line is centered on the average image centers. Each point is spaced sampling_step_mm appart. """
    if sampling_step_mm is None:
        sampling_step_mm = 5.0
    planes = get_image_plane_from_array(affines)
    planes1, planes2 = planes[pair_indices[:, 0]], planes[pair_indices[:, 1]]
    # We assume the plane pairs always intersect
    intersec_scanner_space = plane_intersection(planes1, planes2)
    # The sampling line will be centered on the average image centers projected along the intersection line
    img_centers = image_center_to_scanner_space(affines, shapes)
    img_centers1, img_centers2 = img_centers[pair_indices[:, 0]], img_centers[pair_indices[:, 1]]
    sampling_centers = find_sampling_center(img_centers1, img_centers2, intersec_scanner_space)

    # We sample a line along the intersection N mm at a time for num_samples/2 in each direction
    intersec_dirs = intersec_scanner_space[:, 1] - intersec_scanner_space[:, 0]
    step_vectors = batch_normalize_vector(intersec_dirs)
    step_vectors_ = step_vectors[:, None].tile((1, num_samples, 1))
    dists_from_centers = (torch.arange(0, num_samples, dtype=affines.dtype, device=affines.device) - num_samples / 2)
    dists_from_centers *= sampling_step_mm
    dists_from_centers_ = dists_from_centers[None, :, None].tile((sampling_centers.shape[0], 1, 3))
    sampling_centers_ = sampling_centers[:, None].tile((1, num_samples, 1))
    sample_lines = sampling_centers_ + step_vectors_ * dists_from_centers_

    assert sample_lines.shape[0] == pair_indices.shape[0]
    assert sample_lines.shape[1] == num_samples
    assert sample_lines.shape[2] == 3
    return sample_lines


def sample_image_along_scanner_line(coords_scanner_space: torch.Tensor, images: torch.Tensor,
                                    affines: torch.Tensor, shapes: torch.Tensor, interp_type='linear') \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = coords_scanner_space.dtype
    device = coords_scanner_space.device
    # Add 4th coord dimension in order to multiply by affine
    if coords_scanner_space.shape[-1] == 3:
        last_dim = torch.ones((*coords_scanner_space.shape[:-1], 1), dtype=dtype, device=device)
        coords_scanner_space = torch.cat((coords_scanner_space, last_dim), dim=-1)
    inverse_affines = torch.linalg.inv(affines)  # For (scanner space -> voxel) we need inverse affine
    # Tile affine into (batch, num_point, 4, 4)
    inverse_affines_ = inverse_affines[:, None].tile((1, coords_scanner_space.shape[1], 1, 1))
    # Batch-wise dot product
    points_voxel_space = torch.einsum("ijkl,ijl->ijk", [inverse_affines_, coords_scanner_space])
    # Disregard z dimension, we assume it's a 2D image. Ideally z should always be ~0.0.
    coords_voxel_space = points_voxel_space[:, :, :2]
    # We want to sample these point along all time points in the 2D+time scan. We need to generate the time indices.
    _, _, t = torch.meshgrid(torch.arange(0, coords_scanner_space.shape[0], dtype=dtype, device=device),
                             torch.arange(0, coords_scanner_space.shape[1], dtype=dtype, device=device),
                             torch.arange(0, images.shape[-1], dtype=dtype, device=device))
    # Concatenate time indices to xy coordinates
    coords_voxel_space_t = coords_voxel_space[:, :, None].tile((1, 1, images.shape[-1], 1))
    coords_voxel_space_t = torch.cat((coords_voxel_space_t, t[..., None]), dim=-1)
    if interp_type == 'linear':
        # Treat 2D+time images are volumes and use trilinear interpolation to extract values at each time point
        sampled_points = fast_trilinear_interpolation(images,
                                                      coords_voxel_space_t[..., 0].reshape((images.shape[0], -1)),
                                                      coords_voxel_space_t[..., 1].reshape((images.shape[0], -1)),
                                                      coords_voxel_space_t[..., 2].reshape((images.shape[0], -1)))
    elif interp_type == 'nn':
        # Not differentiable!!
        sampled_points = fast_nearest_neighbor_interpolation(images,
                                                             coords_voxel_space_t[..., 0].reshape((images.shape[0], -1)),
                                                             coords_voxel_space_t[..., 1].reshape((images.shape[0], -1)),
                                                             coords_voxel_space_t[..., 2].reshape((images.shape[0], -1)))
    else:
        raise ValueError(f'Interpolation type "{interp_type}" not implemented')

    sampled_points = sampled_points.reshape((images.shape[0], coords_scanner_space.shape[1], images.shape[-1]))
    # Create mask to deliniate which sampled points were inside/outside image.
    shapes_ = shapes[:, None, :2].tile((1, coords_voxel_space.shape[1], 1))
    out_mask = torch.logical_or((coords_voxel_space < 0.0).any(dim=-1),
                                (coords_voxel_space > (shapes_ - 1)).any(dim=-1))
    in_mask = ~out_mask
    in_mask = in_mask[..., None].tile((1, 1, images.shape[-1]))
    return sampled_points, in_mask, coords_voxel_space


def compute_pairwise_loss(images, affines, pair_indices, shapes, metric='ncc', interp_type='linear', sampling_step_mm=None) -> torch.Tensor:
    if metric == 'l1':
        metric = L1()
    elif metric == 'l2':
        metric = L2()
    elif metric == 'ncc':
        metric = NCC()
    elif metric == 'dice':
        metric = Dice()
    else:
        raise ValueError(f'Metric name "{metric}" not implemented')

    sample_coords_scanner_space = compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_mm=sampling_step_mm)
    images1, images2 = images[pair_indices[:, 0]], images[pair_indices[:, 1]]
    affines1, affines2 = affines[pair_indices[:, 0]], affines[pair_indices[:, 1]]
    shapes1, shapes2 = shapes[pair_indices[:, 0]], shapes[pair_indices[:, 1]]
    sampled_im1, sample_mask1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space, images1,
                                                                          affines1, shapes1, interp_type=interp_type)
    sampled_im2, sample_mask2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space, images2,
                                                                          affines2, shapes2, interp_type=interp_type)
    loss = metric(sampled_im1, sampled_im2, mask1=sample_mask1, mask2=sample_mask2)
    return loss


def optimize_affines(images, affines, pair_indices, shapes, spacings,
                     max_epochs=10000, early_stop_epochs=10000):
    best_affines = affines.clone()
    best_loss = compute_pairwise_loss(images, best_affines, pair_indices, shapes).mean().item()
    best_epoch = 0
    needs_flip = torch.det(affines[:, :3, :3]) <= 0
    # assert (np.abs(affines[:3] - params_to_mat(mat_to_params(affines[:3], spacings[:3], needs_flip[:3]), spacings[:3], needs_flip[:3])).sum((1, 2)) < 1e-5).any()
    # assert (np.abs(affines[3:] - params_to_mat(mat_to_params(affines[3:], spacings[3:], needs_flip[3:]), spacings[3:], needs_flip[3:])).sum((1, 2)) < 1e-5).any()

    params_og = mat_to_params(affines, spacings, needs_flip)
    max_delta = torch.tensor([[torch.pi/8, torch.pi/8, torch.pi/8, 20, 20, 20]],
                             dtype=params_og.dtype, device=params_og.device)
    clamp_low = params_og - max_delta.tile((params_og.shape[0], 1))
    clamp_high = params_og + max_delta.tile((params_og.shape[0], 1))

    params = torch.nn.Parameter(params_og, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=1e-2)
    losses = []
    try:
        for i in tqdm.tqdm(range(max_epochs)):
            optimizer.zero_grad()
            epoch_affines = params_to_mat(params, spacings, needs_flip)
            loss = compute_pairwise_loss(images, epoch_affines, pair_indices, shapes)
            # TODO: Log image-wise and pair-wise losses?
            loss = loss.mean()
            losses.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_affines = epoch_affines.clone().detach()
                best_epoch = i
            if len(losses) > early_stop_epochs and np.min(losses[-early_stop_epochs:]) > best_loss:
                break
            loss.backward()
            optimizer.step()
            params.data.clamp(clamp_low, clamp_high)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping optimization.")
        pass
    print("Best loss:", best_loss, "Best epoch:", best_epoch)
    param_diff = mat_to_params(best_affines.clone(), spacings, needs_flip) - mat_to_params(affines.clone(), spacings, needs_flip)
    print("Rot diff:", param_diff[:, :3])
    print("Tra diff:", param_diff[:, 3:])
    if losses:
        plt.plot(losses)
        plt.show()
    return best_affines


if __name__ == '__main__':
    download_dir = r"D:\UKBB_subjects"
    registered_dataset_dir = r"D:\UKBB_subjects_aligned"
    sax_unregistered_dataset_dir = r"D:\UKBB_subjects_unaligned"
    subject_list = find_subjects(download_dir, sax_unregistered_dataset_dir)
    for idx, sub in enumerate(subject_list[57:]):
        print(f"Subject: {sub.name}     Idx: {idx}")
        subject_data = SubjectData(sub)

        images_pad = subject_data.images_pad.clone()
        affines = subject_data.affines.clone()
        idx_pairs = subject_data.idx_pairs.clone()
        shapes = subject_data.shapes.clone()
        spacings = subject_data.spacings.clone()
        plane_names = [p.name for p in subject_data.planes]

        start_loss = compute_pairwise_loss(images_pad, affines, idx_pairs, shapes)
        print(f"Starting loss: {start_loss.mean()}")
        # continue
        print(f"Starting pair-wise loss:")
        for i in range(start_loss.shape[0]):
            print(f"  {plane_names[idx_pairs[i, 0]]} - {plane_names[idx_pairs[i, 1]]} loss:  {start_loss[i]}")
        # visualize_intersections(images_pad, affines, idx_pairs, shapes, names=plane_names)

        new_affines = optimize_affines(images_pad, affines, idx_pairs, shapes, spacings)
       # str(Path(registered_dataset_dir) / sub.name))


        print("-------------------------------------------")
        final_loss = compute_pairwise_loss(images_pad, new_affines, idx_pairs, shapes)
        print(f"Final loss: {final_loss.mean()}")
        print(f"Final loss change: {final_loss.mean() - start_loss.mean()}")
        print(f"Final pair-wise loss:")
        for i in range(final_loss.shape[0]):
            print(f"  {plane_names[idx_pairs[i, 0]]} - {plane_names[idx_pairs[i, 1]]} loss change:  {final_loss[i] - start_loss[i]}")

        # subject_data.save_niftis(str(Path(registered_dataset_dir) / sub.name), new_affines)
        visualize_intersection_differences(images_pad, affines, new_affines, idx_pairs, shapes, names=plane_names)

