from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Tuple, Union, Iterable
import nibabel as nib
import numpy as np
import tqdm
from dataclasses import dataclass
from scipy.ndimage import map_coordinates
from itertools import combinations, product
import cv2
import random
import pygad


from geo_utils import get_image_plane_from_array, plane_intersection, get_image_edge_planes, plane_line_intersection,\
    Line, PlaneEq
from utils import SubjectFiles, find_subjects, normalize_image, \
    normalize_image_with_mean_lv_value
from metrics import L2, L1


class OptimizableImage:
    def __init__(self, image, affine, spacing, name, seg=None):
        self.image = image
        self.seg = seg
        self.affine = affine
        assert len(spacing) == 3
        self.spacing = np.array(spacing)
        self.needs_flip = np.linalg.det(affine[:3, :3]) <= 0
        self.parameters = self.get_params_from_affine()  # r1, r2, r3, t1, t2, t3
        self.update_parameters()
        self.name = name

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
        self.affine = affine

    def update_parameters(self, new_params: Optional[np.ndarray] = None):
        """
        Update affine matrix based on provided rototranslation params.
        :param new_params: 1D array containing rotation parameters, followed by 3 translation parameters.
        """
        if new_params is not None:
            assert isinstance(new_params, np.ndarray)
            assert self.parameters.shape[0] == new_params.shape[0]
            self.parameters = new_params
        # We assume the last 3 parameters are translation, the rest before that are rotation
        rotation_params, translation_params = self.parameters[:-3], self.parameters[-3:]
        self.update_affine(rotation_params, translation_params)

    def modify_parameters(self, param_delta: np.ndarray):
        new_params = self.parameters + param_delta
        self.update_parameters(new_params)



@dataclass
class PlanePair:
    plane1: OptimizableImage
    plane2: OptimizableImage
    previous_loss: Optional[float] = None


class SubjectData:
    def __init__(self, subject: SubjectFiles):
        self.planes = []
        self.seg_planes = []
        if subject.la4ch is not None:
            try:
                nii_4ch = nib.load(subject.la4ch)
                nii_4ch_seg = nib.load(subject.la4ch_seg)
                self.planes.append(OptimizableImage(nii_4ch.dataobj[:].squeeze(2), nii_4ch.affine, nii_4ch.header.get_zooms()[:3], "la_4ch", seg=nii_4ch_seg.dataobj[:]))
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 4ch image not found")
        if subject.la3ch is not None:
            try:
                nii_3ch = nib.load(subject.la3ch)
                # nii_3ch_seg = nib.load(subject.la3ch_seg)
                self.planes.append(OptimizableImage(nii_3ch.dataobj[:].squeeze(2), nii_3ch.affine, nii_3ch.header.get_zooms()[:3], "la_3ch"))
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 3ch image not found")
        if subject.la2ch is not None:
            try:
                nii_2ch = nib.load(subject.la2ch)
                nii_2ch_seg = nib.load(subject.la2ch_seg)
                self.planes.append(OptimizableImage(nii_2ch.dataobj[:].squeeze(2), nii_2ch.affine, nii_2ch.header.get_zooms()[:3], "la_2ch", seg=nii_2ch_seg.dataobj[:]))
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 2ch image not found")
        self.idx_pairs = [*combinations(list(range(len(self.planes))), 2)]

        sa_la_product = list(product(list(range(len(self.planes))), list(range(len(self.planes), len(self.planes) + len(subject.sax)))))
        self.idx_pairs = self.idx_pairs + sa_la_product
        for i, (sa_lice, sa_lice_seg) in enumerate(zip(subject.sax, subject.sax_seg)):
            im_nii = nib.load(sa_lice)
            im_nii_seg = nib.load(sa_lice_seg)
            self.planes.append(OptimizableImage(im_nii.dataobj[:].squeeze(2), im_nii.affine, im_nii.header.get_zooms()[:3], f"sa{i}-{len(subject.sax)}", seg=im_nii_seg.dataobj[:]))
        assert min([min(i) for i in self.idx_pairs]) == 0
        assert max([max(i) for i in self.idx_pairs]) == len(self.planes) - 1
        self.plane_pairs = [PlanePair(plane1=self.planes[i], plane2=self.planes[j]) for i, j in self.idx_pairs]

        self.plane_param_size = self.planes[0].parameters.shape[0]
        self.param_size = self.plane_param_size * len(self.planes)
        self.parameters = np.zeros((self.param_size,), dtype=float)
        self.update_parameters()
        assert self.parameters.shape[0] == self.param_size

    def update_parameters(self, new_params: Optional[np.ndarray] = None):
        if new_params is not None:
            assert new_params.shape[0] == len(self.planes) * self.planes[0].parameters.shape[0], f"{new_params.shape[0]}, {len(self.planes) * self.planes[0].parameters.shape[0]}"

            for i, plane in enumerate(self.planes):
                plane.update_parameters(new_params[i * self.plane_param_size: (i + 1) * self.plane_param_size])
        self.reload_pameters()

    def modify_parameters(self, param_delta: np.ndarray):
        assert param_delta.shape[0] == len(self.planes) * self.planes[0].parameters.shape[0], f"{param_delta.shape[0]}, {len(self.planes) * self.planes[0].parameters.shape[0]}"
        for i, plane in enumerate(self.planes):
            plane.modify_parameters(param_delta[i * self.plane_param_size: (i + 1) * self.plane_param_size])
        self.reload_pameters()

    def reload_pameters(self):
        self.parameters = np.concatenate([plane.parameters for plane in self.planes], 0)

    def save_niftis(self, directory: str):
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


def find_image_border_intercepts(intersec_line_scan_space, image, affine) -> Line:
    inverse_affine = np.linalg.inv(affine)
    a, b = inverse_affine @ np.array([*intersec_line_scan_space[0], 1.]), inverse_affine @ np.array([*intersec_line_scan_space[1], 1.])

    if np.abs(a[2]) > 1e-3 or np.abs(b[2]) > 1e-3:
        raise ValueError("Intersection line projected onto voxel space appears to be farther from the image z plane "
                         "than we could account for numerical imprecission. Kinda weird.... "
                         "We are skipping this bad boy!")
    a[2], b[2] = 0.0, 0.0
    if a[1] == b[1]:
        a[0], b[0] = 0.0, image.shape[0] - 1.0
        return affine @ a, affine @ b
    if a[0] == b[0]:
        a[1], b[1] = 0.0, image.shape[1] - 1.0
        return affine @ a, affine @ b
    slope = (a[0] - b[0]) / (a[1] - b[1])
    intercept = a[0] - slope * a[1]
    # Calculate intersection points with each border
    x_bottom = (0. - intercept) / slope
    x_top = (image.shape[0] - 1 - intercept) / slope
    y_right = slope * (image.shape[1] - 1) + intercept
    y_left = slope * 0. + intercept

    # Check which intersection points are within the bounds of the image
    intersection_points = []
    if 0 <= x_bottom <= image.shape[1] - 1:  # Line crosses bottom edge
        point1 = np.array([0., x_bottom, 0., 1.])
        intersection_points.append(point1)
    if 0 <= x_top <= image.shape[1] - 1:  # Line crosses top edge
        point2 = np.array([image.shape[0]-1, x_top, 0., 1.])
        intersection_points.append(point2)
    # A line may intersect top and bottom, in which case, we're done looking for intersections.
    # If not check which left/right edges are intersected
    if len(intersection_points) < 2:
        if 0 <= y_right <= image.shape[0] - 1:
            # Avoid placing the same point twice if line intersects bottom-right/top-right corner
            if x_top != image.shape[1] - 1 and x_bottom != image.shape[1] - 1:
                point3 = np.array([y_right, image.shape[1]-1, 0., 1.])
                intersection_points.append(point3)
        if 0 <= y_left <= image.shape[0] - 1:
            # Avoid placing the same point twice if line intersects bottom-left/top-left corner
            if x_top != 0. and x_bottom != 0.:
                point4 = np.array([y_left, 0., 0., 1.])
                intersection_points.append(point4)

    # Number of intersections should be 2. Corner intersection duplicates should have been filtered out
    if len(intersection_points) < 2:
        raise ValueError("Image intersection is not a line. (Don't intersect at all, or intersect at a single point)")
    assert abs(slope - (intersection_points[1][0] - intersection_points[0][0]) / (intersection_points[1][1] - intersection_points[0][1])) < 1e-3, f" {slope}, {(intersection_points[1][0] - intersection_points[0][0]) / (intersection_points[1][1] - intersection_points[0][1])}"
    assert abs(intercept - (intersection_points[0][0] - slope * intersection_points[0][1])) < 1e-3
    assert abs(intercept - (intersection_points[1][0] - slope * intersection_points[1][1])) < 1e-3
    return affine @ intersection_points[0], affine @ intersection_points[1]


def compute_intersection_line(plane_pair: PlanePair) -> Optional[np.ndarray]:
    try:
        plane1 = get_image_plane_from_array(plane_pair.plane1.image, plane_pair.plane1.affine)
        plane2 = get_image_plane_from_array(plane_pair.plane2.image, plane_pair.plane2.affine)
        intersec_scanner_space = plane_intersection(plane1, plane2)
        try:
            im1_border_p1, im1_border_p2 = find_image_border_intercepts(intersec_scanner_space, plane_pair.plane1.image, plane_pair.plane1.affine)
            im2_border_p1, im2_border_p2 = find_image_border_intercepts(intersec_scanner_space, plane_pair.plane2.image, plane_pair.plane2.affine)
        except ValueError:
            return None
        im1_dir = im1_border_p2 - im1_border_p1
        im2_p1_dir = im2_border_p1 - im1_border_p1
        im2_p2_dir = im2_border_p2 - im1_border_p1
        im1_dir_norm = np.linalg.norm(im1_dir)
        im2_p1_dir_norm = np.linalg.norm(im2_p1_dir)
        im2_p2_dir_norm = np.linalg.norm(im2_p2_dir)

        if np.dot(im1_dir, im2_p1_dir) <= 0 and np.dot(im1_dir, im2_p2_dir) <= 0:
            # Images don't overlap
            return None
        elif np.dot(im2_p1_dir, im1_dir) < 0 < np.dot(im1_dir, im2_p2_dir):
            if im1_dir_norm > im2_p2_dir_norm:
                im_intersection_line = (im1_border_p1, im1_border_p2)
            else:
                im_intersection_line = (im1_border_p1, im2_border_p2)
        elif np.dot(im2_p2_dir, im1_dir) < 0 < np.dot(im1_dir, im2_p1_dir):
            if im1_dir_norm > im2_p1_dir_norm:
                im_intersection_line = (im1_border_p1, im1_border_p2)
            else:
                im_intersection_line = (im1_border_p1, im2_border_p1)
        else:
            if max(im2_p1_dir_norm, im2_p2_dir_norm) < im1_dir_norm:
                im_intersection_line = (im2_border_p1, im2_border_p2)
            elif im1_dir_norm < min(im2_p1_dir_norm, im2_p2_dir_norm):
                # Images don't overlap
                return None
            elif im1_dir_norm < im2_p1_dir_norm:
                im_intersection_line = (im2_border_p2, im1_border_p2)
            else:
                im_intersection_line = (im2_border_p1, im1_border_p2)

        num_samples = 100
        step_vector = im_intersection_line[1] - im_intersection_line[0]
        step_vector = step_vector / num_samples
        sample_line = np.array([im_intersection_line[0] + step_vector * i for i in range(num_samples)])
        # assert sample_line.shape[0] == num_samples
        assert sample_line.shape[1] == 4
        assert len(sample_line.shape) == 2
        return sample_line

    except Exception as e:
        print(f"{e} encounter between planes: {plane_pair.plane1.name}, {plane_pair.plane2.name}")
        raise e


def sample_image_along_scanner_line(sample_coords_scanner_space: np.ndarray, image: np.ndarray, affine: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    assert sample_coords_scanner_space.shape[1] == 4
    assert len(sample_coords_scanner_space.shape) == 2
    inverse_affine = np.linalg.inv(affine)
    coords_voxel_space = inverse_affine @ sample_coords_scanner_space.T
    coords_voxel_space = coords_voxel_space[:2]
    sampled_points = map_coordinates(image, coords_voxel_space, mode="nearest")
    return sampled_points, coords_voxel_space


def compute_pairwise_loss(subject, frames: Iterable[int] = (0,), visualize=False) -> float:
    # subject_data.update_parameters(new_params)
    img_pairs = subject.plane_pairs
    loss = 0.
    metric = L1()
    for f in frames:
        loss_frame = 0.
        for i, pair in enumerate(img_pairs):
            sample_coords_scanner_space = compute_intersection_line(pair)
            if sample_coords_scanner_space is None:
                return np.Inf
            sampled_im1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space.copy(), pair.plane1.image[..., f], pair.plane1.affine)
            sampled_im2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space.copy(), pair.plane2.image[..., f], pair.plane2.affine)
            loss_pair = metric(sampled_im1, sampled_im2)
            pair.previous_loss = loss_pair
            loss_frame += loss_pair

            if visualize:
                # Visualize
                scaling = 8
                cat_line = normalize_image_with_mean_lv_value(np.stack((sampled_im1, np.zeros_like(sampled_im1), sampled_im2), axis=0)) * 255
                cat_line = cv2.UMat(cat_line.astype(np.uint8))
                cat_line = cv2.resize(cat_line, (sampled_im1.shape[0] * scaling, 3 * scaling))
                cv2.imshow(pair.plane1.name + "_" + pair.plane2.name + "_line", cat_line)

                scaling = 3
                im1_vis = pair.plane1.image[..., f]
                # im1_vis = np.concatenate((pair.plane1.image[..., f], pair.plane1.image[..., f]), axis=1)
                im1_vis = normalize_image(im1_vis) * 255
                im1_vis_ = cv2.UMat(np.stack([im1_vis.astype(np.uint8)]*3, axis=-1))
                p1 = line_im1[:, 0].round().astype(int)
                p2 = line_im1[:, -1].round().astype(int)
                cv2.line(im1_vis_, (p1[1], p1[0],), (p2[1], p2[0],), (0,255,0))
                im1_vis_ = cv2.resize(im1_vis_, (im1_vis.shape[1] * scaling, im1_vis.shape[0] * scaling))
                cv2.imshow(pair.plane1.name, im1_vis_)

                im2_vis = pair.plane2.image[..., f]
                # im2_vis = np.concatenate((pair.plane2.image[..., f], pair.plane2.image[..., f]), axis=1)
                im2_vis = normalize_image(im2_vis) * 255
                im2_vis_ = cv2.UMat(np.stack([im2_vis.astype(np.uint8)]*3, axis=-1))
                p1 = line_im2[:, 0].round().astype(int)
                p2 = line_im2[:, -1].round().astype(int)
                cv2.line(im2_vis_, (p1[1], p1[0],), ( p2[1], p2[0],), (0,255,0))
                im2_vis_ = cv2.resize(im2_vis_, (im2_vis.shape[1] * scaling, im2_vis.shape[0] * scaling))
                cv2.imshow(pair.plane2.name, im2_vis_)

                cv2.waitKey()

        loss += loss_frame
    return loss


def fitness_func(ga_instance, solution, solution_idx):
    subj = deepcopy(subject_data)
    subj.modify_parameters(np.array(solution))
    loss = compute_pairwise_loss(subj, visualize=False)
    return -loss


def on_generation(ga_instance: pygad.GA):
    subj = deepcopy(subject_data)
    subj.modify_parameters(np.array(ga_instance.best_solutions[0]))
    best_loss = compute_pairwise_loss(subj)
    print(f"\rIteration:  {ga_instance.generations_completed}    Best loss:  {best_loss}", end="")


def optimize_affines(subject: SubjectData, save_dir: str):
    std = np.array([0.01] * (subject_data.plane_param_size - 3) + [2.0] * 3)
    std = np.concatenate([std] * len(subject_data.planes), axis=0)
    max_param_dist = np.array([0.3] * (subject_data.plane_param_size - 3) + [20.0] * 3)
    max_param_dist = np.concatenate([max_param_dist] * len(subject_data.planes), axis=0)
    og_loss = compute_pairwise_loss(subject_data)
    prev_loss = og_loss
    assert prev_loss != np.Inf
    initial_pop = np.random.uniform(low=-std, high=std, size=(100, std.shape[0])).clip(-max_param_dist, max_param_dist)
    initial_pop = np.concatenate([initial_pop, np.zeros(initial_pop[:1].shape)], axis=0)

    ga_instance = pygad.GA(num_generations=1000,
                           num_parents_mating=10,
                           parent_selection_type="rank",
                           fitness_func=fitness_func,
                           initial_population=initial_pop,
                           parallel_processing=None,
                           random_mutation_min_val=-max_param_dist/10,
                           random_mutation_max_val=max_param_dist/10,
                           on_generation=on_generation,
                           save_best_solutions=True,
                           gene_space=[{"low": i, "high": j} for i, j in zip(-max_param_dist, max_param_dist)],
                           keep_elitism=1,
                           )
    ga_instance.run()
    best_params = np.array(ga_instance.best_solutions[0])
    # for i in range(10000):
    #     param_delta = np.random.normal(0.0, scale=std, size=(100, std.shape[0]))
    #     new_params = current_params + param_delta
    #     new_params = np.clip(new_params, og_params - max_param_dist, og_params + max_param_dist)
    #     loss = compute_pairwise_loss(new_params)
    #     if loss < prev_loss or random.uniform(0., 1.) < prob:
    #         if loss < prev_best_loss:
    #             prev_best_params = subject_data.parameters.copy()
    #             prev_best_loss = loss
    #             print(f"New best params found!   Iter: {i}    Loss: {prev_loss}")
    #         prev_loss = loss
    #         current_params = subject_data.parameters.copy()
    #     if prob > min_prob:
    #         prob = p_decay_func(prob)
    #     if i < std_decay_stop_iter:
    #         std = std_decay_func(std)
    #     if i % 10 == 0:
    #         print(f"Iter: {i:06d}    Prob: {prob:.6f}    Std: ({std[0]:.6f}, {std[-1]:.6f})    Original loss: {og_loss:.6f}    Best loss: {prev_best_loss:.6f}    Loss: {loss:.6f}", end="\r")
    #
    # print(f"Iter: {i:06d}    Prob: {prob:.6f}    Std: ({std[0]:.6f}, {std[-1]:.6f})    Original loss: {og_loss:.6f}    Best loss: {prev_best_loss:.6f}")
    subject_data.modify_parameters(best_params)
    final_loss = compute_pairwise_loss(subject_data)
    print(f"\nOriginal loss: {og_loss}     Final loss: {final_loss}")
    for og, new in zip(subject_data_og.planes, subject_data.planes):
        param_diff = og.parameters - new.parameters
        print(og.name, param_diff)
    subject_data.save_niftis(save_dir)
    compute_pairwise_loss(subject_data, visualize=True)


if __name__ == '__main__':
    download_dir = r"D:\UKBB_subjects"
    registered_dataset_dir = r"D:\UKBB_subjects_aligned"
    sax_unregistered_dataset_dir = r"D:\UKBB_subjects_unaligned"
    subject_list = find_subjects(download_dir, sax_unregistered_dataset_dir)
    for sub in subject_list[9:]:
        subject_data = SubjectData(sub)
        # subject_data.modify_parameters(np.random.uniform(0, 0.1, size=subject_data.parameters.shape))
        subject_data_og = SubjectData(sub)
        print(f"Starting loss: {compute_pairwise_loss(subject_data, visualize=True)}")

        optimize_affines(subject_data, str(Path(registered_dataset_dir) / sub.name))



