from typing import Optional, List, Tuple, Union, Iterable
import nibabel as nib
import numpy as np
import tqdm
from dataclasses import dataclass
from scipy.ndimage import map_coordinates
import cv2


from geo_utils import get_image_plane_from_array, plane_intersection, get_image_edge_planes, plane_line_intersection,\
    Line, PlaneEq
from utils import SubjectFiles, SubjectAffines, SubjectSlices, find_subjects, normalize_image, \
    normalize_image_with_mean_lv_value
from metrics import L2


@dataclass
class PlanePair:
    img1: np.ndarray
    img2: np.ndarray
    affine1: np.ndarray
    affine2: np.ndarray
    name1: str = ""
    name2: str = ""
    previous_loss: Optional[float] = None


def create_plane_pairs(subject) -> List[PlanePair]:

    nii_4ch = nib.load(subject.la4ch)
    nii_3ch = nib.load(subject.la3ch)
    nii_2ch = nib.load(subject.la2ch)
    images = SubjectSlices(la4ch=nii_4ch.dataobj[:].squeeze(2),  # We treat images as 2D+t
                           la3ch=nii_3ch.dataobj[:].squeeze(2),
                           la2ch=nii_2ch.dataobj[:].squeeze(2),
                           sax=[])
    affines = SubjectAffines(la4ch=nii_4ch.affine, la3ch=nii_3ch.affine, la2ch=nii_2ch.affine, sax=[])
    for sa_lice in subject.sax:
        im_nii = nib.load(sa_lice)
        images.sax.append(im_nii.dataobj[:].squeeze(2))  # We treat images as 2D+t
        affines.sax.append(im_nii.affine)

    img_pairs = [
                 PlanePair(img1=images.la4ch, affine1=affines.la4ch, name1="4ch",
                           img2=images.la3ch, affine2=affines.la3ch, name2="3ch"),
                 PlanePair(img1=images.la4ch, affine1=affines.la4ch, name1="4ch",
                           img2=images.la2ch, affine2=affines.la2ch, name2="2ch"),
                 PlanePair(img1=images.la3ch, affine1=affines.la3ch, name1="3ch",
                           img2=images.la2ch, affine2=affines.la2ch, name2="2ch"),
                 ]
    for slice_idx, (sa_aff, sa_im) in enumerate(zip(affines.sax, images.sax)):
        sa_la_pair = PlanePair(img1=sa_im, affine1=sa_aff, name1=f"sa{slice_idx}",
                               img2=images.la4ch, affine2=affines.la4ch, name2="4ch")
        img_pairs.append(sa_la_pair)
        sa_la_pair = PlanePair(img1=sa_im, affine1=sa_aff, name1=f"sa{slice_idx}",
                               img2=images.la3ch, affine2=affines.la3ch, name2="3ch")
        img_pairs.append(sa_la_pair)
        sa_la_pair = PlanePair(img1=sa_im, affine1=sa_aff, name1=f"sa{slice_idx}",
                               img2=images.la2ch, affine2=affines.la2ch, name2="2ch")
        img_pairs.append(sa_la_pair)
    return img_pairs


def find_image_border_intercepts(intersec_line_scan_space, image, affine) -> Line:
    inverse_affine = np.linalg.inv(affine)
    a, b = inverse_affine @ np.array([*intersec_line_scan_space[0], 1.]), \
        inverse_affine @ np.array([*intersec_line_scan_space[1], 1.])
    # a, b = np.array((10,50,0, 1)), np.array((100,50,0,1))
    assert np.abs(a[2]) < 1e-3 and np.abs(b[2]) < 1e-3
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
        point = np.array([0., x_bottom, 0., 1.])
        intersection_points.append(point)
    if 0 <= x_top <= image.shape[1] - 1:  # Line crosses top edge
        point = np.array([image.shape[0]-1, x_top, 0., 1.])
        intersection_points.append(point)
    # A line may intersect top and bottom, in which case, we're done looking for intersections.
    # If not check which left/right edges are intersected
    if len(intersection_points) < 2:
        if 0 <= y_right <= image.shape[0] - 1:
            # Avoid placing the same point twice if line intersects bottom-right/top-right corner
            if x_top != image.shape[1] - 1 and x_bottom != image.shape[1] - 1:
                point = np.array([y_right, image.shape[1]-1, 0., 1.])
                intersection_points.append(point)
        if 0 <= y_left <= image.shape[0] - 1:
            # Avoid placing the same point twice if line intersects bottom-left/top-left corner
            if x_top != 0. and x_bottom != 0.:
                point = np.array([y_left, 0., 0., 1.])
                intersection_points.append(point)

    # Number of intersections should be 2. Corner intersection duplicates should have been filtered out
    assert len(intersection_points) == 2
    assert abs(slope - (intersection_points[1][0] - intersection_points[0][0]) / (intersection_points[1][1] - intersection_points[0][1])) < 1e-3, f" {slope}, {(intersection_points[1][0] - intersection_points[0][0]) / (intersection_points[1][1] - intersection_points[0][1])}"
    assert abs(intercept - (intersection_points[0][0] - slope * intersection_points[0][1])) < 1e-3
    assert abs(intercept - (intersection_points[1][0] - slope * intersection_points[1][1])) < 1e-3
    return affine @ intersection_points[0], affine @ intersection_points[1]


def compute_intersection_line(plane_pair: PlanePair) -> np.ndarray:
    try:
        plane1 = get_image_plane_from_array(plane_pair.img1, plane_pair.affine1)
        plane2 = get_image_plane_from_array(plane_pair.img2, plane_pair.affine2)
        intersec_scanner_space = plane_intersection(plane1, plane2)
        im1_border_p1, im1_border_p2 = find_image_border_intercepts(intersec_scanner_space, plane_pair.img1, plane_pair.affine1)
        im2_border_p1, im2_border_p2 = find_image_border_intercepts(intersec_scanner_space, plane_pair.img1, plane_pair.affine1)
        im1_dir = im1_border_p2 - im1_border_p1
        im2_p1_dir = im2_border_p1 - im1_border_p1
        im2_p2_dir = im2_border_p2 - im1_border_p1
        im1_dir_norm = np.linalg.norm(im1_dir)
        im2_p1_dir_norm = np.linalg.norm(im2_p1_dir)
        im2_p2_dir_norm = np.linalg.norm(im2_p2_dir)

        if np.dot(im1_dir, im2_p1_dir) <= 0 and np.dot(im1_dir, im2_p2_dir) <= 0:
            raise ValueError("Images don't overlap")
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
                raise ValueError("Images don't overlap")
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
        print(f"{e} encounter between planes: {plane_pair.name1}, {plane_pair.name2}")
        raise e


def sample_image_along_scanner_line(sample_coords_scanner_space: np.ndarray, image: np.ndarray, affine: np.ndarray) -> np.ndarray:
    assert sample_coords_scanner_space.shape[1] == 4
    assert len(sample_coords_scanner_space.shape) == 2
    inverse_affine = np.linalg.inv(affine)
    coords_voxel_space = inverse_affine @ sample_coords_scanner_space.T
    coords_voxel_space = coords_voxel_space[:2]
    sampled_points = map_coordinates(image, coords_voxel_space, mode="nearest")
    return sampled_points, coords_voxel_space


def compute_pairwise_loss(img_pairs: List[PlanePair], frames: Iterable[int] = (0,), visualize=False):
    loss = 0.
    metric = L2()
    for f in frames:
        loss_frame = 0.
        for pair in img_pairs:
            sample_coords_scanner_space = compute_intersection_line(pair)
            sampled_im1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space.copy(), pair.img1[..., f], pair.affine1)
            sampled_im2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space.copy(), pair.img2[..., f], pair.affine2)
            loss_pair = metric(sampled_im1, sampled_im2)
            loss_frame += loss_pair
            print(loss_frame)

            if visualize:
                # Visualize
                scaling = 8
                cat_line = normalize_image_with_mean_lv_value(np.stack((sampled_im1, np.zeros_like(sampled_im1), sampled_im2), axis=0)) * 255
                cat_line = cv2.UMat(cat_line.astype(np.uint8))
                cat_line = cv2.resize(cat_line, (sampled_im1.shape[0] * scaling, 3 * scaling))
                cv2.imshow(pair.name1 + pair.name2 + "line", cat_line)

                scaling = 3
                im1_vis = np.concatenate((pair.img1[..., f], pair.img1[..., f]), axis=1)
                im1_vis = normalize_image(im1_vis) * 255
                im1_vis_ = cv2.UMat(np.stack([im1_vis.astype(np.uint8)]*3, axis=-1))
                p1 = line_im1[:, 0].round().astype(int)
                p2 = line_im1[:, -1].round().astype(int)
                cv2.line(im1_vis_, (p1[1], p1[0],), (p2[1], p2[0],), (0,255,0))
                im1_vis_ = cv2.resize(im1_vis_, (im1_vis.shape[1] * scaling, im1_vis.shape[0] * scaling))
                cv2.imshow(pair.name1, im1_vis_)

                im2_vis = np.concatenate((pair.img2[..., f], pair.img2[..., f]), axis=1)
                im2_vis = normalize_image(im2_vis) * 255
                im2_vis_ = cv2.UMat(np.stack([im2_vis.astype(np.uint8)]*3, axis=-1))
                p1 = line_im2[:, 0].round().astype(int)
                p2 = line_im2[:, -1].round().astype(int)
                cv2.line(im2_vis_, (p1[1], p1[0],), ( p2[1], p2[0],), (0,255,0))
                im2_vis_ = cv2.resize(im2_vis_, (im2_vis.shape[1] * scaling, im2_vis.shape[0] * scaling))
                cv2.imshow(pair.name2, im2_vis_)

                cv2.waitKey()

        loss += loss_frame


def optimize_affines(subject: SubjectFiles, save_dir: str):
    img_pairs = create_plane_pairs(subject)
    compute_pairwise_loss(img_pairs)


if __name__ == '__main__':
    download_dir = r"D:\UKBB_subjects"
    registered_dataset_dir = r"D:\UKBB_subjects_aligned"
    sax_unregistered_dataset_dir = r"D:\UKBB_subjects_unaligned"
    subject_list = find_subjects(download_dir, sax_unregistered_dataset_dir)
    optimize_affines(subject_list[1], "")



