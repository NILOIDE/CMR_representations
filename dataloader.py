import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
import pickle
import os
import h5py
from tqdm import tqdm
import traceback

from data_utils import array_to_nifti
from dataset import CardiacUKBB, CardiacUKBBValidationFullImage, CardiacUKBBFullImage
from normalization_utils import crop_around_heart, normalize_slice_orientation
from sa_la_interp import interpolate_sa_segs_to_la
from utils import normalize_image_with_percentile, mat_to_params, extract_2dt_contours

UNCERTAIN, AUTO, HAND_ANNOTATED = 'uncertain', 'auto', 'hand_annotated'
PREPR_FILE_NAME = "prep_data.h5"


class CMRDataModule(pl.LightningDataModule):
    def __init__(self,
                 load_la_dir: str,
                 load_sa_dir: str,
                 preprocessed_store_path: str,
                 log_path: str,
                 num_train: int = 100,
                 num_val: int = 8,
                 num_test: int = 1,
                 replace_existing_preprocessed=False,
                 crop_around_heart=True,
                 batch_size: int = 32,
                 num_coords_voxel: int = 40000,
                 num_coords_surface: int = 10000,
                 inf_num_coords: int = 4000,
                 num_workers: int = 0,):
        super().__init__()
        self.load_la_dir = load_la_dir
        self.load_sa_dir = load_sa_dir
        self.store_path = preprocessed_store_path
        self.log_path = log_path
        self.crop_around_heart = crop_around_heart
        self.replace_existing_processed = replace_existing_preprocessed
        self.batch_size = batch_size
        self.num_coords_voxel = num_coords_voxel
        self.num_coords_surface = num_coords_surface
        self.inf_num_coords = inf_num_coords
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.dim_max = None
        self.num_workers = num_workers
        self.subject_data = []
        self.data_prepared = False

    def prepare_data(self) -> None:
        if self.data_prepared:
            return
        num_subjects = self.num_train + self.num_val + self.num_test
        pickle_name = f"dataset_paths_{num_subjects}_{Path(self.store_path).name}.pkl"
        try:
            if self.replace_existing_processed:
                raise FileNotFoundError
            with open(pickle_name, 'rb') as handle:
                subject_data = pickle.load(handle)
        except FileNotFoundError:
            subject_data = self.find_subjects(max_num=num_subjects)
            with open(pickle_name, 'wb') as handle:
                pickle.dump(subject_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(subject_data) == num_subjects
        assert all([Path(i).exists() for i in subject_data])
        shutil.copy(pickle_name, Path(self.log_path) / pickle_name)
        self.subject_data = subject_data
        train_paths = subject_data[:self.num_train]
        val_paths = subject_data[self.num_train:self.num_train+self.num_val]
        test_paths = subject_data[self.num_train+self.num_val:self.num_train+self.num_val+self.num_test]
        self.train_dset = CardiacUKBB(train_paths[:], max_slices=self.get_max_slices(),
                                      max_slice_shape=self.get_max_slice_shape(),
                                      num_coords_voxel=self.num_coords_voxel,
                                      num_coords_surface=self.num_coords_surface,
                                      cache_data=True, cache_to_gpu=False)
        self.val_dset = CardiacUKBB(val_paths[:], max_slices=self.get_max_slices(),
                                    max_slice_shape=self.get_max_slice_shape(),
                                    num_coords_voxel=self.num_coords_voxel,
                                    num_coords_surface=self.num_coords_surface,
                                    cache_data=True, cache_to_gpu=False)
        self.test_dset = CardiacUKBB(test_paths[:], max_slices=self.get_max_slices(),
                                     max_slice_shape=self.get_max_slice_shape(),
                                     num_coords_voxel=self.num_coords_voxel,
                                     num_coords_surface=self.num_coords_surface,
                                     cache_data=True, cache_to_gpu=False)
        self.data_prepared = True

    def setup(self, stage: str):
        self._train_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=self.num_workers, pin_memory=False,
                                            persistent_workers=self.num_workers > 0)
        self._val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size,
                                          num_workers=self.num_workers, pin_memory=False,
                                          persistent_workers=self.num_workers > 0)
        self._test_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, pin_memory=False,
                                           persistent_workers=self.num_workers > 0)

    def get_coord_size(self) -> int:
        return self.train_dset.get_coord_size()

    def extract_max_dims(self):
        if self.dim_max is not None:
            return
        self.dim_max = (-1, -1, -1, -1)
        for p in self.subject_data:
            with h5py.File(p, 'r') as f:
                shape = f['image_padded'].shape
                assert len(shape) == len(self.dim_max)
                self.dim_max = [max(a, b) for a, b in zip(self.dim_max, shape)]

    def get_max_slices(self) -> int:
        if self.dim_max is None:
            self.extract_max_dims()
        return self.dim_max[1]

    def get_max_slice_shape(self) -> Tuple[int, int, int]:
        if self.dim_max is None:
            self.extract_max_dims()
        return self.dim_max[-2], self.dim_max[-1], self.dim_max[0]  # (H, W, T)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def find_subjects(self, max_num=100, **kwargs):
        # Try to see if we have enough preprocessed files
        if not self.replace_existing_processed:
            subj_h5s = list(Path(self.store_path).iterdir())
            subject_data_paths = sorted(subj_h5s, key=lambda x: x.name)
            subject_data_paths = [str(i / PREPR_FILE_NAME) for i in subject_data_paths]
            if len(subject_data_paths) >= max_num:
                return subject_data_paths[:max_num]

        # Look for subjects to preprocess
        images = []
        segs = []
        segs_auto = []
        interp_segs = []
        seg_categories = []
        annotated_subj_ids = [1009169, 1011525, 1012959, 1021869, 1026284, 1037010, 1037287, 1037527, 1043831, 1050481,
                           1053004, 1059837, 1060134, 1060474, 1061311, 1062139, 1063068, 1067227, 1078928, 1083769]
        annotated_subj_ids = [str(i) for i in annotated_subj_ids]
        annotated_subjs = list(sorted([str(Path(self.load_la_dir) / i) for i in annotated_subj_ids]))
        # assert all([Path(i).exists() for i in annotated_subjs])
        subjects = list(sorted(os.listdir(str(self.load_la_dir))))
        subjects = annotated_subj_ids + [i for i in subjects if Path(i).name not in annotated_subj_ids]
        subjects = list(sorted(subjects))
        for i, parent in enumerate(subjects):
            if parent in {"1013493", "1439318"}:
                continue
            # Images
            la_files = sorted(list(Path(os.path.join(self.load_la_dir, parent)).rglob('la*.nii.gz')))
            la_files = [str(x) for x in la_files]

            sa_files = sorted(list(Path(os.path.join(self.load_sa_dir, parent, "sa_slices")).rglob('sa*.nii.gz')))
            sa_files = [str(x) for x in sa_files]
            if len(la_files) != 3 or len(sa_files) < 5:
                continue
            slices = la_files + sa_files
            # Interp segmentations for masking slices where seg is not known.
            # Hopefully been interpolated before, else None (and will be interpolated).
            seg_la_interp_files = [Path(os.path.join(self.load_la_dir, parent)) / 'interp_seg_lv_la_2ch.nii.gz',
                                   Path(os.path.join(self.load_la_dir, parent)) / 'interp_seg_lv_la_3ch.nii.gz',
                                   Path(os.path.join(self.load_la_dir, parent)) / 'interp_seg_lv_la_4ch.nii.gz']
            seg_sa_interp_files = sorted(list(Path(os.path.join(self.load_sa_dir, parent, "sa_slices")).rglob('interp_seg_sa*.nii.gz')))
            seg_interp_files = seg_la_interp_files + seg_sa_interp_files
            seg_interp_files = [str(x) for x in seg_interp_files]
            # Segmentations
            seg_la_files_auto = [Path(os.path.join(self.load_la_dir, parent)) / 'seg_lv_la_2ch.nii.gz',
                                 Path(os.path.join(self.load_la_dir, parent)) / 'seg_lv_la_3ch.nii.gz',
                                 Path(os.path.join(self.load_la_dir, parent)) / 'seg_lv_la_4ch.nii.gz']
            seg_sa_files = sorted(list(Path(os.path.join(self.load_sa_dir, parent, "sa_slices")).rglob('seg_sa*')))
            seg_sa_files_auto = [i for i in seg_sa_files if 'labels' not in i.name]
            seg_files_auto = seg_la_files_auto + seg_sa_files_auto
            if not seg_files_auto:
                continue

            # If hand-annotated files exist, pick them over auto-segmented ones.
            seg_files = []
            for i, p in enumerate(seg_files_auto):
                if i >= 3:
                    seg_files.append(p)
                    continue
                hand_file = p.parent / (p.name[:-len('.nii.gz')] + '-labels.nii')
                if hand_file.exists():
                    seg_files.append(hand_file)
                    continue
                hand_file = p.parent / (p.name[:-len('.nii.gz')] + '-labels.nii.gz')
                if hand_file.exists():
                    seg_files.append(hand_file)
                    continue
                seg_files.append(p)

            # def categorize_seg_files(files, process_backwards=False):
            #     assert all([isinstance(f, Path) for f in files])
            #     annotation_type = []
            #     found = False
            #     files = files[::-1] if process_backwards else files
            #     for s in files:
            #         if 'labels' not in s.name or 'ignore' in s.name:
            #             if not found:
            #                 # Starting from the middle, if we haven't found a hand-annotated yet,
            #                 # we are meant to use this automatically segmented file
            #                 annotation_type.append(AUTO)
            #             else:
            #                 # If we have already found a hand-annotated closer to the center,
            #                 # this region was uncertain and we don't want to supervise this region's segmentation
            #                 annotation_type.append(UNCERTAIN)
            #         else:
            #             # This is a hand-annotated file
            #             found = True
            #             annotation_type.append(HAND_ANNOTATED)
            #     return annotation_type[::-1] if process_backwards else annotation_type
            #
            # midway_sa_idx = 3 + len(seg_files[3:]) // 2
            seg_type_categories = [*[HAND_ANNOTATED if 'labels' in p.name else AUTO for p in seg_files[:3]],
                                   *[AUTO]*len(seg_files[3:])]
                                   # *categorize_seg_files(seg_files[3:midway_sa_idx], process_backwards=True),
                                   # *categorize_seg_files(seg_files[midway_sa_idx:])]
            seg_categories.append(seg_type_categories)
            seg_files = [str(x) for x in seg_files]
            segs.append(seg_files)
            segs_auto.append(seg_files_auto)
            interp_segs.append(seg_interp_files)
            images.append(slices)

        assert len(segs) >= max_num
        print(f"Found {len(images)} subjects.")

        subject_data_paths = self.preprocess_subject_data(images, segs, segs_auto, max_num, interp_segs, seg_categories)
        assert len(subject_data_paths) == max_num,  f"{len(subject_data_paths)} != {max_num}"
        return subject_data_paths

    def preprocess_subject_data(self, subj_paths, seg_paths, seg_paths_auto, max_num, interp_seg_paths=None, seg_type_categories=None, debug=False):
        if self.replace_existing_processed:
            print("Replacing existing preprocessed files.")
        store_path = Path(self.store_path)
        prepr_data_paths = []
        for subj_idx, (subj_slices, subj_seg_slices) in tqdm(list(enumerate(zip(subj_paths, seg_paths))), desc="Preprocessing subject data into torch tensor."):
            if len(prepr_data_paths) == max_num:
                break
            # If file already exists, add path to list and continue
            subject_id = Path([i for i in subj_slices if Path(i).parent.name == "sa_slices"][0]).parent.parent.name
            save_path = store_path / subject_id / PREPR_FILE_NAME
            if save_path.exists() and not self.replace_existing_processed:
                prepr_data_paths.append(str(save_path))
                continue
            try:

                # Otherwise, preprocess subject
                images = []
                segs = []
                gt_available_masks = []
                affines = []
                spacings = []
                # Iterate backwards so that all SA segmentations are loaded by the time we tackle LA
                for idx, slice_path in enumerate(subj_slices):
                    # Load image
                    nib_subj = nib.load(slice_path)
                    img = nib_subj.get_fdata().squeeze()
                    img = torch.from_numpy(normalize_image_with_percentile(img)).to(torch.float32)
                    images.append(img)
                    # Decompose affines into its rotation and translation params
                    aff = torch.tensor(nib_subj.affine, dtype=torch.float32)
                    affines.append(aff)
                    spacing = torch.tensor(nib_subj.header.get_zooms()[:3], dtype=aff.dtype)
                    spacings.append(spacing)
                    # Try to see if segmentations exist
                    seg_found = True
                    try:
                        nib_subj_seg = nib.load(subj_seg_slices[idx])
                        seg = nib_subj_seg.get_fdata().squeeze().astype(np.uint8)
                        seg = torch.tensor(seg, dtype=torch.uint8)
                        segs.append(seg)
                        # If we have a GT LA segmentation, we don't need a training mask
                    except FileNotFoundError:
                        # If we don't have GT segmentations, we set seg as zeros.
                        seg = torch.zeros(img.shape, dtype=torch.uint8)
                        segs.append(seg)
                        seg_found = False
                    # We try to load an interpolated segmentation from SA slices to create a training mask.
                    if seg_found and seg_type_categories[subj_idx][idx] == AUTO:
                        gt_available_mask = torch.ones_like(img, dtype=torch.bool)
                    else:
                        if idx < 3:
                            try:
                                # If interpolated SA seg is not found, we set it to None and will be interpolated below.
                                nib_subj_interp_seg = nib.load(interp_seg_paths[subj_idx][idx])
                                interp_seg = nib_subj_interp_seg.get_fdata().squeeze().astype(np.uint8)
                            except FileNotFoundError:
                                # If we couldn't load the interpolated seg earlier,
                                # we create it and save it so we don't repeat the slow interp process next time
                                segs_auto = [nib.load(p).get_fdata().squeeze().astype(np.uint8) for p in seg_paths_auto[subj_idx][3:]]
                                segs_auto_affs = [nib.load(p).affine for p in seg_paths_auto[subj_idx][3:]]
                                interp_seg, int_interp = interpolate_sa_segs_to_la(segs_auto,
                                                                                   segs_auto,
                                                                                   segs_auto_affs,
                                                                                   target_shape=(
                                                                                       img.shape[0], img.shape[1],
                                                                                       img.shape[-1]),
                                                                                   target_aff=aff.numpy(),
                                                                                   frames=None,
                                                                                   oob_dist_thresh=10.)
                                array_to_nifti(str(interp_seg_paths[subj_idx][idx]), interp_seg[:, :, None], nib_subj.affine)
                            fg_seg = interp_seg > 0
                            fg_dil_seg = np.moveaxis(binary_dilation(np.moveaxis(fg_seg, -1, 0), iterations=10), 0,  -1)
                            gt_bg_seg = ~fg_dil_seg
                            gt_bg_seg = torch.from_numpy(gt_bg_seg)
                        else:
                            segs_auto = [nib.load(p).get_fdata().squeeze().astype(np.uint8) for p in seg_paths_auto[subj_idx][3:]]
                            segs_auto = torch.stack([torch.tensor(s > 0, dtype=torch.bool) for s in segs_auto], dim=0)
                            max_mask = segs_auto.any(0)
                            max_mask = np.moveaxis(binary_dilation(np.moveaxis(max_mask.numpy(), -1, 0), iterations=5), 0,  -1)
                            gt_bg_seg = ~max_mask
                            gt_bg_seg = torch.from_numpy(gt_bg_seg)
                        gt_available_mask = gt_bg_seg
                        if seg_type_categories[subj_idx][idx] == HAND_ANNOTATED:
                            seg_frames = (0, 16, 32)
                            gt_available_mask[..., seg_frames] = True
                    gt_available_masks.append(gt_available_mask)

                if self.crop_around_heart:
                    # Crop images and update affine matrices with new origins
                    affines, segs, images, gt_available_masks = \
                        crop_around_heart(affines, segs, images, gt_available_masks, debug=False)
                # Normalize orientation of planes and store the 6 aff params
                try:
                    affines = normalize_slice_orientation(affines, segs, debug=False)
                except ValueError as e:
                    print(subject_id, e)
                    continue

                contours = extract_2dt_contours(segs, min_area=8, resolution_factor=5)

                flippings = []
                aff_params = []
                coord_max = []
                coord_min = []
                for aff, img, spac in zip(affines, images, spacings):
                    needs_flip = torch.linalg.det(aff[:3, :3]) <= 0
                    flippings.append(needs_flip)
                    aff_p = mat_to_params(aff[None], spac[None], needs_flip[None]).squeeze(0)
                    aff_params.append(aff_p)

                    # Find min and max coordinates of slice
                    (x, y) = img.shape[:2]
                    meshgrid = np.meshgrid(np.arange(0, x), np.arange(0, y), np.array([0]), indexing="ij")
                    coordinates_arr_mesh = np.stack(meshgrid, axis=-1).reshape(-1, 3)
                    w_coordinates = nib.affines.apply_affine(aff.numpy(), coordinates_arr_mesh)
                    # world coordinates
                    coordinates_arr = torch.tensor(w_coordinates, dtype=torch.float32)
                    coord_max.append(torch.amax(coordinates_arr, dim=0))
                    coord_min.append(torch.amin(coordinates_arr, dim=0))

                # Convert images to uin8
                images = [(i * 255).round().to(torch.uint8) for i in images]

                dim_max = np.array([i.shape for i in images]).max(0)
                # Place all subject slices into one combined slice stack (slices, height_max, width_max, time)
                im_pad = torch.zeros((len(images), *dim_max), dtype=torch.uint8)
                im_pad_mask = torch.zeros_like(im_pad, dtype=torch.bool)
                for i, im in enumerate(images):
                    im_pad[i, :im.shape[0], :im.shape[1]] = im.squeeze()
                    im_pad_mask[i, :im.shape[0], :im.shape[1]] = True
                # non_padding_indices = make_masked_coordinate_tensor(im_pad_mask)
                seg_pad = torch.zeros((len(images), *dim_max), dtype=torch.uint8)
                for i, seg in enumerate(segs):
                    seg_pad[i, :seg.shape[0], :seg.shape[1]] = seg.squeeze()
                gt_available_pad = torch.zeros((len(gt_available_masks), *dim_max), dtype=torch.bool)
                for i, a in enumerate(gt_available_masks):
                    gt_available_pad[i, :a.shape[0], :a.shape[1]] = a.squeeze()

                # Get max and min coordinates across subject's slices
                subj_coord_max = torch.concatenate((torch.amax(torch.stack(coord_max, dim=0), dim=0), torch.tensor([50])))
                subj_coord_min = torch.concatenate((torch.amin(torch.stack(coord_min, dim=0), dim=0), torch.tensor([0])))

                # Stack aff_params, spacings and flippings into padded array such that all subjects have equal shapes
                aff_params = torch.stack(aff_params, dim=0)
                aff_params_padded = torch.zeros((len(images), *aff_params.shape[1:]), dtype=aff_params.dtype)
                aff_params_padded[:aff_params.shape[0]] = aff_params

                spacings = torch.stack(spacings, dim=0)
                spacings_padded = torch.zeros((len(images), *spacings.shape[1:]), dtype=spacings.dtype)
                spacings_padded[:spacings.shape[0]] = spacings

                needs_flip = torch.stack(flippings, dim=0)
                needs_flip_padded = torch.zeros((len(images), *needs_flip.shape[1:]), dtype=needs_flip.dtype)
                needs_flip_padded[:needs_flip.shape[0]] = needs_flip

                # Store preprocessed arrays to disk
                pkl_path = store_path / subject_id / "prep_data.pkl"
                if pkl_path.exists():
                    os.remove(str(pkl_path))
                if save_path.parent.exists() and self.replace_existing_processed:
                    shutil.rmtree(str(save_path.parent))
                save_path.parent.parent.mkdir(exist_ok=True)
                save_path.parent.mkdir(exist_ok=True)
                while not save_path.exists() or save_path.stat().st_size < 100:
                    with h5py.File(save_path, 'w') as f:
                        f.create_dataset('image_padded', data=im_pad.moveaxis(-1, 0).numpy(), dtype=np.uint8, compression=1)  # Saving volume as (time, slices, H, W) for faster frame lazy loading
                        f.create_dataset('image_padded_mask', data=im_pad_mask.moveaxis(-1, 0).numpy(), compression=1)# Saving volume as (time, ch, slices, H, W) for faster frame lazy loading
                        f.create_dataset('seg_padded', data=seg_pad.moveaxis(-1, 0).numpy(), dtype=np.uint8, compression=1)
                        f.create_dataset('gt_available_padded', data=gt_available_pad.moveaxis(-1, 0).numpy(), compression=1)
                        f.create_dataset('coord_max', data=subj_coord_max.numpy(), compression=1)
                        f.create_dataset('coord_min', data=subj_coord_min.numpy(), compression=1)
                        f.create_dataset('aff_params_padded', data=aff_params_padded.numpy(), compression=1)
                        f.create_dataset('spacings_padded', data=spacings_padded.numpy(), compression=1)
                        f.create_dataset('flippings_padded', data=needs_flip_padded.numpy(), compression=1)
                        # Contours have structure slice_idx -> frame_idx -> class_dict -> contour_idx -> (N, 2) array
                        g = f.create_group('contours')
                        for slice_idx, slice_contours in enumerate(contours):
                            h = g.create_group(f'{slice_idx:02d}')
                            for frame_idx, frame_contours in enumerate(slice_contours):
                                k = h.create_group(f'{frame_idx:02d}')
                                for label_idx, label_contours in frame_contours.items():
                                    l = k.create_group(f'{int(label_idx):02d}')
                                    for c_idx, c in enumerate(label_contours):
                                        l.create_dataset(f'{c_idx}', data=c, compression=1, dtype=float)

                prepr_data_paths.append(str(save_path))
            except AssertionError as e:
                print(f'Encountered {e}: ',  traceback.print_exc())
                continue
        return prepr_data_paths
