import shutil
import time
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

from data_utils import array_to_nifti
from geo_utils import normalize_slice_orientation
from sa_la_interp import interpolate_sa_segs_to_la
from utils import normalize_image_with_percentile, mat_to_params, make_masked_coordinate_tensor, \
    compute_3d_image_gradients, to_1hot, to_gif, nlm_denoise_multi_parallel


class CMRDataModule(pl.LightningDataModule):
    def __init__(self,
                 load_la_dir: str,
                 load_sa_dir: str,
                 preprocessed_store_path,
                 replace_existing_processed=True,
                 batch_size: int = 32,
                 num_coords: int = 4000,
                 num_workers: int = 0):
        super().__init__()
        self.load_la_dir = load_la_dir
        self.load_sa_dir = load_sa_dir
        self.store_path = preprocessed_store_path
        self.replace_existing_processed = replace_existing_processed
        self.batch_size = batch_size
        self.num_coords = num_coords
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
        self.num_train = 16
        self.num_val = 1
        self.num_test = 1
        self.max_slices = -1
        self.num_workers = num_workers
        self.subject_data = []

    def prepare_data(self) -> None:
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
        self.subject_data = subject_data
        split = (self.num_train / num_subjects, self.num_val / num_subjects, self.num_test / num_subjects)
        train_idxs, val_idxs, test_idxs = [list(s) for s in random_split(list(range(len(self.subject_data))), split)]
        self.train_dset = CardiacUKBB([self.subject_data[i] for i in train_idxs][:],
                                      num_coords=self.num_coords)
        self.val_dset = CardiacUKBB([self.subject_data[i] for i in val_idxs],
                                    num_coords=self.num_coords)
        self.test_dset = CardiacUKBB([self.subject_data[i] for i in test_idxs],
                                     num_coords=self.num_coords)

    def setup(self, stage: str):
        self._train_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=self.num_workers, pin_memory=True,
                                            persistent_workers=self.num_workers > 0)
        self._val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size,
                                          num_workers=self.num_workers, pin_memory=True,
                                          persistent_workers=self.num_workers > 0)
        self._test_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True,
                                           persistent_workers=self.num_workers > 0)

    def get_coord_size(self) -> int:
        return self.train_dset.coord_size

    def get_max_slices(self) -> int:
        dset = self.train_dset if self.train_dset else self.test_dset
        max_slices = -1
        for p in dset.data_paths:
            with h5py.File(p, 'r') as f:
                shape = f['image_padded'].shape
                if shape[1] > max_slices:
                    max_slices = shape[1]
        return max_slices

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def find_subjects(self, max_num=100, **kwargs):
        count = 0
        max_slices = 17
        max_shape = np.array([0,0,0,0])
        images = []
        segs = []
        subjects = list(sorted(os.listdir(str(self.load_la_dir))))
        for i, parent in enumerate(subjects):
            if parent == "1013493":
                continue
            if count == max_num:
                break
            # Images
            la_files = sorted(list(Path(os.path.join(self.load_la_dir, parent)).rglob('la*.nii.gz')))
            la_files = [str(x) for x in la_files]

            sa_files = sorted(list(Path(os.path.join(self.load_sa_dir, parent, "sa_slices")).rglob('sa*.nii.gz')))
            sa_files = [str(x) for x in sa_files]
            if len(la_files) != 3 or len(sa_files) < 5:
                continue

            # Segmentations. LA have hopefully been interpolated before, else None (and will be interpolated).
            # LA segs are used for training masking.
            seg_la_files = [Path(os.path.join(self.load_la_dir, parent)) / 'interp_seg_la_2ch.nii.gz',
                            Path(os.path.join(self.load_la_dir, parent)) / 'interp_seg_la_3ch.nii.gz',
                            Path(os.path.join(self.load_la_dir, parent)) / 'interp_seg_la_4ch.nii.gz']
            seg_la_files = [str(x) for x in seg_la_files]
            seg_sa_files = sorted(list(Path(os.path.join(self.load_sa_dir, parent, "sa_slices")).rglob('seg_sa*.nii.gz')))
            seg_sa_files = [str(x) for x in seg_sa_files]
            if not seg_sa_files:
                continue

            slices = la_files + sa_files
            images.append(slices)
            seg_slices = seg_la_files + seg_sa_files
            segs.append(seg_slices)
            max_slices = max(len(slices), max_slices)
            shape = np.array([nib.load(i).shape for i in slices]).max(0)
            max_shape = np.maximum(shape, max_shape)
            count += 1
        assert count == max_num
        print(f"Found {len(images)} subjects.")

        max_shape = (*max_shape[:2], max_shape[-1])
        subject_data_paths = self.preprocess_subject_data(images, segs, max_slices, max_shape,
                                                          replace_existing=self.replace_existing_processed)

        return subject_data_paths

    def preprocess_subject_data(self, subj_paths, seg_paths, max_slices, dim_max, replace_existing=False, debug=False):
        if replace_existing:
            print("Replacing existing preprocessed files.")
        store_path = Path(self.store_path)
        prepr_data_paths = []
        for subj_slices, subj_seg_slices in tqdm(list(zip(subj_paths, seg_paths)), desc="Preprocessing subject data into torch tensor."):
            # If file already exists, add path to list and continue
            subject_id = Path([i for i in subj_slices if Path(i).parent.name == "sa_slices"][0]).parent.parent.name
            save_path = store_path / subject_id / "prep_data.h5"
            if save_path.exists() and not replace_existing:
                prepr_data_paths.append(str(save_path))
                continue

            # Otherwise, preprocess subject
            images = []
            image_ds = []
            image_dds = []
            segs = []
            la_segs = []
            affines = []
            spacings = []
            # Iterate backwards so that all SA segmentations are loaded by the time we tackle LA
            for idx, slice_path in enumerate(subj_slices):
                # Load image
                nib_subj = nib.load(slice_path)
                img = nib_subj.get_fdata().squeeze()
                img = torch.from_numpy(normalize_image_with_percentile(img)).to(torch.float32)
                # img = nlm_denoise_multi_parallel(img, 27, 1, template_window_size=3,search_window_size=7, num_processes=16)
                img_uint = (img * 255).round().to(torch.uint8)
                images.append(img_uint)
                # Compute image gradients and Hessian
                img_d = compute_3d_image_gradients(img[None, None])[0]
                img_dd_x = compute_3d_image_gradients(img_d[None,0:1])[0]
                img_dd_y = compute_3d_image_gradients(img_d[None,1:2])[0]
                img_dd_t = compute_3d_image_gradients(img_d[None,2:3])[0]
                img_dd = torch.cat((img_dd_x[0:3], img_dd_y[1:3], img_dd_t[2:3]), 0)  # (dxx, dxy, dxz, dyy, dyz, dzz)
                img_d = img_d.moveaxis(0, -1)
                img_dd = img_dd.moveaxis(0, -1)
                image_ds.append(img_d)
                image_dds.append(img_dd)
                if debug:
                    a = [torch.from_numpy(normalize_image_with_percentile(nib_subj.get_fdata().squeeze())).to(torch.float32),
                         img,
                         nlm_denoise_multi_parallel(torch.from_numpy(normalize_image_with_percentile(nib_subj.get_fdata().squeeze())).to(torch.float32), 49, 1, template_window_size=3,search_window_size=15, num_processes=16),
                         nlm_denoise_multi_parallel(torch.from_numpy(normalize_image_with_percentile(nib_subj.get_fdata().squeeze())).to(torch.float32), 49, 1, template_window_size=4,search_window_size=7, num_processes=16),
                         nlm_denoise_multi_parallel(torch.from_numpy(normalize_image_with_percentile(nib_subj.get_fdata().squeeze())).to(torch.float32), 49, 1, template_window_size=4,search_window_size=15, num_processes=16),
                         nlm_denoise_multi_parallel(torch.from_numpy(normalize_image_with_percentile(nib_subj.get_fdata().squeeze())).to(torch.float32), 49, 1, template_window_size=5,search_window_size=7, num_processes=16),
                         nlm_denoise_multi_parallel(torch.from_numpy(normalize_image_with_percentile(nib_subj.get_fdata().squeeze())).to(torch.float32), 49, 1, template_window_size=5,search_window_size=15, num_processes=16),
                         ]
                    b = [compute_3d_image_gradients(i[None, None])[0, -1] for i in a]
                    c = [compute_3d_image_gradients(i[None, None])[0, -1] for i in b]
                    to_gif(torch.cat((torch.cat(a, 0), torch.cat([i*2 for i in b], 0).abs(), torch.cat([i*2 for i in c], 0).abs()), 1), str(idx))

                # Decompose affines into its rotation and translation params
                aff = torch.tensor(nib_subj.affine, dtype=torch.float32)
                affines.append(aff)
                spacing = torch.tensor(nib_subj.header.get_zooms()[:3], dtype=aff.dtype)
                spacings.append(spacing)
                # Get segmentation for SA (when idx>=3)
                if idx >= 3:
                    nib_subj_seg = nib.load(subj_seg_slices[idx])
                    seg = nib_subj_seg.get_fdata().squeeze()
                    seg = torch.from_numpy(seg).to(torch.uint8)
                else:
                    # For LA, we don't have GT segmentations. We set seg as zeros.
                    # We try to load an interpolated segmentation from SA slices to create a training mask.
                    try:
                        nib_subj_seg = nib.load(subj_seg_slices[idx])
                        seg = nib_subj_seg.get_fdata().squeeze().astype(np.uint8)
                        la_segs.append(seg)
                    except FileNotFoundError:
                        # If interpolated SA seg is not found, we set it to None and will be interpolated below.
                        la_segs.append(None)
                    seg = torch.zeros(img.shape, dtype=torch.uint8)
                segs.append(seg)

            # Normalize orientation of planes and store the 6 aff params
            affines = normalize_slice_orientation(affines, segs)
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

            # Because we don't have LA seg, we want to label which points are far enough
            # from foreground to confidently supervise as background
            la_gt_available_masks = []
            for idx, (img, aff) in enumerate(zip(images[:3], affines[:3])):
                if la_segs[idx] is None:
                    # If we couldn't load the interpolated seg earlier,
                    # we create it and save it so we don't repeat the slow interp process next time
                    la_seg_interp, _ = interpolate_sa_segs_to_la([i.numpy() for i in segs[3:]],
                                                                 [i.numpy() for i in images[3:]],
                                                                 [i.numpy() for i in affines[3:]],
                                                                 target_shape=(img.shape[0], img.shape[1], img.shape[-1]),
                                                                 target_aff=aff.numpy(),
                                                                 frames=None,
                                                                 oob_dist_thresh=10.)
                    array_to_nifti(str(subj_seg_slices[idx]), la_seg_interp[:, :, None], aff.numpy())
                else:
                    la_seg_interp = la_segs[idx]
                la_fg = la_seg_interp > 0
                la_fg_dil = np.moveaxis(binary_dilation(np.moveaxis(la_fg, -1, 0), iterations=5), 0, -1)
                la_gt_bg = ~la_fg_dil
                la_gt_bg = torch.from_numpy(la_gt_bg)
                la_gt_available_masks.append(la_gt_bg)

            # Place all subject slices into one combined slice stack (slices, height_max, width_max, time)
            im_pad = torch.zeros((max_slices, *dim_max), dtype=torch.uint8)
            im_pad_mask = torch.zeros_like(im_pad, dtype=torch.bool)
            for i, im in enumerate(images):
                im_pad[i, :im.shape[0], :im.shape[1]] = im.squeeze()
                im_pad_mask[i, :im.shape[0], :im.shape[1]] = True
            # non_padding_indices = make_masked_coordinate_tensor(im_pad_mask)
            img_d_pad = torch.zeros((max_slices, *dim_max, 3), dtype=torch.float32)
            img_dd_pad = torch.zeros((max_slices, *dim_max, 6), dtype=torch.float32)
            for i, (d, dd) in enumerate(zip(image_ds, image_dds)):
                img_d_pad[i, :d.shape[0], :d.shape[1]] = d.squeeze()
                img_dd_pad[i, :dd.shape[0], :dd.shape[1]] = dd.squeeze()
            seg_pad = torch.zeros((max_slices, *dim_max), dtype=torch.uint8)
            for i, seg in enumerate(segs):
                seg_pad[i, :seg.shape[0], :seg.shape[1]] = seg.squeeze()
            gt_available_pad = torch.zeros((len(la_gt_available_masks), *dim_max), dtype=torch.bool)
            for i, a in enumerate(la_gt_available_masks):
                gt_available_pad[i, :a.shape[0], :a.shape[1]] = a.squeeze()

            # Get max and min coordinates across subject's slices
            subj_coord_max = torch.concatenate((torch.amax(torch.stack(coord_max, dim=0), dim=0), torch.tensor([50])))
            subj_coord_min = torch.concatenate((torch.amin(torch.stack(coord_min, dim=0), dim=0), torch.tensor([0])))

            # Stack aff_params, spacings and flippings into padded array such that all subjects have equal shapes
            aff_params = torch.stack(aff_params, dim=0)
            aff_params_padded = torch.zeros((max_slices, *aff_params.shape[1:]), dtype=aff_params.dtype)
            aff_params_padded[:aff_params.shape[0]] = aff_params

            spacings = torch.stack(spacings, dim=0)
            spacings_padded = torch.zeros((max_slices, *spacings.shape[1:]), dtype=spacings.dtype)
            spacings_padded[:spacings.shape[0]] = spacings

            needs_flip = torch.stack(flippings, dim=0)
            needs_flip_padded = torch.zeros((max_slices, *needs_flip.shape[1:]), dtype=needs_flip.dtype)
            needs_flip_padded[:needs_flip.shape[0]] = needs_flip

            # Store preprocessed arrays to disk
            pkl_path = store_path / subject_id / "prep_data.pkl"
            if pkl_path.exists():
                os.remove(str(pkl_path))
            if save_path.parent.exists() and replace_existing:
                shutil.rmtree(str(save_path.parent))
            save_path.parent.parent.mkdir(exist_ok=True)
            save_path.parent.mkdir(exist_ok=True)
            while not save_path.exists() or save_path.stat().st_size < 100:
                with h5py.File(save_path, 'w') as f:
                    f['max_slices'] = max_slices
                    f.create_dataset('image_padded', data=im_pad.moveaxis(-1, 0).numpy(), dtype=np.uint8, compression=1)  # Saving volume as (time, slices, H, W) for faster frame lazy loading
                    f.create_dataset('image_padded_mask', data=im_pad_mask.moveaxis(-1, 0).numpy(), compression=1)
                    f.create_dataset('image_d_padded', data=img_d_pad.moveaxis(-1, 0).moveaxis(-1, 0).numpy(), dtype=np.float32, compression=1)  # Saving volume as (time, ch, slices, H, W) for faster frame lazy loading
                    f.create_dataset('image_dd_padded', data=img_dd_pad.moveaxis(-1, 0).moveaxis(-1, 0).numpy(), dtype=np.float32, compression=1)  # Saving volume as (time, ch, slices, H, W) for faster frame lazy loading
                    f.create_dataset('seg_padded', data=seg_pad.moveaxis(-1, 0).numpy(), dtype=np.uint8, compression=1)
                    f.create_dataset('gt_available_padded', data=gt_available_pad.moveaxis(-1, 0).numpy(), compression=1)
                    f.create_dataset('coord_max', data=subj_coord_max.numpy(), compression=1)
                    f.create_dataset('coord_min', data=subj_coord_min.numpy(), compression=1)
                    f.create_dataset('aff_params_padded', data=aff_params_padded.numpy(), compression=1)
                    f.create_dataset('spacings_padded', data=spacings_padded.numpy(), compression=1)
                    f.create_dataset('flippings_padded', data=needs_flip_padded.numpy(), compression=1)
            prepr_data_paths.append(str(save_path))
        return prepr_data_paths


class CardiacUKBB(Dataset):
    LUT_NAME = "cardiac_mri"

    def __init__(self, subject_data_paths, num_coords=4000, **kwargs):
        super().__init__()
        assert subject_data_paths
        self.data_paths = subject_data_paths
        self.num_coords = num_coords
        coords, values, *_ = self.__getitem__(0)
        self.coord_size = coords.shape[-1]

    def __len__(self):
        return len(self.data_paths)

    def load_subject_data(self, subj_idx: int, frame_idx: Optional[int] = None, **kwargs) \
            -> Tuple[torch.Tensor, ...]:
        """Load image and segmentation files and undersample them according to hold-out rates.
        :param subj_idx: Index of subject in dataset list.
        """
        if frame_idx is None:
            selected_frame = np.random.randint(0, 50)
        else:
            selected_frame = frame_idx

        with h5py.File(self.data_paths[subj_idx], 'r') as f:
            max_slices = f['max_slices'][()]
            # Load only the randomly selected image frame from the (time, slices, H, W) volume
            ims = torch.tensor(f['image_padded'][:, :max_slices], dtype=torch.float32) / 255.
            image = ims[selected_frame]
            image_dt = image
            image_ddt = image
            ims_pad = torch.zeros((max_slices, *image.shape))
            ims_pad[:max_slices] = ims.moveaxis(0, -1)
            # image_dt = torch.tensor(f['image_d_padded'][selected_frame, 2:3, :max_slices], dtype=torch.float32).moveaxis(0, -1)
            # image_ddt = torch.tensor(f['image_dd_padded'][selected_frame, 5:6, :max_slices], dtype=torch.float32).moveaxis(0, -1)
            # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
            image_mask = torch.tensor(f['image_padded_mask'][selected_frame, :max_slices], dtype=torch.bool)
            seg = torch.tensor(f['seg_padded'][selected_frame, :max_slices], dtype=torch.uint8)
            la_gt_available = torch.ones(image.shape, dtype=torch.bool)
            la_gt_available[:3] = torch.tensor(f['gt_available_padded'][selected_frame], dtype=torch.bool)
            # Get available non-padding indices in frame
            non_padding_indices = make_masked_coordinate_tensor(image_mask)
            # Add the time index to get the full volume index
            # full indices (slice, x, y, t)
            full_indices = torch.cat((non_padding_indices, torch.full((non_padding_indices.shape[0], 1), selected_frame)), dim=1)

            # Load in the max/min coord values of volume (used for coord normalization)
            coord_max = torch.tensor(f['coord_max'][:], dtype=torch.float32)
            coord_min = torch.tensor(f['coord_min'][:], dtype=torch.float32)
            # Load in affine-related data
            aff_params_padded = torch.tensor(f['aff_params_padded'][:], dtype=torch.float32).squeeze(-2)
            spacings_padded = torch.tensor(f['spacings_padded'][:], dtype=torch.float32)
            flippings_padded = torch.tensor(f['flippings_padded'][:], dtype=torch.bool)
        return ims_pad, max_slices, image, image_dt, image_ddt, seg, la_gt_available, full_indices, coord_min, coord_max, \
            aff_params_padded, spacings_padded, flippings_padded

    def __getitem__(self, idx: int):
        return self.generate_item(idx)

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        # Load image and seg data
        ims, max_slices, img, img_dt, img_ddt, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded = self.load_subject_data(idx, frame)

        if num_coords is None:
            num_coords = self.num_coords
        elif isinstance(num_coords, float):
            num_coords = num_coords * torch.prod(img.shape)
        else:
            pass  # num_coord is already an int
        # Sample num_coords amount of indices that our batch will consist of
        indices_sample = torch.randint(0, non_padding_indices.shape[0], (num_coords,))
        # indices (slice, x, y)
        indices = non_padding_indices[indices_sample]

        # Get image values at the indices samples
        image_values_sample = img[tuple(indices.T[:-1])]
        image_dt_values_sample = img_dt[tuple(indices.T[:-1])]
        image_ddt_values_sample = img_ddt[tuple(indices.T[:-1])]
        seg_sample = seg[tuple(indices.T[:-1])]
        seg_sample = to_1hot(seg_sample, num_class=4)
        gt_avail_sample = gt_avail[tuple(indices.T[:-1])]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = np.concatenate((indices[:, 1:-1], np.zeros_like(indices[:, :1]), indices[:, -1:]), axis=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (ims, max_slices, voxel_indices, image_values_sample, image_dt_values_sample, image_ddt_values_sample,
                seg_sample, gt_avail_sample, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords)


class CardiacUKBBValidation(CardiacUKBB):
    LUT_NAME = "cardiac_mri_val"

    def __init__(self, subject_data_paths, num_coords=4000, **kwargs):
        super().__init__(subject_data_paths, num_coords=4000)

    def __getitem__(self, idx: int):
        # Load image and seg data
        img_values, indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded = self.load_subject_data(idx)

if __name__ == '__main__':
    a = CMRDataModule()