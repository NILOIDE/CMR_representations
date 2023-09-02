import time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
import pickle
import os
import h5py
from tqdm import tqdm

from utils import normalize_image_with_percentile, mat_to_params, make_masked_coordinate_tensor


class CMRDataModule(pl.LightningDataModule):
    def __init__(self, load_la_dir: str = r"D:\UKBB_subjects", load_sa_dir: str = r"D:\UKBB_subjects_unaligned",
                 batch_size: int = 32, num_coords: int = 4000, num_workers: int = 0):
        super().__init__()
        self.load_la_dir = load_la_dir
        self.load_sa_dir = load_sa_dir
        self.batch_size = batch_size
        self.num_coords = num_coords
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
        self.num_train = 100
        self.num_val = 10
        self.num_test = 10
        self.max_slices = -1
        self.num_workers = num_workers

    def setup(self, stage: str, pickle_name=None):
        num_subjects = self.num_train + self.num_val + self.num_test
        pickle_name = f"dataset_paths_{num_subjects}.pkl"
        try:
            with open(pickle_name, 'rb') as handle:
                subject_data, self.max_slices = pickle.load(handle)
        except FileNotFoundError:
            subject_data, self.max_slices = self.find_subjects(max_num=num_subjects)
            with open(pickle_name, 'wb') as handle:
                pickle.dump([subject_data, self.max_slices], handle, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(subject_data) == num_subjects

        split = (self.num_train / num_subjects, self.num_val / num_subjects, self.num_test / num_subjects)
        train_idxs, val_idxs, test_idxs = [list(s) for s in random_split(list(range(len(subject_data))), split)]

        self.train_dset = CardiacUKBB([subject_data[i] for i in train_idxs],
                                      num_coords=self.num_coords)
        self.val_dset = CardiacUKBB([subject_data[i] for i in val_idxs],
                                    num_coords=self.num_coords)
        self.test_dset = CardiacUKBB([subject_data[i] for i in test_idxs],
                                     num_coords=self.num_coords)

        self._train_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        self._val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        self._test_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def get_coord_size(self) -> int:
        return self.train_dset.coord_size

    def get_max_slices(self) -> int:
        return self.max_slices

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def find_subjects(self, max_num=100, **kwargs):
        count = 0
        max_slices = 0
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
            if len(la_files) != 3 or not sa_files:
                continue

            # Segmentations
            seg_sa_files = sorted(list(Path(os.path.join(self.load_sa_dir, parent, "sa_slices")).rglob('seg_sa*.nii.gz')))
            seg_sa_files = [str(x) for x in seg_sa_files]
            if not seg_sa_files:
                continue

            slices = la_files + sa_files
            images.append(slices)
            segs.append(seg_sa_files)
            if len(slices) > max_slices:
                max_slices = len(slices)
            count += 1
        assert count == max_num
        print(f"Found {len(images)} subjects.")

        subject_data_paths = self.preprocess_subject_data(images, segs, max_slices)

        return subject_data_paths, max_slices

    def preprocess_subject_data(self, subj_paths, seg_paths, max_slices,
                                store_path=r"D:\UKBB_subjects_unaligned", replace_existing=True):
        store_path = Path(store_path)
        prepr_data_paths = []
        for subj_slices in tqdm(subj_paths, desc="Preprocessing subject data into torch tensor."):
            coord_max = []
            coord_min = []
            images = []
            segs = []
            aff_params = []
            spacings = []
            flippings = []
            for idx, slice_path in enumerate(subj_slices):
                nib_subj = nib.load(slice_path)
                img = nib_subj.get_fdata()
                img = torch.from_numpy(normalize_image_with_percentile(img)).to(torch.float32)
                images.append(img)
                # Find min and max coordinates of slice
                aff = nib_subj.affine
                (x, y) = img.shape[:2]
                meshgrid = np.meshgrid(np.arange(0, x), np.arange(0, y), np.array([0]), indexing="ij")
                coordinates_arr_mesh = np.stack(meshgrid, axis=-1).reshape(-1, 3)
                w_coordinates = nib.affines.apply_affine(aff, coordinates_arr_mesh)
                # world coordinates
                coordinates_arr = torch.tensor(w_coordinates, dtype=torch.float32)
                coord_max.append(torch.amax(coordinates_arr, dim=0))
                coord_min.append(torch.amin(coordinates_arr, dim=0))

                # Decompose affines into its rotation and translation params
                aff = torch.tensor(aff, dtype=torch.float32)
                spacing = torch.tensor(nib_subj.header.get_zooms()[:3], dtype=aff.dtype)
                needs_flip = torch.linalg.det(aff[:3, :3]) <= 0
                params = mat_to_params(aff[None], spacing[None], needs_flip[None])
                aff_params.append(params[0])
                spacings.append(spacing)
                flippings.append(needs_flip)

            # Place all subject slices into one combined volume (slices, height_max, width_max)
            dim_max = torch.amax(torch.tensor([i.shape[:2] for i in images]), dim=0)
            im_pad = torch.zeros((len(images), *dim_max, images[-1].shape[-1]))
            im_pad_mask = torch.zeros_like(im_pad, dtype=torch.bool)
            for i, im in enumerate(images):
                im_pad[i, :im.shape[0], :im.shape[1]] = im.squeeze(2)
                im_pad_mask[i, :im.shape[0], :im.shape[1]] = True
            non_padding_indices = make_masked_coordinate_tensor(im_pad_mask)

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
            subject_id = Path([i for i in subj_slices if Path(i).parent.name == "sa_slices"][0]).parent.parent.name
            save_path = store_path / subject_id / "prep_data.pkl"
            if save_path.exists():
                os.remove(str(save_path))
            save_path = store_path / subject_id / "prep_data.h5"
            if save_path.exists() and replace_existing:
                os.remove(str(save_path))
            while not save_path.exists() or save_path.stat().st_size < 100:
                with h5py.File(save_path, 'w') as f:
                    f.create_dataset('image_padded', data=im_pad.moveaxis(-1, 0).numpy(), compression=1)  # Saving volume as (time, slices, H, W) for faster frame lazy loading
                    f.create_dataset('image_padded_mask', data=im_pad_mask.moveaxis(-1, 0).numpy(), compression=1)
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
        assert subject_data_paths
        self.data_paths = subject_data_paths
        self.num_coords = num_coords
        coords, values, *_ = self.__getitem__(0)
        self.coord_size = coords.shape[-1]

    def __len__(self):
        return len(self.data_paths)

    def load_subject_data(self, subj_idx: int, frame_idx: Optional[int] = None, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load image and segmentation files and undersample them according to hold-out rates.
        :param subj_idx: Index of subject in dataset list.
        """
        if frame_idx is None:
            selected_frame = np.random.randint(0, 50)
        else:
            selected_frame = frame_idx

        with h5py.File(self.data_paths[subj_idx], 'r') as f:
            # Load only the randomly selected image frame from the (time, slices, H, W) volume
            image = torch.tensor(f['image_padded'][selected_frame], dtype=torch.float32)
            # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
            image_mask = torch.tensor(f['image_padded_mask'][selected_frame], dtype=torch.bool)
            # Get available non-padding indices in frame
            non_padding_indices = make_masked_coordinate_tensor(image_mask)
            # Sample num_coords amount of indices that our batch will consist of
            indices_sample = torch.randint(0, non_padding_indices.shape[0], (self.num_coords,))
            indices = non_padding_indices[indices_sample]
            # Add the time index to get the full volume index
            full_indices = torch.cat((indices, torch.full((indices.shape[0], 1), selected_frame)), dim=1)
            # Get image values at the indices samples
            image_values_sample = image[tuple(indices.T)]
            # Load in the max/min coord values of volume (used for coord normalization)
            coord_max = torch.tensor(f['coord_max'][:], dtype=torch.float32)
            coord_min = torch.tensor(f['coord_min'][:], dtype=torch.float32)
            # Load in affine-related data
            aff_params_padded = torch.tensor(f['aff_params_padded'][:], dtype=torch.float32)
            spacings_padded = torch.tensor(f['spacings_padded'][:], dtype=torch.float32)
            flippings_padded = torch.tensor(f['flippings_padded'][:], dtype=torch.bool)
        return image_values_sample, full_indices, coord_max, coord_min, \
            aff_params_padded, spacings_padded, flippings_padded

    def __getitem__(self, idx: int):
        # Load image and seg data
        img_values, indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded = self.load_subject_data(idx)

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = np.concatenate((indices[:, 1:3], np.zeros_like(indices[:, :1]), indices[:, -1:]), axis=1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return voxel_indices, img_values, aff_params_padded, spacings_padded, needs_flip_padded, \
            sub_idx, slice_indices, min_coords, max_coords


if __name__ == '__main__':
    a = CMRDataModule()