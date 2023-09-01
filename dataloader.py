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

from tqdm import tqdm

from utils import normalize_image_with_percentile, mat_to_params


class CMRDataModule(pl.LightningDataModule):
    def __init__(self, load_la_dir: str = r"D:\UKBB_subjects", load_sa_dir: str = r"D:\UKBB_subjects_unaligned",
                 batch_size: int = 32, num_coords: int = 4000):
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
        self.num_workers = 0#min(self.batch_size, 8)
        # self.setup("fit")

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

        self._train_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self._val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self._test_dataloader = DataLoader(self.train_dset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

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

    @staticmethod
    def make_masked_coordinate_tensor(mask):
        """Make a coordinate tensor."""
        coordinate_tensor = [torch.arange(0, i) for i in mask.shape]
        coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
        coordinate_tensor = torch.stack(coordinate_tensor, dim=len(mask.shape))
        coordinate_tensor = coordinate_tensor.reshape([np.prod(mask.shape), len(mask.shape)])
        coordinate_tensor = coordinate_tensor[mask.flatten(), :]
        return coordinate_tensor

    def preprocess_subject_data(self, subj_paths, seg_paths, max_slices, store_path=r"D:\UKBB_subjects_unaligned"):
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
            non_padding_indices = self.make_masked_coordinate_tensor(im_pad_mask)

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
            with open(str(save_path), 'wb') as handle:
                pickle.dump([im_pad, non_padding_indices, subj_coord_max, subj_coord_min,
                             aff_params_padded, spacings_padded, needs_flip_padded],
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
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

    def load_subject_data(self, subj_idx: int, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load image and segmentation files and undersample them according to hold-out rates.
        :param subj_idx: Index of subject in dataset list.
        """
        with open(self.data_paths[subj_idx], 'rb') as handle:
            subject_data = pickle.load(handle)
        im_pad, non_padding_indices, subj_coord_max, subj_coord_min, \
            aff_params_padded, spacings_padded, needs_flip_padded = subject_data
        return im_pad, non_padding_indices, subj_coord_max, subj_coord_min, \
            aff_params_padded, spacings_padded, needs_flip_padded

    def __getitem__(self, idx: int):
        # Load image and seg data
        im_pad_mask, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded = self.load_subject_data(idx)
        # Sample random points in the image volume
        indices_sample = np.random.randint(0, non_padding_indices.shape[0], self.num_coords)
        indices = non_padding_indices[indices_sample]
        img_values = im_pad_mask[tuple(indices.T)]
        voxel_indices = indices[:, 1:]
        voxel_indices = np.concatenate((voxel_indices[:, :2], np.zeros_like(voxel_indices[:, :1]), voxel_indices[:, -1:]), axis=1)
        slice_indices = indices[:, :1]

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return voxel_indices, img_values, aff_params_padded, spacings_padded, needs_flip_padded, \
            sub_idx, slice_indices, min_coords, max_coords


if __name__ == '__main__':
    a = CMRDataModule()