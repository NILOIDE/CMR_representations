from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
from utils import make_masked_coordinate_tensor, to_1hot, make_coordinate_tensor


class CardiacUKBB(Dataset):
    def __init__(self, subject_data_paths, max_slices, max_slice_shape, num_coords=4000, **kwargs):
        super().__init__()
        assert subject_data_paths
        self.data_paths = subject_data_paths
        self.num_coords = num_coords
        self.max_slices = max_slices
        self.max_slice_shape = max_slice_shape
        self.coord_size = None

    def get_coord_size(self):
        if self.coord_size is None:
            coords, values, *_ = self.__getitem__(0)
            self.coord_size = coords.shape[-1]
        return self.coord_size

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
            # Load only the randomly selected image frame from the (time, slices, H, W) volume
            image = torch.tensor(f['image_padded'][selected_frame], dtype=torch.float32) / 255.
            S, H, W = image.shape
            image_dt = torch.tensor(f['image_d_padded'][selected_frame, 2:3], dtype=torch.float32).moveaxis(0, -1)
            # image_ddt = torch.tensor(f['image_dd_padded'][selected_frame, 5:6], dtype=torch.float32).moveaxis(0, -1)
            # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
            image_mask = torch.tensor(f['image_padded_mask'][selected_frame], dtype=torch.bool)
            seg = torch.tensor(f['seg_padded'][selected_frame], dtype=torch.uint8)
            la_gt_available = torch.ones_like(seg, dtype=torch.bool)
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
            aff_params_padded = torch.zeros((self.max_slices, 6), dtype=torch.float32)
            aff_params_padded[:S] = torch.tensor(f['aff_params_padded'][:], dtype=torch.float32).squeeze(-2)
            spacings_padded = torch.zeros((self.max_slices, 3), dtype=torch.float32)
            spacings_padded[:S] = torch.tensor(f['spacings_padded'][:], dtype=torch.float32)
            flippings_padded = torch.zeros((self.max_slices,), dtype=torch.bool)
            flippings_padded[:S] = torch.tensor(f['flippings_padded'][:], dtype=torch.bool)
        return image, image_dt, seg, la_gt_available, full_indices, coord_min, coord_max, \
            aff_params_padded, spacings_padded, flippings_padded, torch.tensor((S,), dtype=torch.long)

    def __getitem__(self, idx: int):
        return self.generate_item(idx)

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        # Load image and seg data
        img, img_dt, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded, num_subj_slices = self.load_subject_data(idx, frame)

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
        # image_ddt_values_sample = img_ddt[tuple(indices.T[:-1])]
        seg_sample = seg[tuple(indices.T[:-1])]
        seg_sample = to_1hot(seg_sample, num_class=4)
        gt_avail_sample = gt_avail[tuple(indices.T[:-1])]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = torch.concatenate((indices[:, 1:-1], torch.zeros_like(indices[:, :1]), indices[:, -1:]), dim=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (voxel_indices, image_values_sample, image_dt_values_sample,
                seg_sample, gt_avail_sample, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords, num_subj_slices)


class CardiacUKBBValidation(CardiacUKBB):
    def __init__(self, subj_paths, max_slices, max_slice_shape, num_coords=4000, to_gpu=False, **kwargs):
        super().__init__(subj_paths, max_slices, max_slice_shape, num_coords, **kwargs)
        self.num_subjs = len(subj_paths)
        self.num_coords = num_coords
        device = "cuda" if to_gpu else "cpu"
        self.image_pad = torch.zeros((self.num_subjs, self.max_slices, *self.max_slice_shape), dtype=torch.float32, device=device)
        self.image_dt = torch.zeros(self.image_pad.shape, dtype=torch.float32, device=device).unsqueeze(-1)
        self.image_mask = torch.zeros(self.image_pad.shape, dtype=torch.bool, device=device)
        self.seg = torch.zeros(self.image_pad.shape, dtype=torch.uint8, device=device)
        self.la_gt_available = torch.ones(self.image_pad.shape, dtype=torch.bool, device=device)
        self.non_padding_indices = [None]*self.num_subjs
        self.coord_max = torch.zeros((self.num_subjs, 4), dtype=torch.float32, device=device)
        self.coord_min = torch.zeros((self.num_subjs, 4), dtype=torch.float32, device=device)
        self.aff_params_padded = torch.zeros((self.num_subjs, self.max_slices, 6), dtype=torch.float32, device=device)
        self.spacings_padded = torch.zeros((self.num_subjs, self.max_slices, 3), dtype=torch.float32, device=device)
        self.flippings_padded = torch.zeros((self.num_subjs, self.max_slices,), dtype=torch.bool, device=device)
        self.num_subj_slices = torch.zeros((self.num_subjs,), dtype=torch.uint8, device=device)
        self.cache_data()
        if to_gpu:
            self.non_padding_indices = [i.cuda() for i in self.non_padding_indices]

    def cache_data(self):
        for i, path in enumerate(self.data_paths):
            with h5py.File(path, 'r') as f:
                # Load only the randomly selected image frame from the (time, slices, H, W) volume
                ims = torch.tensor(f['image_padded'][:], dtype=torch.float32) / 255.
                T, S, H, W = ims.shape
                self.num_subj_slices[i] = S
                self.image_pad[i, :S, :H, :W, :T] = ims.moveaxis(0, -1)
                self.image_dt[i, :S, :H, :W, :T] = torch.tensor(f['image_d_padded'][:, 2:3], dtype=torch.float32).moveaxis(0, -1).moveaxis(0, -1)
                # image_ddt = torch.tensor(f['image_dd_padded'][selected_frame, 5:6], dtype=torch.float32).moveaxis(0, -1)
                # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
                self.image_mask[i, :S, :H, :W, :T] = torch.tensor(f['image_padded_mask'][:], dtype=torch.bool).moveaxis(0, -1)
                self.seg[i, :S, :H, :W, :T] = torch.tensor(f['seg_padded'][:], dtype=torch.uint8).moveaxis(0, -1)
                self.la_gt_available[i, :3, :H, :W, :T] = torch.tensor(f['gt_available_padded'][:], dtype=torch.bool).moveaxis(0, -1)
                self.image_pad = self.image_pad[:,:,:H,:W]
                self.image_dt = self.image_dt[:,:,:H,:W]
                self.image_mask = self.image_mask[:,:,:H,:W]
                self.seg = self.seg[:,:,:H,:W]
                self.la_gt_available = self.la_gt_available[:,:,:H,:W]
                # Get available non-padding indices in frame
                non_padding_indices = make_masked_coordinate_tensor(self.image_mask[i])
                # Add the time index to get the full volume index
                # full indices (slice, x, y, t)
                self.non_padding_indices[i] = non_padding_indices

                # Load in the max/min coord values of volume (used for coord normalization)
                self.coord_max[i] = torch.tensor(f['coord_max'][:], dtype=torch.float32)
                self.coord_min[i] = torch.tensor(f['coord_min'][:], dtype=torch.float32)
                # Load in affine-related data
                self.aff_params_padded[i, :S] = torch.tensor(f['aff_params_padded'][:], dtype=torch.float32).squeeze(-2)
                self.spacings_padded[i, :S] = torch.tensor(f['spacings_padded'][:], dtype=torch.float32)
                self.flippings_padded[i, :S] = torch.tensor(f['flippings_padded'][:], dtype=torch.bool)

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        idx=0
        # Load image and seg data
        (img, img_dt, seg, gt_avail,
         non_padding_indices, min_coords, max_coords, num_subj_slices,
         aff_params_padded, spacings_padded, needs_flip_padded) = (
            self.image_pad[idx], self.image_dt[idx], self.seg[idx], self.la_gt_available[idx],
            self.non_padding_indices[idx], self.coord_min[idx], self.coord_max[idx], self.num_subj_slices[idx],
            self.aff_params_padded[idx], self.spacings_padded[idx], self.flippings_padded[idx])

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
        image_values_sample = img[tuple(indices.T)]
        image_dt_values_sample = img_dt[tuple(indices.T)]
        # image_ddt_values_sample = img_ddt[tuple(indices.T)]
        seg_sample = seg[tuple(indices.T)]
        seg_sample = to_1hot(seg_sample, num_class=4)
        gt_avail_sample = gt_avail[tuple(indices.T)]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = torch.concatenate((indices[:, 1:-1], torch.zeros_like(indices[:, :1]), indices[:, -1:]), dim=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)

        subj_idx = torch.tensor(idx, dtype=torch.long)
        return (voxel_indices, image_values_sample, image_dt_values_sample,
                seg_sample, gt_avail_sample, aff_params_padded, spacings_padded, needs_flip_padded,
                subj_idx, slice_indices, min_coords, max_coords, num_subj_slices)


class CardiacUKBBValidationFullImage(CardiacUKBB):

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        if frame is None:
            frame = np.random.randint(0, 50)
        # Load image and seg data
        ims, img, img_dt, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded, num_subj_slices = self.load_subject_data(idx, frame)

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = make_coordinate_tensor(img.shape, img.device)
        slice_indices = voxel_indices[..., :1]
        voxel_indices = torch.cat((voxel_indices[..., 1:], torch.zeros_like(voxel_indices[..., :1]), torch.full_like(voxel_indices[..., :1], frame)), -1)
        # voxel_indices = np.concatenate((indices[:, 1:-1], np.zeros_like(indices[:, :1]), indices[:, -1:]), axis=-1)
        # slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)
        seg = to_1hot(seg.reshape(-1, ), num_class=4).reshape(*seg.shape, 4)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (ims, num_subj_slices, voxel_indices, img, img_dt,
                seg, gt_avail, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords)



class CardiacUKBBFullImage(CardiacUKBB):
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
            # Load only the randomly selected image frame from the (time, slices, H, W) volume
            ims = torch.tensor(f['image_padded'][:], dtype=torch.float32) / 255.
            T, S, H, W = ims.shape
            ims_pad = torch.zeros((self.max_slices, *self.max_slice_shape), dtype=ims.dtype)
            ims_pad[:S, :H, :W, :T] = ims.moveaxis(0, -1)
            image = ims_pad[...,selected_frame]
            image_dt = torch.zeros(image.shape, dtype=ims.dtype).unsqueeze(-1)
            image_dt[:S, :H, :W] = torch.tensor(f['image_d_padded'][selected_frame, 2:3], dtype=torch.float32).moveaxis(0, -1)
            # image_ddt = torch.tensor(f['image_dd_padded'][selected_frame, 5:6], dtype=torch.float32).moveaxis(0, -1)
            # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
            image_mask = torch.zeros(image.shape, dtype=torch.bool)
            image_mask[:S, :H, :W] = torch.tensor(f['image_padded_mask'][selected_frame], dtype=torch.bool)
            seg = torch.zeros(image.shape, dtype=torch.uint8)
            seg[:S, :H, :W] = torch.tensor(f['seg_padded'][selected_frame], dtype=torch.uint8)
            la_gt_available = torch.ones(image.shape, dtype=torch.bool)
            la_gt_available[:3, :H, :W] = torch.tensor(f['gt_available_padded'][selected_frame], dtype=torch.bool)
            # Get available non-padding indices in frame
            non_padding_indices = make_masked_coordinate_tensor(image_mask)
            # Add the time index to get the full volume index
            # full indices (slice, x, y, t)
            full_indices = torch.cat((non_padding_indices, torch.full((non_padding_indices.shape[0], 1), selected_frame)), dim=1)

            # Load in the max/min coord values of volume (used for coord normalization)
            coord_max = torch.tensor(f['coord_max'][:], dtype=torch.float32)
            coord_min = torch.tensor(f['coord_min'][:], dtype=torch.float32)
            # Load in affine-related data
            aff_params_padded = torch.zeros((self.max_slices, 6), dtype=torch.float32)
            aff_params_padded[:S] = torch.tensor(f['aff_params_padded'][:], dtype=torch.float32).squeeze(-2)
            spacings_padded = torch.zeros((self.max_slices, 3), dtype=torch.float32)
            spacings_padded[:S] = torch.tensor(f['spacings_padded'][:], dtype=torch.float32)
            flippings_padded = torch.zeros((self.max_slices,), dtype=torch.bool)
            flippings_padded[:S] = torch.tensor(f['flippings_padded'][:], dtype=torch.bool)
        return image, image_dt, seg, la_gt_available, full_indices, coord_min, coord_max, \
            aff_params_padded, spacings_padded, flippings_padded, torch.tensor(S, dtype=torch.long), ims_pad

    def __getitem__(self, idx: int):
        return self.generate_item(idx)

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        # Load image and seg data
        img, img_dt, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded, num_subj_slices, full_imgs \
            = self.load_subject_data(idx, frame)

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
        # image_ddt_values_sample = img_ddt[tuple(indices.T[:-1])]
        seg_sample = seg[tuple(indices.T[:-1])]
        seg_sample = to_1hot(seg_sample, num_class=4)
        gt_avail_sample = gt_avail[tuple(indices.T[:-1])]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = torch.concatenate((indices[:, 1:-1], torch.zeros_like(indices[:, :1]), indices[:, -1:]), dim=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (voxel_indices, image_values_sample, image_dt_values_sample,
                seg_sample, gt_avail_sample, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords, num_subj_slices, full_imgs)


class CardiacUKBBValidationFullImage(CardiacUKBBFullImage, CardiacUKBBValidation):
    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        if frame is None:
            frame = np.random.randint(0, 50)
        # Load image and seg data
        img, img_dt, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded, num_subj_slices, full_imgs \
            = self.load_subject_data(idx, frame)

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = make_coordinate_tensor(img.shape, img.device)
        slice_indices = voxel_indices[..., :1]
        voxel_indices = torch.cat((voxel_indices[..., 1:],
                                   torch.zeros_like(voxel_indices[..., :1]),
                                   torch.full_like(voxel_indices[..., :1], frame)), dim=-1)
        # voxel_indices = np.concatenate((indices[:, 1:-1], np.zeros_like(indices[:, :1]), indices[:, -1:]), axis=-1)
        # slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)
        seg = to_1hot(seg.reshape(-1, ), num_class=4).reshape(*seg.shape, 4)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (num_subj_slices, voxel_indices, img, img_dt,
                seg, gt_avail, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords, full_imgs)

