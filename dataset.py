from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset
import h5py
from utils import make_masked_coordinate_tensor, to_1hot, make_coordinate_tensor


class CardiacUKBB(Dataset):
    def __init__(self, subject_data_paths, max_slices, max_slice_shape,
                 num_coords_voxel=30000, num_coords_surface=10000, cache_data=True, cache_to_gpu=False, **kwargs):
        super().__init__()
        self.data_paths = subject_data_paths
        self.num_subjs = len(subject_data_paths)
        self.num_coords = num_coords_voxel
        self.num_coords_contour = num_coords_surface
        self.max_slices = max_slices
        self.max_slice_shape = max_slice_shape
        device = "cuda" if cache_to_gpu else "cpu"
        self.cache_data = cache_data
        if cache_data:
            H, W, T = self.max_slice_shape
            self.image_pad = torch.zeros((self.num_subjs, T, self.max_slices, H, W), dtype=torch.uint8, device=device)
            self.image_mask = torch.zeros(self.image_pad.shape, dtype=torch.bool, device=device)
            self.seg = torch.zeros(self.image_pad.shape, dtype=torch.uint8, device=device)
            self.gt_available = torch.ones(self.image_pad.shape, dtype=torch.bool, device=device)
            self.non_padding_indices = [None]*self.num_subjs
            self.coord_max = torch.zeros((self.num_subjs, 4), dtype=torch.float32, device=device)
            self.coord_min = torch.zeros((self.num_subjs, 4), dtype=torch.float32, device=device)
            self.aff_params_padded = torch.zeros((self.num_subjs, self.max_slices, 6), dtype=torch.float32, device=device)
            self.spacings_padded = torch.zeros((self.num_subjs, self.max_slices, 3), dtype=torch.float32, device=device)
            self.flippings_padded = torch.zeros((self.num_subjs, self.max_slices,), dtype=torch.bool, device=device)
            self.num_subj_slices = torch.zeros((self.num_subjs,), dtype=torch.uint8, device=device)
            self.contours = [[] for _ in range(self.num_subjs)]
            self.load_data_to_cache()
            if cache_to_gpu:
                self.non_padding_indices = [i.cuda() for i in self.non_padding_indices]
        self.coord_size = self.load_subject_data(0, 0)[4].shape[-1]


    def load_data_to_cache(self):
        for i, path in tqdm.tqdm(list(enumerate(self.data_paths)), desc="Loading data to cache"):
            with h5py.File(path, 'r') as f:
                # Load only the randomly selected image frame from the (time, slices, H, W) volume
                ims = torch.tensor(f['image_padded'][:], dtype=torch.uint8)
                T, S, H, W = ims.shape
                self.num_subj_slices[i] = S
                self.image_pad[i, :T, :S, :H, :W] = ims
                # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
                self.image_mask[i, :T, :S, :H, :W] = torch.tensor(f['image_padded_mask'][:], dtype=torch.bool)
                # self.image_mask[i, :T,3:] = False
                self.seg[i, :T, :S, :H, :W] = torch.tensor(f['seg_padded'][:], dtype=torch.uint8)
                self.gt_available[i, :T, :S, :H, :W] = torch.tensor(f['gt_available_padded'][:], dtype=torch.bool)
                # Get available non-padding indices in frame
                non_padding_indices = make_masked_coordinate_tensor(self.image_mask[i, 0])
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
                # Load contours for each class
                for t in range(T):
                    contour_per_class = {1: [[] for _ in range(S)],
                                         2: [[] for _ in range(S)],
                                         3: [[] for _ in range(S)]}
                    for class_idx in contour_per_class.keys():
                        for s in range(S):
                            # Stored as 'contours' -> 'slice' -> 'frame' -> 'class' -> 'coordinates'
                            d = f['contours'][f'{s:02d}'][f'{t:02d}'][f'{class_idx:02d}']
                            for c_idx in d.keys():
                                c = torch.tensor(d[c_idx][:], dtype=torch.float32)
                                # full indices (slice, x, y, t)
                                c = torch.cat((torch.full((c.shape[0], 1), s), c,
                                               torch.full((c.shape[0], 1), t)), dim=1)
                                contour_per_class[class_idx][s].append(c)
                    self.contours[i].append(contour_per_class)

    def __len__(self):
        return len(self.data_paths)

    def load_subject_data(self, subj_idx: int, frame_idx: Optional[int] = None, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, List[List[torch.Tensor]]]]:
        """Load image and segmentation files and undersample them according to hold-out rates.
        :param subj_idx: Index of subject in dataset list.
        """
        if frame_idx is None:
            selected_frame = np.random.randint(0, 50)
        else:
            selected_frame = frame_idx
        if self.cache_data:
            image = self.image_pad[subj_idx, selected_frame].float() / 255.
            seg = self.seg[subj_idx, selected_frame]
            gt_avail = self.gt_available[subj_idx, selected_frame]
            non_padding_indices = self.non_padding_indices[subj_idx]
            # full indices (slice, x, y, t)
            full_indices = torch.cat((non_padding_indices, torch.full_like(non_padding_indices[:, :1], selected_frame)),dim=1)
            coord_min = self.coord_min[subj_idx]
            coord_max = self.coord_max[subj_idx]
            num_subj_slices = self.num_subj_slices[subj_idx]
            aff_params_padded = self.aff_params_padded[subj_idx]
            spacings_padded = self.spacings_padded[subj_idx]
            needs_flip_padded = self.flippings_padded[subj_idx]
            contour_per_class = self.contours[subj_idx][selected_frame]
        else:
            with h5py.File(self.data_paths[subj_idx], 'r') as f:
                # Load only the randomly selected image frame from the (time, slices, H, W) volume
                image = torch.tensor(f['image_padded'][selected_frame], dtype=torch.float32) / 255.
                S, H, W = image.shape
                # Load only the randomly selected padding mask frame from the (time, slices, H, W) volume
                image_mask = torch.tensor(f['image_padded_mask'][selected_frame], dtype=torch.bool)
                seg = torch.zeros_like(image, dtype=torch.bool)
                seg[:S] = torch.tensor(f['seg_padded'][selected_frame], dtype=torch.uint8)
                gt_avail = torch.ones_like(seg, dtype=torch.bool)
                gt_avail[:S] = torch.tensor(f['gt_available_padded'][selected_frame], dtype=torch.bool)

                # Get available non-padding indices in frame
                non_padding_indices = make_masked_coordinate_tensor(image_mask)
                # Add the time index to get the full volume index
                # full indices (slice, x, y, t)
                full_indices = torch.cat((non_padding_indices, torch.full_like(non_padding_indices[:, :1], selected_frame)),dim=1)

                # Load in the max/min coord values of volume (used for coord normalization)
                coord_max = torch.tensor(f['coord_max'][:], dtype=torch.float32)
                coord_min = torch.tensor(f['coord_min'][:], dtype=torch.float32)
                # Load in affine-related data
                aff_params_padded = torch.zeros((self.max_slices, 6), dtype=torch.float32)
                aff_params_padded[:S] = torch.tensor(f['aff_params_padded'][:], dtype=torch.float32).squeeze(-2)
                spacings_padded = torch.zeros((self.max_slices, 3), dtype=torch.float32)
                spacings_padded[:S] = torch.tensor(f['spacings_padded'][:], dtype=torch.float32)
                needs_flip_padded = torch.zeros((self.max_slices,), dtype=torch.bool)
                needs_flip_padded[:S] = torch.tensor(f['flippings_padded'][:], dtype=torch.bool)
                num_subj_slices = torch.tensor((S,), dtype=torch.long)

                # Load contours for each class
                contour_per_class = {1: [[] for _ in range(S)],
                                     2: [[] for _ in range(S)],
                                     3: [[] for _ in range(S)]}
                for class_idx in contour_per_class.keys():
                    for s in range(S):
                        # Stored as 'contours' -> 'slice' -> 'frame' -> 'class' -> 'coordinates'
                        d = f['contours'][f'{s:02d}'][f'{selected_frame:02d}'][f'{class_idx:02d}']
                        for c_idx in d.keys():
                            c = torch.tensor(d[c_idx], dtype=torch.float32)
                            # full indices (slice, x, y, t)
                            c = torch.cat((torch.full((c.shape[0], 1), s), c,
                                           torch.full((c.shape[0], 1), selected_frame)), dim=1)
                            contour_per_class[class_idx][s].append(c)
        return (image,
                seg, gt_avail,
                full_indices, coord_min, coord_max,
                aff_params_padded, spacings_padded, needs_flip_padded, num_subj_slices, contour_per_class)

    def __getitem__(self, idx: int):
        return self.generate_item(idx)

    def sample_image_points(self, img, non_padding_indices, num_coords: Optional[Union[int, float]] = None,):
        if num_coords is None:
            num_coords = self.num_coords
        elif isinstance(num_coords, float):
            num_coords = int(num_coords * torch.prod(img.shape))
        else:
            pass  # num_coord is already an int
        # Sample num_coords amount of indices that our batch will consist of
        indices_sample = torch.randint(0, non_padding_indices.shape[0], (num_coords,))
        # indices (slice, x, y)
        indices = non_padding_indices[indices_sample]

        # Get image values at the indices samples
        image_values_sample = img[tuple(indices.T[:-1])]
        # image_ddt_values_sample = img_ddt[tuple(indices.T[:-1])]
        # seg_sample = seg[tuple(indices.T[:-1])]
        # seg_sample = to_1hot(seg_sample, num_class=4)
        # gt_avail_sample = gt_avail[tuple(indices.T[:-1])]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = torch.concatenate((indices[:, 1:-1], torch.zeros_like(indices[:, :1]), indices[:, -1:]), dim=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)
        return voxel_indices, image_values_sample, slice_indices

    def sample_contour_points(self, contours, num_coords: Optional[Union[int, float]] = None,):
        # Contours
        surface_points_per_class = []
        for k, frame_contours in contours.items():
            slice_surf_points = []
            for slice_idx, slice_contours in enumerate(frame_contours):
                if slice_contours:
                    merged_points = torch.cat(slice_contours, dim=0)
                else:
                    merged_points = torch.zeros((0, self.coord_size), dtype=torch.float32)
                slice_surf_points.append(merged_points)
            surface_points = torch.cat(slice_surf_points, dim=0)

            if num_coords is None:
                num_coords = self.num_coords_contour
            elif isinstance(num_coords, float):
                num_coords = int(num_coords * surface_points.shape[0] * 0.2)
            else:
                pass  # num_coord is already an int
            contour_sample = torch.randint(0, surface_points.shape[0], (num_coords,))
            surface_points_sample = surface_points[contour_sample]
            surface_points_per_class.append(surface_points_sample)
        surface_points_per_class = torch.stack(surface_points_per_class, dim=-2)
        surface_points_slice_idx = surface_points_per_class[..., :1].long()
        surface_points_slice_idx_ = surface_points_slice_idx.reshape(-1, 1)
        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        surface_coords_per_class = torch.cat((surface_points_per_class[..., 1:-1],
                                              torch.zeros_like(surface_points_per_class[..., :1]),
                                              surface_points_per_class[..., -1:]), dim=-1)
        surface_coords_per_class_ = surface_coords_per_class.reshape(-1, surface_coords_per_class.shape[-1])
        surface_points_class = torch.arange(0, len(contours.keys()))[None, :].tile(surface_points_per_class.shape[0], 1)
        surface_points_class_ = surface_points_class.reshape(-1, 1).long()
        return surface_coords_per_class_, surface_points_slice_idx_, surface_points_class_

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None,
                      num_coords_contour: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        # Load image and seg data
        (img,
         _, _,
         non_padding_indices, min_coords, max_coords, aff_params_padded, spacings_padded, needs_flip_padded,
         num_subj_slices, contours) = self.load_subject_data(idx, frame)
        voxel_indices, image_values_sample, slice_indices \
            = self.sample_image_points(img, non_padding_indices, num_coords)
        surface_coords_per_class_, surface_points_slice_idx_, surface_points_class_ = self.sample_contour_points(contours, num_coords)
        sub_idx = torch.tensor(idx, dtype=torch.long)

        return (voxel_indices, image_values_sample, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords, num_subj_slices,
                surface_coords_per_class_, surface_points_slice_idx_, surface_points_class_)


class CardiacUKBBValidation(CardiacUKBB):

    def sample_image_points(self, img, non_padding_indices, seg, gt_avail,
                            num_coords: Optional[Union[int, float]] = None,):
        if num_coords is None:
            num_coords = self.num_coords
        elif isinstance(num_coords, float):
            num_coords = int(num_coords * torch.prod(img.shape))
        else:
            pass  # num_coord is already an int
        # Sample num_coords amount of indices that our batch will consist of
        indices_sample = torch.randint(0, non_padding_indices.shape[0], (num_coords,))
        # indices (slice, x, y)
        indices = non_padding_indices[indices_sample]

        # Get image values at the indices samples
        image_values_sample = img[tuple(indices.T[:-1])]
        seg_sample = seg[tuple(indices.T[:-1])]
        seg_sample = to_1hot(seg_sample, num_class=4)
        gt_avail_sample = gt_avail[tuple(indices.T[:-1])]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = torch.concatenate((indices[:, 1:-1], torch.zeros_like(indices[:, :1]), indices[:, -1:]), dim=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)
        return voxel_indices, image_values_sample, slice_indices, seg_sample, gt_avail_sample

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None,
                      num_coords_contour: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        # Load image and seg data
        (img,
         seg, gt_avail,
         non_padding_indices, min_coords, max_coords, aff_params_padded, spacings_padded, needs_flip_padded,
         num_subj_slices, contours) = self.load_subject_data(idx, frame)
        voxel_indices, image_values_sample, slice_indices, seg_sample, gt_avail_sample \
            = self.sample_image_points(img, non_padding_indices, seg, gt_avail  , num_coords)
        surface_coords_per_class_, surface_points_slice_idx_, surface_points_class_ \
            = self.sample_contour_points(contours, num_coords)
        sub_idx = torch.tensor(idx, dtype=torch.long)

        return (voxel_indices, image_values_sample, seg_sample, gt_avail_sample, aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords, num_subj_slices,
                surface_coords_per_class_, surface_points_slice_idx_, surface_points_class_)

class CardiacUKBBValidationFullImage(CardiacUKBB):

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        if frame is None:
            frame = np.random.randint(0, 50)
        # Load image and seg data
        ims, img, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
            aff_params_padded, spacings_padded, needs_flip_padded, num_subj_slices = self.load_subject_data(idx, frame)

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = make_coordinate_tensor(img.shape, img.device)
        slice_indices = voxel_indices[..., :1]
        voxel_indices = torch.cat((voxel_indices[..., 1:], torch.zeros_like(voxel_indices[..., :1]), torch.full_like(voxel_indices[..., :1], frame)), -1)
        # voxel_indices = np.concatenate((indices[:, 1:-1], np.zeros_like(indices[:, :1]), indices[:, -1:]), axis=-1)
        # slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)
        seg = to_1hot(seg.reshape(-1, ), num_class=4).reshape(*seg.shape, 4)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (ims, num_subj_slices, voxel_indices, img, seg, gt_avail,
                aff_params_padded, spacings_padded, needs_flip_padded,
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
            image = ims_pad[..., selected_frame]
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
        return image, seg, la_gt_available, full_indices, coord_min, coord_max, \
            aff_params_padded, spacings_padded, flippings_padded, torch.tensor(S, dtype=torch.long), ims_pad

    def __getitem__(self, idx: int):
        return self.generate_item(idx)

    def generate_item(self, idx: int, num_coords: Optional[Union[int, float]] = None, frame: Optional[int] = None):
        # Load image and seg data
        img, seg, gt_avail, non_padding_indices, min_coords, max_coords, \
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
        seg_sample = seg[tuple(indices.T[:-1])]
        seg_sample = to_1hot(seg_sample, num_class=4)
        gt_avail_sample = gt_avail[tuple(indices.T[:-1])]

        # Create coordinates of point in the slice (x, y, z, t) where z == 0. Shape: (N, 4)
        voxel_indices = torch.concatenate((indices[:, 1:-1], torch.zeros_like(indices[:, :1]), indices[:, -1:]), dim=-1)
        slice_indices = indices[:, :1]  # Get which slice does each point belong to. Shape: (N, 1)

        sub_idx = torch.tensor(idx, dtype=torch.long)
        return (voxel_indices, image_values_sample, seg_sample, gt_avail_sample,
                aff_params_padded, spacings_padded, needs_flip_padded,
                sub_idx, slice_indices, min_coords, max_coords, num_subj_slices, full_imgs)


class CardiacUKBBValidationFullImage(CardiacUKBBFullImage):
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

