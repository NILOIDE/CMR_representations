import os
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil

ROOT_DIR = Path(r"D:\annotations\handmade")  # change this to your root directory
DATA_DIR = Path(
    r"D:\unaligned_subjects\vol\miltank\projects\ukbb\data\cardiac\slice_alignment\unaligned_subjects")  # change this to your root directory
subjs = list(DATA_DIR.iterdir())
subjs.sort()
for subj in subjs:
    subj_id = subj.name
    print(subj_id)
    hand_made_dir = ROOT_DIR / subj_id / "epoch_080000" / 'niftis'

    sa_slices = subj / "sa_slices"
    sample = str(list(sa_slices.iterdir())[0].name)

    conv_slices = []
    name = sa_slices.parent / "interp_seg_lv_la_2ch.nii.gz"
    hand_made_path = hand_made_dir / f'seg_conv_slice_{0:02d}.nii.gz'
    shutil.copy(name, str(hand_made_path))
    conv_slices.append(hand_made_path)
    name = sa_slices.parent / "interp_seg_lv_la_3ch.nii.gz"
    hand_made_path = hand_made_dir / f'seg_conv_slice_{1:02d}.nii.gz'
    shutil.copy(name, str(hand_made_path))
    conv_slices.append(hand_made_path)
    name = sa_slices.parent / "interp_seg_lv_la_4ch.nii.gz"
    hand_made_path = hand_made_dir / f'seg_conv_slice_{2:02d}.nii.gz'
    shutil.copy(name, str(hand_made_path))
    conv_slices.append(hand_made_path)

    for i in range(17):
        name = f"sa_{i:02d}{sample[-10:]}"
        name = sa_slices / name
        if not name.exists():
            break
        hand_made_path = hand_made_dir / f'seg_conv_slice_{i + 3:02d}.nii.gz'
        print(hand_made_path)
        shutil.copy(name, str(hand_made_path))
        conv_slices.append(hand_made_path)
        assert hand_made_path.exists()
    for i in conv_slices:
        a = nib.load(str(i))
        b_path = i.parent / ("seg_" + i.name[len("seg_conv_"):])
        b = nib.load(str(b_path))
        assert (a.affine != b.affine).sum() > 0
        print(i, b_path)
        new_img = nib.Nifti1Image(b.get_fdata(), a.affine, b.header)
        # Overwrite the original file
        nib.save(new_img, str(b_path))
        b_path = i.parent / 'original' / i.name[len("seg_conv_"):]
        b = nib.load(str(b_path))
        assert (a.affine != b.affine).sum() > 0
        print(i, b_path)
        new_img = nib.Nifti1Image(b.get_fdata(), a.affine, b.header)
        # Overwrite the original file
        nib.save(new_img, str(b_path))
    break
