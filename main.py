import copy
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import tyro
import torch

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dataloader import CMRDataModule
from model_autoreg import INR_AutoReg


@dataclass
class Params:
    """" Default params """
    # Epochs -------------------------------------------------------------------
    max_epochs: int = 1_000_000
    logging_disabled: bool = False
    logging_wandb_disabled: bool = False
    replace_existing_preprocessed: bool = False
    logging_rate: int = 2_000
    logging_start_rate: int = 10_000
    addit_log_epochs: Tuple[int, ...] = (1000, 2000, 5000)
    num_train: int = 100
    num_val: int = 2
    num_test: int = 1
    num_workers: int = 4
    batch_size: int = 4

    num_coords_voxel: int = 40_000
    num_coords_surface: int = 10_000
    # Point spread function ------------------------------------------------------------
    point_spread_start_epoch: int = 10000
    point_spread_size_before: int = 1
    point_spread_size_after: int = 16
    num_coords_during_point_spread: int = 10000
    point_spread_std_before: Tuple[float, float, float, float] = (0.01, 0.01, 0.01, 0.01)#(0.3, 0.3, 0.3, 0.3)
    point_spread_std_after: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)
    # Model -------------------------------------------------------------------
    num_hidden_layers: int = 16
    hidden_size: int = 512
    latent_size: int = 16
    int_scale_range: float = 0.3  # applied via: int_scaled = int * (1 + tanh(x)*(scale_range/2))
    spatial_functa_resolution: int = 4  # If 1, a single global vec is used. If >1, latent size is split between the 4 dims (n^4)
    # Regularization -------------------------------------------------------------------
    weight_reg_inr: float = 1e-5
    weight_reg_aff: float = 1e-4
    weight_reg_lat: float = 1e-1
    weight_reg_int_scale: float = 1e-2
    # Segmentation ----------------------------------------------------------------
    weight_loss_seg: float = 1e2
    weight_loss_deriv: float = 1e0
    weight_seg_class: Tuple[float, float, float] = (4,4,3)  # Will be normalized
    # Learning rates -------------------------------------------------------------------
    learning_rate: float = 1e-4
    learning_rate_aff: float = 1e-4
    learning_rate_def: float = 1e-4
    # Positional encoder -------------------------------------------------------------------
    pe_num_frequencies: Tuple[int, int, int, int, int] = (8,8,8,5,5)
    pe_anneal_max_iter: int = 200_000
    pe_anneal_start_prop: float = 0.2
    pe_freq_scale: float = 1.0
    # Paths
    job_name: str = ''
    data_dir: str = r"/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/unaligned_subjects"
    preprocessed_h5_dir: str = r"/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/unaligned_h5_crop_rotate"
    trained_models_dir: str = "/u/home/stol/Documents/Projects/CMR_intensity_alignment/trained_models"
    resume_checkpoint_path: str = ""
    # resume_checkpoint_path: str = "/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251125-034718-psf20k_100subj-20ann/checkpoints/epoch-epoch=029999.ckpt"
    # Inference ----------
    inference: bool = False
    inference_path: str = "/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251125-034718-psf20k_100subj-20ann/checkpoints/epoch-epoch=024999.ckpt"
    inf_max_epochs: int = 2000
    inf_num_coords: int = 35_000
    inf_learning_rate_inr: float = 1e-5
    inf_learning_rate_latent: float = 1e-3
    inf_learning_rate_aff: float = 1e-3
    inf_learning_rate_def: float = 1e-3
    inf_point_spread_start_epoch: int = 2500
    inf_weight_loss_seg: float = 0e0


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Pass arguments using the command line like:
    # python main.py --conv_channels 32 64 128 --no-use_conv
    # For bools such as 'use_conv' passing --use_conv will make it True, passing --no-use_conv will make it False
    params = tyro.cli(Params)
    print(params)

    if params.inference and 'inference' not in params.job_name:
        params.job_name = f'inference_{params.job_name}' if params.job_name else 'inference'
    model_path_parent = Path(params.trained_models_dir)
    model_path_parent.mkdir(exist_ok=True)
    model_path = model_path_parent / (f'{datetime.now().strftime("%Y%m%d-%H%M%S")}' + (f'-{params.job_name}' if params.job_name else ""))
    model_path.mkdir(exist_ok=True)
    with open(str(model_path / "params.json"), "w") as f:
        json.dump(params.__dict__, f, indent=4)
    print('Model path:', model_path)

    data_module = CMRDataModule(load_la_dir=params.data_dir,
                                load_sa_dir=params.data_dir,
                                preprocessed_store_path=params.preprocessed_h5_dir,
                                replace_existing_preprocessed=params.replace_existing_preprocessed,
                                log_path=model_path,
                                num_train=params.num_train,
                                num_val=params.num_val,
                                num_test=params.num_test,
                                batch_size=params.batch_size,
                                num_coords_voxel=params.num_coords_voxel,
                                num_coords_surface=params.num_coords_surface,
                                inf_num_coords=params.inf_num_coords,
                                num_workers=params.num_workers,
                                cache_data=params.cache_data)
    data_module.prepare_data()

    checkpoint_path = model_path / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=str(checkpoint_path),
                                          filename='epoch-{epoch:06d}',
                                          every_n_epochs=params.logging_rate//2,
                                          save_top_k=-1)
    log_path = model_path / 'logs'
    log_path.mkdir(exist_ok=True)
    model = INR_AutoReg(coord_size=data_module.get_coord_size(), num_subjects=data_module.num_train,
                        max_slices=data_module.get_max_slices(), regist_cache_dims=data_module.get_max_slice_shape(),
                        log_path=log_path, **params.__dict__)

    os.environ['WANDB_DISABLED'] = str(params.logging_wandb_disabled)
    logger = WandbLogger(project="CMR-Align")
    logger.log_hyperparams(params.__dict__)
    print('Params', params)
    print('Model path:', model_path)

    if not params.inference:
        trainer = Trainer(
            logger=logger,
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            max_epochs=params.point_spread_start_epoch,
            fast_dev_run=False,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            num_sanity_val_steps=1,
        )
        ckpt_path = params.resume_checkpoint_path if params.resume_checkpoint_path else None
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        # Then continue with updated datasets ready for point-spread
        trainer.datamodule.train_dset.num_coords = params.num_coords_during_point_spread
        trainer.fit_loop.max_epochs = params.max_epochs
        trainer.fit(model, datamodule=data_module)
        # First train up until point-spread start epochs
    else:
        ckpt = torch.load(params.inference_path)
        model.load_state_dict(ckpt['state_dict'])
        model.canonical_inr = model.canonical_inr.to('cuda')
        model.target_net = deepcopy(model.canonical_inr)
        model.regist_inr = model.regist_inr.to('cuda')
        model.do_testing(data_module.test_dset, )


if __name__ == '__main__':
    main()
