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
from model_conv import INR_Conv


@dataclass
class Params:
    """" Default params """
    # Epochs -------------------------------------------------------------------
    max_epochs: int = 1_000_000
    logging_disabled: bool = False
    logging_wandb_disabled: bool = False
    replace_existing_preprocessed: bool = False
    logging_rate: int = 2_000
    logging_start_rate: int = 20_000
    addit_log_epochs: Tuple[int, ...] = ()
    num_train: int = 100
    num_val: int = 0
    num_test: int = 6
    num_workers: int = 8
    batch_size: int = 4
    num_coords: int = 70_000
    # Point spread function ------------------------------------------------------------
    point_spread_start_epoch: int = 20_000
    point_spread_size_before: int = 1
    point_spread_size_after: int = 16
    num_coords_during_point_spread: int = 35000
    point_spread_std_before: Tuple[float, float, float, float] = (0.01, 0.01, 0.01, 0.01)#(0.3, 0.3, 0.3, 0.3)
    point_spread_std_after: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.4)
    # Model -------------------------------------------------------------------
    num_hidden_layers: int = 16
    hidden_size: int = 256
    latent_size: int = 128
    int_scale_range: float = 0.3  # applied via: int_scaled = int * (1 + tanh(x)*(scale_range/2))
    # Conv latent prediction -------------------------------------------------------------------
    use_conv: bool = False
    conv_channels: Tuple[int, ...] = (32,64,64,128,128)
    # Regularization -------------------------------------------------------------------
    weight_reg_inr: float = 1e-5
    weight_reg_aff: float = 1e-4
    weight_reg_lat: float = 1e-1
    weight_reg_int_scale: float = 1e-2
    weight_loss_deriv: float = 0e0
    # Segmentation ----------------------------------------------------------------
    weight_loss_seg: float = 1e0
    weight_seg_class: Tuple[float, float, float, float] = (1,2,4,3)  # Will be normalized
    # Registration ---------------------------------------------------------------
    regist_task_start_epoch: int = 999000
    regist_weights_std: float = 1e-3
    weight_loss_regist_recon: float = 1e-1
    weight_loss_regist_seg: float = 1e-1
    weight_loss_regist_jac_reg: float = 1e-1
    weight_loss_regist_mag_reg: float = 1e-1
    # Learning rates -------------------------------------------------------------------
    learning_rate: float = 1e-4
    learning_rate_aff: float = 1e-4
    learning_rate_def: float = 1e-4
    # Positional encoder -------------------------------------------------------------------
    pe_num_frequencies: Tuple[int, int, int, int, int] = (7,7,7,5,5)
    pe_anneal_max_iter: int = 100_000
    pe_anneal_start_prop: float = 0.2
    pe_freq_scale: float = 1.0
    # Paths
    job_name: str = ''
    data_dir: str = r"/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/unaligned_subjects"
    preprocessed_h5_dir: str = r"/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/unaligned_h5_crop_la"
    trained_models_dir: str = "/u/home/stol/Documents/Projects/CMR_intensity_alignment/trained_models"
    resume_checkpoint_path: str = ""
    # resume_checkpoint_path: str = "/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251125-034718-psf20k_100subj-20ann/checkpoints/epoch-epoch=029999.ckpt"
    # Inference ----------
    inference: bool = False
    inference_path: str = "/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251125-034718-psf20k_100subj-20ann/checkpoints/epoch-epoch=029999.ckpt"
    inf_max_epochs: int = 2500
    inf_num_coords: int = 35_000
    inf_learning_rate: float = 1e-3
    inf_learning_rate_aff: float = 1e-3
    inf_learning_rate_def: float = 1e-3
    inf_point_spread_start_epoch: int = 9999


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Pass arguments using the command line like:
    # python main.py --conv_channels 32 64 128 --no-use_conv
    # For bools such as 'use_conv' passing --use_conv will make it True, passing --no-use_conv will make it False
    params = tyro.cli(Params)
    print(params)

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
                                full_seq_dataset=params.use_conv,
                                batch_size=params.batch_size,
                                num_coords=params.num_coords,
                                inf_num_coords=params.inf_num_coords,
                                num_workers=params.num_workers)
    data_module.prepare_data()
    os.environ['WANDB_DISABLED'] = str(params.logging_disabled)
    logger = WandbLogger(project="CMR-Align")
    logger.log_hyperparams(params.__dict__)
    print('Params', params)
    print('Model path:', model_path)

    checkpoint_path = model_path / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=str(checkpoint_path),
                                          filename='epoch-{epoch:06d}',
                                          every_n_epochs=params.logging_rate//2,
                                          save_top_k=-1)
    log_path = model_path / 'logs'
    log_path.mkdir(exist_ok=True)
    if params.use_conv:
        model = INR_Conv(coord_size=data_module.get_coord_size(), num_subjects=data_module.num_train,
                         max_slices=data_module.get_max_slices(), regist_cache_dims=data_module.get_max_slice_shape(),
                         log_path=log_path, **params.__dict__)
    else:
        model = INR_AutoReg(coord_size=data_module.get_coord_size(), num_subjects=data_module.num_train,
                            max_slices=data_module.get_max_slices(), regist_cache_dims=data_module.get_max_slice_shape(),
                            log_path=log_path, **params.__dict__)
    if not params.inference:
        trainer = Trainer(
            logger=logger,
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            devices=1,
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
        # prefix = "canonical_inr."
        # # Filter + rename keys
        # filtered = {k[len(prefix):]: v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)}
        # model.canonical_inr.load_state_dict(filtered, strict=True)
        model.canonical_inr = model.canonical_inr.to('cuda')
        model.target_net = deepcopy(model.canonical_inr)
        model.regist_inr = model.regist_inr.to('cuda')
        model.do_testing(data_module.test_dset, )


if __name__ == '__main__':
    main()
