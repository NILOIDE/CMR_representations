import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import tyro

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dataloader import CMRDataModule
from model_autoreg import INR_AutoReg
from model_conv import INR_Conv


@dataclass
class Params:
    # Epochs -------------------------------------------------------------------
    max_epochs: int = 1_000_000
    logging_disabled: bool = False
    logging_wandb_disabled: bool = False
    logging_rate: int = 10_000
    addit_log_epochs: Tuple[int, ...] = (1, 10, 100, 1000, 5000)
    num_workers: int = 8
    batch_size: int = 4
    num_coords: int = 30_000
    # Point spread function ------------------------------------------------------------
    point_spread_size: int = 32
    point_spread_std: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)
    # Model -------------------------------------------------------------------
    num_hidden_layers: int = 16
    hidden_size: int = 256
    latent_size: int = 128
    int_scale_range: float = 0.3  # applied via: int_scaled = int * (1 + tanh(x)*(scale_range/2))
    # Conv latent prediction -------------------------------------------------------------------
    use_conv: bool = True
    conv_channels: Tuple[int, ...] = (32,64,64,128,128)
    # Regularization -------------------------------------------------------------------
    weight_reg_inr: float = 1e-5
    weight_reg_aff: float = 1e-2
    weight_reg_lat: float = 1e-4
    weight_reg_int_scale: float = 1e-2
    weight_loss_deriv: float = 0e0
    weight_loss_hess: float = 0e0
    weight_loss_seg: float = 1e0
    weight_seg_class: Tuple[float, float, float, float] = (1,2,4,3)  # Will be normalized
    # Learning rates -------------------------------------------------------------------
    learning_rate: float = 1e-4
    learning_rate_aff: float = 1e-4
    learning_rate_def: float = 1e-4
    # Inference ----------
    inf_max_epochs: int = 500
    inf_num_coords: int = 75_000
    inf_learning_rate: float = 1e-3
    inf_learning_rate_aff: float = 1e-3
    inf_learning_rate_def: float = 1e-3
    # Positional encoder -------------------------------------------------------------------
    pe_num_frequencies: Tuple[int, int, int, int, int] = (8,8,8,5,5)
    pe_anneal_max_iter: int = 50_000
    pe_anneal_start_prop: float = 0.2
    pe_freq_scale: float = 1.0
    # Paths
    job_name: str = ''
    data_dir: str = r"/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/unaligned_subjects"
    preprocessed_h5_dir: str = r"/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/unaligned_h5_crop"


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Pass arguments using the command line like:
    # python main.py --conv_channels 32 64 128 --no-use_conv
    # For bools such as 'use_conv' passing --use_conv will make it True, passing --no-use_conv will make it False
    params = tyro.cli(Params)

    model_path_parent = Path('trained_models')
    model_path_parent.mkdir(exist_ok=True)
    model_path = model_path_parent / (f'{datetime.now().strftime("%Y%m%d-%H%M%S")}' + params.job_name)
    model_path.mkdir(exist_ok=True)
    with open(str(model_path / "params.json"), "w") as f:
        json.dump(params.__dict__, f, indent=4)

    data_module = CMRDataModule(load_la_dir=params.data_dir,
                                load_sa_dir=params.data_dir,
                                preprocessed_store_path=params.preprocessed_h5_dir,
                                full_seq_dataset=params.use_conv,
                                batch_size=params.batch_size,
                                num_coords=params.num_coords,
                                inf_num_coords=params.inf_num_coords,
                                num_workers=params.num_workers)
    data_module.prepare_data()

    os.environ['WANDB_DISABLED'] = str(params.logging_disabled)
    logger = WandbLogger(project="CMR-Align")
    logger.log_hyperparams(params.__dict__)

    checkpoint_path = model_path / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=True)
    checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                          save_last=True,
                                          dirpath=str(checkpoint_path),
                                          verbose=True,
                                          monitor='val_metrics/dice_FG',
                                          mode='max',
                                          every_n_epochs=params.logging_rate,
                                          )
    log_path = model_path / 'logs'
    log_path.mkdir(exist_ok=True)
    if params.use_conv:
        model = INR_Conv(coord_size=data_module.get_coord_size(), num_subjects=data_module.num_train,
                         max_slices=data_module.get_max_slices(), log_path=log_path, **params.__dict__)
    else:
        model = INR_AutoReg(coord_size=data_module.get_coord_size(), num_subjects=data_module.num_train,
                            max_slices=data_module.get_max_slices(), log_path=log_path, **params.__dict__)

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=1,
        max_epochs=params.max_epochs,
        # check_val_every_n_epoch=params.check_val_every_n_epoch,
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
