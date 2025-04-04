import json
from glob import glob
from pathlib import Path

import click
import torch
from asapdiscovery.data.util.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.cli_args import (
    ds_cache_overwrite,
    ds_config_cache_overwrite,
    ds_split_args,
    e3nn_args,
    es_args,
    gat_args,
    graph_ds_args,
    kvp_list_to_dict,
    loss_args,
    model_config_cache,
    model_rand_seed,
    model_tag,
    mtenn_args,
    optim_args,
    output_dir,
    overwrite_args,
    representation_config_cache_args,
    s3_args,
    save_weights,
    schnet_args,
    struct_ds_args,
    trainer_args,
    trainer_config_cache,
    visnet_args,
    wandb_args,
    weights_path,
)

# from asapdiscovery.ml.cli_mlops import mlops
# from asapdiscovery.ml.cli_sweep import sweep
from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterType,
    EarlyStoppingType,
    OptimizerType,
)
from asapdiscovery.ml.trainer import Trainer
from mtenn.config import CombinationConfig, ModelType, ReadoutConfig, StrategyConfig
from pydantic import ValidationError


@click.group()
def ml():
    """Tools to build and train ML models and run inference."""
    pass


# Function for training using an already built Trainer
@ml.command()
def train():
    pass


# Functions for just building a Dataset and DatasetConfig
@click.group(name="build-dataset")
def build_ds():
    pass


# ml.add_command(build)
# ml.add_command(build_and_train)
# ml.add_command(build_ds)
# ml.add_command(sweep)
# ml.add_command(mlops)


# Functions for just building a Trainer and then dumping it
@ml.command()
@output_dir
@save_weights
@weights_path
@trainer_config_cache
@optim_args
@wandb_args
@model_config_cache
@representation_config_cache_args
@model_rand_seed
@model_tag
@mtenn_args
@es_args
@graph_ds_args
@ds_split_args
@loss_args
@trainer_args
@overwrite_args
@s3_args
def build(
    output_dir: Path | None = None,
    save_weights: str | None = None,
    weights_path: Path | None = None,
    trainer_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    optimizer_config_cache: Path | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: tuple[str] | None = None,
    model_type: ModelType | None = None,
    representation: str | None = None,
    complex_representation: str | None = None,
    ligand_representation: str | None = None,
    protein_representation: str | None = None,
    strategy: StrategyConfig | None = None,
    strategy_layer_norm: bool | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    model_config_cache: Path | None = None,
    representation_config_cache: Path | None = None,
    complex_representation_config_cache: Path | None = None,
    ligand_representation_config_cache: Path | None = None,
    protein_representation_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    model_tag: str | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_burnin: int | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_dict: Path | None = None,
    ds_split_config_cache: Path | None = None,
    loss: tuple[str] = (),
    loss_weights: tuple[float] = (),
    eval_loss_weights: tuple[float] = (),
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    data_aug: tuple[str] = (),
    trainer_weight_decay: float | None = None,
    batch_norm: bool | None = None,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    s3_path: str | None = None,
    upload_to_s3: bool | None = None,
):
    # Build each dict and pass to Trainer
    optim_config = {
        "cache": optimizer_config_cache,
        "overwrite_cache": overwrite_optimizer_config_cache,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dampening": dampening,
        "b1": b1,
        "b2": b2,
        "eps": eps,
        "rho": rho,
    }
    representation = (
        kvp_list_to_dict(representation) if representation is not None else {}
    )
    representation |= {"cache": representation_config_cache}
    complex_representation = (
        kvp_list_to_dict(complex_representation)
        if complex_representation is not None
        else {}
    )
    complex_representation |= {"cache": complex_representation_config_cache}
    ligand_representation = (
        kvp_list_to_dict(ligand_representation)
        if ligand_representation is not None
        else {}
    )
    ligand_representation |= {"cache": ligand_representation_config_cache}
    protein_representation = (
        kvp_list_to_dict(protein_representation)
        if protein_representation is not None
        else {}
    )
    protein_representation |= {"cache": protein_representation_config_cache}
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": model_type,
        "representation": representation,
        "complex_representation": complex_representation,
        "ligand_representation": ligand_representation,
        "protein_representation": protein_representation,
        "strategy": strategy,
        "strategy_layer_norm": strategy_layer_norm,
        "pred_readout": pred_readout,
        "combination": combination,
        "comb_readout": comb_readout,
        "max_comb_neg": max_comb_neg,
        "max_comb_scale": max_comb_scale,
        "pred_substrate": pred_substrate,
        "pred_km": pred_km,
        "comb_substrate": comb_substrate,
        "comb_km": comb_km,
        "rand_seed": model_rand_seed,
    }
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
        "burnin": es_burnin,
    }
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "exp_file": exp_file,
        "is_structural": False,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": model_type == ModelType.grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
        "split_dict": ds_split_dict,
    }
    loss_configs = [kvp_list_to_dict(loss_str) for loss_str in loss]
    data_aug_configs = [kvp_list_to_dict(aug_str) for aug_str in data_aug]

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    # Gather all the configs
    trainer_kwargs = {
        "optimizer_config": optim_config,
        "mtenn_model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_configs": loss_configs,
        "loss_weights": loss_weights,
        "eval_loss_weights": eval_loss_weights,
        "data_aug_configs": data_aug_configs,
        "weight_decay": trainer_weight_decay,
        "batch_norm": batch_norm,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "save_weights": save_weights,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": Trainer.parse_extra_config(extra_config),
        "s3_path": s3_path,
        "upload_to_s3": upload_to_s3,
        "model_tag": model_tag,
    }

    t = _build_trainer(
        trainer_kwargs, trainer_config_cache, overwrite_trainer_config_cache
    )
    return t


# Functions for building a Trainer and subsequently training the model
@ml.command(name="build-and-train")
@output_dir
@save_weights
@weights_path
@trainer_config_cache
@optim_args
@wandb_args
@model_config_cache
@representation_config_cache_args
@model_rand_seed
@model_tag
@mtenn_args
@es_args
@graph_ds_args
@ds_split_args
@loss_args
@trainer_args
@overwrite_args
@s3_args
def build_and_train(*args, **kwargs):
    t = build(*args, **kwargs)
    t.initialize()
    t.train()


def _build_trainer(
    trainer_kwargs: dict,
    trainer_config_cache: Path = None,
    overwrite_trainer_config_cache: bool = False,
):
    """
    Helper function to build a Trainer from kwargs and (optionally) a JSON Trainer
    config file. If a config file is given, those args will be used as the default, to
    be overwritten by anything in trainer_kwargs.

    Parameters
    ----------
    trainer_kwargs : dict
        Args to be passed to the Trainer constructor. These will supersede anything in
        trainer_config_cache
    trainer_config_cache : Path, optional
        Trainer Config JSON cache file. Any other CLI args that are passed will
        supersede anything in this file
    overwrite_trainer_config_cache : bool, default=False
        Overwrite any existing Trainer JSON cache file

    Returns
    -------
    Trainer
    """

    # Filter out None Trainer kwargs
    trainer_kwargs = {
        k: v
        for k, v in trainer_kwargs.items()
        if not ((v is None) or (isinstance(v, tuple) and len(v) == 0))
    }

    # If we got a config for the Trainer, load those args and merge with CLI args
    if trainer_config_cache and trainer_config_cache.exists():
        print("loading trainer args from cache", flush=True)
        config_trainer_kwargs = json.loads(trainer_config_cache.read_text())

        for config_name, config_val in config_trainer_kwargs.items():
            # Arg wasn't passed at all, so got filtered out before
            if config_name not in trainer_kwargs:
                continue

            if isinstance(config_val, dict):
                config_val.update(
                    {
                        k: v
                        for k, v in trainer_kwargs[config_name].items()
                        if not ((v is None) or (isinstance(v, tuple) and len(v) == 0))
                    }
                )
            elif (isinstance(config_val, list) or isinstance(config_val, tuple)) and (
                len(trainer_kwargs[config_name]) == 0
            ):
                # If no values are passed to CLI keep config values
                pass
            else:
                config_trainer_kwargs[config_name] = trainer_kwargs[config_name]

        trainer_kwargs = config_trainer_kwargs

    try:
        t = Trainer(**trainer_kwargs)
    except ValidationError as exc:
        # Only want to handle missing values, so if anything else went wrong just raise
        #  the pydantic error
        if any([err["type"] != "value_error.missing" for err in exc.errors()]):
            raise exc

        # Gather all missing values
        missing_vals = [err["loc"][0] for err in exc.errors()]

        raise ValueError(
            "Tried to build Trainer but missing required values: ["
            + ", ".join(missing_vals)
            + "]"
        )

    # Save Trainer
    if trainer_config_cache and (
        (not trainer_config_cache.exists()) or overwrite_trainer_config_cache
    ):
        trainer_config_cache.write_text(t.json())

    return t
