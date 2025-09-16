from functools import partial
import multiprocessing as mp
from pathlib import Path
import time
from typing import List, Union
import yaml

import click
import numpy as np
import pandas
from scipy.stats import bootstrap, kendalltau, spearmanr

import drugforge.ml.schema as mlschema


@click.group()
def analysis():
    pass


################################################################################
## build_results_dfs
@analysis.command()
@click.option(
    "--collection-args-fn",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--target-prop",
    default="pIC50",
    type=click.Choice(["pIC50", "pKi"], case_sensitive=False),
)
@click.option(
    "--extract-epochs",
    default=["-1", "best_mae"],
    multiple=True,
    help="Individual epochs to extract all preds for.",
)
@click.option("--gat", is_flag=True)
@click.option("--n-workers", type=int, default=16)
def build_results_dfs(
    collection_args_fn: Path,
    output_dir: Path,
    target_prop: str = "pIC50",
    extract_epochs: List[Union[str, int]] = [-1, "best_mae"],
    gat: bool = False,
    n_workers: int = 16,
):
    """
    Build and save results df CSV files for an experiment, combining results from all
    specified runs.

    Parameters
    ----------
    collection_args_fn : Path
        Path to yaml file that must contain the following entries, which will be passed
        to mlschema.load_collection_df:
        - top_level_dir
        - model_dir_str
        - model_spec_kwargs
        - spec_name_to_output_name
        - spec_lab_to_output_lab
    output_dir : Path
        Path to store output files
    target_prop : str, default="pIC50"
        Property to load from the pred trackers (passed to mlschema.load_collection_df)
    extract_epochs : List[Union[str, int]], default=[-1, "best_mae"]
        Individual epochs to extract all pred values for (passed to
        mlschema.load_collection_df). Pass None or an empty list to disable
    gat : bool, default=False
        Also load entries for GAT models. Only needed if the model_dir_str has an entry
        in it for Strategy, which will be omitted when trying to load for GAT models
    n_workers : int, default=16
        Number of concurrent processes to run when loading files
    """
    collection_kwargs = yaml.safe_load(collection_args_fn.read_text())
    top_level_dir = Path(collection_kwargs["top_level_dir"])
    model_dir_str = collection_kwargs["model_dir_str"]
    model_spec_kwargs = collection_kwargs["model_spec_kwargs"]
    spec_name_to_output_name = collection_kwargs["spec_name_to_output_name"]
    spec_lab_to_output_lab = collection_kwargs["spec_lab_to_output_lab"]

    if extract_epochs is None:
        extract_epochs = []
    parsed_extract_epochs = []
    for epoch in extract_epochs:
        try:
            parsed_extract_epochs.append(int(epoch))
        except ValueError:
            if epoch not in {"all", "best_loss", "best_mae"}:
                raise ValueError(f"Unknown value for extract_epoch: {epoch}")
            parsed_extract_epochs.append(epoch)

    s = time.time()
    per_epoch_df, (last_epoch_df, best_mae_epoch_df) = mlschema.load_collection_df(
        top_level_dir=top_level_dir,
        model_dir_str=model_dir_str,
        model_spec_kwargs=model_spec_kwargs,
        spec_name_to_output_name=spec_name_to_output_name,
        spec_lab_to_output_lab=spec_lab_to_output_lab,
        extract_epochs=parsed_extract_epochs,
        target_prop=target_prop,
        n_workers=16,
    )
    e = time.time()
    print(f"took {(e - s) // 60} minutes", flush=True)

    if gat:
        if "_{strat}" in model_dir_str:
            model_dir_str = model_dir_str.replace("_{strat}", "")
        elif "{strat}_" in model_dir_str:
            model_dir_str = model_dir_str.replace("{strat}_", "")

        for d in [model_spec_kwargs, spec_name_to_output_name, spec_lab_to_output_lab]:
            try:
                d.pop("strat")
            except KeyError:
                pass

        model_spec_kwargs["model"] = ["gat"]
        spec_lab_to_output_lab["model"] = {"gat": "GAT"}

        s = time.time()
        gat_per_epoch_df, (
            gat_last_epoch_df,
            gat_best_mae_epoch_df,
        ) = mlschema.load_collection_df(
            top_level_dir=top_level_dir,
            model_dir_str=model_dir_str,
            model_spec_kwargs=model_spec_kwargs,
            spec_name_to_output_name=spec_name_to_output_name,
            spec_lab_to_output_lab=spec_lab_to_output_lab,
            extract_epochs=parsed_extract_epochs,
            target_prop=target_prop,
            n_workers=16,
        )
        e = time.time()
        print(f"took {(e - s) // 60} minutes", flush=True)

        per_epoch_df = pandas.concat([per_epoch_df, gat_per_epoch_df])
        last_epoch_df = pandas.concat([last_epoch_df, gat_last_epoch_df])
        best_mae_epoch_df = pandas.concat([best_mae_epoch_df, gat_best_mae_epoch_df])

    # Output DFs
    output_dir.mkdir(parents=True, exist_ok=True)
    per_epoch_df.to_csv(output_dir / "per_epoch_df.csv", index=False)
    last_epoch_df.to_csv(output_dir / "last_epoch_df.csv", index=False)
    best_mae_epoch_df.to_csv(output_dir / "best_mae_epoch_df.csv", index=False)


################################################################################


################################################################################
## calc_stats
# Function to calculate a statistic (for multiprocessing)
def calc_one_stat(stat_func, target_vals, preds):
    val = stat_func(target_vals, preds)
    try:
        conf_interval = bootstrap(
            (target_vals, preds),
            statistic=lambda target, pred: stat_func(target, pred),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
    except ValueError as e:
        print(target_vals, preds, flush=True)
        raise e

    print("finished", stat_func, flush=True)
    return val, conf_interval


# Different stat functions
def calc_mae(target_vals, preds):
    return np.abs(target_vals - preds).mean()


def calc_rmse(target_vals, preds):
    return np.sqrt(np.power(target_vals - preds, 2).mean())


def calc_spearmanr(target_vals, preds):
    return spearmanr(target_vals, preds).statistic


def calc_kendalltau(target_vals, preds):
    return kendalltau(target_vals, preds).statistic


@analysis.command()
@click.option(
    "--in-fn",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    required=True,
    help="Input csv file.",
)
@click.option(
    "--out-fn",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    help="Output csv file.",
)
@click.option(
    "--gb-keys",
    type=str,
    required=True,
    help="Comma separated list of DF columns to group by when calculating stats.",
)
def calc_stats(in_fn: Path, out_fn: Path, gb_keys: str):
    gb_keys = gb_keys.split(",")
    dtypes = {k: str for k in gb_keys} | {"Model Seed": str, "Dataset Seed": str}

    # Load DF
    df = pandas.read_csv(in_fn, dtype=dtypes)
    df = df.fillna(value={"Strategy": ""})
    print("loaded df", flush=True)

    # Loop through each split and run the stats calculations
    stats_df = []
    in_range_stats_df = []
    for keys, g in df.groupby(gb_keys):
        target_vals = g["target"].values
        preds = g["pred"].values

        num_compounds = len(preds)

        # Values and low/high bounds of 95% CIs for all stats
        stat_names = []
        stat_vals = []
        stat_95ci_lows = []
        stat_95ci_highs = []

        mp_func = partial(calc_one_stat, target_vals=target_vals, preds=preds)
        stats_funcs = [calc_mae, calc_rmse, calc_spearmanr, calc_kendalltau]
        with mp.Pool(processes=4) as pool:
            stat_res = pool.map(mp_func, stats_funcs)
        # stat_res = [mp_func(f) for f in stats_funcs]
        for stat_name, (val, conf_interval) in zip(
            ["MAE", "RMSE", r"Spearman's $\rho$", r"Kendall's $\tau$"], stat_res
        ):
            stat_names.append(stat_name)
            stat_vals.append(val)
            stat_95ci_lows.append(conf_interval.low)
            stat_95ci_highs.append(conf_interval.high)

        stats_dict = {
            "Num Compounds": num_compounds,
            "Statistic": stat_names,
            "Value": stat_vals,
            "95ci_low": stat_95ci_lows,
            "95ci_high": stat_95ci_highs,
        }
        stats_df.append(pandas.DataFrame(dict(zip(gb_keys, keys)) | stats_dict))

        if "in_range" not in g:
            continue
        # Use only in range values
        range_idx = (g["in_range"] == 0).values
        target_vals = target_vals[range_idx]
        preds = preds[range_idx]

        num_compounds = len(preds)

        # Values and low/high bounds of 95% CIs for all stats
        stat_names = []
        stat_vals = []
        stat_95ci_lows = []
        stat_95ci_highs = []

        mp_func = partial(calc_one_stat, target_vals=target_vals, preds=preds)
        stats_funcs = [calc_mae, calc_rmse, calc_spearmanr, calc_kendalltau]
        with mp.Pool(processes=4) as pool:
            stat_res = pool.map(mp_func, stats_funcs)
        # stat_res = [mp_func(f) for f in stats_funcs]
        for stat_name, (val, conf_interval) in zip(
            ["MAE", "RMSE", r"Spearman's $\rho$", r"Kendall's $\tau$"], stat_res
        ):
            stat_names.append(stat_name)
            stat_vals.append(val)
            stat_95ci_lows.append(conf_interval.low)
            stat_95ci_highs.append(conf_interval.high)

        stats_dict = {
            "Num Compounds": num_compounds,
            "Statistic": stat_names,
            "Value": stat_vals,
            "95ci_low": stat_95ci_lows,
            "95ci_high": stat_95ci_highs,
        }
        in_range_stats_df.append(
            pandas.DataFrame(dict(zip(gb_keys, keys)) | stats_dict)
        )

    stats_df = pandas.concat(stats_df, axis=0, ignore_index=True)
    stats_df.to_csv(out_fn, index=False)
    in_range_out_fn = out_fn.with_stem(f"{out_fn.stem}_in_range")
    in_range_stats_df = pandas.concat(in_range_stats_df, axis=0, ignore_index=True)
    in_range_stats_df.to_csv(in_range_out_fn, index=False)


################################################################################
