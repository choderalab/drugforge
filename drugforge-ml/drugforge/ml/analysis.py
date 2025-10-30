import datetime
from itertools import product
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
@click.option(
    "--filter-date",
    type=click.DateTime(),
    help="Don't load results from before this date.",
)
def build_results_dfs(
    collection_args_fn: Path,
    output_dir: Path = None,
    target_prop: str = "pIC50",
    extract_epochs: List[Union[str, int]] = [-1, "best_mae"],
    gat: bool = False,
    n_workers: int = 16,
    filter_date: datetime.date = None,
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
    per_epoch_df, extract_epoch_dfs = mlschema.load_collection_df(
        top_level_dir=top_level_dir,
        model_dir_str=model_dir_str,
        model_spec_kwargs=model_spec_kwargs,
        spec_name_to_output_name=spec_name_to_output_name,
        spec_lab_to_output_lab=spec_lab_to_output_lab,
        extract_epochs=parsed_extract_epochs,
        target_prop=target_prop,
        run_date=filter_date,
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
        gat_per_epoch_df, gat_extract_epoch_dfs = mlschema.load_collection_df(
            top_level_dir=top_level_dir,
            model_dir_str=model_dir_str,
            model_spec_kwargs=model_spec_kwargs,
            spec_name_to_output_name=spec_name_to_output_name,
            spec_lab_to_output_lab=spec_lab_to_output_lab,
            extract_epochs=parsed_extract_epochs,
            target_prop=target_prop,
            run_date=filter_date,
            n_workers=16,
        )
        e = time.time()
        print(f"took {(e - s) // 60} minutes", flush=True)

        if gat_per_epoch_df is not None:
            if per_epoch_df is None:
                per_epoch_df = gat_per_epoch_df
                extract_epoch_dfs = gat_extract_epoch_dfs
            else:
                per_epoch_df = pandas.concat([per_epoch_df, gat_per_epoch_df])
                extract_epoch_dfs = [
                    pandas.concat([df, gat_df], axis=0, ignore_index=True)
                    for df, gat_df in zip(extract_epoch_dfs, gat_extract_epoch_dfs)
                ]

    if per_epoch_df is None:
        raise RuntimeError("No pred trackers were loaded")

    if output_dir:
        # Output DFs
        output_dir.mkdir(parents=True, exist_ok=True)
        per_epoch_df.to_csv(output_dir / "per_epoch_df.csv", index=False)
        for df, epoch in zip(extract_epoch_dfs, parsed_extract_epochs):
            if epoch == -1:
                epoch = "last"

            df.to_csv(output_dir / f"{epoch}_epoch_df.csv", index=False)

    return per_epoch_df, extract_epoch_dfs


################################################################################


################################################################################
## calc_stats
# Function to calculate a statistic (for multiprocessing)
def calc_one_stat(stat_func, target_vals, preds, in_range):
    val = stat_func(target_vals, preds, in_range)
    try:
        conf_interval = bootstrap(
            (target_vals, preds, in_range),
            statistic=lambda target, pred, r: stat_func(target, pred, r),
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
def calc_mae(target_vals, preds, _):
    return np.abs(target_vals - preds).mean()


def calc_mae_stepped(target_vals, preds, in_range):
    zero_loss_mask = ((in_range < 0) & (preds <= target_vals)) | (
        (in_range > 0) & (preds >= target_vals)
    )
    abs_err_vals = np.abs(target_vals - preds)
    abs_err_vals[zero_loss_mask] = 0
    return abs_err_vals.mean()


def calc_rmse(target_vals, preds, _):
    return np.sqrt(np.power(target_vals - preds, 2).mean())


def calc_spearmanr(target_vals, preds, _):
    return spearmanr(target_vals, preds).statistic


def calc_kendalltau(target_vals, preds, _):
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
        if ("in_range" not in g) or g["in_range"].isna().all():
            use_range = False
            in_range = np.zeros(len(preds))
        else:
            use_range = True
            in_range = g["in_range"].values

        num_compounds = len(preds)

        # Values and low/high bounds of 95% CIs for all stats
        stat_names = []
        stat_vals = []
        stat_95ci_lows = []
        stat_95ci_highs = []

        mp_func = partial(
            calc_one_stat, target_vals=target_vals, preds=preds, in_range=in_range
        )
        stats_funcs = [
            calc_mae,
            calc_mae_stepped,
            calc_rmse,
            calc_spearmanr,
            calc_kendalltau,
        ]
        with mp.Pool(processes=5) as pool:
            stat_res = pool.map(mp_func, stats_funcs)
        # stat_res = [mp_func(f) for f in stats_funcs]
        for stat_name, (val, conf_interval) in zip(
            ["MAE", "Adjusted MAE", "RMSE", r"Spearman's $\rho$", r"Kendall's $\tau$"],
            stat_res,
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

        if not use_range:
            continue
        # Use only in range values
        range_idx = (g["in_range"] == 0).values
        target_vals = target_vals[range_idx]
        preds = preds[range_idx]
        in_range = in_range[range_idx]

        num_compounds = len(preds)

        # Values and low/high bounds of 95% CIs for all stats
        stat_names = []
        stat_vals = []
        stat_95ci_lows = []
        stat_95ci_highs = []

        stats_funcs = [calc_mae, calc_rmse, calc_spearmanr, calc_kendalltau]
        mp_func = partial(
            calc_one_stat, target_vals=target_vals, preds=preds, in_range=in_range
        )
        with mp.Pool(processes=5) as pool:
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
    if len(in_range_stats_df) > 0:
        in_range_out_fn = out_fn.with_stem(f"{out_fn.stem}_in_range")
        in_range_stats_df = pandas.concat(in_range_stats_df, axis=0, ignore_index=True)
        in_range_stats_df.to_csv(in_range_out_fn, index=False)


################################################################################


################################################################################
## subset_by_strat
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
    "--model-strat",
    type=str,
    multiple=True,
    help=(
        "Which strategy to keep for each model. Pass as multiple <model>:<strategy> "
        "pairs."
    ),
)
def subset_by_strat(in_fn, out_fn, model_strat):
    df = pandas.read_csv(in_fn, dtype=str)
    df = df.fillna(value={"Strategy": ""})

    model_strat = [kvp.split(":") for kvp in model_strat]
    idx = (df["Model"] == model_strat[0][0]) & (df["Strategy"] == model_strat[0][1])
    for model, strat in model_strat:
        idx |= (df["Model"] == model) & (df["Strategy"] == strat)

    df = df.loc[idx, :]
    df.to_csv(out_fn, index=False)


################################################################################


################################################################################
## subset_general
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
    "--filters",
    type=str,
    multiple=True,
    help=(
        "Filters to use to subset DF. Pass as a comma separated list of <key>:<value> "
        "pairs. The final index will be the intersection of these fitlers. If multiple "
        "values are passed, for this option, the final index will be the union of the "
        "index from each individual list of filters."
    ),
)
def subset_general(in_fn, out_fn, model_strat, filters):
    df = pandas.read_csv(in_fn, dtype=str)
    df = df.fillna(value={"Strategy": ""})

    filter_pairs = [[kvp.split(":") for kvp in f] for f in filters]
    filter_idxs = []
    for filt_pair_list in filter_pairs:
        idx = df[filt_pair_list[0][0]] == df[filt_pair_list[0][1]]
        for col, val in filt_pair_list[1:]:
            idx &= df[col] == val
        filter_idxs.append(idx)

    final_idx = filter_idxs[0]
    for idx in filter_idxs[1:]:
        final_idx |= idx

    df = df.loc[final_idx, :]
    df.to_csv(out_fn, index=False)


################################################################################


################################################################################
## Check how many epochs each model has trained for
@analysis.command()
@click.option(
    "--collection-args-fn",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-fn",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    required=True,
)
def training_progress(collection_args_fn, out_fn):
    collection_kwargs = yaml.safe_load(collection_args_fn.read_text())
    top_level_dir = Path(collection_kwargs["top_level_dir"])
    print(top_level_dir, flush=True)
    model_dir_str = collection_kwargs["model_dir_str"]
    model_spec_kwargs = collection_kwargs["model_spec_kwargs"]
    spec_name_to_output_name = collection_kwargs["spec_name_to_output_name"]
    spec_lab_to_output_lab = collection_kwargs["spec_lab_to_output_lab"]

    model_metadata_dict = {}
    for full_spec in product(*model_spec_kwargs.values()):
        d = dict(zip(model_spec_kwargs.keys(), full_spec))
        full_model_spec = model_dir_str.format(**d)

        # Dict mapping formatted model_spec key -> formatted val
        formatted_d = {
            spec_name_to_output_name.get(orig_k, orig_k): spec_lab_to_output_lab.get(
                orig_k, {}
            ).get(orig_v, orig_v)
            for orig_k, orig_v in d.items()
        }
        model_metadata_dict[full_model_spec] = formatted_d

    try:
        # Just check if there's some GAT results there
        next(iter(top_level_dir.glob("gat*")))

        if r"_{strat}" in model_dir_str:
            gat_model_dir_str = model_dir_str.replace(r"_{strat}", "")
        elif r"{strat}_" in model_dir_str:
            gat_model_dir_str = model_dir_str.replace(r"{strat}_", "")
        elif r"{strat}" in model_dir_str:
            gat_model_dir_str = model_dir_str.replace(r"{strat}", "")
        else:
            gat_model_dir_str = model_dir_str

        model_spec_kwargs["model"] = ["gat"]
        if "strat" in model_spec_kwargs:
            model_spec_kwargs["strat"] = [""]

        spec_lab_to_output_lab["model"]["gat"] = "GAT"

        for full_spec in product(*model_spec_kwargs.values()):
            d = dict(zip(model_spec_kwargs.keys(), full_spec))
            full_model_spec = gat_model_dir_str.format(**d)

            # Dict mapping formatted model_spec key -> formatted val
            formatted_d = {
                spec_name_to_output_name.get(
                    orig_k, orig_k
                ): spec_lab_to_output_lab.get(orig_k, {}).get(orig_v, orig_v)
                for orig_k, orig_v in d.items()
            }
            model_metadata_dict[full_model_spec] = formatted_d
    except StopIteration:
        pass

    n_epochs_df = []
    for full_model_spec, formatted_d in model_metadata_dict.items():
        model_wts_dir = top_level_dir / full_model_spec
        model_wts_dir /= (model_wts_dir / "run_id").read_text()
        model_wts = [
            int(p.stem) for p in model_wts_dir.glob("*.th") if p.stem.isdecimal()
        ]
        if len(model_wts) == 0:
            max_epoch = -1
        else:
            max_epoch = max(model_wts)
        formatted_d["Epochs"] = max_epoch
        df = pandas.DataFrame(formatted_d, index=[0])
        n_epochs_df.append(df)

    n_epochs_df = pandas.concat(n_epochs_df, axis=0, ignore_index=True)
    n_epochs_df.to_csv(out_fn, index=False)


################################################################################
