from pathlib import Path
import time
from typing import List, Union
import yaml

import click
import pandas

import drugforge.ml.schema as mlschema


@click.group()
def analysis():
    pass


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
