import click


def use_dask(func):
    return click.option(
        "--use-dask",
        is_flag=True,
        default=False,
        help="Whether to use dask for parallelism.",
    )(func)


def dask_type(func):
    from drugforge.data.util.dask_utils import DaskType

    return click.option(
        "--dask-type",
        type=click.Choice(DaskType.get_values(), case_sensitive=False),
        default=DaskType.LOCAL,
        help="The type of dask cluster to use. Local mode is reccommended for most use cases.",
    )(func)


def failure_mode(func):
    from drugforge.data.util.dask_utils import FailureMode

    return click.option(
        "--failure-mode",
        type=click.Choice(FailureMode.get_values(), case_sensitive=False),
        default=FailureMode.SKIP,
        help="The failure mode for dask. Can be 'raise' or 'skip'.",
        show_default=True,
    )(func)


def dask_n_workers(func):
    return click.option(
        "--dask-n-workers",
        type=int,
        default=None,
        help="The number of workers to use with dask.",
    )(func)


def dask_args(func):
    return use_dask(dask_type(dask_n_workers(failure_mode(func))))


def target(func):
    from drugforge.data.services.postera.manifold_data_validation import TargetTags

    return click.option(
        "--target",
        type=click.Choice(TargetTags.get_values(), case_sensitive=True),
        help="The target for the workflow",
        required=True,
    )(func)


def ligands(func):
    return click.option(
        "-l",
        "--ligands",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="File containing ligands",
    )(func)


def output_dir(func):
    return click.option(
        "--output-dir",
        type=click.Path(
            resolve_path=True, exists=False, file_okay=False, dir_okay=True
        ),
        help="The directory to output results to.",
        default="output",
    )(func)


def overwrite(func):
    return click.option(
        "--overwrite/--no-overwrite",
        default=True,
        help="Whether to overwrite the output directory if it exists.",
    )(func)


def input_json(func):
    return click.option(
        "--input-json",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="Path to a json file containing the inputs to the workflow,  WARNING: overrides all other inputs.",
    )(func)


def fragalysis_dir(func):
    return click.option(
        "--fragalysis-dir",
        type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
        help="Path to a directory containing fragments to dock.",
    )(func)


def structure_dir(func):
    return click.option(
        "--structure-dir",
        type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
        help="Path to a directory containing structures.",
    )(func)


def pdb_file(func):
    return click.option(
        "--pdb-file",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="Path to a pdb file containing a structure",
    )(func)


def cache_dir(func):
    return click.option(
        "--cache-dir",
        type=click.Path(
            resolve_path=True, exists=False, file_okay=False, dir_okay=True
        ),
        help="Path to a directory where design units are cached.",
    )(func)


def use_only_cache(func):
    return click.option(
        "--use-only-cache",
        is_flag=True,
        default=False,
        help="Whether to only use the cache.",
    )(func)


def save_to_cache(func):
    return click.option(
        "--save-to-cache/--no-save-to-cache",
        help="If the newly generated structures should be saved to the cache folder.",
        default=True,
    )(func)


def loglevel(func):
    return click.option(
        "--loglevel",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        help="The log level to use.",
        default="INFO",
        show_default=True,
    )(func)
