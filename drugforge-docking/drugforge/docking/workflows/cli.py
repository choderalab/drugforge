import logging
from typing import Union

import click
from drugforge.data.services.postera.manifold_data_validation import TargetTags
from drugforge.data.util.dask_utils import DaskType, FailureMode
from drugforge.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE
from drugforge.docking.selectors.selector_list import StructureSelector
from drugforge.docking.workflows.cli_args import (
    cache_dir,
    dask_args,
    fragalysis_dir,
    input_json,
    ligands,
    loglevel,
    output_dir,
    overwrite,
    pdb_file,
    save_to_cache,
    structure_dir,
    use_only_cache,
)
from drugforge.docking.workflows.cross_docking import (
    CrossDockingWorkflowInputs,
    cross_docking_workflow,
)


@click.group()
def cli(help="Command-line interface for drugforge-docking"):
    ...


@cli.command()
@click.option(
    "--target",
    type=click.Choice(TargetTags.get_values(), case_sensitive=True),
    help="The target for the workflow",
    required=False,
)
@click.option(
    "--use-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@click.option(
    "--omega-dense",
    is_flag=True,
    default=False,
    help="Whether to use dense conformer enumeration with OEOmega (slower, more accurate)",
)
@click.option(
    "--posit-method",
    type=click.Choice(POSIT_METHOD.get_names(), case_sensitive=False),
    default="all",
    help="The set of methods POSIT can use. Defaults to all.",
)
@click.option(
    "--relax-mode",
    type=click.Choice(POSIT_RELAX_MODE.get_names(), case_sensitive=False),
    default="none",
    help="When to check for relaxation either, 'clash', 'all', 'none'",
)
@click.option(
    "--allow-retries",
    is_flag=True,
    default=False,
    help="Whether to allow POSIT to retry with relaxed parameters if docking fails (slower, more likely to succeed)",
)
@click.option(
    "--allow-final-clash",
    is_flag=True,
    default=False,
    help="Allow clashing poses in last stage of docking",
)
@click.option(
    "--multi-reference",
    is_flag=True,
    default=False,
    help="Whether to pass multiple references to the docker for each ligand instead of just one at a time",
)
@click.option(
    "--structure-selector",
    type=click.Choice(StructureSelector.get_values(), case_sensitive=False),
    default=StructureSelector.LEAVE_SIMILAR_OUT.value,
    help="The type of structure selector to use.",
)
@click.option("--num-poses", type=int, default=1, help="Number of poses to generate")
@ligands
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@use_only_cache
@dask_args
@output_dir
@overwrite
@input_json
@loglevel
def cross_docking(
    target: TargetTags,
    multi_reference: bool = False,
    structure_selector: StructureSelector = StructureSelector.LEAVE_SIMILAR_OUT,
    use_omega: bool = False,
    omega_dense: bool = False,
    posit_method: str | None = POSIT_METHOD.ALL.name,
    relax_mode: str | None = POSIT_RELAX_MODE.NONE.name,
    num_poses: int = 1,
    allow_retries: bool = False,
    allow_final_clash: bool = False,
    ligands: str | None = None,
    pdb_file: str | None = None,
    fragalysis_dir: str | None = None,
    structure_dir: str | None = None,
    use_only_cache: bool = False,
    save_to_cache: bool | None = True,
    cache_dir: str | None = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: str | None = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    dask_n_workers: int | None = None,
    failure_mode: FailureMode = FailureMode.SKIP,
    loglevel: Union[int, str] = logging.INFO,
):
    """
    Runs docking on the provided set of targets and ligands.
    Uses the specified structure selector to select with protein-ligand combinations to run.
    By default, uses the LeaveSimilarOutSelector, which excludes query-reference pairs where the ligands are
        identical or stereoisomers, protonation states isomers, or tautomers. All other query-reference pairs
        are docked. If no similar ligands are present in a dataset of N structures with N ligands,
        N * (N-1) pairs will be docked.
    By default, each poses is scored with the ChemGauss4 scorer.
    """

    if input_json is not None:
        logging.info(
            f"Loading inputs from {input_json}...Will override all other inputs."
        )
        inputs = CrossDockingWorkflowInputs.from_json_file(input_json)

    else:
        inputs = CrossDockingWorkflowInputs(
            target=target,
            multi_reference=multi_reference,
            structure_selector=structure_selector,
            use_dask=use_dask,
            dask_type=dask_type,
            dask_n_workers=dask_n_workers,
            failure_mode=failure_mode,
            use_omega=use_omega,
            omega_dense=omega_dense,
            posit_method=POSIT_METHOD[posit_method],
            relax_mode=POSIT_RELAX_MODE[relax_mode],
            num_poses=num_poses,
            allow_retries=allow_retries,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            cache_dir=cache_dir,
            use_only_cache=use_only_cache,
            save_to_cache=save_to_cache,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_final_clash=allow_final_clash,
            loglevel=loglevel,
        )

    cross_docking_workflow(inputs)
