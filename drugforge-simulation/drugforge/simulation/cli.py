import itertools
import logging
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union

import click
from drugforge.cli.cli_args import (
    ligands,
    loglevel,
    md_openmm_platform,
    md_steps,
    output_dir,
    pdb_file,
    use_dask,
    target
)
from drugforge.data.readers.molfile import MolFileFactory
from drugforge.data.schema.complex import Complex
from drugforge.data.util.dask_utils import DaskType, make_dask_client_meta
from drugforge.data.util.logging import FileLogger
from drugforge.simulation.simulate import OpenMMPlatform, VanillaMDSimulator
from drugforge.simulation.simulate import minimize_from_pdb
from openmm import unit


@click.group()
def simulation():
    """Run simulations on molecular systems."""
    pass


@simulation.command()
@ligands
@pdb_file
@md_steps
@md_openmm_platform
@output_dir
@loglevel
@use_dask
def vanilla_md(
    ligands: Optional[str] = None,
    pdb_file: Optional[str] = None,
    md_steps: int = 2500000,  # 10 ns @ 4.0 fs timestep
    md_openmm_platform: OpenMMPlatform = OpenMMPlatform.Fastest,
    output_dir: str = "output",
    loglevel: Union[int, str] = logging.INFO,
    use_dask: bool = False,
):

    # make output directory
    output_dir = Path(output_dir)

    if output_dir.exists():
        rmtree(output_dir)

    output_dir.mkdir()

    if not pdb_file:
        raise ValueError("Please provide a pdb file")

    logger = FileLogger(
        "",  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="vanilla_md.log",
        stdout=True,
        level=loglevel,
    ).getLogger()

    logger.info("Running vanilla MD simulations")
    logger.info(f"Using {md_openmm_platform} as the OpenMM platform")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log level: {loglevel}")
    logger.info(f"Use dask: {use_dask}")

    if use_dask:
        logger.info("Using dask")
        dask_client = make_dask_client_meta(DaskType.LOCAL_GPU, loglevel=loglevel)
    else:
        dask_client = None

    pdb_path = Path(pdb_file)

    complex = Complex.from_pdb(
        pdb_file,
        target_kwargs={"target_name": pdb_path.stem},
        ligand_kwargs={"compound_name": f"{pdb_path.stem}_ligand"},
    )

    if ligands:
        # in case its a multisdf
        ligs = MolFileFactory(filename=ligands).load()
        logger.info(f"Read {len(ligs)} ligands")

        # save each ligand to a file
        lig_paths = []
        lig_dir = output_dir / "ligands"
        lig_dir.mkdir(exist_ok=True)
        for lig in ligs:
            path = lig_dir / f"{lig.compound_name}.sdf"
            lig.to_sdf(path)
            lig_paths.append(path)

    else:
        # use the ligand that was in the pdb
        if complex.ligand:
            if complex.ligand.to_rdkit().GetNumAtoms() == 0:
                logger.info(
                    "Protein ligand has no atoms, will simulate only protein ..."
                )
            lig_path = output_dir / f"{pdb_path.stem}_ligand.sdf"
            complex.ligand.to_sdf(lig_path)
            lig_paths = [lig_path]

        else:
            raise ValueError("No ligands provided and none found in pdb")

    protein_processed_path = output_dir / f"{pdb_path.stem}_processed.pdb"
    complex.to_pdb(protein_processed_path)

    combo = list(itertools.product([protein_processed_path], lig_paths))

    simulator = VanillaMDSimulator(
        output_dir=output_dir,
        openmm_platform=md_openmm_platform,
        num_steps=md_steps,
        progressbar=True,
    )
    logger.info(f"Simulator: {simulator}")
    logger.info(f"Number of steps: {md_steps}")
    logger.info(f"OpenMM Platform: {md_openmm_platform}")
    logger.info(f"Time step (fs): {simulator.timestep}")
    logger.info(
        f"Simulation length (ns): {simulator.total_simulation_time.value_in_unit(unit.nanoseconds)}"
    )

    logger.info("running simulations ..., please wait")
    simulator.simulate(
        combo, use_dask=use_dask, dask_client=dask_client, failure_mode="skip"
    )
    logger.info("done")


@simulation.command()
@ligands
@pdb_file
def szybki():
    raise NotImplementedError("Szybki simulation not yet implemented")

@simulation.command()
@pdb_file
@click.option(
    "--min-out",
    type=str,
    default="minimized.pdb",
    help="Optional file name for saving result of BLAST search",
)
@output_dir
@md_openmm_platform
@target
@loglevel
def minimize(
    pdb_file: Union[Path, str],
    min_out: Union[Path, str],
    output_dir: Union[Path, str],
    md_openmm_platform: str,
    target: str,
    loglevel: Union[int, str] = logging.INFO,
):
    logger = FileLogger(
        "",  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="vanilla_md.log",
        stdout=True,
        level=loglevel,
    ).getLogger()

    logger.info("Running minimization")

    minimize_from_pdb(
        pdb_file=pdb_file,
        min_out=min_out,
        output_dir=output_dir,
        md_openmm_platform=md_openmm_platform,
        target=target,
    )
    logger.info(f"Minimized structure saved to {min_out}")
    