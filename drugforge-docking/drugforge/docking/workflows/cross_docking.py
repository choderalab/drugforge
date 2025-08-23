"""
A test-oriented docking workflow for testing the docking pipeline.
Removes all the additional layers in the other workflows and adds some features to make running cross-docking easier
"""

import logging
from pathlib import Path
from typing import Optional, Union
from pydantic.v1 import BaseModel, Field, PositiveInt, root_validator
from shutil import rmtree
from drugforge.docking.selectors.selector_list import StructureSelector
from drugforge.data.readers.meta_structure_factory import MetaStructureFactory
from drugforge.data.readers.molfile import MolFileFactory
from drugforge.data.util.dask_utils import BackendType, make_dask_client_meta
from drugforge.data.util.logging import FileLogger
from drugforge.docking.docking import (
    DockingInputMultiStructure,
    write_results_to_multi_sdf,
)
from drugforge.data.metadata.resources import active_site_chains
from drugforge.data.services.postera.manifold_data_validation import TargetTags
from drugforge.data.util.dask_utils import DaskType, FailureMode
from drugforge.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from drugforge.docking.scorer import ChemGauss4Scorer
from drugforge.docking.meta_scorer import MetaScorer
from drugforge.modeling.protein_prep import ProteinPrepper


class DockingWorkflowInputsBase(BaseModel):
    ligands: Optional[str] = Field(
        None, description="Path to a molecule file containing query ligands."
    )

    pdb_file: Optional[Path] = Field(
        None, description="Path to a PDB file to prep and dock to."
    )

    fragalysis_dir: Optional[Path] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )
    structure_dir: Optional[Path] = Field(
        None,
        description="Path to a directory containing structures to dock instead of a full fragalysis database.",
    )

    cache_dir: Optional[str] = Field(
        None, description="Path to a directory where a cache has been generated"
    )

    use_only_cache: bool = Field(
        False,
        description="Whether to only use the cached structures, otherwise try to prep uncached structures.",
    )

    save_to_cache: bool = Field(
        True,
        description="Generate a cache from structures prepped in this workflow run in this directory",
    )

    target: TargetTags = Field(None, description="The target to dock against.")

    write_final_sdf: bool = Field(
        default=True,
        description="Whether to write the final docked poses to an SDF file.",
    )
    use_dask: bool = Field(True, description="Whether to use dask for parallelism.")

    dask_type: DaskType = Field(
        DaskType.LOCAL, description="Dask client to use for parallelism."
    )

    dask_n_workers: Optional[PositiveInt] = Field(None, description="Number of workers")

    failure_mode: FailureMode = Field(
        FailureMode.SKIP, description="Dask failure mode."
    )

    n_select: PositiveInt = Field(
        5, description="Number of targets to dock each ligand against."
    )
    logname: str = Field(
        "", description="Name of the log file."
    )  # use root logger for proper forwarding of logs from dask

    loglevel: Union[int, str] = Field(logging.INFO, description="Logging level")

    output_dir: Path = Field(Path("output"), description="Output directory")

    overwrite: bool = Field(
        False, description="Whether to overwrite existing output directory."
    )
    ref_chain: Optional[str] = Field(
        None,
        description="Chain ID to align to in reference structure containing the active site",
    )
    active_site_chain: Optional[str] = Field(
        None,
        description="Active site chain ID to align to ref_chain in reference structure",
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_json_file(cls, file: str | Path):
        return cls.parse_file(str(file))

    def to_json_file(self, file: str | Path):
        with open(file, "w") as f:
            f.write(self.json(indent=2))

    @root_validator
    @classmethod
    def check_inputs(cls, values):
        """
        Validate inputs
        """
        ligands = values.get("ligands")
        fragalysis_dir = values.get("fragalysis_dir")
        structure_dir = values.get("structure_dir")
        postera = values.get("postera")
        pdb_file = values.get("pdb_file")

        if postera and ligands:
            raise ValueError("Cannot specify both ligands and postera.")

        if not postera and not ligands:
            raise ValueError("Must specify either ligands or postera.")

        # can only specify one of fragalysis dir, structure dir and PDB file
        if sum([bool(fragalysis_dir), bool(structure_dir), bool(pdb_file)]) != 1:
            raise ValueError(
                "Must specify exactly one of fragalysis_dir, structure_dir or pdb_file"
            )

        return values

    @root_validator(pre=True)
    def check_and_set_chains(cls, values):
        active_site_chain = values.get("active_site_chain")
        ref_chain = values.get("ref_chain")
        target = values.get("target")
        if target:
            if not active_site_chain:
                values["active_site_chain"] = active_site_chains[target]
            # set same chain for active site if not specified
            if not ref_chain:
                values["ref_chain"] = active_site_chains[target]
        return values


class CrossDockingWorkflowInputs(DockingWorkflowInputsBase):
    logname: str = Field("", description="Name of the log file.")
    structure_selector: StructureSelector = Field(
        StructureSelector.LEAVE_SIMILAR_OUT,
        description="Structure selector to use for docking",
    )
    multi_reference: bool = Field(
        False,
        description="Whether to use multi reference docking, in which the docking_method "
        "recieves a DockingInputMultiStructure object instead of a DockingInputPair object",
    )
    # Copied from POSITDocker
    relax_mode: POSIT_RELAX_MODE = Field(
        POSIT_RELAX_MODE.NONE,
        description="When to check for relaxation either, 'clash', 'all', 'none'",
    )
    posit_method: POSIT_METHOD = Field(
        POSIT_METHOD.ALL, description="POSIT method to use"
    )
    use_omega: bool = Field(False, description="Use omega to generate conformers")
    omega_dense: bool = Field(False, description="Use dense conformer generation")
    num_poses: PositiveInt = Field(1, description="Number of poses to generate")
    allow_low_posit_prob: bool = Field(False, description="Allow low posit probability")
    low_posit_prob_thresh: float = Field(
        0.1,
        description="Minimum posit probability threshold if allow_low_posit_prob is False",
    )
    allow_final_clash: bool = Field(
        False, description="Allow clashing poses in last stage of docking"
    )
    allow_retries: bool = Field(
        True,
        description="Allow retries with different options if docking fails initially",
    )


def cross_docking_workflow(inputs: CrossDockingWorkflowInputs):
    """
    Run cross docking on a set of ligands, against multiple targets
    Parameters
    ----------
    inputs : CrossDockingWorkflowInputs
        Inputs to cross docking
    Returns
    -------
    None
    """
    output_dir = inputs.output_dir
    new_directory = True
    if output_dir.exists():
        if inputs.overwrite:
            rmtree(output_dir)
        else:
            new_directory = False
    # this won't overwrite the existing directory
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = FileLogger(
        inputs.logname,  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="cross-docking.log",
        stdout=True,
        level=inputs.loglevel,
    ).getLogger()
    if new_directory:
        logger.info(f"Writing to / overwriting output directory: {output_dir}")
    else:
        logger.info(f"Writing to existing output directory: {output_dir}")
    logger.info(f"Running cross docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")
    inputs.to_json_file(output_dir / "cross_docking_inputs.json")
    if inputs.use_dask:
        dask_client = make_dask_client_meta(
            inputs.dask_type,
            loglevel=inputs.loglevel,
            n_workers=inputs.dask_n_workers,
        )
    else:
        dask_client = None
    # make a directory to store intermediate CSV results
    data_intermediates = Path(output_dir / "data_intermediates")
    data_intermediates.mkdir(exist_ok=True)
    # load from file
    logger.info(f"Loading ligands from file: {inputs.ligands}")
    molfile = MolFileFactory(filename=inputs.ligands)
    query_ligands = molfile.load()
    # read structures
    structure_factory = MetaStructureFactory(
        structure_dir=inputs.structure_dir,
        fragalysis_dir=inputs.fragalysis_dir,
        pdb_file=inputs.pdb_file,
        use_dask=inputs.use_dask,
        failure_mode=inputs.failure_mode,
        dask_client=dask_client,
    )
    complexes = structure_factory.load()
    n_query_ligands = len(query_ligands)
    logger.info(f"Loaded {n_query_ligands} query ligands")
    n_complexes = len(complexes)
    logger.info(f"Loaded {n_complexes} complexes")
    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(cache_dir=inputs.cache_dir)
    prepped_complexes = prepper.prep(
        complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        cache_dir=inputs.cache_dir,
        use_only_cache=inputs.use_only_cache,
    )
    del complexes
    n_prepped_complexes = len(prepped_complexes)
    logger.info(f"Prepped {n_prepped_complexes} complexes")
    if inputs.save_to_cache and inputs.cache_dir is not None:
        logger.info(f"Writing prepped complexes to global cache {inputs.cache_dir}")
        prepper.cache(prepped_complexes, inputs.cache_dir)
    # define selector and select pairs
    # using dask here is too memory intensive as each worker needs a copy of all the complexes in memory
    # which are quite large themselves, is only effective for large numbers of ligands and small numbers of complexes
    logger.info("Selecting pairs for docking")
    # TODO: MCS takes an n_select arg but Pairwise does not...meaning we are losing that option the way this is written
    selector = inputs.structure_selector.selector_cls()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
    )
    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")
    if inputs.multi_reference:
        # Generate one-to-many objects for multi-reference docking
        logger.info("Generating one-to-many objects for multi-reference docking")
        sets = DockingInputMultiStructure.from_pairs(pairs)
        logger.info(f"Generated {len(sets)} ligand-protein sets for docking")
    else:
        sets = pairs
    del prepped_complexes
    # dock pairs
    logger.info("Running docking on selected pairs")
    docker = POSITDocker(
        relax_mode=inputs.relax_mode,
        posit_method=inputs.posit_method,
        use_omega=inputs.use_omega,
        omega_dense=inputs.omega_dense,
        num_poses=inputs.num_poses,
        allow_low_posit_prob=inputs.allow_low_posit_prob,
        low_posit_prob_thresh=inputs.low_posit_prob_thresh,
        allow_final_clash=inputs.allow_final_clash,
        allow_retries=inputs.allow_retries,
        last_ditch_fred=False,
    )
    results = docker.dock(
        sets,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_for_disk_backend=False,
    )
    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs
    # add chemgauss4 scorer
    scorers = [ChemGauss4Scorer()]
    # score results
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)
    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_results_to_multi_sdf(
            output_dir / "docking_results.sdf",
            results,
            backend=BackendType.IN_MEMORY,
            reconstruct_cls=docker.result_cls,
        )

    logger.info("Running scoring")
    scores_df = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_df=True,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=docker.result_cls,
    )

    del results

    scores_df.to_csv(output_dir / "docking_scores_raw.csv", index=False)
    logger.info("Finished successfully!")
