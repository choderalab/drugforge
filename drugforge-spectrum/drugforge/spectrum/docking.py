"""
drugforge/spectrum/docking.py
==============================
Ligand transfer docking for predicted protein structures.

Strips away the target-tag, ML scoring, MD, Dask, HTML visualisation, and
caching concerns from the full drugforge-workflows implementation so that the
workflow can be run against novel viral protease sequences without a registered
TargetTags entry.

Public API
----------
ligand_transfer_docking  – run POSIT ligand transfer docking for a directory
                           of predicted structures against Fragalysis reference
                           complexes, returning a results DataFrame and SDF.
"""

import logging
import warnings
from pathlib import Path
from shutil import rmtree

import pandas as pd

from drugforge.data.readers.meta_structure_factory import MetaStructureFactory
from drugforge.data.readers.structure_dir import StructureDirFactory
from drugforge.data.util.dask_utils import BackendType, FailureMode
from drugforge.data.util.logging import FileLogger
from drugforge.docking.docking import write_results_to_multi_sdf
from drugforge.docking.docking_data_validation import DockingResultCols
from drugforge.docking.meta_scorer import MetaScorer
from drugforge.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from drugforge.docking.scorer import ChemGauss4Scorer
from drugforge.docking.selectors.selector_list import StructureSelector
from drugforge.modeling.protein_prep import LigandTransferProteinPrepper

logger = logging.getLogger(__name__)


def ligand_transfer_docking(
    target_structure_dir: str | Path,
    reference_fragalysis_dir: str | Path,
    output_dir: str | Path,
    ref_chain: str = "A",
    active_site_chain: str = "A",
    posit_method: POSIT_METHOD = POSIT_METHOD.ALL,
    relax_mode: POSIT_RELAX_MODE = POSIT_RELAX_MODE.NONE,
    use_omega: bool = False,
    num_poses: int = 1,
    allow_retries: bool = True,
    allow_final_clash: bool = True,
    posit_confidence_cutoff: float = 0.1,
    overwrite: bool = True,
    failure_mode: FailureMode = FailureMode.SKIP,
    loglevel: int = logging.INFO,
) -> pd.DataFrame:
    """Run POSIT ligand transfer docking for predicted structures.

    Loads predicted protein structures (AF3 / Boltz PDB files) from
    ``target_structure_dir``, aligns each to every reference complex in
    ``reference_fragalysis_dir`` using ``LigandTransferProteinPrepper``,
    then runs POSIT self-docking on each transferred pose.

    No target tag, ML scoring, MD, HTML visualisation, Dask, or caching is
    required – this function is intentionally minimal.

    Parameters
    ----------
    target_structure_dir:
        Directory of predicted PDB files (one per sequence). Globs ``*.pdb``.
    reference_fragalysis_dir:
        Fragalysis-format directory of reference crystal complexes
        (``<ID>/<ID>.pdb`` + ``<ID>/<ID>.sdf``).
    output_dir:
        Directory to write results. Created if absent; overwritten if
        ``overwrite=True``.
    ref_chain:
        Chain ID in the reference complex used for structural alignment.
    active_site_chain:
        Chain ID in the target structure used for structural alignment.
    posit_method:
        POSIT method(s) to use. Defaults to ``ALL``.
    relax_mode:
        When to relax clashing atoms. Defaults to ``NONE``.
    use_omega:
        Whether to enumerate conformers with OEOmega before docking.
    num_poses:
        Number of docked poses to return per pair.
    allow_retries:
        Whether POSIT may retry with relaxed settings on failure.
    allow_final_clash:
        Whether to keep poses that clash in the final docking stage.
    posit_confidence_cutoff:
        Minimum POSIT confidence score to keep a result.
    overwrite:
        Whether to wipe and recreate ``output_dir`` if it already exists.
    failure_mode:
        How to handle per-structure failures – ``SKIP`` or ``RAISE``.
    loglevel:
        Python logging level.

    Returns
    -------
    pd.DataFrame
        Final scored and filtered docking results, sorted by ChemGauss4 score.
        Also written to ``output_dir/docking_results_final.csv``.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and overwrite:
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_logger = FileLogger(
        logname="ligand_transfer_docking",
        path=str(output_dir),
        logfile="ligand_transfer_docking.log",
        level=loglevel,
        stdout=True,
    )
    log = file_logger.getLogger()
    data_intermediates = output_dir / "data_intermediates"
    data_intermediates.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load reference complexes from Fragalysis directory
    # ------------------------------------------------------------------
    log.info(f"Loading reference complexes from {reference_fragalysis_dir}")
    ref_factory = MetaStructureFactory(
        fragalysis_dir=reference_fragalysis_dir,
        structure_dir=None,
        pdb_file=None,
    )
    ref_complexes = ref_factory.load(use_dask=False, failure_mode=failure_mode)
    log.info(f"Loaded {len(ref_complexes)} reference complexes")

    # ------------------------------------------------------------------
    # Load predicted target structures from flat PDB directory
    # ------------------------------------------------------------------
    log.info(f"Loading target structures from {target_structure_dir}")
    target_factory = StructureDirFactory.from_dir(target_structure_dir)
    targets = target_factory.load(use_dask=False, failure_mode=failure_mode)
    log.info(f"Loaded {len(targets)} target structures")

    # ------------------------------------------------------------------
    # Prep: align each target to each reference, transfer ligand coordinates
    # ------------------------------------------------------------------
    log.info("Running LigandTransferProteinPrepper (align + ligand transfer)...")
    prepper = LigandTransferProteinPrepper(
        reference_complexes=ref_complexes,
        ref_chain=ref_chain,
        active_site_chain=active_site_chain,
        seqres_yaml=None,  # no mutation – predicted structures are complete
        loop_db=None,  # no loop filling – predicted structures are complete
    )
    prepped = prepper.prep(
        targets,
        use_dask=False,
        failure_mode=failure_mode,
        cache_dir=None,
        use_only_cache=False,
    )
    log.info(f"Prepped {len(prepped)} target-reference pairs")

    # ------------------------------------------------------------------
    # Select pairs for docking (self-docking: each ligand docked back into
    # the structure it was transferred from)
    # ------------------------------------------------------------------
    selector = StructureSelector.SELF_DOCKING.selector_cls()

    # De-duplicate ligands by InChIKey so pivot() doesn't fail when the same
    # reference ligand appears in multiple prepped complexes.
    seen: set[str] = set()
    unique_ligands = []
    for pc in prepped:
        ik = pc.ligand.inchikey
        if ik not in seen:
            seen.add(ik)
            unique_ligands.append(pc.ligand)

    pairs = selector.select(unique_ligands, prepped)
    log.info(
        f"Selected {len(pairs)} pairs from {len(unique_ligands)} unique ligands "
        f"and {len(prepped)} prepped complexes"
    )

    # ------------------------------------------------------------------
    # Dock
    # ------------------------------------------------------------------
    log.info("Running POSIT docking...")
    docker = POSITDocker(
        relax_mode=relax_mode,
        posit_method=posit_method,
        use_omega=use_omega,
        omega_dense=False,
        num_poses=num_poses,
        allow_low_posit_prob=True,
        low_posit_prob_thresh=posit_confidence_cutoff,
        allow_final_clash=allow_final_clash,
        allow_retries=allow_retries,
        last_ditch_fred=False,
    )
    results = docker.dock(
        pairs,
        output_dir=output_dir / "docking_results",
        use_dask=False,
        failure_mode=failure_mode,
    )
    log.info(f"Docked {len(results)} pairs successfully")

    if not results:
        raise ValueError("No docking results generated – check structures and inputs.")

    # ------------------------------------------------------------------
    # Write SDF of all poses before filtering
    # ------------------------------------------------------------------
    sdf_path = output_dir / "docking_results.sdf"
    write_results_to_multi_sdf(
        sdf_path,
        results,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=docker.result_cls,
    )
    log.info(f"Wrote all poses to {sdf_path}")

    # ------------------------------------------------------------------
    # Score with ChemGauss4
    # ------------------------------------------------------------------
    log.info("Scoring with ChemGauss4...")
    scorer = MetaScorer(scorers=[ChemGauss4Scorer()])
    scores_df = scorer.score(
        results,
        use_dask=False,
        failure_mode=failure_mode,
        return_df=True,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=docker.result_cls,
        return_for_disk_backend=True,
    )
    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)

    # ------------------------------------------------------------------
    # Filter by POSIT confidence
    # ------------------------------------------------------------------
    n_before = len(scores_df)
    scores_df = scores_df[
        scores_df[DockingResultCols.DOCKING_CONFIDENCE_POSIT.value]
        > posit_confidence_cutoff
    ]
    log.info(
        f"POSIT confidence filter: {len(scores_df)} / {n_before} results kept "
        f"(cutoff={posit_confidence_cutoff})"
    )

    if scores_df.empty:
        warnings.warn(
            "No docking results passed the POSIT confidence cutoff – "
            "raw results written to data_intermediates/docking_scores_raw.csv"
        )

    # ------------------------------------------------------------------
    # Optionally filter clashes (ChemGauss4 > 0 → clash)
    # ------------------------------------------------------------------
    if not allow_final_clash:
        n_before = len(scores_df)
        scores_df = scores_df[
            scores_df[DockingResultCols.DOCKING_SCORE_POSIT.value] <= 0
        ]
        log.info(f"Clash filter: {len(scores_df)} / {n_before} results kept")

    # ------------------------------------------------------------------
    # Sort and write final CSV
    # ------------------------------------------------------------------
    scores_df = scores_df.sort_values(
        DockingResultCols.DOCKING_SCORE_POSIT.value, ascending=True
    )
    scores_df.to_csv(
        data_intermediates / "docking_scores_filtered_sorted.csv", index=False
    )

    final_csv = output_dir / "docking_results_final.csv"
    scores_df.to_csv(final_csv, index=False)
    log.info(f"Wrote final results to {final_csv}")

    return scores_df
