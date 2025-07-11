from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.simulation.simulate import OpenMMPlatform
from asapdiscovery.docking.scorer import ChemGauss4Scorer
from asapdiscovery.spectrum.score import (
    ScoreSpectrumInputsBase,
    dock_and_score,
    get_ligand_rmsd,
    score_autodock_vina,
    minimize_structure,
    score_gnina,
)
from asapdiscovery.spectrum.calculate_rmsd import get_binding_site_rmsd

import pandas as pd
from pathlib import Path
import re

import logging
from typing import Optional
from pydantic.v1 import Field

class ScoreInputs(ScoreSpectrumInputsBase):
    """Schema for inputs for scoring complexes

    Parameters
    ----------
    docking_csv : Path
        Path to docking output csv file, from previoud asap-docking step.
    ligand_regex : str
        Pattern for extracting ligand ID from file string.
    protein_regex : str        
        Pattern for extracting protein ID from file string.
    bsite_rmsd : bool
        Whether to calculate binding site RMSD.
    ml_score : bool
        Whether to employ asap-ml models to score poses.
    minimize : bool
        Whether to minimize the pdb structures before running scoring.
    md_openmm_platform : OpenMMPlatform
        OpenMM platform to use for MD minimization
    run_vina : bool = False
        Whether to run Autodock Vina on the docked poses.
    vina_box_x : Optional[float] = None
        X coordinate of the center of the Vina docking box.
    vina_box_y : Optional[float] = None
        Y coordinate of the center of the Vina docking box.
    vina_box_z : Optional[float] = None
        Z coordinate of the center of the Vina docking box.
    dock_vina : bool = False
        Whether to run extra docking step with autodock vina.
    gnina_score : bool = False
        Whether to run Gnina on the docked poses (requires separate installation).
    gnina_script : Path = Path("gnina_score.sh")
        Path to bash script to run gnina.
    gnina_out_dir : Path = Path("gnina_out")
        Path to directory to save gnina output.
    """

    docking_csv: Path = Field(
        None, description="Path to docking output csv file."
    )

    ligand_regex: str = Field(
        "ASAP-[0-9]+", description="Pattern for extracting ligand ID from file string."
    )

    protein_regex: str = Field(
        "YP_[0-9]+_[0-9]+|NP_[0-9]+_[0-9]+", description="Pattern for extracting protein ID from file string."
    )

    bsite_rmsd: bool = Field(
        False, description="Whether to calculate binding site RMSD."
    )

    ml_score: bool = Field(
        False, description="Whether to employ asap-implemented ML models to score poses."
    )

    minimize: bool = Field(
        False,
        description="Whether to minimize the pdb structures before running scoring.",
    )

    md_openmm_platform: OpenMMPlatform = Field(
        OpenMMPlatform.Fastest, description="OpenMM platform to use for MD minimization"
    )

def score_complex_workflow(inputs: ScoreInputs):
    """Run scoring workflow for a set of docked complexes according to spectrum.
    Prepares a csv file with different types of scores, including:
    - Pre-docking score (if available)
    - Docking score
    - ML scores (if requested)
    - RMSD of the ligand
    - RMSD of the binding site (if requested)
    - Affinity score with AutoDock Vina (if requested)
    - Affinity score with Gnina (if requested)

    Parameters
    ----------
    inputs : ScoreInputs
        Input to spectrum scoring workflow

    Returns
    -------
    None
    """
    path_ref = inputs.pdb_ref
    docking_dir = inputs.docking_dir

    comp_name = "MOL"

    if inputs.docking_csv.exists():
        try:
            # Match the protein and ligand regex on docking output file
            logging.info("Reading docking CSV file: %s", inputs.docking_csv)

            df_dock = pd.read_csv(inputs.docking_csv)
            df_dock["lig-ID"] = df_dock["input"].apply(
                lambda s: (
                    re.search(inputs.ligand_regex, s).group(0)
                    if re.search(inputs.ligand_regex, s)
                    else None
                )
            )
            df_dock["prot-ID"] = df_dock["input"].apply(
                lambda s: (
                    re.search(inputs.protein_regex, s).group(0)
                    if re.search(inputs.protein_regex, s)
                    else None
                )
            )
            df_dock = df_dock[["lig-ID", "prot-ID", "docking-score-POSIT"]]
            logging.debug("Docking CSV file processed successfully.")
        except Exception as e:
            print("Error %e", e)
            df_dock = pd.DataFrame({})
    else:
        df_dock = pd.DataFrame({})

    logging.info("Starting scoring for docking directory %s", docking_dir.name)

    first_write = True
    for file_min in docking_dir.glob("*.pdb"):
        # Extracting protein and ligand IDs and pre-calculated scores
        prot_id = re.search(inputs.protein_regex, file_min.stem)
        prot_id = prot_id.group(0) if prot_id else None

        ligand = re.search(inputs.ligand_regex, file_min.stem)
        ligand = ligand.group(0) if ligand else None

        # Default pre-min score if not found
        pre_min_score = 0

        if prot_id is not None:
            try:
                # 1 ligand, multiple targets
                pre_min_score = df_dock.loc[
                    df_dock["prot-ID"] == prot_id, "docking-score-POSIT"
                ].iloc[0]
            except (IndexError, KeyError):
                logging.warning("No pre-docking score found for protein %s", prot_id)
            tag = prot_id
        else:
            tag = ""

        if ligand is not None:
            try:
                # 1 target, multiple ligands
                pre_min_score = df_dock.loc[
                    df_dock["lig-ID"] == ligand, "docking-score-POSIT"
                ].iloc[0]
            except (IndexError, KeyError):
                logging.warning("No pre-docking score found for ligand %s", ligand)

            if prot_id is not None:
                # multiple targets, multiple ligands
                try:
                    pre_min_score = df_dock.loc[
                        (df_dock["lig-ID"] == ligand) & (df_dock["prot-ID"] == prot_id),
                        "docking-score-POSIT",
                    ].iloc[0]
                except (IndexError, KeyError):
                    logging.warning(
                        "No pre-docking score found for ligand %s and protein %s",
                        ligand,
                        prot_id,
                    )
                tag += "_"

            tag += f"{ligand}"

        logging.info(
            "Scoring protein %s and ligand %s from file %s", prot_id, ligand, file_min
        )
        # Reference structure
        if path_ref.is_dir():
            files_in_dir = list(path_ref.glob(f"*{ligand}*.pdb"))
            if len(files_in_dir) > 0:
                file_ref = files_in_dir[0]  # return first find
                logging.info("The ref %s was found for %s", file_ref, tag)
            else:
                logging.error("A reference was not found for %s", tag)
                continue
        else:
            file_ref = path_ref

        # Run minimization if requested
        min_out = file_min
        if inputs.minimize:
            min_folder = docking_dir / "minimized"
            min_folder.mkdir(parents=True, exist_ok=True)
            new_docking_dir = min_folder
            md_openmm_platform = inputs.md_openmm_platform
            try:
                min_out = f"{min_folder}/{tag}_min.pdb"
                logging.info("Running MD minimization of %s", tag)
                minimize_structure(
                    file_min,
                    min_out,
                    min_folder,
                    md_openmm_platform,
                    inputs.target,
                    comp_name,
                )
                chain_dock = "1"  # Standard in OpenMM output file
            except Exception as error:
                logging.error("Couldn't minimize %s: %s", file_min, error)
                continue
        else:
            new_docking_dir = docking_dir
        # Directory to save aligned complexes
        docked_aligned = new_docking_dir / "aligned"
        docked_aligned.mkdir(parents=True, exist_ok=True)

        scorers = [ChemGauss4Scorer()]
        # load addtional ml scorers
        if inputs.ml_score:
            from asapdiscovery.ml.models import ASAPMLModelRegistry
            from asapdiscovery.docking.scorer import MLModelScorer
            logging.info("Loading additional ML scorers")
            # check which endpoints are availabe for the target
            models = ASAPMLModelRegistry.reccomend_models_for_target(inputs.target)
            ml_scorers = MLModelScorer.load_model_specs(models=models)
            scorers.extend(ml_scorers)

        # Prepare complex, re-dock and score
        logging.info("Running protein prep, docking and scoring of %s", min_out)
        scores_df, prepped_cmp, ligand_pose, aligned = dock_and_score(
            min_out,
            comp_name,
            inputs.target,
            scorers,
            label=tag,
            allow_clashes=True,
            pdb_ref=file_ref,
            aligned_folder=docked_aligned,
            align_chain=inputs.dock_chain,
            align_chain_ref=inputs.ref_chain,
        )
        logging.debug(
            "Columns of scoring dataset from asapdiscovery: %s", scores_df.columns
        )
        scores_df["premin-score-POSIT"] = pre_min_score
        df_save = scores_df[["premin-score-POSIT", "docking-score-POSIT"]]
        if inputs.ml_score:  # Add ML scores  
            if scores_df["docking-score-POSIT"].values:
                df_save = scores_df[
                    [
                        "premin-score-POSIT",
                        "docking-score-POSIT",
                        "computed-SchNet-pIC50",
                        "computed-E3NN-pIC50",
                        "computed-GAT-pIC50",
                    ]
                ]
            else:
                df_save = pd.DataFrame(
                    columns=[
                        "premin-score-POSIT",
                        "docking-score-POSIT",
                        "computed-SchNet-pIC50",
                        "computed-E3NN-pIC50",
                        "computed-GAT-pIC50",
                    ]
                )

        if ligand is not None:
            df_save.insert(loc=0, column="lig-ID", value=ligand)
        if prot_id is not None:
            df_save.insert(loc=0, column="prot-ID", value=prot_id)
        logging.debug("The DataFrame after docking is %s", df_save)

        # Now save files for later scoring steps
        docked_prepped = new_docking_dir / "prepped"
        docked_prepped.mkdir(parents=True, exist_ok=True)
        pdb_prep = docked_prepped / (aligned.stem + "_target.pdb")
        sdf_ligand = docked_prepped / (aligned.stem + "_ligand.sdf")
        # Really annoying, but Target and PreppedTarget have different functions for gen the PDB
        if type(prepped_cmp) is Complex:
            prepped_cmp.target.to_pdb(pdb_prep)
        else:
            prepped_cmp.target.to_pdb_file(pdb_prep)
        ligand_pose.to_sdf(sdf_ligand)
        logging.info(
            "Saved prepped target as %s and ligand as %s",
            pdb_prep.stem,
            sdf_ligand.stem,
        )

        # RMSD score
        logging.debug("The aligned file was saved in %s", aligned)
        logging.info("Calculating RMSD of the ligand")
        lig_rmsd = get_ligand_rmsd(
            str(aligned), str(file_ref), True, rmsd_mode="oechem", overlay=False
        )
        df_save.insert(loc=len(df_save.columns), column="Lig-RMSD", value=lig_rmsd)
        pout = f"Calculated RMSD of POSIT ligand pose = {lig_rmsd} with ref {file_ref.stem}"

        if lig_rmsd < 0:
            # Retry ligand rmsd with RDKit
            logging.info("Trying Ligand RMSD on a different method")
            sdf_ref = docked_prepped / (file_ref.stem + "_ligand.sdf")
            lig_rmsd = get_ligand_rmsd(
                str(aligned),
                str(file_ref),
                True,
                rmsd_mode="rdkit",
                overlay=False,
                pathT=str(sdf_ligand),
                pathR=str(sdf_ref),
            )

            if lig_rmsd < 0:
                logging.warning("Retry ligand RMSD calculation failed.")

        if inputs.bsite_rmsd:
            logging.info("Calculating RMSD of the binding site")
            try:
                bsite_rmsd = get_binding_site_rmsd(
                    aligned,
                    file_ref,
                   bsite_dist=5.0,
                    ligres=inputs.lig_resname,
                    chain_mob=inputs.dock_chain,
                    chain_ref=inputs.ref_chain,
                    rmsd_mode="heavy",
                    aligned_temp=aligned,
                )
            except Exception as e:
                bsite_rmsd = -1
                logging.warning("Binding site RMSD couldn't be calculated: %s", e)
            df_save.insert(
                loc=len(df_save.columns), column="Bsite-RMSD", value=bsite_rmsd
            )
            pout += f" and {bsite_rmsd} for binding site"
        logging.info(pout)

        if inputs.run_vina:
            logging.info("Calculating the affinity score with AutoDock Vina")
            if inputs.vina_box_x is None:
                vina_box = None
                logging.info("The grid box will be calculated for the complex")
            else:
                vina_box = [inputs.vina_box_x, inputs.vina_box_y, inputs.vina_box_z]

            df_vina, out_pose = score_autodock_vina(
                pdb_prep,
                sdf_ligand,
                vina_box,
                box_size=[20, 20, 20],
                dock=inputs.dock_vina,
                path_to_prepare_file=str(inputs.path_to_grid_prep),
            )
            if out_pose is not None:
                logging.info("Vina docking pose was successfully generated")
                try:
                    lig_rmsd = get_ligand_rmsd(
                        out_pose,
                        str(file_ref),
                        True,
                        rmsd_mode="oechem",
                        overlay=False,
                    )
                    logging.info("The RMSD of the vina pose was: %s", lig_rmsd)
                except Exception as e:
                    lig_rmsd = -1
                    logging.warning("The vina RMSD couldn't be calculated: %s", e)
                df_vina["Vina-pose-RMSD"] = lig_rmsd
            df_save = pd.concat([df_save, df_vina], axis=1, join="inner")

        if inputs.gnina_score:
            if inputs.gnina_script.exists():
                logging.info("Calculating the affnity with Gnina")
                try:
                    df_gnina = score_gnina(
                        f"{pdb_prep.stem}.pdb",
                        f"{sdf_ligand.stem}.sdf",
                        docked_prepped,
                        inputs.gnina_out_dir,
                        inputs.gnina_script,
                    )
                    logging.debug("Gnina df output is: %s", df_gnina)
                    df_save = pd.concat([df_save, df_gnina], axis=1, join="inner")
                except Exception as e:
                    logging.error("The Gnina calculation failed: %s", e)
            else:
                logging.error(
                    "A gnina bash script must be provided to calculate gnina scores. Won't calculate."
                )
        df_save.to_csv(inputs.output_csv, mode='a', index=False, header=first_write)
        first_write = False

    return