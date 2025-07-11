from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.modeling.schema import PreppedComplex
from asapdiscovery.docking.docking import DockingInputPair
from asapdiscovery.docking.meta_scorer import MetaScorer
from asapdiscovery.docking.analysis import calculate_rmsd_openeye

from asapdiscovery.docking.openeye import POSITDocker

from asapdiscovery.data.backend.openeye import (
    load_openeye_pdb,
    save_openeye_sdf,
)
from asapdiscovery.data.backend.openeye import oechem, split_openeye_mol
from asapdiscovery.spectrum.calculate_rmsd import rmsd_alignment
from asapdiscovery.simulation.simulate import VanillaMDSimulator
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.metadata.resources import active_site_chains

import os
import shutil
from rdkit import Chem
from typing import Union
import pandas as pd
from pathlib import Path
import subprocess

import logging
from typing import Optional
from pydantic.v1 import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)

class ScoreSpectrumInputsBase(BaseModel):
    """Inputs for scoring workflow

    Parameters
    ----------
    docking_dir : Path
        Path to directory where docked structures are stored.
    pdb_ref : Path
        Path to directory or file where crystal structures are stored.
    target : TargetTags
        The target to dock against.
    logname : str
        Name of the log file.
    loglevel : Union[int, str]      
        Logging level
    output_csv : Path
        CSV where scoring results will be stored.
    overwrite : bool    
        Whether to overwrite existing output.
    ref_chain : Optional[str]
        Chain ID to align to in reference structure containing the active site
    dock_chain : Optional[str]
        Active site chain ID to align to ref_chain in reference structure
    lig_resname : Optional[str] 
        Name of residue with Ligand
    run_vina : bool
        Whether to run vina scoring.
    vina_box_x : Optional[float]    
        Coordinate x of vina box
    vina_box_y : Optional[float]
        Coordinate y of vina box
    vina_box_z : Optional[float]
        Coordinate z of vina box
    path_to_grid_prep : Optional[Path]
        Path to file for grid prepping
    dock_vina : bool
        Optionally run extra docking step with autodock vina
    gnina_score : bool
        Whether to run gnina scoring.
    gnina_script : Optional[Path]
        Path to bash script that runs Gnina CLI.
    gnina_out_dir : Optional[Path]      
        Path to directory to process gnina files. 
        Gnina has problems with remote directories so location in $HOME is recommended when running in a remote cluster.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Either especify ALL coordinates of the box, ot the path to grid prepper function
    ValueError
        If Gnina scoring is requested, a gnina_script to run the CLI, and a directory to save intermediate files must be provided
    """
    docking_dir: Path = Field(
        None, description="Path to directory where docked structures are stored."
    )

    pdb_ref: Path = Field(
        None, description="Path to directory/file where crystal structures are stored."
    )

    target: TargetTags = Field(None, description="The target to dock against.")

    logname: str = Field(
        "", description="Name of the log file."
    )  # use root logger for proper forwarding of logs from dask

    loglevel: Union[int, str] = Field(logging.INFO, description="Logging level")

    output_csv: Path = Field(Path("scores.csv"), description="CSV where scoring results will be stored.")

    overwrite: bool = Field(
        False, description="Whether to overwrite existing output."
    )
    ref_chain: Optional[str] = Field(
        None,
        description="Chain ID to align to in reference structure containing the active site",
    )
    dock_chain: Optional[str] = Field(
        None,
        description="Active site chain ID to align to ref_chain in reference structure",
    )
    lig_resname: Optional[str] = Field(
        None,
        description="Name of residue with Ligand",
    )

    # Running Vina
    run_vina: bool = Field(
        False,
        description="Whether to run vina scoring."
    )
    vina_box_x: Optional[float] = Field(
        None,
        description="Coordinate x of vina box"
    ) 
    vina_box_y: Optional[float] = Field(
        None,
        description="Coordinate y of vina box"
    ) 
    vina_box_z: Optional[float] = Field(
        None,
        description="Coordinate z of vina box"
    ) 
    path_to_grid_prep: Optional[Path] = Field(
        None, description="Path to file for grid prepping"
    )
    dock_vina: bool = Field(
        False,
        description="Optionally run extra docking step with autodock vina "
    )
    
    # Running Gnina
    gnina_score: bool = Field(
        False, description="Whether to run gnina scoring."
    )

    gnina_script: Optional[Path] = Field(
        None, description="Path to bash script that runs Gnina CLI."
    )
    
    gnina_out_dir: Optional[Path] = Field(
        None, description="Path to directory to process gnina files. Gnina has problems with remote directories so location in $HOME is recommended when running in a remote cluster."
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
    def check_inputs_vina(cls, values):
        """
        Validate Vina inputs. If vina scoring is requested, either all box coordinates or a path to a grid prepper function must be provided.
        """
        vina_box_x = values.get("vina_box_x")
        vina_box_y = values.get("vina_box_y")
        vina_box_z = values.get("vina_box_z")
        path_to_grid_prep = values.get("path_to_grid_prep")

        if (not vina_box_x or not vina_box_y or not vina_box_z) and not path_to_grid_prep:
            raise ValueError("Either especify ALL coordinates of the box, ot the path to grid prepper function")
    

        return values
    
    @root_validator
    @classmethod
    def check_inputs_gnina(cls, values):
        """
        Validate gnina inputs. A bash script to run the gnina CLI is required, as well as a directory to store gnina outputs.
        """
        gnina_score = values.get("gnina_score")
        gnina_script = values.get("gnina_script")
        gnina_out_dir = values.get("gnina_out_dir")
        
        if gnina_score and (not gnina_script or not gnina_out_dir):
            raise ValueError("If Gnina scoring is requested, a gnina_script to run the CLI, and a directory to save intermediate files must be provided")

        return values

    @root_validator(pre=True)
    def check_and_set_chains(cls, values):
        dock_chain = values.get("dock_chain")
        ref_chain = values.get("ref_chain")
        target = values.get("target")
        lig_resname = values.get("lig_resname")
        if not dock_chain:
            values["dock_chain"] = active_site_chains[target]
        # set same chain for active site if not specified
        if not ref_chain:
            values["ref_chain"] = active_site_chains[target]
        if not lig_resname:
            values["lig_resname"] = "LIG"
        return values

def dock_and_score(
    pdb_complex: Union[Path, str],
    comp_name: str,
    target_name: str,
    scorers: list,
    label: str,
    pdb_ref: Union[Path, str] = None,
    aligned_folder: Path = None,
    allow_clashes: bool = True,
    align_chain: str = "A",
    align_chain_ref: str = "A",
):
    """Re-dock ligand in a complex and return pose scores, using the POSITDocker and given 'scorers'. 
    Optionally aligns complex to a reference structure before docking.

    Parameters
    ----------
    pdb_complex : Union[Path, str]
        PDB path for protein-ligand complex to score
    comp_name : str
        Name of to give to complex. Can be arbitrary.
    target_name : str
        Name of reference target (see asapdiscovery documentation).
    scorers : List
        List with scorer objects. For ChemGauss use ChemGauss4Scorer().
    pdb_ref : Union[Path, str], optional
        PDB of reference structure that will be used to align pdb_complex, by default None
    aligned_folder : Path, optional
        Folder where aligned complex PDB will be save, only if pdb_ref is provided, by default None
    allow_clashes : bool, optional
       Dock allowing clashes on final pose, by default True
    align_chain : str, optional
        Chain by which , by default "A"
    align_chain : str, optional
        Chain in target to align with ref, by default "A"
    align_chain_ref : str, optional
        Chain in ref to align with target, by default "A"

    Returns
    -------
    tuple : (Pandas DataFrame, PreppedComplex, Ligand, Path)
        DataFrame with scores, Prepared complex, Ligand pose, Path to aligned PDB
    """
    if pdb_ref:
        if aligned_folder is not None:
            pdb_complex, aligned = rmsd_alignment(
                str(pdb_complex),
                str(pdb_ref),
                str(aligned_folder / f"{label}_a.pdb"),
                align_chain,
                align_chain_ref,
            )
        else: 
            logging.warning(
                f"A folder to store aligned PDB must be provided if target is to be aligned to ref_pdb. Alignment won't be made."
            )
            aligned = pdb_complex
    else:
        aligned = pdb_complex
    cmp = Complex.from_pdb(
        aligned,
        ligand_kwargs={"compound_name": comp_name},
        target_kwargs={"target_name": target_name},
    )
    try:
        prepped_cmp = PreppedComplex.from_complex(cmp)
        docker = POSITDocker(allow_final_clash=allow_clashes)
        input_pair = DockingInputPair(ligand=cmp.ligand, complex=prepped_cmp)
        results = docker.dock(inputs=[input_pair])
        ligand_pose = results[0].posed_ligand

        metascorer = MetaScorer(scorers=scorers)
        scores_df = metascorer.score(results, return_df=True)
    except (ValueError, RuntimeError, AttributeError, IndexError) as e:
        logger.warning(f"Prep and dock unsuccessful: {e}")
        scores_df = pd.DataFrame([None], columns=["docking-score-POSIT"])
        prepped_cmp = cmp
        ligand_pose = cmp.ligand
    return scores_df, prepped_cmp, ligand_pose, aligned


def ligand_rmsd_oechem(
    refmol: oechem.OEGraphMol, fitmol: oechem.OEGraphMol, overlay=False
):
    """Helper function to calculate ligand RMSD with OEChem"""
    nConfs = 1
    _ = oechem.OEDoubleArray(nConfs)
    automorf = True
    heavyOnly = True
    rotmat = oechem.OEDoubleArray(9 * nConfs)
    transvec = oechem.OEDoubleArray(3 * nConfs)

    success = oechem.OERMSD(
        refmol, fitmol, automorf, heavyOnly, overlay, rotmat, transvec
    )
    if not success:
        logger.warning(f"RMSD calculation failed")
    return success


def ligand_rmsd_rdkit(target_sdf, ref_sdf):
    """Helper function to calculate ligand RMSD with RDKit"""
    target_sdf = str(target_sdf)
    ref_sdf = str(ref_sdf)
    with Chem.SDMolSupplier(target_sdf) as supp:
        mol_target = supp[0]
    with Chem.SDMolSupplier(ref_sdf) as supp:
        mol_ref = supp[0]
    try:
        rmsd = Chem.rdMolAlign.CalcRMS(mol_target, mol_ref)
    except Exception:
        try:
            rmsd = Chem.AllChem.AlignMol(mol_target, mol_ref)
        except Exception:
            rmsd = -1
            logger.warning(f"RMSD calculation failed")
    return rmsd


def get_ligand_rmsd(
    target_pdb: str,
    ref_pdb: str,
    addHs=True,
    pathT="",
    pathR="",
    rmsd_mode="oechem",
    overlay=False,
) -> float:
    """Calculate RMSD of a molecule against a reference

    Parameters
    ----------
    target_pdb : str
        Path to PDB of protein with ligand to align.
    ref_pdb : str
        Path to PDB to align target against.
    addHs : bool, optional
        Add explicit Hs with OEChem tools, by default True
    pathT : str, optional
        Temporary path to save the protein target pdb, as needed for rdkit rmsd mode, by default ""
    pathR : str, optional
        Temporary path to save the ligand sdf, as needed for rdkit rmsd mode, by default ""
    rmsd_mode : str, optional
        Tool to use for RMSD calculation between ["oechem", "rdkit"], by default "oechem"
    overlay : bool, optional
        Whether to overlay pose for RMSD, by default False

    Returns
    -------
    float
        RMSD after alignment

    Raises
    ------
    ValueError
        When pathT and pathR aren't provided in rdkit mode.
    NotImplementedError
        When incorrect rmsd_mode is provided.
    """
    target_complex = load_openeye_pdb(target_pdb)
    ref_complex = load_openeye_pdb(ref_pdb)

    target_dict = split_openeye_mol(target_complex)
    ref_dict = split_openeye_mol(ref_complex)

    # Add Hs
    target_lig = target_dict["lig"]
    ref_lig = ref_dict["lig"]
    if addHs:
        oechem.OEAddExplicitHydrogens(target_lig)
        oechem.OEAddExplicitHydrogens(ref_lig)
    else:
        oechem.OESuppressHydrogens(target_lig)
        oechem.OESuppressHydrogens(ref_lig)

    path_target = path_ref = ""
    if pathT and pathR:
        path_target = save_openeye_sdf(target_lig, pathT)
        path_ref = save_openeye_sdf(ref_lig, pathR)

    if ref_lig.NumAtoms() != target_lig.NumAtoms():
        logger.warning(
            f"Ref({ref_lig.NumAtoms()}) and  target({target_lig.NumAtoms()}) ligands have different number of atoms"
        )

    if not (pathT and pathR) and (rmsd_mode == "rdkit" or rmsd_mode == "both"):
        raise ValueError(
            "for rdkit mode. a path to save/load sdf mols must be provided"
        )

    rmsd_oechem = calculate_rmsd_openeye(ref_lig, target_lig) # ligand_rmsd_oechem(ref_lig, target_lig, overlay)
    if rmsd_mode == "oechem":
        return rmsd_oechem
    elif rmsd_mode == "rdkit":
        rmsd_rdkit = ligand_rmsd_rdkit(path_target, path_ref)
        return rmsd_rdkit
    elif rmsd_mode == "both":
        rmsd_rdkit = ligand_rmsd_rdkit(path_target, path_ref)
        return [rmsd_oechem, rmsd_rdkit]
    else:
        raise NotImplementedError("Must provide a valid value for rmsd_mode")


def score_autodock_vina(
    receptor_pdb: Union[str, Path],
    ligand_sdf: Path,
    box_center = None,
    box_size = [20, 20, 20],
    dock = False,
    path_to_prepare_file = "./",
):
    """ Score ligand pose with AutoDock Vina. 
    This function will take a receptor PDB and a ligand SDF, and prepare them to pdbqt files with the MGLTools, which are needed for Vina.
    If the receptor and/or ligand is already in pdbqt format, it will be used as is. 
    The dimensions of the grid box for Vina can be specified or calculated, provided a path to a grid file (which can be downloaded from Vina).
    A dataframe with the scores will be returned, including the scores before and after minimization, as well as the path to a Vina docked pose if dock=True.

    Parameters
    ----------
    receptor_pdb : Path
        Path to pdb of target (no ligand).
    ligand_sdf : Path
        Path to sdf of ligand.
    box_center : List, optional
        Center of ligand box as [x, y, z], by default None and the box will be calculated.
    box_size : list, optional
        Size of docking box, by default [20, 20, 20]
    dock : bool, optional
        Whether to redock ligand with AutoDock Vina, by default False
    path_to_prepare_file : str, optional
        Path to Python file which prepares ligand box if not provided (copied from AutoDock Vina repo), by default "./"

    Returns
    -------
    tuple: (pd.DataFrame, str)
        (DataFrame with scores, path to docked pose)

    Raises
    ------
    ValueError
        Path to target file is neither of pdb or pdbqt allowed formats
    ValueError
        Path to ligand file is neither of sdf or pdbqt allowed formats
    """
    from vina import Vina

    df_scores = pd.DataFrame(index=[0])

    receptor_pdb = Path(receptor_pdb)
    ligand_sdf = Path(ligand_sdf)
    if receptor_pdb.suffix == ".pdb":
        # Prepare receptor
        receptor_pdbqt = receptor_pdb.with_suffix("pdbqt")
        subprocess.run(
            f"prepare_receptor -r {receptor_pdb} -o {receptor_pdbqt}", shell=True
        )
    elif receptor_pdb.suffix == ".pdbqt":
        receptor_pdbqt = receptor_pdb
        logger.info(f"Prepped target provided")
    else:
        raise ValueError("Only allowed formats are .pdb and .pdbqt")
    # Prepare ligand
    if ligand_sdf.suffix == ".sdf":
        ligand_pdbqt = ligand_sdf.with_suffix("pdbqt")
        subprocess.run(
            f"mk_prepare_ligand.py -i {ligand_sdf} -o {ligand_pdbqt}",
            shell=True,
        )
    elif ligand_sdf.suffix == ".pdbqt":
        ligand_pdbqt = ligand_sdf
        logger.info(f"Prepped ligand provided")
    else:
        raise ValueError("Only allowed formats are .pdb and .pdbqt")
    v = Vina(sf_name="vina")

    # First check if prep was successful
    if (
        not receptor_pdbqt.is_file()
        or not ligand_pdbqt.is_file()
    ):
        df_scores["Vina-score-premin"] = None
        df_scores["Vina-score-min"] = None
        if dock:
            df_scores["Vina-dock-score"] = None
        return df_scores, None

    # get coordinates of box
    if box_center is None:
        parent_dir = ligand_sdf.resolve().parents[0]
        p = subprocess.Popen(
            f"pythonsh {path_to_prepare_file}/prepare_gpf.py -l {ligand_pdbqt} -r {receptor_pdbqt} -y",
            cwd=parent_dir,
            shell=True,
            stdout=subprocess.PIPE,
        )
        (output, err) = p.communicate()
        # The grid needs some time to compute
        p.wait()
        x, y, z = -1, -1, -1
        with open(f"{parent_dir/receptor_pdb.stem}.gpf", "r") as f:
            for line in f:
                if line.startswith("gridcenter"):
                    # Split the line into columns
                    comps = line.split()
                    x = float(comps[1])
                    y = float(comps[2])
                    z = float(comps[3])
                    break
        if x<0 and y<0 and z<0:
            logger.warning(f"Could not generate grid box for Vina calculation because .gpf file was incorrect."
            )
            df_scores["Vina-score-premin"] = None
            df_scores["Vina-score-min"] = None
            if dock:
                df_scores["Vina-dock-score"] = None
            return df_scores, None
        box_center = [x, y, z]
    v.set_receptor(str(receptor_pdbqt))

    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(center=box_center, box_size=box_size)

    # Score the current pose
    energy = v.score()
    logger.info(
        f"Score before minimization: {energy[0]} (kcal/mol)"
    )
    df_scores["Vina-score-premin"] = energy[0]

    # Minimized locally the current pose
    energy_minimized = v.optimize()
    logger.info(
        f"Score after minimization: {energy_minimized[0]} (kcal/mol)"
    )
    df_scores["Vina-score-min"] = energy_minimized[0]
    parent_dir = receptor_pdb.resolve().parents[0]
    v.write_pose(f"{parent_dir/receptor_pdb.stem}_minimized.pdbqt", overwrite=True)
    out_pose = None

    if dock:
        # Dock the ligand
        v.dock(exhaustiveness=32, n_poses=20)
        v.write_poses(
            f"{receptor_pdb.stem}_vina_out.pdbqt", n_poses=1, overwrite=True
        )
        df_scores["Vina-dock-score"] = v.score()[0]
        # Convert pose in pdbqt to calculate rmsd
        out_pose = f"{parent_dir/receptor_pdb.stem}_vina_out.pdb"
        subprocess.run(
            f"babel -ipdbqt '{parent_dir/receptor_pdb.stem}_vina_out.pdbqt' -opdb '{out_pose}'",
            shell=True,
        )
    return df_scores, out_pose


def score_gnina(pdb_target:str, 
                sdf_ligand:str, 
                pdb_dir:str, 
                home_dir:str,
                gnina_script:str,
) -> pd.DataFrame:
    """Score a ligand pose with Gnina CNN scoring function.

    Parameters
    ----------
    pdb_target : str
        Path to PDB of target
    sdf_ligand : str
        Path to SDF of ligand
    pdb_dir : str
        Directory where prepped pdbs are located.
    home_dir : str
        Directory where gnina will be run. This directory should be in $HOME, as gnina has problems with remote directories.
    gnina_script : str
        Path to bash script that runs Gnina CLI. 

    Returns
    -------
    pd.DataFrame
        DataFrame with gnina scores. Columns are:
        - gnina-RMSD: RMSD of the ligand pose to the crystal structure.
        - gnina-Affinity: Gnina affinity score.
        - gnina-Affinity-var: Gnina affinity score variance.
        - CNNscore: Gnina CNN score.
        - CNNaffinity: Gnina CNN affinity score.
        - CNNvariance: Gnina CNN affinity score variance.
    
    """
    logfile = f"out_{pdb_target[:-4]}.log"
    env = os.environ.copy()
    env["SDF"] = sdf_ligand
    env["PDB"] = pdb_target
    env["PDB_DIR"] = pdb_dir
    env["home_data"] = home_dir
    env["LOGFILE"] = logfile

    process = subprocess.Popen(
        ["bash", gnina_script],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    # keep only the last line
    last_line = None
    for line in process.stdout:
        last_line = line.strip()
    process.wait()
    if last_line:
        data = [last_line.split(",")]
        df = pd.DataFrame(
            data,
            columns=[
                "gnina-RMSD",
                "gnina-Affinity",
                "gnina-Affinity-var",
                "CNNscore",
                "CNNaffinity",
                "CNNvariance",
            ],
        )
    else:
        df = pd.DataFrame(
            columns=[
                "gnina-RMSD",
                "gnina-Affinity",
                "gnina-Affinity-var",
                "CNNscore",
                "CNNaffinity",
                "CNNvariance",
            ]
        )
    return df


def minimize_structure(
    pdb_complex: Union[Path, str],
    min_out: Union[Path, str],
    out_dir: Union[Path, str],
    md_platform: str,
    comp_name: str,
    target_name: str,
) -> Union[Path, str]:
    """MD energy minimization a protein ligand complex. 
    Energy minimization is performed with OpenMM (no equilibration or production MD is performed).

    Parameters
    ----------
    pdb_complex : Union[Path, str]
        Path to protein ligand complex pdb
    min_out : Union[Path, str]
        Output file with minimized pdb
    out_dir : Union[Path, str]
        Directory to save output minimized complex
    md_platform : str
        MD OpenMM platform [CPU, CUDA, OpenCL, Reference, Fastest]
    comp_name : str
        Name of to give to complex. Can be arbitrary.
    target_name : str
        Name of reference target (see asapdiscovery documentation).

    Returns
    -------
    str
       Path to minimized file
    """

    if Path(min_out).is_file():
        logger.warning(
            f"The file {min_out} already exists. The minimization will be skipped"
        )
        return min_out
    cmp = Complex.from_pdb(
        pdb_complex,
        ligand_kwargs={"compound_name": comp_name},
        target_kwargs={"target_name": target_name},
    )
    out_dir = Path(out_dir)
    prepped_cmp = PreppedComplex.from_complex(cmp)
    prepped_cmp.target.to_pdb_file(out_dir / "target.pdb")
    cmp.ligand.to_sdf(out_dir / "ligand.sdf")

    md_simulator = VanillaMDSimulator(
        output_dir=out_dir,
        openmm_platform=md_platform,
        minimize_only=True,
        num_steps=1,
    )
    simulation_results = md_simulator.simulate(
        [(out_dir / "target.pdb", out_dir / "ligand.sdf")],
        outpaths=[out_dir],
        failure_mode="skip",
    )
    min_path = simulation_results[0].minimized_pdb_path
    shutil.move(min_path, min_out)
    shutil.rmtree(f"{out_dir}/target_ligand", ignore_errors=True)
    for file_path in [f"{out_dir}/target.pdb", f"{out_dir}/ligand.sdf"]:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    return min_out
