import os
import traceback
from pathlib import Path

import pytest
from drugforge.spectrum.score import (
    dock_and_score,
    get_ligand_rmsd,
    score_autodock_vina,
)
from drugforge.spectrum.calculate_rmsd import get_binding_site_rmsd
from drugforge.docking.scorer import ChemGauss4Scorer
from drugforge.data.schema.ligand import Ligand


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_bsite_rmsd_CA(protein_apo_path, protein_path):
    """Test binding site RMSD calculation based on C-alpha atoms.
    """
    rmsd = get_binding_site_rmsd(
        file_ref=protein_path, 
        file_mob=protein_apo_path,
        bsite_dist=4.5,
        rmsd_mode="CA",
        ligres="LIG",
        chain_mob="A",
        chain_ref="A",
    )
    assert rmsd > 0


def test_bsite_rmsd_heavy(protein_apo_path, protein_path):
    """Test binding site RMSD calculation based on heavy atoms."""
    rmsd = get_binding_site_rmsd(
        file_ref=protein_path, 
        file_mob=protein_apo_path,
        bsite_dist=4.5,
        rmsd_mode="heavy",
        ligres="LIG",
        chain_mob="A",
        chain_ref="A",
    )
    assert rmsd > 0


def test_dock_score(protein_path):
    """Test docking and scoring of a protein-ligand complex using ChemGauss4Scorer."""
    scorers = [ChemGauss4Scorer()]
    scores_df, prepped_cmp, ligand_pose, aligned = dock_and_score(
        pdb_complex=protein_path,
        comp_name="MOL",
        target_name="SARS-CoV-2",
        scorers=scorers,
        label="test",
        pdb_ref=None,
        aligned_folder=None,
        allow_clashes=True,
        align_chain="A",
        align_chain_ref="A",
    )
    assert scores_df["docking-score-POSIT"].values[0]
    assert type(ligand_pose) == Ligand
    assert Path(aligned).exists()
    

def test_lig_rmsd_oechem(protein_path): 
    """Test ligand RMSD calculation using OpenEye toolkit method."""
    lig_rmsd = get_ligand_rmsd(
        ref_pdb=str(protein_path), 
        target_pdb=str(protein_path), 
        addHs=True, 
        rmsd_mode="oechem", 
    )
    assert lig_rmsd == 0


def test_lig_rmsd_rdkit(protein_path, tmp_path):
    """Test ligand RMSD calculation using RDKit toolkit method."""
    lig_rmsd = get_ligand_rmsd(
        ref_pdb=str(protein_path), 
        target_pdb=str(protein_path), 
        addHs=True, 
        rmsd_mode="rdkit", 
        pathT=str(tmp_path/"target.sdf"),
        pathR=str(tmp_path/"ref.sdf"),
    )
    assert lig_rmsd == 0


def test_vina_score(target_prepped_vina, ligand_prepped_vina):
    """Test AutoDock Vina scoring workflow of a protein-ligand complex."""
    df_vina, out_pose = score_autodock_vina(
        receptor_pdb=target_prepped_vina,
        ligand_sdf=ligand_prepped_vina,
        box_center=[-22,5,25],
        box_size=[20, 20, 20],
        dock=False,
    )
    assert df_vina["Vina-score-premin"].values[0] < 0
