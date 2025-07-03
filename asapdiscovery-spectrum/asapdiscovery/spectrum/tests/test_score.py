import os
import traceback
from pathlib import Path

import pytest
from asapdiscovery.spectrum.score import (
    dock_and_score,
    get_ligand_rmsd,
    score_autodock_vina,
    minimize_structure
)
from asapdiscovery.spectrum.calculate_rmsd import get_binding_site_rmsd
from asapdiscovery.spectrum.cli import spectrum as cli
from asapdiscovery.docking.scorer import ChemGauss4Scorer
from asapdiscovery.data.schema.ligand import Ligand
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_bsite_rmsd_CA(protein_apo_path, protein_path):
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
    lig_rmsd = get_ligand_rmsd(
        ref_pdb=str(protein_path), 
        target_pdb=str(protein_path), 
        addHs=True, 
        rmsd_mode="oechem", 
        overlay=False
    )
    assert lig_rmsd == 0


def test_lig_rmsd_rdkit(protein_path, tmp_path):
    lig_rmsd = get_ligand_rmsd(
        ref_pdb=str(protein_path), 
        target_pdb=str(protein_path), 
        addHs=True, 
        rmsd_mode="oechem", 
        overlay=False,
        pathT=str(tmp_path/"target.sdf"),
        pathR=str(tmp_path/"ref.sdf"),
    )
    assert lig_rmsd == 0


def test_vina_score(target_prepped_vina, ligand_prepped_vina):
    df_vina, out_pose = score_autodock_vina(
        receptor_pdb=target_prepped_vina,
        ligand_sdf=ligand_prepped_vina,
        box_center=[-22,5,25],
        box_size=[20, 20, 20],
        dock=False,
    )
    assert df_vina["Vina-score-premin"].values[0] < 0


def test_minimize():
    return


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_score_docking_only(structure_dir, pdb_file, tmp_path, docking_results_csv_path):  
    runner = CliRunner()
    struct_dir, _ = structure_dir
    csv_save = tmp_path / "scores.csv"
    result = runner.invoke(
        cli,
        [
            "score",
            "-d",
            struct_dir, 
            "-f",
            pdb_file,
            "-o",
            csv_save,
            "--docking-csv",
            docking_results_csv_path,
            "--target",
            "SARS-CoV-2-Mpro",
            "--dock-chain",
            "A",
            "--ref-chain",
            "A"
        ],
    )
    assert csv_save.exists()
    assert click_success(result)


@pytest.mark.xfail(reason="Vina CLI won't run without all the software requirements installed")
def test_score_vina(structure_dir, pdb_file, tmp_path, docking_results_csv_path):
    runner = CliRunner()
    struct_dir, _ = structure_dir
    csv_save = tmp_path / "scores.csv"
    result = runner.invoke(
        cli,
        [
            "score",
            "-d",
            struct_dir, 
            "-f",
            pdb_file,
            "-o",
            tmp_path / "scores.csv",
            "--docking-csv",
            docking_results_csv_path,
            "--dock-chain",
            "1",
            "--ref-chain",
            "A"
            "--vina-score",
            True,
            "--vina-box-x",
            "22",
            "--vina-box-y",
            "5",
            "--vina-box-z",
            "25",
        ],
    )
    assert csv_save.exists()
    assert click_success(result)