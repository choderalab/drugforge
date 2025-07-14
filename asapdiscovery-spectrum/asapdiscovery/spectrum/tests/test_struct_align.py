import os
import traceback

import pytest
from asapdiscovery.spectrum.align_seq_match import (
    pairwise_alignment,
    save_pymol_seq_align,
)
from asapdiscovery.spectrum.calculate_rmsd import rmsd_alignment, save_alignment_pymol


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_rmsd_alignment(protein_path, protein_apo_path, tmp_path):
    rmsd, pdb_out = rmsd_alignment(
        target_pdb=protein_apo_path,
        ref_pdb=protein_path,
        final_pdb=tmp_path / "file.pdb",
        target_chain="A",
        ref_chain="A",
    )
    assert isinstance(rmsd, float)
    assert pdb_out.exists()


def test_save_alignment(protein_path, protein_apo_path, tmp_path):
    pse_out = tmp_path / "file.pse"
    save_alignment_pymol(
        pdbs=[protein_apo_path],
        labels=["pdb"],
        reference=protein_path,
        session_save=pse_out,
        align_chain="A",
    )
    assert pse_out.exists()


def test_pairwise_alignment(protein_path):
    # Test of pairwise alignment with the same protein file twice
    start_idx = 1
    pdb_align, colorsA, colorsB = pairwise_alignment(
        pdb_file=protein_path,
        pdb_align=protein_path,
        start_idxA=start_idx,
        start_idxB=start_idx,
    )
    assert len(pdb_align) == 1
    assert len(set(colorsA.values())) == 1  # All should be white
    assert len(set(colorsB.values())) == 1
    assert colorsA[start_idx] == "white"
    assert colorsB[start_idx] == "white"


def test_pymol_seq_align(protein_path, tmp_path):
    import MDAnalysis as mda

    u = mda.Universe(protein_path)
    nres = len(u.select_atoms("protein").residues)
    colorsA = {(index + 1): string for index, string in enumerate(["white"] * nres)}
    pse_out = tmp_path / "file.pse"

    save_pymol_seq_align(
        pdbs=[protein_path],
        labels=["ref", "pdb"],
        reference=protein_path,
        color_dict=[colorsA, colorsA],
        session_save=pse_out,
    )
    assert pse_out.exists()