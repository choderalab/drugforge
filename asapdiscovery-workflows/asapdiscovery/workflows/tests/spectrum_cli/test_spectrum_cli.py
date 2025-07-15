import os
import traceback

import pytest
from asapdiscovery.workflows.spectrum_workflows.cli import spectrum as cli
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_seq_alignment_pre_calc(blast_xml_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "seq-alignment",
            "-f",
            blast_xml_path,
            "-t",
            "pre-calc",
            "--sel-key",
            "",
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_seq_alignment_multimer(blast_xml_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "seq-alignment",
            "-f",
            blast_xml_path,
            "-t",
            "pre-calc",
            "--sel-key",
            "",
            "--multimer",
            "--n-chains",
            '2',
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)

@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_single_pdb(blast_csv_path, protein_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_path,
            "--pymol-save",
            tmp_path / "file.pse",
            "--chain",
            "both",
            "--color-by-rmsd",
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_one_chain(
    blast_csv_path, protein_path, protein_apo_path, tmp_path
):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_apo_path,
            "--output-dir",
            tmp_path,
            "--pymol-save",
            tmp_path / "file.pse",
            "--chain",
            "A",
            "--color-by-rmsd",
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_struct_dir(
    blast_csv_path, protein_path, structure_dir, tmp_path
):
    runner = CliRunner()
    struct_dir, _ = structure_dir
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--struct-dir",
            struct_dir,
            "--output-dir",
            tmp_path,
            "--pymol-save",
            tmp_path / "file.pse",
            "--chain",
            "both",
            "--color-by-rmsd",
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_cfold_dir(blast_csv_path, protein_path, cfold_dir, tmp_path):
    runner = CliRunner()
    cfold_dir, _ = cfold_dir
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--cfold-results",
            cfold_dir,
            "--output-dir",
            tmp_path,
            "--pymol-save",
            tmp_path / "file.pse",
            "--chain",
            "both",
            "--color-by-rmsd",
            "--cf-format",
            "alphafold2_multimer_v3",
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_fitness_alignment_pairwise(protein_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "fitness-alignment",
            "-t",
            "pwise",
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_path,
            "--pdb-label",
            "ref,pdb",
            "--pymol-save",
            tmp_path / "file.pse",
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_fitness_alignment_fasta(
    fasta_alignment_path, protein_path, protein_mers_path, tmp_path
):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "fitness-alignment",
            "-t",
            "fasta",
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_mers_path,
            "--pdb-label",
            "ref,pdb",
            "--pymol-save",
            tmp_path / "file.pse",
            "--fasta-a",
            fasta_alignment_path,
            "--fasta-b",
            fasta_alignment_path,
            "--fasta-sel",
            "0,4",
        ],
    )
    assert click_success(result)


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