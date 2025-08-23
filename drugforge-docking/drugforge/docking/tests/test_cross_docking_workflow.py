import os
import traceback
import pytest
from drugforge.docking.workflows.cli import cli
from click.testing import CliRunner
from drugforge.data.testing.test_resources import fetch_test_file


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_cross_docking_cli_structure_directory_du_cache(
    ligand_file, structure_dir, du_cache, tmp_path
):
    runner = CliRunner()
    struct_dir, _ = structure_dir
    du_cache_dir, _ = du_cache
    result = runner.invoke(
        cli,
        [
            "cross-docking",
            "--target",
            "SARS-CoV-2-Mpro",
            "--ligands",
            ligand_file,
            "--structure-dir",
            struct_dir,
            "--cache-dir",
            du_cache_dir,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


# drugforge-docking cross-docking --target SARS-CoV-2-Mpro --pdb-file tyk2_ligand.pdb --ligands ligands.sdf --output-dir tyk2_docked


@pytest.fixture()
def tyk2_complex():
    return fetch_test_file("tyk2_lig_ejm_54.pdb")


@pytest.fixture()
def tyk2_ligands():
    return fetch_test_file("tyk2_ligands.sdf")


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_non_asap_target(tyk2_complex, tyk2_ligands, du_cache, tmp_path):
    runner = CliRunner()
    du_cache_dir, _ = du_cache
    result = runner.invoke(
        cli,
        [
            "cross-docking",
            "--ligands",
            tyk2_ligands,
            "--pdb-file",
            tyk2_complex,
            "--cache-dir",
            du_cache_dir,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)
