import os
import traceback
import pytest
from drugforge.docking.workflows.cli import cross_docking as cli
from click.testing import CliRunner

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