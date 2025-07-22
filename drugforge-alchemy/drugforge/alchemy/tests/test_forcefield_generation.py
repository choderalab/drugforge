import pytest
from drugforge.alchemy.schema.forcefield import DefaultForceFieldParams
from drugforge.alchemy.utils import create_protein_only_system
from drugforge.data.testing.test_resources import fetch_test_file
from openmm import System


@pytest.mark.parametrize(
    "pdb",
    [
        "Mpro-P2660_0A_bound-prepped_protein.pdb",
        "rcsb_8czv-assembly1-prepped_protein.pdb",
    ],
)
def test_forcefield_generation(pdb):
    pdb_path = fetch_test_file(pdb)

    system = create_protein_only_system(str(pdb_path), DefaultForceFieldParams)
    assert isinstance(system, System)
