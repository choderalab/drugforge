import pytest
from drugforge.data.schema.complex import Complex
from drugforge.modeling.schema import PreppedComplex
from drugforge.data.schema.ligand import Ligand
from drugforge.data.schema.pairs import CompoundStructurePair
from drugforge.data.testing.test_resources import fetch_test_file
from drugforge.docking.docking import DockingInputPair  # TODO: move to data


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


def test_compoundstructure_pair(complex_pdb):
    cmplx = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    lig = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    _ = CompoundStructurePair(complex=cmplx, ligand=lig)


def test_dockinginput_pair_from_compoundstructure_pair(complex_pdb):
    cmplx = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    lig = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    cs_pair = CompoundStructurePair(complex=cmplx, ligand=lig)
    _ = DockingInputPair.from_compound_structure_pair(cs_pair)


def test_dockinginput_pair(complex_pdb):
    cmplx = PreppedComplex.from_complex(
        Complex.from_pdb(
            complex_pdb,
            target_kwargs={"target_name": "test"},
            ligand_kwargs={"compound_name": "test"},
        )
    )
    lig = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    _ = DockingInputPair(complex=cmplx, ligand=lig)
