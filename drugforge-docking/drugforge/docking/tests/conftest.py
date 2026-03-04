import pytest
from drugforge.data.backend.openeye import oechem
from drugforge.data.schema.complex import Complex
from drugforge.data.schema.ligand import Ligand
from drugforge.data.testing.test_resources import fetch_test_file
from drugforge.docking.docking import DockingInputMultiStructure, DockingInputPair
from drugforge.docking.openeye import POSITDockingResults
from drugforge.modeling.schema import PreppedComplex


@pytest.fixture()
def ligand():
    """A real-world Ligand loaded from SDF (Mpro-P0008 fragment hit).

    Used as the primary ligand in docking input fixtures and scorer tests
    that exercise code paths requiring a 3D conformer.
    """
    return Ligand.from_sdf(
        fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"), compound_name="test"
    )


@pytest.fixture()
def ligand_simple():
    """A minimal Ligand constructed from a SMILES string (no 3D conformer).

    Used in ``docking_input_pair_simple`` to test docking code paths that
    generate a conformer on-the-fly rather than reading one from disk.
    """
    return Ligand.from_smiles("CCCOCO", compound_name="test2")


@pytest.fixture()
def prepped_complex():
    """A single PreppedComplex loaded from an OE design-unit file (Mpro-P2660).

    The primary receptor fixture.  Used by ``docking_input_pair`` and
    ``docking_input_pair_simple`` to supply a ready-to-dock receptor.
    """
    return PreppedComplex.from_oedu_file(
        fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu"),
        ligand_kwargs={"compound_name": "test"},
        target_kwargs={"target_name": "test", "target_hash": "mock_hash"},
    )


@pytest.fixture()
def prepped_complexes():
    """Two PreppedComplex objects (Mpro-x1002 and Mpro-x0354) loaded from OE design-unit files.

    Used by ``docking_multi_structure`` to exercise multi-receptor docking,
    where POSIT picks the best receptor for a given ligand.
    """
    cached_dus = {
        "Mpro-x1002": "du_cache/Mpro-x1002_0A_bound.oedu",
        "Mpro-x0354": "du_cache/Mpro-x0354_0A_bound.oedu",
    }
    return [
        PreppedComplex.from_oedu_file(
            fetch_test_file(cached_du),
            ligand_kwargs={"compound_name": "test"},
            target_kwargs={"target_name": name, "target_hash": "mock_hash"},
        )
        for name, cached_du in cached_dus.items()
    ]


@pytest.fixture()
def docking_input_pair(ligand, prepped_complex):
    """The canonical DockingInputPair: real SDF ligand paired with the Mpro-P2660 receptor.

    This is the primary input fixture for most docking tests (single-receptor
    docking, caching, multi-pose, timing).
    """
    return DockingInputPair(complex=prepped_complex, ligand=ligand)


@pytest.fixture()
def docking_input_pair_simple(ligand_simple, prepped_complex):
    """A DockingInputPair using the SMILES-only ligand paired with the Mpro-P2660 receptor.

    Used in ``test_docking_dask`` to cover the conformer-generation code path
    alongside the SDF-based ``docking_input_pair``.
    """
    return DockingInputPair(complex=prepped_complex, ligand=ligand_simple)


@pytest.fixture()
def docking_multi_structure(prepped_complexes, ligand):
    """A DockingInputMultiStructure pairing the real SDF ligand with two Mpro receptors.

    Used in ``test_multireceptor_docking`` to verify that POSIT selects the
    best receptor (expected: Mpro-x1002) from the two candidates.
    """
    return DockingInputMultiStructure(complexes=prepped_complexes, ligand=ligand)


@pytest.fixture()
def results():
    """Pre-computed POSITDockingResults for the real SDF ligand, loaded from JSON.

    Used in scorer tests (e.g. ``test_meta_scorer``) and as one half of
    ``results_multi`` to avoid running POSIT during unit tests.
    """
    return [POSITDockingResults.from_json_file(fetch_test_file("docking_results.json"))]


@pytest.fixture()
def results_simple():
    """Pre-computed POSITDockingResults for the SMILES-only ligand, loaded from JSON.

    Used in docking file-write tests (``test_docking_with_file_write``),
    scorer tests, and as the source for ``results_simple_nolist``.
    """
    return [
        POSITDockingResults.from_json_file(
            fetch_test_file("docking_results_simple.json")
        )
    ]


@pytest.fixture()
def results_simple_nolist(results_simple):
    """The single POSITDockingResults object unwrapped from ``results_simple``.

    Passed directly to scorer tests that expect a scalar result rather than a list.
    """
    return results_simple[0]


@pytest.fixture()
def complex_simple():
    """A Complex (non-prepped) loaded from PDB (Mpro-P0008 crystal structure).

    Used in scorer tests that accept a ``Complex`` directly (e.g.
    ``test_chemgauss_scorer``, ``test_schnet_scorer``).
    """
    return Complex.from_pdb(
        fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb"),
        ligand_kwargs={"compound_name": "test"},
        target_kwargs={"target_name": "test", "target_hash": "mock_hash"},
    )


@pytest.fixture()
def pdb_simple():
    """The raw ``Path`` to the Mpro-P0008 PDB file.

    Used in scorer tests that accept a path directly rather than a parsed
    ``Complex`` object (e.g. ``test_chemgauss_scorer``).
    """
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")


@pytest.fixture()
def smiles():
    """A bare SMILES string (``CCCOCO``).

    Used in scorer tests that accept a SMILES string as input (e.g.
    ``test_gat_scorer``).
    """
    return "CCCOCO"


@pytest.fixture()
def results_multi(results, results_simple):
    """Concatenation of ``results`` and ``results_simple`` (two heterogeneous result sets).

    Used in ``test_meta_scorer_df`` to verify that the MetaScorer correctly
    handles multiple result objects with different ligands.
    """
    return results + results_simple


@pytest.fixture()
def mol_with_constrained_confs() -> oechem.OEMol:
    """A multiconformer OEMol (ASAP-0008650) loaded from SDF with isomeric conformer test applied.

    Used in ``test_prune_clashes`` and ``test_select_best_chemgauss`` to verify
    that clash-pruning and Chemgauss scoring correctly reduce the 187-conformer
    ensemble.
    """
    mol = oechem.OEMol()
    ifs = oechem.oemolistream(
        str(fetch_test_file("constrained_conformer/ASAP-0008650.sdf"))
    )
    ifs.SetConfTest(oechem.OEIsomericConfTest())
    oechem.OEReadMolecule(ifs, mol)
    return mol


@pytest.fixture()
def mac1_complex():
    """A PreppedComplex for the MAC1 target, loaded from a cached JSON file.

    Used alongside ``mol_with_constrained_confs`` in constrained pose-generation
    tests (e.g. ``test_prune_clashes``, ``test_select_best_chemgauss``).
    """
    return PreppedComplex.parse_file(
        fetch_test_file("constrained_conformer/complex.json")
    )


# For workflow tests
@pytest.fixture
def ligand_file():
    """Path to the Mpro-P0008 SDF file, used as the ligand input for CLI workflow tests.

    Passed directly to the ``cross-docking`` CLI command via ``--ligands``.
    """
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")


@pytest.fixture()
def all_structure_dir_fns():
    """Relative paths to two Mpro PDB structures (Mpro-x0354 and Mpro-x1002).

    A helper fixture consumed by ``structure_dir`` to build the full resolved
    paths and the parent directory for CLI workflow tests.
    """
    return [
        "structure_dir/Mpro-x0354_0A_bound.pdb",
        "structure_dir/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture()
def structure_dir(all_structure_dir_fns):
    """The resolved structure directory and its list of PDB paths.

    Returns a ``(directory, [Path, ...])`` tuple.  The directory is passed to
    the CLI via ``--structure-dir``; the paths are used to confirm the files
    are present.  Used in ``test_cross_docking_cli_structure_directory_du_cache``.
    """
    all_paths = [fetch_test_file(f) for f in all_structure_dir_fns]
    return all_paths[0].parent, all_paths


@pytest.fixture()
def du_cache_files():
    """Relative paths to two pre-built OE design-unit files (Mpro-x0354 and Mpro-x1002).

    A helper fixture consumed by ``du_cache`` to resolve the full paths and
    parent directory for CLI workflow tests.
    """
    return ["du_cache/Mpro-x0354_0A_bound.oedu", "du_cache/Mpro-x1002_0A_bound.oedu"]


@pytest.fixture()
def du_cache(du_cache_files):
    """The resolved design-unit cache directory and its list of ``.oedu`` paths.

    Returns a ``(directory, [Path, ...])`` tuple.  The directory is passed to
    the CLI via ``--cache-dir``; used in
    ``test_cross_docking_cli_structure_directory_du_cache`` and
    ``test_non_asap_target``.
    """
    all_paths = [fetch_test_file(f) for f in du_cache_files]
    return all_paths[0].parent, all_paths
