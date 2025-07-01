from pathlib import Path

import pytest

from asapdiscovery.data.backend.openeye import load_openeye_pdb
from asapdiscovery.dataviz.fint_score import FINTScorer
from asapdiscovery.dataviz.plip import compute_fint_score
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.testing.test_resources import fetch_test_file


def test_fint_score():
    fint_score = compute_fint_score(
        load_openeye_pdb(
            Path(
                fetch_test_file(
                    "Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb"
                )
            )
        ),
        MolFileFactory(
            filename=Path(fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"))
        )
        .load()[0]
        .to_oemol(),
        "SARS-CoV-2-Mpro",
    )
    # should return a tuple
    assert isinstance(fint_score, tuple)

    # both should be floats
    assert isinstance(fint_score[0], float)
    assert isinstance(fint_score[1], float)

    # should both fall between 0 and 1
    assert 0 <= fint_score[0] <= 1.0
    assert 0 <= fint_score[1] <= 1.0


@pytest.mark.parametrize(
    "data_fixture", ["results_simple_nolist", "complex_simple", "pdb_simple"]
)
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_FINT_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = FINTScorer(target="SARS-CoV-2-Mpro")
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1
