import pytest
from asapdiscovery.docking.scorer import (
    ChemGauss4Scorer,
)


# parametrize over fixtures
@pytest.mark.parametrize(
    "data_fixture", ["results_simple_nolist", "complex_simple", "pdb_simple"]
)
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_chemgauss_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = ChemGauss4Scorer()
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1
