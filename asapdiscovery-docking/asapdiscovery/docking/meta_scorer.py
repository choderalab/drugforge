import numpy as np
from pydantic.v1 import BaseModel, Field

from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.docking.scorer import ScorerBase, Score
from util.dask_utils import FailureMode, BackendType


class MetaScorer(BaseModel):
    """
    Score from a combination of other scorers, the scorers must share an input type,
    """

    scorers: list[ScorerBase] = Field(..., description="Scorers to score with")

    def score(
        self,
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        return_df: bool = False,
        return_for_disk_backend: bool = False,
    ) -> list[Score]:
        """
        Score the inputs using all the scorers provided in the constructor
        """
        results = []
        for scorer in self.scorers:
            vals = scorer.score(
                inputs=inputs,
                use_dask=use_dask,
                dask_client=dask_client,
                failure_mode=failure_mode,
                backend=backend,
                reconstruct_cls=reconstruct_cls,
                return_df=return_df,
                pivot=False,
                return_for_disk_backend=return_for_disk_backend,
            )
            results.append(vals)

        if return_df:
            return Score._combine_and_pivot_scores_df(results)

        return np.ravel(results).tolist()
