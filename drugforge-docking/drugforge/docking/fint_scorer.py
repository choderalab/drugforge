from pathlib import Path
from typing import ClassVar, Union

from multimethod import multimethod
from pydantic.v1 import Field, validator

from drugforge.dataviz.plip import compute_fint_score
from drugforge.docking.docking import DockingResult
from drugforge.docking.scorer import (
    ScorerBase,
    ScoreType,
    ScoreUnits,
    Score,
    _get_disk_path_from_docking_result,
)
from drugforge.spectrum.fitness import target_has_fitness_data
from drugforge.data.schema.complex import Complex
from drugforge.data.services.postera.manifold_data_validation import TargetTags
from drugforge.data.util.dask_utils import dask_vmap, backend_wrapper


class FINTScorer(ScorerBase):
    """
    Score using Fitness Interaction Score

    Overloaded to accept DockingResults, Complexes, or Paths to PDB files.
    """

    score_type: ScoreType = Field(ScoreType.FINT, description="Type of score")
    units: ClassVar[ScoreUnits.arbitrary] = ScoreUnits.arbitrary
    target: TargetTags = Field(..., description="Which target to use for scoring")

    @validator("target")
    @classmethod
    def validate_target(cls, v):
        if not target_has_fitness_data(v):
            raise ValueError(
                "target does not have fitness data so cannot use FINTScorer"
            )
        return v

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(
        self,
        inputs: Union[list[DockingResult], list[Complex], list[Path]],
        return_for_disk_backend: bool = False,
        **kwargs,
    ) -> list[Score]:
        """
        Score the inputs, dispatching based on type.
        """
        return self._dispatch(
            inputs, return_for_disk_backend=return_for_disk_backend, **kwargs
        )

    @multimethod
    def _dispatch(
        self,
        inputs: list[DockingResult],
        return_for_disk_backend: bool = False,
        **kwargs,
    ) -> list[Score]:
        """
        Dispatch for DockingResults
        """
        results = []
        for inp in inputs:
            _, fint_score = compute_fint_score(
                inp.to_protein(), inp.posed_ligand.to_oemol(), self.target
            )

            sc = Score.from_score_and_docking_result(
                fint_score, self.score_type, self.units, inp
            )
            # overwrite the input with the path to the file
            if return_for_disk_backend:
                sc.input = _get_disk_path_from_docking_result(inp)

            results.append(sc)

        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Complex], **kwargs):
        """
        Dispatch for Complexes
        """
        results = []
        for inp in inputs:
            _, fint_score = compute_fint_score(
                inp.target.to_oemol(), inp.ligand.to_oemol(), self.target
            )
            results.append(
                Score.from_score_and_complex(
                    fint_score, self.score_type, self.units, inp
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Path], **kwargs):
        """
        Dispatch for PDB files from disk
        """
        # assuming reading PDB files from disk
        complexes = [
            Complex.from_pdb(
                p,
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
                target_kwargs={"target_name": f"{p.stem}_target"},
            )
            for p in inputs
        ]

        return self._dispatch(complexes, **kwargs)
