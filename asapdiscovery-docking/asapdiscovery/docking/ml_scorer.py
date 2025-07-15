from pathlib import Path
from typing import ClassVar, Optional, Any, Union

from mtenn.config import ModelType
from multimethod import multimethod
from pydantic.v1 import Field

from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.docking.scorer import (
    ScoreType,
    ScorerBase,
    ScoreUnits,
    logger,
    Score,
    _get_disk_path_from_docking_result,
)
from asapdiscovery.ml.inference import InferenceBase, get_inference_cls_from_model_type
from asapdiscovery.ml.models import MLModelSpecBase
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.dask_utils import dask_vmap, backend_wrapper


def endpoint_and_model_type_to_score_type(endpoint: str, model_type: str) -> ScoreType:
    """
    Convert an endpoint to a score type.

    Parameters
    ----------
    endpoint : str
        Endpoint to convert

    Returns
    -------
    ScoreType
        Score type
    """
    if model_type == ModelType.GAT:
        if endpoint == "pIC50":  # TODO: make this an enum
            return ScoreType.GAT_pIC50
        elif endpoint == "LogD":
            return ScoreType.GAT_LogD
        else:
            raise ValueError(f"Endpoint {endpoint} not recognized, for GAT")
    elif model_type == ModelType.schnet:
        if endpoint == "pIC50":
            return ScoreType.schnet_pIC50
        else:
            raise ValueError(f"Endpoint {endpoint} not recognized, for Schnet")
    elif model_type == ModelType.e3nn:
        if endpoint == "pIC50":
            return ScoreType.e3nn_pIC50
        else:
            raise ValueError(f"Endpoint {endpoint} not recognized for E3NN")
    else:
        raise ValueError(f"Model type {model_type} not recognized")


_ml_scorer_classes_meta = []


def register_ml_scorer(cls):
    _ml_scorer_classes_meta.append(cls)
    return cls


class MLModelScorer(ScorerBase):
    """
    Baseclass to score from some kind of ML model, including 2D or 3D models
    """

    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    score_type: ScoreType = Field(..., description="Type of score")
    endpoint: Optional[str] = Field(None, description="Endpoint biological property")
    units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    targets: Any = Field(
        ...,
        description="Which targets can this model do predictions for",  # FIXME: Optional[set[TargetTags]]
    )
    model_name: str = Field(..., description="String indicating which model to use")
    inference_cls: InferenceBase = Field(..., description="Inference class")

    @classmethod
    def from_latest_by_target(cls, target: TargetTags):
        if cls.model_type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_latest_by_target(target)
        if inference_instance is None:
            logger.warn(
                f"no ML model of type {cls.model_type} found for target: {target}, skipping"
            )
            return None
        else:
            try:
                instance = cls(
                    targets=inference_instance.targets,
                    model_name=inference_instance.model_name,
                    inference_cls=inference_instance,
                    endpoint=inference_instance.model_spec.endpoint,
                    score_type=endpoint_and_model_type_to_score_type(
                        inference_instance.model_spec.endpoint, cls.model_type
                    ),
                )
                return instance
            except Exception as e:
                logger.error(f"error instantiating MLModelScorer: {e}")
                return None

    @staticmethod
    def from_latest_by_target_and_type(target: TargetTags, type: ModelType):
        """
        Get the latest ML Scorer by target and type.

        Parameters
        ----------
        target : TargetTags
            Target to get the scorer for
        type : ModelType
            Type of model to get the scorer for
        """
        if type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        scorer_class = get_ml_scorer_cls_from_model_type(type)
        return scorer_class.from_latest_by_target(target)

    @classmethod
    def from_model_name(cls, model_name: str):
        if cls.model_type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_model_name(model_name)
        if inference_instance is None:
            logger.warn(
                f"no ML model of type {cls.model_type} found for model_name: {model_name}, skipping"
            )
            return None
        else:
            try:
                instance = cls(
                    targets=inference_instance.targets,
                    model_name=inference_instance.model_name,
                    inference_cls=inference_instance,
                    endpoint=inference_instance.model_spec.endpoint,
                    score_type=endpoint_and_model_type_to_score_type(
                        inference_instance.model_spec.endpoint, cls.model_type
                    ),
                )
                return instance
            except Exception as e:
                logger.error(f"error instantiating MLModelScorer: {e}")
                return None

    @staticmethod
    def load_model_specs(
        models: list[MLModelSpecBase],
    ) -> list["MLModelScorer"]:  # noqa: F821
        """
        Load a list of models into scorers.

        Parameters
        ----------
        models : list[MLModelSpecBase]
            List of models to load
        """
        scorers = []
        for model in models:
            scorer_class = get_ml_scorer_cls_from_model_type(model.type)
            scorer = scorer_class.from_model_name(model.name)
            if scorer is not None:
                scorers.append(scorer)
        return scorers


@register_ml_scorer
class GATScorer(MLModelScorer):
    """
    Scoring using GAT ML Model
    """

    model_type: ClassVar[ModelType.GAT] = ModelType.GAT
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(
        self,
        inputs: Union[list[DockingResult], list[str], list[Ligand]],
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
            gat_score = self.inference_cls.predict_from_smiles(inp.posed_ligand.smiles)
            sc = Score.from_score_and_docking_result(
                gat_score,
                self.score_type,
                self.units,
                inp,
            )
            # overwrite the input with the path to the file
            if return_for_disk_backend:
                sc.input = _get_disk_path_from_docking_result(inp)
            results.append(sc)
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[str], **kwargs) -> list[Score]:
        """
        Dispatch for SMILES strings
        """
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp)
            results.append(
                Score.from_score_and_smiles(
                    gat_score,
                    inp,
                    self.score_type,
                    self.units,
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Ligand], **kwargs) -> list[Score]:
        """
        Dispatch for Ligands
        """
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp.smiles)
            results.append(
                Score.from_score_and_ligand(
                    gat_score,
                    inp,
                    self.score_type,
                    self.units,
                )
            )
        return results


class E3MLModelScorer(MLModelScorer):
    """
    Scoring using ML Models that operate over 3D structures
    These all share an interface so we can use multimethods to dispatch
    for the different input types for all subclasses.
    """

    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(
        self,
        inputs: Union[list[DockingResult], list[Complex], list[Path]],
        return_for_disk_backend: bool = False,
        **kwargs,
    ) -> list[Score]:
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
        results = []
        for inp in inputs:
            score = self.inference_cls.predict_from_oemol(inp.to_posed_oemol())

            sc = Score.from_score_and_docking_result(
                score, self.score_type, self.units, inp
            )
            # overwrite the input with the path to the file
            if return_for_disk_backend:
                sc.input = _get_disk_path_from_docking_result(inp)
            results.append(sc)

        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Complex], **kwargs) -> list[Score]:
        results = []
        for inp in inputs:
            score = self.inference_cls.predict_from_oemol(inp.to_combined_oemol())
            results.append(
                Score.from_score_and_complex(score, self.score_type, self.units, inp)
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Path], **kwargs) -> list[Score]:
        # assuming reading PDB files from disk
        complexes = [
            Complex.from_pdb(
                p,
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
                target_kwargs={"target_name": f"{p.stem}_target"},
            )
            for i, p in enumerate(inputs)
        ]
        return self._dispatch(complexes, **kwargs)


@register_ml_scorer
class SchnetScorer(E3MLModelScorer):
    """
    Scoring using Schnet ML Model
    """

    model_type: ClassVar[ModelType.schnet] = ModelType.schnet
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50


@register_ml_scorer
class E3NNScorer(E3MLModelScorer):
    """
    Scoring using e3nn ML Model
    """

    model_type: ClassVar[ModelType.e3nn] = ModelType.e3nn
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50


def get_ml_scorer_cls_from_model_type(model_type: ModelType):
    instantiable_classes = [
        m for m in _ml_scorer_classes_meta if m.model_type != ModelType.INVALID
    ]
    scorer_class = [m for m in instantiable_classes if m.model_type == model_type]
    if len(scorer_class) != 1:
        raise Exception("Somehow got multiple scorers")
    return scorer_class[0]
