import abc
import json
from pathlib import Path
from typing import ClassVar, List, Optional, Set, Union

import mtenn
import numpy as np
import torch
from drugforge.data.backend.openeye import oechem
from drugforge.data.schema.complex import Complex
from drugforge.data.schema.ligand import Ligand
from drugforge.data.services.postera.manifold_data_validation import TargetTags
from drugforge.ml.config import DatasetConfig
from drugforge.ml.dataset import DockedDataset, GraphDataset, SplitDockedDataset
from drugforge.ml.models import (
    ASAPMLModelRegistry,
    LocalMLModelSpecBase,
    MLModelRegistry,
    MLModelSpec,
    MLModelSpecBase,
)

# static import of models from base yaml here
from dgllife.utils import CanonicalAtomFeaturizer
from mtenn.config import (
    LigandOnlyModelConfig,
    ModelConfig,
    ModelType,
    RepresentationType,
    SplitModelConfig,
)
from pydantic import BaseModel, Field

"""
TODO

Need to adjust inference model construction to use new ModelConfigs. Can create one for
each model and store in S3 to use during testing.
"""


class InferenceBase(abc.ABC, BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    targets: Optional[Set[TargetTags]] = Field(
        None,
        description="Targets that them model can predict for",  # FIXME: should be Optional[Set[TargetTags]] but this causes issues with pydantic
    )
    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    representation_type: Optional[RepresentationType] = Field(
        ..., description="Representation type of the underlying model(s)."
    )
    complex_representation_type: Optional[RepresentationType] = Field(
        ...,
        description="Representation type of complex in the underlying split model(s).",
    )
    ligand_representation_type: Optional[RepresentationType] = Field(
        ...,
        description="Representation type of ligand in the underlying split model(s).",
    )
    protein_representation_type: Optional[RepresentationType] = Field(
        ...,
        description="Representation type of protein in the underlying split model(s).",
    )
    model_name: str = Field(..., description="Name of model to use")
    model_spec: Optional[MLModelSpecBase] = Field(
        ..., description="Model spec used to create Model to use"
    )
    local_model_spec: LocalMLModelSpecBase = Field(
        ..., description="Local model spec used to create Model to use"
    )
    device: str = Field("cpu", description="Device to use for inference")
    models: Optional[List[torch.nn.Module]] = Field(..., description="PyTorch model(s)")

    @property
    def is_ensemble(self):
        return len(self.models) > 1

    @property
    def ensemble_size(self):
        return len(self.models)

    @classmethod
    def from_latest_by_target(
        cls,
        target: TargetTags,
        model_registry: MLModelRegistry = ASAPMLModelRegistry,
        representation_type: RepresentationType = None,
        complex_representation_type: RepresentationType = None,
        ligand_representation_type: RepresentationType = None,
        protein_representation_type: RepresentationType = None,
        **kwargs,
    ):
        """
        Create an InferenceBase object from the latest model for the latest target.

        Returns
        -------
        InferenceBase
            InferenceBase object created from latest model for latest target.
        """
        if all(
            [
                representation_type is None,
                complex_representation_type is None,
                ligand_representation_type is None,
                protein_representation_type is None,
            ]
        ):
            model_spec = model_registry.get_latest_model_for_target_and_type(
                target, cls.model_type
            )
        else:
            model_spec = (
                model_registry.get_latest_model_for_target_and_type_and_rep_type(
                    target,
                    cls.model_type,
                    representation_type,
                    complex_representation_type,
                    ligand_representation_type,
                    protein_representation_type,
                )
            )

        if model_spec is None:  # No model found, return None
            return None
        else:
            return cls.from_ml_model_spec(model_spec, **kwargs)

    @classmethod
    def from_latest_by_target_and_endpoint(
        cls,
        target: TargetTags,
        endpoint: str,
        model_registry: MLModelRegistry = ASAPMLModelRegistry,
        **kwargs,
    ):
        """
        Create an InferenceBase object from the latest model for the latest target.

        Returns
        -------
        InferenceBase
            InferenceBase object created from latest model for latest target.
        """
        model_spec = model_registry.get_latest_model_for_target_and_endpoint_and_type(
            target, endpoint, cls.model_type
        )

        if model_spec is None:
            return None
        else:
            return cls.from_ml_model_spec(model_spec, **kwargs)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        model_registry: MLModelRegistry = ASAPMLModelRegistry,
        **kwargs,
    ):
        """
        Create an InferenceBase object from a model name.

        Returns
        -------
        InferenceBase
            InferenceBase object created from model name.
        """
        model_spec = model_registry.get_model(model_name)
        return cls.from_ml_model_spec(model_spec, **kwargs)

    @classmethod
    def from_ml_model_spec(
        cls,
        model_spec: MLModelSpec,
        device: str = "cpu",
        local_dir: Optional[Union[str, Path]] = None,
        build_model_kwargs: Optional[dict] = {},
    ) -> "InferenceBase":
        """
        Create an InferenceBase object from an MLModelSpec.

        Parameters
        ----------
        model_spec : MLModelSpec
            MLModelSpec to use to create InferenceBase object.

        Returns
        -------
        InferenceBase
            InferenceBase object created from MLModelSpec.
        """
        model_components = model_spec.pull(local_dir=local_dir)
        return cls.from_local_model_spec(
            model_components,
            device=device,
            model_spec=model_spec,
            build_model_kwargs=build_model_kwargs,
        )

    @classmethod
    def from_local_model_spec(
        cls,
        local_model_spec: LocalMLModelSpecBase,
        device: str = "cpu",
        model_spec: Optional[MLModelSpec] = None,
        build_model_kwargs: Optional[dict] = {},
    ) -> "InferenceBase":
        """
        Create an InferenceBase object from a LocalMLModelSpec.

        Parameters
        ----------
        local_model_spec : LocalMLModelSpec
            LocalMLModelSpec to use to create InferenceBase object.

        Returns
        -------
        InferenceBase
            InferenceBase object created from LocalMLModelSpec.
        """

        # First make sure mtenn versions are compatible
        if not local_model_spec.check_mtenn_version():
            lower_pin = (
                f">={local_model_spec.mtenn_lower_pin}"
                if local_model_spec.mtenn_lower_pin
                else ""
            )
            upper_pin = (
                f"<{local_model_spec.mtenn_upper_pin}"
                if local_model_spec.mtenn_upper_pin
                else ""
            )
            sep = "," if lower_pin and upper_pin else ""

            raise ValueError(
                f"Installed mtenn version ({mtenn.__version__}) "
                "is incompatible with the version specified in the MLModelSpec "
                f"({lower_pin}{sep}{upper_pin})"
            )

        # Select appropriate Config class
        match local_model_spec.type:
            case ModelType.model:
                config_cls = ModelConfig
            case ModelType.grouped:
                raise ValueError(
                    "Inference is not currently supported for grouped models."
                )
            case ModelType.ligand:
                config_cls = LigandOnlyModelConfig
            case ModelType.split:
                config_cls = SplitModelConfig
            case other:
                raise ValueError(f"Can't instantiate model config for type {other}.")

        models = []
        representation_type = None

        if local_model_spec.ensemble:
            for model in local_model_spec.models:
                config_kwargs = json.loads(model.config_file.read_text())

                # warnings.warn(f"failed to parse model config file, {model.config_file}")
                # config_kwargs = {}
                config_kwargs["model_weights"] = torch.load(
                    model.weights_file, map_location=device
                )
                model_config = config_cls(**config_kwargs)
                cur_rep_type = tuple(
                    r.representation_type if r is not None else None
                    for r in (
                        model_config.representation,
                        model_config.complex_representation,
                        model_config.ligand_representation,
                        model_config.protein_representation,
                    )
                )

                if representation_type is None:
                    representation_type = cur_rep_type
                elif cur_rep_type != representation_type:
                    raise ValueError(
                        "Mismatched model representation types: "
                        f"{representation_type} and {cur_rep_type}."
                    )

                model = model_config.build()
                model.eval()
                models.append(model)
        else:
            config_kwargs = json.loads(local_model_spec.config_file.read_text())
            config_kwargs["model_weights"] = torch.load(
                local_model_spec.weights_file, map_location=device
            )
            model_config = config_cls(**config_kwargs)
            representation_type = tuple(
                r.representation_type if r is not None else None
                for r in (
                    model_config.representation,
                    model_config.complex_representation,
                    model_config.ligand_representation,
                    model_config.protein_representation,
                )
            )

            model = model_config.build()
            model.eval()
            models.append(model)

        # Any issues with appropriately specified representations will be handled in
        #  the model_config.build() step so no need to manually check them here
        (
            representation,
            complex_representation,
            ligand_representation,
            protein_representation,
        ) = representation_type
        return cls(
            targets=local_model_spec.targets,
            representation_type=representation,
            complex_representation_type=complex_representation,
            ligand_representation_type=ligand_representation,
            protein_representation_type=protein_representation,
            model_name=local_model_spec.name,
            model_spec=model_spec,
            local_model_spec=local_model_spec,
            device=device,
            models=models,
        )

    @abc.abstractmethod
    def _check_input_data(self, input_data):
        """
        Make sure that the provided input_data is in the right format for the
        corresponding underlying model.

        Parameters
        ----------
        input_data: dict

        Returns
        -------
        bool
            True if data is ok, False otherwise
        """
        ...

    def predict_ds(self, ds, return_err):
        # always return a 2D array, then we can mask out the err dimension
        data = [self.predict(pose, return_err=True) for _, pose in ds]
        data = np.asarray(data, dtype=np.float32)
        # if it is 1D array, we need to convert to 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        preds = data[:, 0]
        if return_err:
            errs = data[:, 1]
        # return a scalar float value if we only have one input
        if np.all(np.array(preds.shape) == 1):
            preds = preds.item()
            if return_err:
                errs = errs.item()

        else:
            # flatten the array if we have multiple inputs
            preds = preds.flatten()
            if return_err:
                errs = errs.flatten()

        if return_err:
            return preds, errs
        else:
            return preds

    def predict(self, input_data, aggfunc=np.mean, errfunc=np.std, return_err=False):
        """Predict on data, needs to be overloaded in child classes most of
        the time

        Parameters
        ----------

        input_data: dict
            Input data to predict on.
        aggfunc: function, default=np.mean
            Function to aggregate predictions from multiple models.
        errfunc: function, default=np.std
            Function to calculate error from multiple models.
        return_err: bool, default=False
            Return error in addition to prediction.

        Returns
        -------
        np.ndarray
            Prediction from model.
        float
            Error from model.
        """
        if not self._check_input_data(input_data):
            raise ValueError("Passed input_data had incorrect keys.")

        with torch.no_grad():
            # for model ensemble, we need to loop through each model and get the
            # prediction from each, then aggregate
            aggregate_preds = []
            for model in self.models:
                output_tensor = model(input_data)[0].cpu().numpy().flatten()
                aggregate_preds.append(output_tensor)
            if self.is_ensemble:
                aggregate_preds = np.array(aggregate_preds)
                pred = aggfunc(aggregate_preds, axis=0)
                err = errfunc(aggregate_preds, axis=0)
            else:
                # iterates only once, just return the prediction
                pred = output_tensor
                err = np.asarray([np.nan])
            if return_err:
                return pred, err
            else:
                return pred


class ModelInference(InferenceBase):
    model_type: ClassVar[ModelType.model] = ModelType.model

    def _check_input_data(self, input_data):
        """
        Make sure that the provided input_data is in the right format for the
        corresponding underlying model.

        Parameters
        ----------
        input_data: dict

        Returns
        -------
        bool
            True if data is ok, False otherwise
        """
        match self.representation_type:
            case RepresentationType.gat:
                return "g" in input_data
            case RepresentationType.schnet:
                return ("z" in input_data) and ("pos" in input_data)
            case RepresentationType.e3nn:
                return ("x" in input_data) and ("pos" in input_data)
            case RepresentationType.visnet:
                return ("z" in input_data) and ("pos" in input_data)
            case other:
                raise ValueError(f"Unknown representation type {other}")

    def predict_from_smiles(
        self,
        smiles: Union[str, List[str]],
        node_featurizer=None,
        edge_featurizer=None,
        return_err=False,
    ) -> Union[np.ndarray, float]:
        """Predict on a list of SMILES strings, or a single SMILES string.

        Parameters
        ----------
        smiles : Union[str, List[str]]
            SMILES string or list of SMILES strings.
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Predictions for each graph, or a single prediction if only one SMILES string is provided.
        np.ndarray or float
            Errors for each prediction, or a single error if only one SMILES string is provided.
        """
        if self.representation_type != RepresentationType.gat:
            raise ValueError(
                "Predicting from a SMILES string is only supported for GAT-based "
                "inference models."
            )

        if isinstance(smiles, str):
            smiles = [smiles]

        ligands = [
            Ligand.from_smiles(smi, compound_name=f"eval_{i}")
            for i, smi in enumerate(smiles)
        ]

        if not node_featurizer:
            node_featurizer = CanonicalAtomFeaturizer()
        ds = GraphDataset.from_ligands(
            ligands, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer
        )

        return self.predict_ds(ds, return_err)

    def predict_from_structure_file(
        self, pose: Union[Path, List[Path]], return_err=False
    ) -> Union[np.ndarray, float]:
        """Predict on a list of poses or a single pose.

        Parameters
        ----------
        pose : Union[Path, List[Path]]
            Path to pose file or list of paths to pose files.
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Prediction for poses, or a single prediction if only one pose is provided.
        np.ndarray or float
            Error for poses, or a single error if only one pose is provided.
        """
        if self.representation_type == RepresentationType.gat:
            raise ValueError(
                "Predicting from a structure file is not currently supported for "
                "GAT-based inference models."
            )

        if isinstance(pose, Path):
            pose = [pose]

        ds = DockedDataset.from_files(
            pose, compounds=[("pose", str(i)) for i in range(len(pose))]
        )

        if self.representation_type == RepresentationType.e3nn:
            ds = DatasetConfig.fix_e3nn_labels(ds)

        return self.predict_ds(ds, return_err)

    def predict_from_oemol(
        self, pose: Union[oechem.OEMol, List[oechem.OEMol]], return_err=False
    ) -> Union[np.ndarray, float]:
        """
        Predict on a (list of) OEMol objects.

        Parameters
        ----------
        pose : Union[oechem.OEMol, List[oechem.OEMol]]
            (List of) OEMol pose(s)
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Model prediction(s)
        np.ndarray or float
            Model error(s)
        """
        if isinstance(pose, oechem.OEMolBase):
            pose = [pose]

        # Build each complex
        complexes = [
            Complex.from_oemol(
                complex_mol=p,
                target_kwargs={"target_name": "pose"},
                ligand_kwargs={"compound_name": str(i)},
            )
            for i, p in enumerate(pose)
        ]

        # Build each pose from complex
        ds = DockedDataset.from_complexes(complexes)
        if self.representation_type == RepresentationType.e3nn:
            ds = DatasetConfig.fix_e3nn_labels(ds)

        return self.predict_ds(ds, return_err)


class LigandOnlyModelInference(ModelInference):
    model_type: ClassVar[ModelType.ligand] = ModelType.ligand


class SplitModelInference(InferenceBase):
    model_type: ClassVar[ModelType.split] = ModelType.split

    def _check_input_data(self, input_data):
        """
        Make sure that the provided input_data is in the right format for the
        corresponding underlying model.

        Parameters
        ----------
        input_data: dict

        Returns
        -------
        bool
            True if data is ok, False otherwise
        """
        if (
            ("complex" not in input_data)
            or ("protein" not in input_data)
            or ("ligand" not in input_data)
        ):
            return False

        for rep_type in (
            self.complex_representation_type,
            self.ligand_representation_type,
            self.protein_representation_type,
        ):
            match rep_type:
                case RepresentationType.gat:
                    if "g" not in input_data:
                        return False
                case RepresentationType.schnet:
                    if ("z" not in input_data) or ("pos" not in input_data):
                        return False
                case RepresentationType.e3nn:
                    if ("x" not in input_data) or ("pos" not in input_data):
                        return False
                case RepresentationType.visnet:
                    if ("z" not in input_data) or ("pos" not in input_data):
                        return False
                case other:
                    raise ValueError(f"Unknown representation type {other}")

        return True

    def predict_from_structure_file(
        self, pose: Union[Path, List[Path]], return_err=False
    ) -> Union[np.ndarray, float]:
        """Predict on a list of poses or a single pose.

        Parameters
        ----------
        pose : Union[Path, List[Path]]
            Path to pose file or list of paths to pose files.
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Prediction for poses, or a single prediction if only one pose is provided.
        np.ndarray or float
            Error for poses, or a single error if only one pose is provided.
        """
        if self.representation_type == RepresentationType.gat:
            raise ValueError(
                "Predicting from a structure file is not currently supported for "
                "GAT-based inference models."
            )

        if isinstance(pose, Path):
            pose = [pose]

        ds = SplitDockedDataset.from_files(
            pose, compounds=[("pose", str(i)) for i in range(len(pose))]
        )

        if self.representation_type == RepresentationType.e3nn:
            ds = DatasetConfig.fix_e3nn_labels(ds)

        return self.predict_ds(ds, return_err)

    def predict_from_oemol(
        self, pose: Union[oechem.OEMol, List[oechem.OEMol]], return_err=False
    ) -> Union[np.ndarray, float]:
        """
        Predict on a (list of) OEMol objects.

        Parameters
        ----------
        pose : Union[oechem.OEMol, List[oechem.OEMol]]
            (List of) OEMol pose(s)
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Model prediction(s)
        np.ndarray or float
            Model error(s)
        """
        if isinstance(pose, oechem.OEMolBase):
            pose = [pose]

        # Build each complex
        complexes = [
            Complex.from_oemol(
                complex_mol=p,
                target_kwargs={"target_name": "pose"},
                ligand_kwargs={"compound_name": str(i)},
            )
            for i, p in enumerate(pose)
        ]

        # Build each pose from complex
        ds = SplitDockedDataset.from_complexes(complexes)
        if self.representation_type == RepresentationType.e3nn:
            ds = DatasetConfig.fix_e3nn_labels(ds)

        return self.predict_ds(ds, return_err)


_inferences_classes_meta = [
    InferenceBase,
    ModelInference,
    LigandOnlyModelInference,
    SplitModelInference,
]


def get_inference_cls_from_model_type(model_type: ModelType):
    instantiable_classes = [
        m for m in _inferences_classes_meta if m.model_type != ModelType.INVALID
    ]
    model_class = [m for m in instantiable_classes if m.model_type == model_type]
    if len(model_class) != 1:
        raise Exception("Somehow got multiple models")
    return model_class[0]
