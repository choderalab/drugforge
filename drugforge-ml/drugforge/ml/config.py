from __future__ import annotations

import json
import pickle as pkl
from collections.abc import Iterator
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from drugforge.data.schema.complex import Complex
from drugforge.data.schema.experimental import ExperimentalCompoundData
from drugforge.data.schema.ligand import Ligand
from drugforge.data.util.stringenum import StringEnum
from drugforge.data.util.utils import extract_compounds_from_filenames
from drugforge.ml.dataset import (
    DockedDataset,
    GraphDataset,
    GroupedDockedDataset,
    SplitDockedDataset,
)
from drugforge.ml.es import (
    BestEarlyStopping,
    ConvergedEarlyStopping,
    PatientConvergedEarlyStopping,
)
from pydantic import (
    BaseModel,
    Field,
    confloat,
    field_serializer,
    field_validator,
    model_validator,
)


class ConfigBase(BaseModel):
    """
    Base class to provide update functionality.
    """

    def update(self, config_updates={}) -> ConfigBase:
        return self._update(config_updates)

    def _update(self, config_updates={}) -> ConfigBase:
        """
        Default version of this function. Just update original config with new options,
        and generate new object. Designed to be overloaded if there are specific things
        that a class needs to handle (see GATModelConfig as an example).
        """

        orig_config = self.model_dump()

        # Get new config by overwriting old stuff with any new stuff
        new_config = orig_config | config_updates

        return type(self)(**new_config)


class OptimizerType(StringEnum):
    """
    Enum for training optimizers.
    """

    sgd = "sgd"
    adam = "adam"
    adadelta = "adadelta"
    adamw = "adamw"


class OptimizerConfig(ConfigBase):
    """
    Class for constructing an ML optimizer. All parameter defaults are their defaults in
    pytorch.

    NOTE: some of the parameters have different defaults between different optimizers,
    need to figure out how to deal with that
    """

    optimizer_type: OptimizerType = Field(
        OptimizerType.adam,
        description=(
            "Type of optimizer to use. "
            f"Options are [{', '.join(OptimizerType.get_values())}]."
        ),
    )
    # Common parameters
    lr: float = Field(0.0001, description="Optimizer learning rate.")
    weight_decay: confloat(ge=0.0, allow_inf_nan=False) = Field(
        0, description="Optimizer weight decay (L2 penalty)."
    )

    # SGD-only parameters
    momentum: float = Field(0, description="Momentum for SGD optimizer.")
    dampening: float = Field(0, description="Dampening for momentum for SGD optimizer.")

    # Adam* parameters
    b1: float = Field(0.9, description="B1 parameter for Adam and AdamW optimizers.")
    b2: float = Field(0.999, description="B2 parameter for Adam and AdamW optimizers.")
    eps: float = Field(
        1e-8, description="Epsilon parameter for Adam, AdamW, and Adadelta optimizers."
    )

    # Adadelta parameters
    rho: float = Field(0.9, description="Rho parameter for Adadelta optimizer.")

    def build(
        self, parameters: Iterator[torch.nn.parameter.Parameter]
    ) -> torch.optim.Optimizer:
        """
        Build the Optimizer object.

        Parameters
        ----------
        parameters : Iterator[torch.nn.parameter.Parameter]
            Model parameters that will be adjusted by the optimizer

        Returns
        -------
        torch.optim.Optimizer
        Optimizer object
        """
        match self.optimizer_type:
            case OptimizerType.sgd:
                return torch.optim.SGD(
                    parameters,
                    lr=self.lr,
                    momentum=self.momentum,
                    dampening=self.dampening,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.adam:
                return torch.optim.Adam(
                    parameters,
                    lr=self.lr,
                    betas=(self.b1, self.b2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.adadelta:
                return torch.optim.Adadelta(
                    parameters,
                    rho=self.rho,
                    eps=self.eps,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.adamw:
                return torch.optim.AdamW(
                    parameters,
                    lr=self.lr,
                    betas=(self.b1, self.b2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case optimizer_type:
                # Shouldn't be possible but just in case
                raise ValueError(f"Unknown value for optimizer_type: {optimizer_type}")


class EarlyStoppingType(StringEnum):
    """
    Enum for early stopping classes.
    """

    none = "none"
    best = "best"
    converged = "converged"
    patient_converged = "patient_converged"


class EarlyStoppingConfig(ConfigBase):
    """
    Class for constructing an early stopping class.
    """

    es_type: EarlyStoppingType = Field(
        EarlyStoppingType.none,
        description=(
            "Type of early stopping to use. "
            f"Options are [{', '.join(EarlyStoppingType.get_values())}]."
        ),
    )
    # Parameters for best
    patience: int = Field(
        None,
        description=(
            "The maximum number of epochs to continue training with no improvement in "
            "the val loss. Used only in BestEarlyStopping."
        ),
    )
    # Parameters for converged
    n_check: int = Field(
        None,
        description=(
            "Number of past epochs to keep track of when calculating divergence. "
            "Used only in ConvergedEarlyStopping."
        ),
    )
    divergence: float = Field(
        None,
        description=(
            "Max allowable difference from the mean of the losses. "
            "Used only in ConvergedEarlyStopping."
        ),
    )
    burnin: int = Field(
        0,
        description=(
            "Minimum number of epochs to train for regardless of early "
            "stopping criteria."
        ),
    )

    @model_validator(mode="after")
    def check_args(self):
        match self.es_type:
            case EarlyStoppingType.none:
                pass
            case EarlyStoppingType.best:
                if self.patience is None:
                    raise ValueError(
                        "Value required for patience when using BestEarlyStopping."
                    )
            case EarlyStoppingType.converged:
                if (self.n_check is None) or (self.divergence is None):
                    raise ValueError(
                        "Values required for n_check and divergence when using "
                        "ConvergedEarlyStopping."
                    )
            case EarlyStoppingType.patient_converged:
                if (
                    (self.n_check is None)
                    or (self.divergence is None)
                    or (self.patience is None)
                ):
                    raise ValueError(
                        "Values required for n_check, divergence, and patience when "
                        "using PatientConvergedEarlyStopping."
                    )
            case other:
                raise ValueError(f"Unknown EarlyStoppingType: {other}")

        return self

    def build(
        self,
    ) -> BestEarlyStopping | ConvergedEarlyStopping | PatientConvergedEarlyStopping:
        match self.es_type:
            case EarlyStoppingType.none:
                return None
            case EarlyStoppingType.best:
                return BestEarlyStopping(self.patience, self.burnin)
            case EarlyStoppingType.converged:
                return ConvergedEarlyStopping(
                    self.n_check, self.divergence, self.burnin
                )
            case EarlyStoppingType.patient_converged:
                return PatientConvergedEarlyStopping(
                    self.n_check, self.divergence, self.patience, self.burnin
                )
            case other:
                raise ValueError(f"Unknown EarlyStoppingType: {other}")


class DatasetType(StringEnum):
    """
    Enum for different Dataset types.
    """

    graph = "graph"
    structural = "structural"
    split = "split"


class DatasetConfig(ConfigBase):
    """
    Class for constructing an ML Dataset class.
    """

    # Graph or structure-based dataset
    ds_type: DatasetType = Field(
        ...,
        description=(
            "Type of dataset to build. "
            f"Options are [{', '.join(DatasetType.get_values())}]."
        ),
    )

    # Required inputs used to build the dataset
    exp_data: dict[str, dict[str, Any]] = Field(
        {},
        description=(
            "Dict mapping from compound_id to another dict containing "
            "experimental data."
        ),
    )
    input_data: list[Complex] | list[Ligand] | None = Field(
        None,
        description=(
            "List of Complex objects (structure-based models) or Ligand objects "
            "(graph-based models), which will be used to generate the input structures."
        ),
    )
    export_input_data: bool = Field(
        False,
        description=(
            "Whether to actually save the input_data. Note that if this is "
            "True, you are essentially saving multiple SDF/PDB files. If this is set "
            "to False (default), a value must be provided for cache_file, otherwise "
            "constructing the dataset will be impossible."
        ),
    )
    export_exp_data: bool = Field(
        False,
        description=(
            "Whether to actually save the exp_data. If this is set to False (default), "
            "a value must be provided for cache_file, otherwise constructing the "
            "dataset will be impossible."
        ),
    )

    # Cache file to save to/load from. Probably want to move away from pickle eventually
    cache_file: Path | None = Field(
        None, description="Pickle cache file of the actual dataset object."
    )
    cache_protein: bool = Field(
        False,
        description=(
            "Actually save the coordinates when caching the Dataset. Leaving this as "
            "False will make loading a bit longer, but will save a significant amount "
            "of disk space when caching."
        ),
    )

    # Parallelize data processing
    num_workers: int = Field(
        1, description="Number of threads to use for dataset processing."
    )

    # Multi-pose or not
    grouped: bool = Field(False, description="Build a GroupedDockedDataset.")

    # Dataset will be used in an e3nn model, so make sure the fields have the correct
    #  names
    for_e3nn: bool = Field(False, description="Dataset will be used in an e3nn model.")

    # Don't use (and overwrite) any existing cache_file
    overwrite: bool = Field(False, description="Overwrite any existing cache_file.")

    # Torch device to send loaded dataset to
    device: torch.device = Field("cpu", description="Device to send loaded dataset to.")

    # Randomly iterate through the dataset each time
    random_iter: bool = Field(
        False, description="Randomly iterate through the dataset each time."
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("device", when_used="always")
    def serialize_torch_device(self, device: torch.device):
        return str(device)

    @model_validator(mode="after")
    def check_input_data_exists(self):
        if self.input_data is None:
            if self.cache_file is None:
                raise ValueError(
                    "A value must be supplied for cache_file if no data is passed to "
                    "input_data."
                )
            if not self.cache_file.exists():
                raise ValueError(
                    "If no data is passed to input_data the passed cache_file must "
                    "exist."
                )

        return self

    @model_validator(mode="after")
    def check_data_type(self):
        # Just return here, make sure cache_file is supplied in a different function
        if self.input_data is None:
            return self

        inp = self.input_data[0]
        match self.ds_type:
            case DatasetType.graph:
                if not isinstance(inp, Ligand):
                    raise ValueError(
                        "Expected Ligand input data for graph-based model, but got "
                        f"{type(inp)}."
                    )

            case DatasetType.structural:
                if not isinstance(inp, Complex):
                    raise ValueError(
                        "Expected Complex input data for structure-based model, but "
                        f"got {type(inp)}."
                    )
            case DatasetType.split:
                # Still need to make sure we have Complex data
                if not isinstance(inp, Complex):
                    raise ValueError(
                        "Expected Complex input data for structure-based model, but "
                        f"got {type(inp)}."
                    )
            case other:
                raise ValueError(f"Unknown dataset type {other}.")

        return self

    @field_validator("device", mode="before")
    def fix_device(cls, v):
        """
        The torch device gets serialized as a string and the Trainer class doesn't
        automatically cast it back to a device.
        """
        return torch.device(v)

    # Override the pydantic functions to conditionally exclude input_data from being
    #  serialized
    def model_dump(self, **kwargs):
        for check_field, export_field in zip(
            ["input_data", "exp_data"], [self.export_input_data, self.export_exp_data]
        ):
            if (
                ("include" not in kwargs)
                or (kwargs["include"] is None)
                or (check_field not in kwargs["include"])
            ):
                if not export_field:
                    try:
                        kwargs["exclude"].add(check_field)
                    except (KeyError, AttributeError):
                        kwargs["exclude"] = {check_field}

        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        for check_field, export_field in zip(
            ["input_data", "exp_data"], [self.export_input_data, self.export_exp_data]
        ):
            if (
                ("include" not in kwargs)
                or (kwargs["include"] is None)
                or (check_field not in kwargs["include"])
            ):
                if not export_field:
                    try:
                        kwargs["exclude"].add(check_field)
                    except (KeyError, AttributeError):
                        kwargs["exclude"] = {check_field}

        return super().model_dump_json(**kwargs)

    @classmethod
    def from_exp_file(cls, exp_file: Path, **config_kwargs):
        """
        Build a graph DatasetConfig from an experimental data file.

        Parameters
        ----------
        exp_file : Path
            JSON file giving a list of ExperimentalDataCompound objects
        config_kwargs
            Other kwargs that are passed to the class constructor


        Returns
        -------
        DatasetConfig
        """

        # Parse experimental data
        exp_compounds = [
            ExperimentalCompoundData(**d) for d in json.loads(exp_file.read_text())
        ]
        exp_data = {
            c.compound_id: c.experimental_data | {"date_created": c.date_created}
            for c in exp_compounds
        }

        # Update parsed exp_data with anything passed explicitly
        if "exp_data" in config_kwargs:
            exp_data |= config_kwargs["exp_data"]

        # Create Ligand objects from SMILES
        input_data = [
            Ligand.from_smiles(c.smiles, compound_name=c.compound_id)
            for c in exp_compounds
        ]

        # Get rid of the kwargs that we are passing explicitly
        config_kwargs = {
            k: v
            for k, v in config_kwargs.items()
            if k not in {"ds_type", "exp_data", "input_data"}
        }

        return cls(
            ds_type=DatasetType.graph,
            exp_data=exp_data,
            input_data=input_data,
            **config_kwargs,
        )

    @classmethod
    def from_str_files(
        cls,
        structures: str,
        xtal_regex: str,
        cpd_regex: str,
        for_training: bool = False,
        exp_file: Path | None = None,
        ds_type: DatasetType = DatasetType.structural,
        **config_kwargs,
    ):
        """
        Build a structural DatasetConfig from structure files.

        Parameters
        ----------
        structures : str
            Glob or directory containing PDB files
        xtal_regex : str
            Regex for extracting crystal structure name from filename
        cpd_regex : str
            Regex for extracting compound id from filename
        for_training : bool, default=False
            This dataset will be used for training. Forces exp_file to exist, and
            filters structures to only include those with exp data
        exp_file : str, optional
            JSON file giving a list of ExperimentalDataCompound objects
        config_kwargs
            Other kwargs that are passed to the class constructor


        Returns
        -------
        DatasetConfig
        """

        # Parse experimental data
        if exp_file and exp_file.exists():
            exp_compounds = [
                ExperimentalCompoundData(**d) for d in json.loads(exp_file.read_text())
            ]
            exp_data = {
                c.compound_id: c.experimental_data | {"date_created": c.date_created}
                for c in exp_compounds
            }

            # Update parsed exp_data with anything passed explicitly
            if "exp_data" in config_kwargs:
                exp_data |= config_kwargs["exp_data"]
        elif for_training:
            raise ValueError(
                "exp_file must be passed if the dataset will be used for training."
            )
        else:
            exp_data = {}

        # Get structure files
        if Path(structures).is_dir():
            all_str_fns = glob(f"{structures}/*.pdb")
        else:
            all_str_fns = glob(structures)
        compounds = extract_compounds_from_filenames(
            all_str_fns, xtal_pat=xtal_regex, compound_pat=cpd_regex, fail_val="NA"
        )

        # Filter compounds to only include datat that we have experimental data for
        if for_training:
            idx = [c[1] in exp_data for c in compounds]
            print(
                f"Filtering {len(idx) - sum(idx)} structures that we don't have",
                "experimental data for.",
                flush=True,
            )
            compounds = [c for i, c in zip(idx, compounds) if i]
            all_str_fns = [fn for i, fn in zip(idx, all_str_fns) if i]

        # Create Complex objects from PDB files
        print(len(all_str_fns), len(compounds), flush=True)
        input_data = [
            Complex.from_pdb(
                fn,
                target_kwargs={"target_name": cpd[0]},
                ligand_kwargs={"compound_name": cpd[1]},
            )
            for fn, cpd in zip(all_str_fns, compounds)
        ]

        # Get rid of the kwargs that we are passing explicitly
        config_kwargs = {
            k: v
            for k, v in config_kwargs.items()
            if k not in {"ds_type", "exp_data", "input_data"}
        }

        return cls(
            ds_type=ds_type,
            exp_data=exp_data,
            input_data=input_data,
            **config_kwargs,
        )

    def build(self):
        # Load from the cache file if it exists
        if self.cache_file and self.cache_file.exists() and (not self.overwrite):
            print("loading from cache", flush=True)
            ds = pkl.loads(self.cache_file.read_bytes())

            # Set in case ds doesn't have this attr (older version of the class) or
            #  there's a mismatch
            ds.random_iter = self.random_iter

            # Protein data might not exist if we're not caching it to save space, so
            #  make sure
            if (self.ds_type == DatasetType.split) and (
                "protein" not in next(iter(ds))[1]
            ):
                for pose in ds.structures:
                    # Extract the protein atoms
                    lig_idx = pose["complex"]["lig"]
                    prot_pose = {
                        k: pose["complex"][k][~lig_idx] for k in ["pos", "z", "x", "b"]
                    }
                    pose["protein"] = prot_pose

            if self.for_e3nn:
                ds = DatasetConfig.fix_e3nn_labels(ds, grouped=self.grouped)

            # Shuttle data to desired device
            for _, d in ds:
                for k, v in d.items():
                    try:
                        d[k] = v.to(self.device)
                    except AttributeError:
                        # Not a tensor so just let it go
                        pass

            return ds

        # Build directly from Complexes/Ligands
        match self.ds_type:
            case DatasetType.graph:
                from dgllife.utils import CanonicalAtomFeaturizer

                ds = GraphDataset.from_ligands(
                    self.input_data,
                    exp_dict=self.exp_data,
                    node_featurizer=CanonicalAtomFeaturizer(),
                )
            case DatasetType.structural:
                if self.grouped:
                    ds = GroupedDockedDataset.from_complexes(
                        self.input_data, exp_dict=self.exp_data
                    )
                else:
                    ds = DockedDataset.from_complexes(
                        self.input_data,
                        exp_dict=self.exp_data,
                        random_iter=self.random_iter,
                    )
                if self.for_e3nn:
                    ds = DatasetConfig.fix_e3nn_labels(ds, grouped=self.grouped)
            case DatasetType.split:
                ds = SplitDockedDataset.from_complexes(
                    self.input_data,
                    exp_dict=self.exp_data,
                    random_iter=self.random_iter,
                )
            case other:
                raise ValueError(f"Unknwon dataset type {other}.")

        if self.cache_file:
            if (self.ds_type == DatasetType.split) and (not self.cache_protein):
                tmp_ds = deepcopy(ds)
                for pose in tmp_ds.structures:
                    del pose["protein"]
                self.cache_file.write_bytes(pkl.dumps(tmp_ds))
                del tmp_ds
            else:
                self.cache_file.write_bytes(pkl.dumps(ds))

        # Shuttle data to desired device
        for _, d in ds:
            for k, v in d.items():
                try:
                    d[k] = v.to(self.device)
                except AttributeError:
                    # Not a tensor so just let it go
                    pass
        return ds

    @staticmethod
    def fix_e3nn_labels(ds, grouped=False):
        new_ds = deepcopy(ds)
        for _, data in new_ds:
            if isinstance(ds, GroupedDockedDataset):
                for pose in data["poses"]:
                    # Check if this pose has already been adjusted
                    if pose["z"].is_floating_point():
                        # Assume it'll only be floats if we've already run this function
                        continue

                    pose["x"] = torch.nn.functional.one_hot(pose["z"] - 1, 100).float()
                    pose["z"] = pose["lig"].reshape((-1, 1)).float()
            elif isinstance(ds, SplitDockedDataset):
                for lab in ["complex", "protein"]:
                    # Check if this data has already been adjusted
                    if data[lab]["z"].is_floating_point():
                        # Assume it'll only be floats if we've already run this function
                        continue

                    data[lab]["x"] = torch.nn.functional.one_hot(
                        data[lab]["z"] - 1, 100
                    ).float()
                    data[lab]["z"] = data[lab]["lig"].reshape((-1, 1)).float()
            else:
                # Check if this data has already been adjusted
                if data["z"].is_floating_point():
                    # Assume it'll only be floats if we've already run this function
                    continue

                data["x"] = torch.nn.functional.one_hot(data["z"] - 1, 100).float()
                data["z"] = data["lig"].reshape((-1, 1)).float()

        return new_ds


class DatasetSplitterType(StringEnum):
    """
    Enum for different methods of splitting a dataset.
    """

    random = "random"
    temporal = "temporal"
    manual = "manual"


class DatasetSplitterConfig(ConfigBase):
    """
    Class for splitting an ML Dataset class.
    """

    # Parameter for splitting
    split_type: DatasetSplitterType = Field(
        ...,
        description=(
            "Method to use for splitting. "
            f"Options are [{', '.join(DatasetSplitterType.get_values())}]."
        ),
    )

    # Multi-pose or not
    grouped: bool = Field(False, description="Build a GroupedDockedDataset.")

    # Split sizes
    train_frac: float = Field(
        0.8, description="Fraction of dataset to put in the train split."
    )
    val_frac: float = Field(
        0.1, description="Fraction of dataset to put in the val split."
    )
    test_frac: float = Field(
        0.1, description="Fraction of dataset to put in the test split."
    )
    enforce_one: bool = Field(
        True, description="Make sure that all split fractions add up to 1."
    )

    # Seed for randomly splitting data
    rand_seed: int | None = Field(
        None, description="Random seed to use if randomly splitting data."
    )

    # Dict giving splits for manual splitting
    split_dict: dict | Path | None = Field(
        None,
        description=(
            "Dict giving splits for manual splitting. If a Path is passed, "
            "the Path should be a JSON file and its contents will be loaded directly "
            "as split_dict."
        ),
    )

    @model_validator(mode="after")
    def check_frac_sum(self):
        # Don't need to check if we're doing manual split mode
        if self.split_type is DatasetSplitterType.manual:
            return self

        frac_sum = sum([self.train_frac, self.val_frac, self.test_frac])
        if frac_sum < 0:
            raise ValueError("Can't have negative split fractions.")

        if not np.isclose(frac_sum, 1):
            warn_msg = f"Split fractions add to {frac_sum:0.2f}, not 1."
            if self.enforce_one:
                raise ValueError(warn_msg)
            else:
                from warnings import warn

                warn(warn_msg, RuntimeWarning)

            if frac_sum > 1:
                raise ValueError("Can't have split fractions adding to > 1.")

        return self

    @model_validator(mode="after")
    def check_split_dict(self):
        if self.split_type is DatasetSplitterType.manual:
            if self.split_dict is None:
                raise ValueError(
                    "Must pass value for split_dict if using manual splitting."
                )

            if isinstance(self.split_dict, Path):
                try:
                    self.split_dict = json.loads(self.split_dict.read_text())
                except json.decoder.JSONDecodeError:
                    raise ValueError(
                        "Path given by split_dict must be a JSON file storing a dict."
                    )

            if set(self.split_dict.keys()) != {"train", "val", "test"}:
                raise ValueError(
                    "Keys in split_dict must be exactly [train, val, test]."
                )

            # Cast to tuples to match compounds in the datasets
            self.split_dict = {
                split: list(map(tuple, compound_list))
                for split, compound_list in self.split_dict.items()
            }

        return self

    def split(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Dispatch method for spliting ds into train, val, and test splits.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """
        match self.split_type:
            case DatasetSplitterType.random:
                return self._split_random(ds)
            case DatasetSplitterType.temporal:
                return self._split_temporal(ds)
            case DatasetSplitterType.manual:
                return self._split_manual(ds)
            case other:
                raise ValueError(f"Unknown DatasetSplitterType {other}.")

    @staticmethod
    def _make_subsets(ds, idx_lists, split_lens):
        """
        Helper script for making subsets of a dataset.

        Parameters
        ----------
        ds : Union[cml.data.DockedDataset, cml.data.GraphDataset]
            Molecular dataset to split
        idx_dict : List[List[int]]
            List of lists of indices into `ds`
        split_lens : List[int]
            List of split lengths

        Returns
        -------
        List[torch.utils.data.Subset]
            List of Subsets of original dataset
        """
        # For each Subset, grab all molecules with the included compound_ids
        all_subsets = []
        # Keep track of which structure indices we've seen so we don't double count in the
        #  end splits
        seen_idx = set()
        prev_idx = 0
        # Go up to the last split so we can add anything that got left out from rounding
        for i, n_mols in enumerate(split_lens):
            n_mols_cur = 0
            subset_idx = []
            cur_idx = prev_idx
            # Keep adding groups until the split is as big as it needs to be, or we reach
            #  the end of the array (making sure to save at least some molecules for the
            #  rest of the splits)
            while (n_mols_cur < n_mols) and (
                cur_idx < (len(idx_lists) - (len(split_lens) - i - 1))
            ):
                subset_idx.extend(idx_lists[cur_idx])
                n_mols_cur += len(idx_lists[cur_idx])
                cur_idx += 1

            # Make sure we're not including something that's in another split
            subset_idx = [i for i in subset_idx if i not in seen_idx]
            seen_idx.update(subset_idx)
            all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

            # Update counter
            prev_idx = cur_idx

        return all_subsets

    def _split_random(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Random split.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """

        if self.rand_seed is None:
            g = torch.Generator()
            g.seed()
        else:
            g = torch.Generator().manual_seed(self.rand_seed)
        print("splitting with random seed:", g.initial_seed(), flush=True)

        if self.grouped:
            # Grouped models are already grouped by compound, so don't need to do
            #  anything fancy here
            ds_train, ds_val, ds_test = torch.utils.data.random_split(
                ds, [self.train_frac, self.val_frac, self.test_frac], g
            )
        else:
            # Calculate how many molecules we want covered through each split
            n_mols_split = np.floor(
                np.asarray([self.train_frac, self.val_frac, self.test_frac]) * len(ds)
            )

            # First get all the unique compound_ids
            compound_ids_dict = {}
            for c, idx_list in ds.compounds.items():
                try:
                    compound_ids_dict[c[1]].extend(idx_list)
                except KeyError:
                    compound_ids_dict[c[1]] = idx_list
            all_compound_ids = np.asarray(list(compound_ids_dict.keys()))
            print("compound_ids_dict", next(iter(compound_ids_dict.items())))
            print(
                "all_compound_ids",
                len(all_compound_ids),
                len(set(all_compound_ids)),
                flush=True,
            )

            # Shuffle the indices
            indices = torch.randperm(len(all_compound_ids), generator=g)
            idx_lists = [compound_ids_dict[all_compound_ids[i]] for i in indices]

            # For each Subset, grab all molecules with the included compound_ids
            ds_train, ds_val, ds_test = DatasetSplitterConfig._make_subsets(
                ds, idx_lists, n_mols_split
            )

        return ds_train, ds_val, ds_test

    def _split_temporal(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Temporal split.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """
        split_fracs = [self.train_frac, self.val_frac, self.test_frac]
        # Check that split_fracs adds to 1, padding if it doesn't
        # Add an allowance for floating point inaccuracies
        total_splits = sum(split_fracs)
        if not np.isclose(total_splits, 1):
            sink_frac = 1 - total_splits
            split_fracs = split_fracs[:2] + [sink_frac] + split_fracs[2:]
            sink_split = True
        else:
            sink_split = False

        # Calculate how many molecules we want covered through each split
        n_mols_split = np.floor(np.asarray(split_fracs) * len(ds))

        # First get all the unique created dates
        dates_dict = {}
        # If we have a grouped dataset, we want to iterate through compound_ids, which will
        #  allow us to access a group of structures. Otherwise, loop through the structures
        #  directly
        if self.grouped:
            iter_list = ds.structures.values()
        else:
            iter_list = ds.structures
        for i, iter_item in enumerate(iter_list):
            try:
                date_created = iter_item["date_created"]
            except KeyError:
                raise ValueError("Dataset doesn't contain dates.")
            try:
                dates_dict[date_created].append(i)
            except KeyError:
                dates_dict[date_created] = [i]

        # check if dates_dict is empty
        if all(d == None for d in dates_dict.keys()) or not dates_dict:  # noqa: E711
            raise ValueError("No dates found in dataset.")

        all_dates = np.asarray(list(dates_dict.keys()))

        # Sort the dates
        all_dates_sorted = sorted(all_dates)

        # Make subsets
        idx_lists = [dates_dict[d] for d in all_dates_sorted]
        all_subsets = DatasetSplitterConfig._make_subsets(ds, idx_lists, n_mols_split)

        # Take out the sink split
        if sink_split:
            all_subsets = all_subsets[:2] + [all_subsets[3]]

        return all_subsets

    def _split_manual(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Manual split, based on ``self.split_dict``.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """
        all_subset_idxs = {}
        for i, (compound, _) in enumerate(ds):
            if compound in self.split_dict["train"]:
                split = "train"
            elif compound in self.split_dict["val"]:
                split = "val"
            elif compound in self.split_dict["test"]:
                split = "test"
            else:
                continue

            try:
                all_subset_idxs[split].append(i)
            except KeyError:
                all_subset_idxs[split] = [i]

        return [
            torch.utils.data.Subset(ds, all_subset_idxs[split])
            for split in ["train", "val", "test"]
        ]


class LossFunctionType(StringEnum):
    """
    Enum for different methods of splitting a dataset.
    """

    # Standard L1 loss
    l1 = "l1"
    # Stepped L1 loss (adjusted loss for values outside assay range)
    l1_step = "l1_step"
    # Standard MSE loss
    mse = "mse"
    # Stepped MSE loss (adjusted loss for values outside assay range)
    mse_step = "mse_step"
    # Standard smooth L1 loss
    smooth_l1 = "smooth_l1"
    # Stepped smooth L1 loss (adjusted loss for values outside assay range)
    smooth_l1_step = "smooth_l1_step"
    # Gaussian NLL loss (ignoring semiquant values)
    gaussian = "gaussian"
    # Gaussian NLL loss (including semiquant values)
    gaussian_sq = "gaussian_sq"
    # Squared difference penalty for pred outside range
    range_penalty = "range"
    # Pose cross entropy loss
    cross_entropy = "cross_entropy"


class LossFunctionConfig(ConfigBase):
    """
    Class for building a loss function.
    """

    # Parameter for loss type
    loss_type: LossFunctionType = Field(
        ...,
        description=(
            "Loss function to use. "
            f"Options are [{', '.join(LossFunctionType.get_values())}]."
        ),
    )

    # Value to fill in for semiquant uncertainty values in gaussian_sq loss
    semiquant_fill: float | None = Field(
        None,
        description=(
            "Value to fill in for semiquant uncertainty values in gaussian_sq loss."
        ),
    )

    # Range values for RangeLoss
    range_lower_lim: float | None = Field(
        None, description="Lower range of acceptable prediction values."
    )
    range_upper_lim: float | None = Field(
        None, description="Upper range of acceptable prediction values."
    )

    @model_validator(mode="after")
    def check_range_lims(self):
        if self.loss_type is not LossFunctionType.range_penalty:
            return self

        if (self.range_lower_lim is None) or (self.range_upper_lim is None):
            raise ValueError(
                "Values must be passed for range_lower_lim and range_upper_lim."
            )

        if self.range_lower_lim >= self.range_upper_lim:
            raise ValueError(
                "Value given for range_lower_lim >= value given for range_upper_lim."
            )

        return self

    def build(self):
        from drugforge.ml.loss import (
            GaussianNLLLoss,
            L1Loss,
            MSELoss,
            PoseCrossEntropyLoss,
            RangeLoss,
            SmoothL1Loss,
        )

        match self.loss_type:
            case LossFunctionType.l1:
                return L1Loss()
            case LossFunctionType.l1_step:
                return L1Loss("step")
            case LossFunctionType.mse:
                return MSELoss()
            case LossFunctionType.mse_step:
                return MSELoss("step")
            case LossFunctionType.smooth_l1:
                return SmoothL1Loss()
            case LossFunctionType.smooth_l1_step:
                return SmoothL1Loss("step")
            case LossFunctionType.gaussian:
                return GaussianNLLLoss(keep_sq=False)
            case LossFunctionType.gaussian_sq:
                return GaussianNLLLoss(keep_sq=True, semiquant_fill=self.semiquant_fill)
            case LossFunctionType.range_penalty:
                return RangeLoss(
                    lower_lim=self.range_lower_lim, upper_lim=self.range_upper_lim
                )
            case LossFunctionType.cross_entropy:
                return PoseCrossEntropyLoss()
            case other:
                raise ValueError(f"Unknown LossFunctionType {other}.")


class DataAugType(StringEnum):
    """
    Enum for different methods of data augmentation.
    """

    # Jitter all coordinates by a fixed amount
    jitter_fixed = "jitter_fixed"


class DataAugConfig(BaseModel):
    """
    Class for building a data augmentation module.
    """

    # Parameter for augmentation type
    aug_type: DataAugType = Field(
        ...,
        description=(
            "Type of augmentation."
            f"Options are [{', '.join(DataAugType.get_values())}]."
        ),
    )

    # Define the distribution of random noise to jitter with
    jitter_fixed_mean: float | None = Field(
        None, description="Mean of gaussian distribution to draw noise from."
    )
    jitter_fixed_std: float | None = Field(
        None,
        description="Standard deviation of gaussian distribution to draw noise from.",
    )

    # Seed for randomly jittering
    jitter_rand_seed: int | None = Field(
        None, description="Random seed to use for reproducbility, if desired."
    )

    # Dict key for the positions
    jitter_pos_key: str | None = Field(
        None, description="Key to access the coords in pose dict."
    )

    def build(self):
        from drugforge.ml.data_augmentation import JitterFixed

        match self.aug_type:
            case DataAugType.jitter_fixed:
                build_class = JitterFixed
                kwargs = {
                    "mean": "jitter_fixed_mean",
                    "std": "jitter_fixed_std",
                    "rand_seed": "jitter_rand_seed",
                    "dict_key": "jitter_pos_key",
                }

        # Remove any None kwargs
        kwargs = {
            k: getattr(self, v)
            for k, v in kwargs.items()
            if getattr(self, v) is not None
        }
        return build_class(**kwargs)
