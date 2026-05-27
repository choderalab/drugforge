from copy import deepcopy

import torch


class JitterFixed:
    """
    Jitter an input pose by drawing noise from a fixed normal distribution. Noise is
    generated per-atom.
    """

    def __init__(
        self,
        mean: float = 0,
        std: float = 1,
        rand_seed: int | None = None,
        dict_key: str = "pos",
    ):
        """
        Parameters
        ----------
        mean : float, default=1
            Mean of noise distribution
        std : float, default=0.1
            Standard deviation of noise distribution
        rand_seed : int, optional
            Random seed for noise generation
        dict_key : str, default="pos"
            If the inputs are a dict, this will be the key used to access the coords in
            the dict
        """

        self.mean = mean
        self.std = std
        self.rand_seed = rand_seed
        if rand_seed is not None:
            self.g = torch.Generator().manual_seed(rand_seed)
        else:
            self.g = torch.Generator().manual_seed(torch.random.seed())
        self.dict_key = dict_key

    def __call__(self, coords, inplace=False):
        """
        Apply noise to each atom. Unless inplace is True, this method will create a copy
        of the input coordinate Tensor.

        Parameters
        ----------
        coords : torch.Tensor | dict
            Initial coordinates to be jittered. Noise will be generated independently
            for each
        inplace : bool, default=False
            Modify the passed Tensor in place, rather than first copying

        Returns
        -------
        torch.Tensor
            Jittered coordinates
        """

        # Figure out if we're working with a dict or raw Tensor inputs
        if isinstance(coords, dict):
            dict_inp = True
        else:
            dict_inp = False

        # Fist make a copy of the input coords (if inplace is False)
        if not inplace:
            if dict_inp:
                dict_copy = deepcopy(coords)
                coords_copy = dict_copy[self.dict_key]
            else:
                coords_copy = coords.clone().detach()
        else:
            # Should just be a reference so inputs should get modified
            if dict_inp:
                dict_copy = coords
                coords_copy = coords_copy[self.dict_key]
            else:
                coords_copy = coords
        # Create the mean Tensor, which should just have the same shape as the coords
        mean = torch.full_like(coords_copy, self.mean).to("cpu")
        # Generate noise (the std will be broadcast to the same shape as mean)
        noise = torch.normal(mean=mean, std=self.std, generator=self.g)
        # Add the noise
        coords_copy += noise.to(coords_copy.device)

        if dict_inp:
            return dict_copy
        else:
            return coords_copy


class PositionShuffle:
    """
    Shuffle all atom coordinates.
    """

    def __init__(
        self,
        which: str = "both",
        rand_seed: int | None = None,
        dict_key: str = "pos",
        lig_idx_key: str = "lig",
    ):
        """
        Parameters
        ----------
        which : str, default=both
            Whether to shuffle ligand coordinates ("lig"), protein coordinates ("prot"),
            or both ("both", default)
        rand_seed : int, optional
            Random seed for noise generation
        dict_key : str, default="pos"
            If the inputs are a dict, this will be the key used to access the coords in
            the dict
        lig_idx_key : str, default="lig"
            If the inputs are a dict, this will be the key used to access the lig_idx in
            the dict
        """

        which = which.lower()
        if which not in {"both", "lig", "prot"}:
            raise ValueError(
                f"unsupported value {which} for which (supported values are "
                '["both", "lig", "prot"])'
            )

        self.which = which
        self.rand_seed = rand_seed
        if rand_seed is not None:
            self.g = torch.Generator().manual_seed(rand_seed)
        else:
            self.g = torch.Generator().manual_seed(torch.random.seed())
        self.dict_key = dict_key
        self.lig_idx_key = lig_idx_key

    def __call__(self, coords, lig_idx=None, inplace=False):
        """
        Shuffle positions of each atom. Unless inplace is True, this method will create
        a copy of the input coordinate Tensor.

        Parameters
        ----------
        coords : torch.Tensor | dict
            Initial coordinates to be jittered. Noise will be generated independently
            for each
        lig_idx : torch.Tensor, optional
            Index for which atoms belong to the ligand. This is required if self.which
            is not "both" and coords is not a dict that contains the ligand index
        inplace : bool, default=False
            Modify the passed Tensor in place, rather than first copying

        Returns
        -------
        torch.Tensor
            Shuffled coordinates
        """

        # Figure out if we're working with a dict or raw Tensor inputs
        if isinstance(coords, dict):
            dict_inp = True
        else:
            dict_inp = False

        if (not dict_inp) and (lig_idx is None) and (self.which != "both"):
            raise RuntimeError("lig_idx must be passed if input is not a dictionary")
        if dict_inp and (self.lig_idx_key not in coords) and (self.which != "both"):
            raise RuntimeError("lig_idx_key must be present in dictionary input")

        # Fist make a copy of the input coords (if inplace is False)
        if not inplace:
            if dict_inp:
                dict_copy = deepcopy(coords)
                coords_copy = dict_copy[self.dict_key]
            else:
                coords_copy = coords.clone().detach()
        else:
            # Should just be a reference so inputs should get modified
            if dict_inp:
                dict_copy = coords
                coords_copy = coords_copy[self.dict_key]
            else:
                coords_copy = coords

        if dict_inp and (self.which != "both"):
            lig_idx = coords[self.lig_idx_key]

        # Get indices that we'll actually be shuffling
        shuffle_indices = torch.arange(coords_copy.shape[0])
        if self.which == "lig":
            shuffle_indices = shuffle_indices[lig_idx]
        elif self.which == "prot":
            shuffle_indices = shuffle_indices[~lig_idx]

        # Generate new index
        idx = torch.randperm(len(shuffle_indices), generator=self.g)

        # Shuffle the positions
        coords_copy = coords_copy[shuffle_indices[idx], :]

        if dict_inp:
            return dict_copy
        else:
            return coords_copy
