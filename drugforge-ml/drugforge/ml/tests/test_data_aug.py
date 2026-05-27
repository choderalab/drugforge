import pytest
import torch

from drugforge.ml.data_augmentation import PositionShuffle


def test_shuffle_both():
    pos = torch.rand((5, 3))

    shuff = PositionShuffle()
    assert not torch.allclose(pos, shuff(pos))


def test_shuffle_lig():
    pos = torch.rand((5, 3))
    lig_idx = torch.tensor([True, True, False, False, False])

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])
    assert not torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])


def test_shuffle_prot():
    pos = torch.rand((5, 3))
    lig_idx = torch.tensor([True, True, False, False, False])

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])
    assert not torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])
