import pytest
import torch

from drugforge.ml.data_augmentation import PositionRandomize, PositionShuffle


@pytest.fixture()
def data():
    pos = torch.rand((100, 3))
    lig = torch.randint(0, 2, (100,), dtype=bool)

    return pos, lig


@pytest.fixture()
def data_dict():
    pos = torch.rand((100, 3))
    lig = torch.randint(0, 2, (100,), dtype=bool)

    return {"pos": pos, "lig": lig}


def test_shuffle_both(data):
    pos = data[0]

    shuff = PositionShuffle()
    assert not torch.allclose(pos, shuff(pos))


def test_shuffle_lig(data):
    pos, lig_idx = data

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])
    assert not torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])


def test_shuffle_prot(data):
    pos, lig_idx = data

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])
    assert not torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])


def test_shuffle_lig_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])
    assert not torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])


def test_shuffle_prot_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])
    assert not torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])


def test_shuffled_fixed_seed(data):
    pos = data[0]

    shuff = PositionShuffle(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionShuffle(rand_seed=0)
    pos2 = shuff(pos)

    assert torch.allclose(pos1, pos2)


def test_randomize_both(data):
    pos = data[0]

    shuff = PositionRandomize()
    assert not torch.allclose(pos, shuff(pos))


def test_randomize_lig(data):
    pos, lig_idx = data

    shuff = PositionRandomize(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])
    assert not torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])


def test_randomize_prot(data):
    pos, lig_idx = data

    shuff = PositionRandomize(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])
    assert not torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])


def test_randomize_lig_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionRandomize(which="lig")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])
    assert not torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])


def test_randomize_prot_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionRandomize(which="prot")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[lig_idx, :], shuff_pos[lig_idx, :])
    assert not torch.allclose(pos[~lig_idx, :], shuff_pos[~lig_idx, :])


def test_randomized_fixed_seed(data):
    pos = data[0]

    shuff = PositionRandomize(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionRandomize(rand_seed=0)
    pos2 = shuff(pos)

    assert torch.allclose(pos1, pos2)
