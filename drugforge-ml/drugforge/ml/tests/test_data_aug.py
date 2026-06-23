import pytest
import torch
from drugforge.ml.data_augmentation import (
    PositionRandomize,
    PositionShuffle,
    SplitComplex,
)


@pytest.fixture()
def data():
    # Position, ligand index, integer atomic number, and one-hot encoding for passing
    #  directly to the data aug class
    pos = torch.rand((100, 3))
    lig = torch.randint(0, 2, (100,), dtype=bool)
    atomic_num = torch.randint(1, 101, (100,))
    one_hot = torch.nn.functional.one_hot(atomic_num - 1)

    return pos, lig, atomic_num, one_hot


@pytest.fixture()
def data_dict():
    # Position, ligand index, integer atomic number, and one-hot encoding for passing as
    #  a dict to the data aug class
    pos = torch.rand((100, 3))
    lig = torch.randint(0, 2, (100,), dtype=bool)
    atomic_num = torch.randint(1, 101, (100,))
    one_hot = torch.nn.functional.one_hot(atomic_num - 1)

    return {"pos": pos, "lig": lig, "atomic_num": atomic_num, "one_hot": one_hot}


def test_shuffle_pos_both(data):
    # Test shuffling all positions when input is passed directly
    pos = data[0]

    shuff = PositionShuffle()
    assert not torch.allclose(pos, shuff(pos))


def test_shuffle_pos_lig(data):
    # Test shuffling only ligand positions when input is passed directly
    pos, lig_idx, *_ = data

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    # Make sure only ligand atoms have changed
    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_shuffle_pos_prot(data):
    # Test shuffling only protein positions when input is passed directly
    pos, lig_idx, *_ = data

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    # Make sure only protein atoms have changed
    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_shuffle_pos_lig_dict(data_dict):
    # Test shuffling only ligand positions when input is passed as a dict
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(data_dict)["pos"]

    # Make sure only ligand atoms have changed
    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_shuffle_pos_prot_dict(data_dict):
    # Test shuffling only protein positions when input is passed as a dict
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(data_dict)["pos"]

    # Make sure only protein atoms have changed
    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_shuffle_atomic_num_both(data):
    # Test shuffling all atomic numbers when input is passed directly
    atomic_num = data[2]

    shuff = PositionShuffle()
    assert not torch.allclose(atomic_num, shuff(atomic_num))


def test_shuffle_atomic_num_lig(data):
    # Test shuffling only ligand atomic numbers when input is passed directly
    _, lig_idx, atomic_num, _ = data

    shuff = PositionShuffle(which="lig")
    shuff_atomic_num = shuff(atomic_num, lig_idx)

    # Make sure only ligand atoms have changed
    assert torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])


def test_shuffle_atomic_num_prot(data):
    # Test shuffling only protein atomic numbers when input is passed directly
    _, lig_idx, atomic_num, _ = data

    shuff = PositionShuffle(which="prot")
    shuff_atomic_num = shuff(atomic_num, lig_idx)

    # Make sure only protein atoms have changed
    assert torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])


def test_shuffle_atomic_num_lig_dict(data_dict):
    # Test shuffling only ligand atomic numbers when input is passed as a dict
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="atomic_num", which="lig")
    shuff_atomic_num = shuff(data_dict)["atomic_num"]

    # Make sure only ligand atoms have changed
    assert torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])


def test_shuffle_atomic_num_prot_dict(data_dict):
    # Test shuffling only protein atomic numbers when input is passed as a dict
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="atomic_num", which="prot")
    shuff_atomic_num = shuff(data_dict)["atomic_num"]

    # Make sure only protein atoms have changed
    assert torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])


def test_shuffle_one_hot_both(data):
    # Test shuffling a one-hot encoding when input is passed directly
    one_hot = data[3]

    shuff = PositionShuffle()
    assert not torch.allclose(one_hot, shuff(one_hot))


def test_shuffle_one_hot_lig(data):
    # Test shuffling a one-hot encoding for only ligand atoms when input is passed
    #  directly
    _, lig_idx, _, one_hot = data

    shuff = PositionShuffle(which="lig")
    shuff_one_hot = shuff(one_hot, lig_idx)

    # Make sure only ligand atoms have changed
    assert torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])


def test_shuffle_one_hot_prot(data):
    # Test shuffling a one-hot encoding for only protein atoms when input is passed
    #  directly
    _, lig_idx, _, one_hot = data

    shuff = PositionShuffle(which="prot")
    shuff_one_hot = shuff(one_hot, lig_idx)

    # Make sure only protein atoms have changed
    assert torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])


def test_shuffle_one_hot_lig_dict(data_dict):
    # Test shuffling a one-hot encoding for only ligand atoms when input is passed as a
    #  dict
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="one_hot", which="lig")
    shuff_one_hot = shuff(data_dict)["one_hot"]

    # Make sure only ligand atoms have changed
    assert torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])


def test_shuffle_one_hot_prot_dict(data_dict):
    # Test shuffling a one-hot encoding for only protein atoms when input is passed as a
    #  dict
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="one_hot", which="prot")
    shuff_one_hot = shuff(data_dict)["one_hot"]

    # Make sure only protein atoms have changed
    assert torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])


def test_shuffled_pos_fixed_seed(data):
    # Make sure that using a fixed seed shuffles reproducibly
    pos = data[0]

    shuff = PositionShuffle(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionShuffle(rand_seed=0)
    pos2 = shuff(pos)

    assert torch.allclose(pos1, pos2)


def test_shuffled_pos_diff_seeds(data):
    # Make sure that using different fixed seeds produces different shuffles
    pos = data[0]

    shuff = PositionShuffle(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionShuffle(rand_seed=1)
    pos2 = shuff(pos)

    assert not torch.allclose(pos1, pos2)


def test_randomize_pos_both(data):
    # Test randomizing all positions when input is passed directly
    pos = data[0]

    shuff = PositionRandomize()
    assert not torch.allclose(pos, shuff(pos))


def test_randomize_pos_lig(data):
    # Test randomizing ligand positions when input is passed directly
    pos, lig_idx, *_ = data

    shuff = PositionRandomize(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    # Make sure only ligand atoms have changed
    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_randomize_pos_prot(data):
    # Test randomizing protein positions when input is passed directly
    pos, lig_idx, *_ = data

    shuff = PositionRandomize(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    # Make sure only protein atoms have changed
    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_randomize_pos_lig_dict(data_dict):
    # Test randomizing ligand positions when input is passed as a dict
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionRandomize(which="lig")
    shuff_pos = shuff(data_dict)["pos"]

    # Make sure only ligand atoms have changed
    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_randomize_pos_prot_dict(data_dict):
    # Test randomizing protein positions when input is passed as a dict
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionRandomize(which="prot")
    shuff_pos = shuff(data_dict)["pos"]

    # Make sure only protein atoms have changed
    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_randomize_int_both(data):
    # Test randomizing integer data when input is passed directly
    atomic_num = data[2]

    rand = PositionRandomize(dict_key="atomic_num", data_type="int")
    assert not torch.allclose(atomic_num, rand(atomic_num))


def test_randomize_int_lig(data):
    # Test randomizing integer data for only ligand atoms when input is passed directly
    _, lig_idx, atomic_num, _ = data

    rand = PositionRandomize(which="lig", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(atomic_num, lig_idx)

    # Make sure only ligand atoms have changed
    assert torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])


def test_randomize_int_prot(data):
    # Test randomizing integer data for only protein atoms when input is passed directly
    _, lig_idx, atomic_num, _ = data

    rand = PositionRandomize(which="prot", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(atomic_num, lig_idx)

    # Make sure only protein atoms have changed
    assert torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])


def test_randomize_int_lig_dict(data_dict):
    # Test randomizing integer data for only ligand atoms when input is passed as a dict
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="lig", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(data_dict)["atomic_num"]

    # Make sure only ligand atoms have changed
    assert torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])


def test_randomize_int_prot_dict(data_dict):
    # Test randomizing integer data for only protein atoms when input is passed as a
    #  dict
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="prot", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(data_dict)["atomic_num"]

    # Make sure only protein atoms have changed
    assert torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])


def test_randomize_one_hot_both(data):
    # Test randomizing one-hot data when input is passed directly
    one_hot = data[3]

    rand = PositionRandomize(dict_key="one_hot", data_type="onehot")
    assert not torch.allclose(one_hot, rand(one_hot))


def test_randomize_one_hot_lig(data):
    # Test randomizing one-hot data for only ligand atoms when input is passed directly
    _, lig_idx, _, one_hot = data

    rand = PositionRandomize(which="lig", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(one_hot, lig_idx)

    # Make sure only ligand atoms have changed
    assert torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])


def test_randomize_one_hot_prot(data):
    # Test randomizing one-hot data for only protein atoms when input is passed directly
    _, lig_idx, _, one_hot = data

    rand = PositionRandomize(which="prot", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(one_hot, lig_idx)

    # Make sure only protein atoms have changed
    assert torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])


def test_randomize_one_hot_lig_dict(data_dict):
    # Test randomizing one-hot data for only ligand atoms when input is passed as a dict
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="lig", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(data_dict)["one_hot"]

    # Make sure only ligand atoms have changed
    assert torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])


def test_randomize_one_hot_prot_dict(data_dict):
    # Test randomizing one-hot data for only protein atoms when input is passed as a
    #  dict
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="prot", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(data_dict)["one_hot"]

    # Make sure only protein atoms have changed
    assert torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])


def test_randomized_fixed_seed(data):
    # Make sure that using a fixed seed randomizes reproducibly
    pos = data[0]

    shuff = PositionRandomize(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionRandomize(rand_seed=0)
    pos2 = shuff(pos)

    assert torch.allclose(pos1, pos2)


def test_randomized_diff_seeds(data):
    # Make sure that using different fixed seeds produces different randomizations
    pos = data[0]

    shuff = PositionRandomize(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionRandomize(rand_seed=1)
    pos2 = shuff(pos)

    assert not torch.allclose(pos1, pos2)


def test_split(data):
    # Test that SplitComplex is moving only the ligand atoms the correct distance away
    #  for input passed directly
    pos, lig_idx, *_ = data

    split = SplitComplex()
    split_pos = split(pos, lig_idx)

    assert torch.allclose(pos[~lig_idx], split_pos[~lig_idx])
    assert torch.allclose(pos[lig_idx] + split.split_dist, split_pos[lig_idx])


def test_split_dict(data_dict):
    # Test that SplitComplex is moving only the ligand atoms the correct distance away
    #  for input passed as a dict
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    split = SplitComplex()
    split_pos = split(data_dict)["pos"]

    assert torch.allclose(pos[~lig_idx], split_pos[~lig_idx])
    assert torch.allclose(pos[lig_idx] + split.split_dist, split_pos[lig_idx])
