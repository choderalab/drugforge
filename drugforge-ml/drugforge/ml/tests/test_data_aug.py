import pytest
import torch

from drugforge.ml.data_augmentation import PositionRandomize, PositionShuffle


@pytest.fixture()
def data():
    pos = torch.rand((100, 3))
    lig = torch.randint(0, 2, (100,), dtype=bool)
    atomic_num = torch.randint(1, 101, (100,))
    one_hot = torch.nn.functional.one_hot(atomic_num - 1)

    return pos, lig, atomic_num, one_hot


@pytest.fixture()
def data_dict():
    pos = torch.rand((100, 3))
    lig = torch.randint(0, 2, (100,), dtype=bool)
    atomic_num = torch.randint(1, 101, (100,))
    one_hot = torch.nn.functional.one_hot(atomic_num - 1)

    return {"pos": pos, "lig": lig, "atomic_num": atomic_num, "one_hot": one_hot}


def test_shuffle_pos_both(data):
    pos = data[0]

    shuff = PositionShuffle()
    assert not torch.allclose(pos, shuff(pos))


def test_shuffle_pos_lig(data):
    pos, lig_idx, *_ = data

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_shuffle_pos_prot(data):
    pos, lig_idx, *_ = data

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_shuffle_pos_lig_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(which="lig")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_shuffle_pos_prot_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(which="prot")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_shuffle_atomic_num_both(data):
    atomic_num = data[2]

    shuff = PositionShuffle()
    assert not torch.allclose(atomic_num, shuff(atomic_num))


def test_shuffle_atomic_num_lig(data):
    _, lig_idx, atomic_num, _ = data

    shuff = PositionShuffle(which="lig")
    shuff_atomic_num = shuff(atomic_num, lig_idx)

    assert torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])


def test_shuffle_atomic_num_prot(data):
    _, lig_idx, atomic_num, _ = data

    shuff = PositionShuffle(which="prot")
    shuff_atomic_num = shuff(atomic_num, lig_idx)

    assert torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])


def test_shuffle_atomic_num_lig_dict(data_dict):
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="atomic_num", which="lig")
    shuff_atomic_num = shuff(data_dict)["atomic_num"]

    assert torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])


def test_shuffle_atomic_num_prot_dict(data_dict):
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="atomic_num", which="prot")
    shuff_atomic_num = shuff(data_dict)["atomic_num"]

    assert torch.allclose(atomic_num[lig_idx], shuff_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], shuff_atomic_num[~lig_idx])


def test_shuffle_one_hot_both(data):
    one_hot = data[3]

    shuff = PositionShuffle()
    assert not torch.allclose(one_hot, shuff(one_hot))


def test_shuffle_one_hot_lig(data):
    _, lig_idx, _, one_hot = data

    shuff = PositionShuffle(which="lig")
    shuff_one_hot = shuff(one_hot, lig_idx)

    assert torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])


def test_shuffle_one_hot_prot(data):
    _, lig_idx, _, one_hot = data

    shuff = PositionShuffle(which="prot")
    shuff_one_hot = shuff(one_hot, lig_idx)

    assert torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])


def test_shuffle_one_hot_lig_dict(data_dict):
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="one_hot", which="lig")
    shuff_one_hot = shuff(data_dict)["one_hot"]

    assert torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])


def test_shuffle_one_hot_prot_dict(data_dict):
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    shuff = PositionShuffle(dict_key="one_hot", which="prot")
    shuff_one_hot = shuff(data_dict)["one_hot"]

    assert torch.allclose(one_hot[lig_idx], shuff_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], shuff_one_hot[~lig_idx])


def test_shuffled_pos_fixed_seed(data):
    pos = data[0]

    shuff = PositionShuffle(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionShuffle(rand_seed=0)
    pos2 = shuff(pos)

    assert torch.allclose(pos1, pos2)


def test_randomize_pos_both(data):
    pos = data[0]

    shuff = PositionRandomize()
    assert not torch.allclose(pos, shuff(pos))


def test_randomize_pos_lig(data):
    pos, lig_idx, *_ = data

    shuff = PositionRandomize(which="lig")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_randomize_pos_prot(data):
    pos, lig_idx, *_ = data

    shuff = PositionRandomize(which="prot")
    shuff_pos = shuff(pos, lig_idx)

    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_randomize_pos_lig_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionRandomize(which="lig")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])
    assert not torch.allclose(pos[lig_idx], shuff_pos[lig_idx])


def test_randomize_pos_prot_dict(data_dict):
    pos = data_dict["pos"]
    lig_idx = data_dict["lig"]

    shuff = PositionRandomize(which="prot")
    shuff_pos = shuff(data_dict)["pos"]

    assert torch.allclose(pos[lig_idx], shuff_pos[lig_idx])
    assert not torch.allclose(pos[~lig_idx], shuff_pos[~lig_idx])


def test_randomize_int_both(data):
    atomic_num = data[2]

    rand = PositionRandomize(dict_key="atomic_num", data_type="int")
    assert not torch.allclose(atomic_num, rand(atomic_num))


def test_randomize_int_lig(data):
    _, lig_idx, atomic_num, _ = data

    rand = PositionRandomize(which="lig", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(atomic_num, lig_idx)

    assert torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])


def test_randomize_int_prot(data):
    _, lig_idx, atomic_num, _ = data

    rand = PositionRandomize(which="prot", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(atomic_num, lig_idx)

    assert torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])


def test_randomize_int_lig_dict(data_dict):
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="lig", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(data_dict)["atomic_num"]

    assert torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])
    assert not torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])


def test_randomize_int_prot_dict(data_dict):
    atomic_num = data_dict["atomic_num"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="prot", dict_key="atomic_num", data_type="int")
    rand_atomic_num = rand(data_dict)["atomic_num"]

    assert torch.allclose(atomic_num[lig_idx], rand_atomic_num[lig_idx])
    assert not torch.allclose(atomic_num[~lig_idx], rand_atomic_num[~lig_idx])


def test_randomize_one_hot_both(data):
    one_hot = data[3]

    rand = PositionRandomize(dict_key="one_hot", data_type="onehot")
    assert not torch.allclose(one_hot, rand(one_hot))


def test_randomize_one_hot_lig(data):
    _, lig_idx, _, one_hot = data

    rand = PositionRandomize(which="lig", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(one_hot, lig_idx)

    assert torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])


def test_randomize_one_hot_prot(data):
    _, lig_idx, _, one_hot = data

    rand = PositionRandomize(which="prot", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(one_hot, lig_idx)

    assert torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])


def test_randomize_one_hot_lig_dict(data_dict):
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="lig", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(data_dict)["one_hot"]

    assert torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])
    assert not torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])


def test_randomize_one_hot_prot_dict(data_dict):
    one_hot = data_dict["one_hot"]
    lig_idx = data_dict["lig"]

    rand = PositionRandomize(which="prot", dict_key="one_hot", data_type="onehot")
    rand_one_hot = rand(data_dict)["one_hot"]

    assert torch.allclose(one_hot[lig_idx], rand_one_hot[lig_idx])
    assert not torch.allclose(one_hot[~lig_idx], rand_one_hot[~lig_idx])


def test_randomized_fixed_seed(data):
    pos = data[0]

    shuff = PositionRandomize(rand_seed=0)
    pos1 = shuff(pos)

    shuff = PositionRandomize(rand_seed=0)
    pos2 = shuff(pos)

    assert torch.allclose(pos1, pos2)
