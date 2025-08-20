import asapdiscovery.ml
import mtenn
import numpy as np
import pytest
import torch
from asapdiscovery.data.backend.openeye import load_openeye_pdb
from asapdiscovery.data.testing.test_resources import fetch_test_file
from numpy.testing import assert_allclose

from asapdiscovery.ml.inference import (
    ModelInference,
    LigandOnlyModelInference,
    # SplitModelInference, # still need to add tests for these
)
from asapdiscovery.ml.models import MLModelRegistry


@pytest.fixture()
def docked_structure_file(scope="session"):
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")


@pytest.fixture()
def smiles():
    smiles = [
        "CC1=CC(=O)C(=C(C1=O)C)C",
        "CC1=CC(=O)C(=C(C1=O)C",
        "CC1=CC(=O)C(=C(C1=O)C",
    ]
    return smiles


@pytest.fixture()
def test_registry():
    reg_yaml_file = fetch_test_file(
        "ml_testing/inference_testing_mtenn_0_7_0/test_models.yaml"
    )
    return MLModelRegistry.from_yaml(str(reg_yaml_file))


@pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro"])
def test_ligandonlyinference_construct_by_latest(target, test_registry):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        target, model_registry=test_registry
    )
    assert inference_cls is not None
    assert inference_cls.model_type == "ligand"
    assert target in inference_cls.targets


def test_ligandonlyinference_construct_from_name(tmp_path, test_registry):
    inference_cls = LigandOnlyModelInference.from_model_name(
        "ligand_gat", local_dir=tmp_path, model_registry=test_registry
    )
    assert inference_cls is not None
    assert inference_cls.local_model_spec.local_dir == tmp_path


def test_ligandonlyinference_weights(tmp_path, test_registry):
    inference_cls = LigandOnlyModelInference.from_model_name(
        "ligand_gat", local_dir=tmp_path, model_registry=test_registry
    )
    wts_file_params = torch.load(
        inference_cls.local_model_spec.weights_file,
        map_location=inference_cls.device,
    )

    param_mismatches = []
    for model in inference_cls.models:
        for k, model_param in model.named_parameters():
            if not torch.allclose(model_param, wts_file_params[k]):
                param_mismatches.append(k)

    assert len(param_mismatches) == 0, param_mismatches


def test_ligandonlyinference_predict(test_data, test_registry):
    inference_cls = LigandOnlyModelInference.from_model_name(
        "ligand_gat", model_registry=test_registry
    )
    g1, _, _, _ = test_data
    assert inference_cls is not None
    output = inference_cls.predict(g1)
    assert output is not None


def test_ligandonlyinference_predict_err(test_data, test_registry):
    inference_cls = LigandOnlyModelInference.from_model_name(
        "ligand_gat", model_registry=test_registry
    )
    g1, _, _, _ = test_data
    assert inference_cls is not None
    pred, err = inference_cls.predict(g1, return_err=True)
    assert pred is not None
    assert err is not None


@pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro"])
def test_ligandonlyinference_predict_smiles_equivariant(
    test_data, target, test_registry
):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        target, model_registry=test_registry
    )
    g1, g2, _, _ = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2, rtol=1e-5)


def test_ligandonlyinference_predict_from_smiles_err_gds(test_data, test_registry):
    inference_cls = LigandOnlyModelInference.from_model_name(
        "ligand_gat", model_registry=test_registry
    )
    g1, g2, g3, gds = test_data
    # same data different smiles order
    assert inference_cls is not None
    smiles = [pose["smiles"] for _, pose in gds]
    assert inference_cls is not None
    pred, err = inference_cls.predict_from_smiles(smiles, return_err=True)
    # check they are flat arrays
    assert pred is not None
    assert err is not None
    assert len(pred.shape) == 1
    assert len(err.shape) == 1


# test inference dataset cls against training dataset cls
@pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro"])
def test_ligandonlyinference_predict_dataset(test_data, target, test_registry):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        target, model_registry=test_registry
    )
    g1, g2, g3, _ = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    output3 = inference_cls.predict(g3)
    assert_allclose(output1, output2, rtol=1e-5)

    # test that the ones that should be different are
    assert not np.allclose(output3, output1, rtol=1e-5)
    assert not np.allclose(output3, output2, rtol=1e-5)


@pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro"])
def test_ligandonlyinference_predict_from_smiles_dataset(
    test_data, target, test_registry
):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        target, model_registry=test_registry
    )
    g1, g2, g3, gds = test_data
    # same data different smiles order
    assert inference_cls is not None
    smiles = [pose["smiles"] for _, pose in gds]
    s1 = smiles[0]
    s2 = smiles[1]
    # smiles one and two are the same
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2, rtol=1e-5)

    # smiles one and three are different
    s3 = smiles[2]
    output3 = inference_cls.predict(g3)
    assert not np.allclose(output3, output1, rtol=1e-5)

    # test predicting directly from smiles
    output_smiles_1 = inference_cls.predict_from_smiles(s1)
    output_smiles_2 = inference_cls.predict_from_smiles(s2)
    output_smiles_3 = inference_cls.predict_from_smiles(s3)

    assert_allclose(output_smiles_1, output_smiles_2, rtol=1e-5)
    assert_allclose(output1, output_smiles_1, rtol=1e-5)

    assert_allclose(output3, output_smiles_3, rtol=1e-5)
    assert not np.allclose(output_smiles_3, output_smiles_1, rtol=1e-5)
    assert not np.allclose(output3, output_smiles_1, rtol=1e-5)

    # test predicting list of smiles
    output_arr = inference_cls.predict_from_smiles([s1, s2, s3], return_err=False)
    smiles_arr = np.array([output_smiles_1, output_smiles_2, output_smiles_3])
    assert_allclose(output_arr, smiles_arr, rtol=1e-5),
    # check they are the same shape
    assert output_arr.shape == smiles_arr.shape


def test_ligandonlyinference_predict_from_subset(test_data, test_registry):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", model_registry=test_registry
    )

    _, _, _, gids = test_data
    gids_subset = gids[0:2:1]
    for _, g in gids_subset:
        res = inference_cls.predict(g)
        assert res


def test_ligandonlyinference_predict_from_smiles_err(test_registry):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", model_registry=test_registry
    )
    pred, err = inference_cls.predict_from_smiles("CCC", return_err=True)
    # check both are single floats
    assert isinstance(pred, float)
    assert isinstance(err, float)


def test_ligandonlyinference_predict_from_smiles_err_multi(smiles, test_registry):
    inference_cls = LigandOnlyModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", model_registry=test_registry
    )

    pred, err = inference_cls.predict_from_smiles(smiles, return_err=True)
    assert pred is not None
    assert err is not None
    assert len(pred.shape) == 1
    assert len(err.shape) == 1


def test_model_schnet_inference_construct(test_registry):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="schnet", model_registry=test_registry
    )
    assert inference_cls is not None
    assert inference_cls.model_type == "model"
    assert inference_cls.representation_type == "schnet"
    assert type(inference_cls.models[0].readout) is mtenn.readout.PIC50Readout


def test_model_schnet_inference_weights(tmp_path, test_registry):
    inference_cls = ModelInference.from_model_name(
        "model_schnet", local_dir=tmp_path, model_registry=test_registry
    )
    wts_file_params = torch.load(
        inference_cls.local_model_spec.weights_file,
        map_location=inference_cls.device,
    )

    param_mismatches = []
    for model in inference_cls.models:
        for k, model_param in model.named_parameters():
            if not torch.allclose(model_param, wts_file_params[k]):
                param_mismatches.append(k)

    assert len(param_mismatches) == 0, param_mismatches


def test_model_schnet_inference_predict_from_structure_file(
    docked_structure_file, test_registry
):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="schnet", model_registry=test_registry
    )
    assert inference_cls is not None
    output = inference_cls.predict_from_structure_file(docked_structure_file)
    # check its a single float
    assert isinstance(output, float)


def test_model_schnet_inference_predict_from_structure_file_err(
    docked_structure_file, test_registry
):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="schnet", model_registry=test_registry
    )
    assert inference_cls is not None
    pred, err = inference_cls.predict_from_structure_file(
        docked_structure_file, return_err=True
    )
    # check both are single floats
    assert isinstance(pred, float)
    assert isinstance(err, float)


def test_model_schnet_inference_predict_from_structure_file_err_multi(
    docked_structure_file, test_registry
):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="schnet", model_registry=test_registry
    )
    assert inference_cls is not None
    pred, err = inference_cls.predict_from_structure_file(
        [docked_structure_file, docked_structure_file], return_err=True
    )
    assert pred is not None
    assert err is not None
    assert len(pred.shape) == 1
    assert len(err.shape) == 1
    np.all(np.isclose(pred, pred[0]))
    np.all(np.isclose(err, err[0]))


def test_model_schnet_inference_predict_from_pose(docked_structure_file, test_registry):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="schnet", model_registry=test_registry
    )

    dataset = asapdiscovery.ml.dataset.DockedDataset.from_files(
        str_fns=[docked_structure_file],
        compounds=[("Mpro-P0008_0A", "ERI-UCB-ce40166b-17")],
    )
    assert inference_cls is not None
    c, pose = dataset[0]
    output = inference_cls.predict(pose)
    assert output is not None


def test_model_schnet_inference_predict_from_oemol(
    docked_structure_file, test_registry
):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="schnet", model_registry=test_registry
    )

    pose_oemol = load_openeye_pdb(docked_structure_file)
    assert inference_cls is not None
    output = inference_cls.predict_from_oemol(pose_oemol)
    assert output is not None


def test_model_e3nn_inference_construct(test_registry):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="e3nn", model_registry=test_registry
    )
    assert inference_cls is not None
    assert inference_cls.model_type == "model"
    assert inference_cls.representation_type == "e3nn"
    assert type(inference_cls.models[0].readout) is mtenn.readout.PIC50Readout


def test_model_e3nn_predict_from_structure_file(docked_structure_file, test_registry):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="e3nn", model_registry=test_registry
    )
    assert inference_cls is not None
    output = inference_cls.predict_from_structure_file(docked_structure_file)
    # check its a single float
    assert isinstance(output, float)


def test_model_e3nn_predict_from_structure_file_err(
    docked_structure_file, test_registry
):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="e3nn", model_registry=test_registry
    )
    assert inference_cls is not None
    pred, err = inference_cls.predict_from_structure_file(
        docked_structure_file, return_err=True
    )
    # check both are single floats
    assert isinstance(pred, float)
    assert isinstance(err, float)


def test_model_e3nn_predict_from_structure_file_err_multi(
    docked_structure_file, test_registry
):
    inference_cls = ModelInference.from_latest_by_target(
        "SARS-CoV-2-Mpro", representation_type="e3nn", model_registry=test_registry
    )
    assert inference_cls is not None
    pred, err = inference_cls.predict_from_structure_file(
        [docked_structure_file, docked_structure_file], return_err=True
    )
    assert pred is not None
    assert err is not None
    assert len(pred.shape) == 1
    assert len(err.shape) == 1
    np.all(np.isclose(pred, pred[0]))
    np.all(np.isclose(err, err[0]))
