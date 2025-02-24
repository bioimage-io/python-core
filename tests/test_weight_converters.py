# type: ignore  # TODO enable type checking
import os
import zipfile
from pathlib import Path

import pytest

from bioimageio.core import test_model
from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5


def test_pytorch_to_torchscript(any_torch_model, tmp_path):
    from bioimageio.core.weight_converters.pytorch_to_torchscript import convert

    model_descr = load_description(any_torch_model)
    out_path = tmp_path / "weights.pt"
    ret_val = convert(model_descr, out_path)
    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TorchscriptWeightsDescr)
    assert ret_val.source == out_path
    model_descr.weights.torchscript = ret_val
    summary = test_model(model_descr, weight_format="torchscript")
    assert summary.status == "passed", summary.display()


def test_pytorch_to_onnx(convert_to_onnx, tmp_path):
    from bioimageio.core.weight_converters.pytorch_to_onnx import convert

    model_descr = load_description(convert_to_onnx, format_version="latest")
    out_path = tmp_path / "weights.onnx"
    opset_version = 15
    ret_val = convert(
        model_descr=model_descr,
        output_path=out_path,
        opset_version=opset_version,
    )
    assert os.path.exists(out_path)
    assert isinstance(ret_val, v0_5.OnnxWeightsDescr)
    assert ret_val.opset_version == opset_version
    assert ret_val.source == out_path

    model_descr.weights.onnx = ret_val
    summary = test_model(model_descr, weight_format="onnx")
    assert summary.status == "passed", summary.display()


def test_keras_to_tensorflow(any_keras_model: Path, tmp_path: Path):
    from bioimageio.core.weight_converters.keras_to_tensorflow import convert

    model_descr = load_description(any_keras_model)
    out_path = tmp_path / "weights"
    ret_val = convert(model_descr, output_path=out_path)
    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)
    assert ret_val.source == out_path

    model_descr.weights.keras = ret_val
    summary = test_model(model_descr, weight_format="keras_hdf5")
    assert summary.status == "passed", summary.display()


@pytest.mark.skip()
def test_keras_to_tensorflow_zipped(any_keras_model: Path, tmp_path: Path):
    from bioimageio.core.weight_converters.keras_to_tensorflow import convert

    out_path = tmp_path / "weights.zip"
    model_descr = load_description(any_keras_model)
    ret_val = convert(model_descr, out_path)

    assert out_path.exists()
    assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)

    expected_names = {"saved_model.pb", "variables/variables.index"}
    with zipfile.ZipFile(out_path, "r") as f:
        names = set([name for name in f.namelist()])
    assert len(expected_names - names) == 0

    model_descr.weights.keras = ret_val
    summary = test_model(model_descr, weight_format="keras_hdf5")
    assert summary.status == "passed", summary.display()


# TODO: add tensorflow_to_keras converter
# def test_tensorflow_to_keras(any_tensorflow_model: Path, tmp_path: Path):
#     from bioimageio.core.weight_converters.tensorflow_to_keras import convert

#     model_descr = load_description(any_tensorflow_model)
#     out_path = tmp_path / "weights.h5"
#     ret_val = convert(model_descr, output_path=out_path)
#     assert out_path.exists()
#     assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)
#     assert ret_val.source == out_path

#     model_descr.weights.keras = ret_val
#     summary = test_model(model_descr, weight_format="keras_hdf5")
#     assert summary.status == "passed", summary.display()


# @pytest.mark.skip()
# def test_tensorflow_to_keras_zipped(any_tensorflow_model: Path, tmp_path: Path):
#     from bioimageio.core.weight_converters.tensorflow_to_keras import convert

#     out_path = tmp_path / "weights.zip"
#     model_descr = load_description(any_tensorflow_model)
#     ret_val = convert(model_descr, out_path)

#     assert out_path.exists()
#     assert isinstance(ret_val, v0_5.TensorflowSavedModelBundleWeightsDescr)

#     expected_names = {"saved_model.pb", "variables/variables.index"}
#     with zipfile.ZipFile(out_path, "r") as f:
#         names = set([name for name in f.namelist()])
#     assert len(expected_names - names) == 0

#     model_descr.weights.keras = ret_val
#     summary = test_model(model_descr, weight_format="keras_hdf5")
#     assert summary.status == "passed", summary.display()
