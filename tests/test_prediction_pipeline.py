from pathlib import Path

from numpy.testing import assert_array_almost_equal

from bioimageio.core.common import SupportedWeightsFormat
from bioimageio.spec import load_description
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr04
from bioimageio.spec.model.v0_5 import ModelDescr


def _test_prediction_pipeline(
    model_package: Path, weights_format: SupportedWeightsFormat
):
    from bioimageio.core._prediction_pipeline import create_prediction_pipeline
    from bioimageio.core.digest_spec import get_test_inputs, get_test_outputs

    bio_model = load_description(model_package)
    assert isinstance(
        bio_model, (ModelDescr, ModelDescr04)
    ), bio_model.validation_summary.format()
    pp = create_prediction_pipeline(
        bioimageio_model=bio_model, weight_format=weights_format
    )

    inputs = get_test_inputs(bio_model)
    outputs = pp.predict_sample_without_blocking(inputs)

    expected_outputs = get_test_outputs(bio_model)
    assert len(outputs.shape) == len(expected_outputs.shape)
    for m in expected_outputs.members:
        out = outputs.members[m].data
        assert out is not None
        exp = expected_outputs.members[m].data
        assert_array_almost_equal(out, exp, decimal=4)


def test_prediction_pipeline_torch(any_torch_model: Path):
    _test_prediction_pipeline(any_torch_model, "pytorch_state_dict")


def test_prediction_pipeline_torchscript(any_torchscript_model: Path):
    _test_prediction_pipeline(any_torchscript_model, "torchscript")


def test_prediction_pipeline_onnx(any_onnx_model: Path):
    _test_prediction_pipeline(any_onnx_model, "onnx")


def test_prediction_pipeline_tensorflow(any_tensorflow_model: Path):
    _test_prediction_pipeline(any_tensorflow_model, "tensorflow_saved_model_bundle")


def test_prediction_pipeline_keras(any_keras_model: Path):
    _test_prediction_pipeline(any_keras_model, "keras_hdf5")
