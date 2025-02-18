# type: ignore  # TODO: type
from __future__ import annotations

import abc
from bioimageio.spec.model.v0_5 import WeightsEntryDescrBase
from typing import Any, List, Sequence, cast, Union
from typing_extensions import assert_never
import numpy as np
from numpy.testing import assert_array_almost_equal
from bioimageio.spec.model import v0_4, v0_5
from torch.jit import ScriptModule
from bioimageio.core.digest_spec import get_test_inputs, get_member_id
from bioimageio.core.model_adapters._pytorch_model_adapter import PytorchModelAdapter
import os
import shutil
from pathlib import Path
from typing import no_type_check
from zipfile import ZipFile
from bioimageio.spec._internal.version_type import Version
from bioimageio.spec._internal.io_utils import download

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow.saved_model
except Exception:
    tensorflow = None


# additional convenience for pytorch state dict, eventually we want this in python-bioimageio too
# and for each weight format
def load_torch_model(  # pyright: ignore[reportUnknownParameterType]
    node: Union[v0_4.PytorchStateDictWeightsDescr, v0_5.PytorchStateDictWeightsDescr],
):
    assert torch is not None
    model = (  # pyright: ignore[reportUnknownVariableType]
        PytorchModelAdapter.get_network(node)
    )
    state = torch.load(download(node.source).path, map_location="cpu")
    model.load_state_dict(state)  # FIXME: check incompatible keys?
    return model.eval()  # pyright: ignore[reportUnknownVariableType]


class WeightConverter(abc.ABC):
    @abc.abstractmethod
    def convert(
        self, model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr], output_path: Path
    ) -> WeightsEntryDescrBase:
        raise NotImplementedError


class Pytorch2Onnx(WeightConverter):
    def __init__(self):
        super().__init__()
        assert torch is not None

    def convert(
        self,
        model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        output_path: Path,
        use_tracing: bool = True,
        test_decimal: int = 4,
        verbose: bool = False,
        opset_version: int = 15,
    ) -> v0_5.OnnxWeightsDescr:
        """
        Convert model weights from the PyTorch state_dict format to the ONNX format.

        Args:
            model_descr (Union[v0_4.ModelDescr, v0_5.ModelDescr]):
                The model description object that contains the model and its weights.
            output_path (Path):
                The file path where the ONNX model will be saved.
            use_tracing (bool, optional):
                Whether to use tracing or scripting to export the ONNX format. Defaults to True.
            test_decimal (int, optional):
                The decimal precision for comparing the results between the original and converted models.
                This is used in the `assert_array_almost_equal` function to check if the outputs match.
                Defaults to 4.
            verbose (bool, optional):
                If True, will print out detailed information during the ONNX export process. Defaults to False.
            opset_version (int, optional):
                The ONNX opset version to use for the export. Defaults to 15.

        Raises:
            ValueError:
                If the provided model does not have weights in the PyTorch state_dict format.
            ImportError:
                If ONNX Runtime is not available for checking the exported ONNX model.
            ValueError:
                If the results before and after weights conversion do not agree.

        Returns:
            v0_5.OnnxWeightsDescr:
                A descriptor object that contains information about the exported ONNX weights.
        """

        state_dict_weights_descr = model_descr.weights.pytorch_state_dict
        if state_dict_weights_descr is None:
            raise ValueError(
                "The provided model does not have weights in the pytorch state dict format"
            )

        assert torch is not None
        with torch.no_grad():
            sample = get_test_inputs(model_descr)
            input_data = [
                sample.members[get_member_id(ipt)].data.data
                for ipt in model_descr.inputs
            ]
            input_tensors = [torch.from_numpy(ipt) for ipt in input_data]
            model = load_torch_model(state_dict_weights_descr)

            expected_tensors = model(*input_tensors)
            if isinstance(expected_tensors, torch.Tensor):
                expected_tensors = [expected_tensors]
            expected_outputs: List[np.ndarray[Any, Any]] = [
                out.numpy() for out in expected_tensors
            ]

            if use_tracing:
                torch.onnx.export(
                    model,
                    (
                        tuple(input_tensors)
                        if len(input_tensors) > 1
                        else input_tensors[0]
                    ),
                    str(output_path),
                    verbose=verbose,
                    opset_version=opset_version,
                )
            else:
                raise NotImplementedError

        try:
            import onnxruntime as rt  # pyright: ignore [reportMissingTypeStubs]
        except ImportError:
            raise ImportError(
                "The onnx weights were exported, but onnx rt is not available and weights cannot be checked."
            )

        # check the onnx model
        sess = rt.InferenceSession(str(output_path))
        onnx_input_node_args = cast(
            List[Any], sess.get_inputs()
        )  # fixme: remove cast, try using rt.NodeArg instead of Any
        onnx_inputs = {
            input_name.name: inp
            for input_name, inp in zip(onnx_input_node_args, input_data)
        }
        outputs = cast(
            Sequence[np.ndarray[Any, Any]], sess.run(None, onnx_inputs)
        )  # FIXME: remove cast

        try:
            for exp, out in zip(expected_outputs, outputs):
                assert_array_almost_equal(exp, out, decimal=test_decimal)
        except AssertionError as e:
            raise ValueError(
                f"Results before and after weights conversion do not agree:\n {str(e)}"
            )

        return v0_5.OnnxWeightsDescr(
            source=output_path, parent="pytorch_state_dict", opset_version=opset_version
        )


class Pytorch2Torchscipt(WeightConverter):
    def __init__(self):
        super().__init__()
        assert torch is not None

    def convert(
        self,
        model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        output_path: Path,
        use_tracing: bool = True,
    ) -> v0_5.TorchscriptWeightsDescr:
        """
        Convert model weights from the PyTorch `state_dict` format to TorchScript.

        Args:
            model_descr (Union[v0_4.ModelDescr, v0_5.ModelDescr]):
                The model description object that contains the model and its weights in the PyTorch `state_dict` format.
            output_path (Path):
                The file path where the TorchScript model will be saved.
            use_tracing (bool):
                Whether to use tracing or scripting to export the TorchScript format.
                - `True`: Use tracing, which is recommended for models with straightforward control flow.
                - `False`: Use scripting, which is better for models with dynamic control flow (e.g., loops, conditionals).

        Raises:
            ValueError:
                If the provided model does not have weights in the PyTorch `state_dict` format.

        Returns:
            v0_5.TorchscriptWeightsDescr:
                A descriptor object that contains information about the exported TorchScript weights.
        """
        state_dict_weights_descr = model_descr.weights.pytorch_state_dict
        if state_dict_weights_descr is None:
            raise ValueError(
                "The provided model does not have weights in the pytorch state dict format"
            )

        input_data = model_descr.get_input_test_arrays()

        with torch.no_grad():
            input_data = [torch.from_numpy(inp.astype("float32")) for inp in input_data]
            model = load_torch_model(state_dict_weights_descr)
            scripted_module: ScriptModule = (
                torch.jit.trace(model, input_data)
                if use_tracing
                else torch.jit.script(model)
            )
            self._check_predictions(
                model=model,
                scripted_model=scripted_module,
                model_spec=model_descr,
                input_data=input_data,
            )

        scripted_module.save(str(output_path))

        return v0_5.TorchscriptWeightsDescr(
            source=output_path,
            pytorch_version=Version(torch.__version__),
            parent="pytorch_state_dict",
        )

    def _check_predictions(
        self,
        model: Any,
        scripted_model: Any,
        model_spec: v0_4.ModelDescr | v0_5.ModelDescr,
        input_data: Sequence[torch.Tensor],
    ):
        assert torch is not None

        def _check(input_: Sequence[torch.Tensor]) -> None:
            expected_tensors = model(*input_)
            if isinstance(expected_tensors, torch.Tensor):
                expected_tensors = [expected_tensors]
            expected_outputs: List[np.ndarray[Any, Any]] = [
                out.numpy() for out in expected_tensors
            ]

            output_tensors = scripted_model(*input_)
            if isinstance(output_tensors, torch.Tensor):
                output_tensors = [output_tensors]
            outputs: List[np.ndarray[Any, Any]] = [
                out.numpy() for out in output_tensors
            ]

            try:
                for exp, out in zip(expected_outputs, outputs):
                    assert_array_almost_equal(exp, out, decimal=4)
            except AssertionError as e:
                raise ValueError(
                    f"Results before and after weights conversion do not agree:\n {str(e)}"
                )

        _check(input_data)

        if len(model_spec.inputs) > 1:
            return  # FIXME: why don't we check multiple inputs?

        input_descr = model_spec.inputs[0]
        if isinstance(input_descr, v0_4.InputTensorDescr):
            if not isinstance(input_descr.shape, v0_4.ParameterizedInputShape):
                return
            min_shape = input_descr.shape.min
            step = input_descr.shape.step
        else:
            min_shape: List[int] = []
            step: List[int] = []
            for axis in input_descr.axes:
                if isinstance(axis.size, v0_5.ParameterizedSize):
                    min_shape.append(axis.size.min)
                    step.append(axis.size.step)
                elif isinstance(axis.size, int):
                    min_shape.append(axis.size)
                    step.append(0)
                elif axis.size is None:
                    raise NotImplementedError(
                        f"Can't verify inputs that don't specify their shape fully: {axis}"
                    )
                elif isinstance(axis.size, v0_5.SizeReference):
                    raise NotImplementedError(f"Can't handle axes like '{axis}' yet")
                else:
                    assert_never(axis.size)

        input_data = input_data[0]
        max_shape = input_data.shape
        max_steps = 4

        # check that input and output agree for decreasing input sizes
        for step_factor in range(1, max_steps + 1):
            slice_ = tuple(
                (
                    slice(None)
                    if step_dim == 0
                    else slice(0, max_dim - step_factor * step_dim, 1)
                )
                for max_dim, step_dim in zip(max_shape, step)
            )
            sliced_input = input_data[slice_]
            if any(
                sliced_dim < min_dim
                for sliced_dim, min_dim in zip(sliced_input.shape, min_shape)
            ):
                return
            _check([sliced_input])


class Tensorflow2Bundled(WeightConverter):
    def __init__(self):
        super().__init__()
        assert tensorflow is not None

    def convert(
        self, model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr], output_path: Path
    ) -> v0_5.TensorflowSavedModelBundleWeightsDescr:
        """
        Convert model weights from the 'keras_hdf5' format to the 'tensorflow_saved_model_bundle' format.

        This method handles the conversion of Keras HDF5 model weights into a TensorFlow SavedModel bundle,
        which is the recommended format for deploying TensorFlow models. The method supports both TensorFlow 1.x
        and 2.x versions, with appropriate checks to ensure compatibility.

        Adapted from:
        https://github.com/deepimagej/pydeepimagej/blob/5aaf0e71f9b04df591d5ca596f0af633a7e024f5/pydeepimagej/yaml/create_config.py

        Args:
            model_descr (Union[v0_4.ModelDescr, v0_5.ModelDescr]):
                The bioimage.io model description containing the model's metadata and weights.
            output_path (Path):
                The directory where the TensorFlow SavedModel bundle will be saved.
                This path must not already exist and, if necessary, will be zipped into a .zip file.
            use_tracing (bool):
                Placeholder argument; currently not used in this method but required to match the abstract method signature.

        Raises:
            ValueError:
                - If the specified `output_path` already exists.
                - If the Keras HDF5 weights are missing in the model description.
            RuntimeError:
                If there is a mismatch between the TensorFlow version used by the model and the version installed.
            NotImplementedError:
                If the model has multiple inputs or outputs and TensorFlow 1.x is being used.

        Returns:
             v0_5.TensorflowSavedModelBundleWeightsDescr:
                A descriptor object containing information about the converted TensorFlow SavedModel bundle.
        """
        assert tensorflow is not None
        tf_major_ver = int(tensorflow.__version__.split(".")[0])

        if output_path.suffix == ".zip":
            output_path = output_path.with_suffix("")
            zip_weights = True
        else:
            zip_weights = False

        if output_path.exists():
            raise ValueError(f"The ouptut directory at {output_path} must not exist.")

        if model_descr.weights.keras_hdf5 is None:
            raise ValueError("Missing Keras Hdf5 weights to convert from.")

        weight_spec = model_descr.weights.keras_hdf5
        weight_path = download(weight_spec.source).path

        if weight_spec.tensorflow_version:
            model_tf_major_ver = int(weight_spec.tensorflow_version.major)
            if model_tf_major_ver != tf_major_ver:
                raise RuntimeError(
                    f"Tensorflow major versions of model {model_tf_major_ver} is not {tf_major_ver}"
                )

        if tf_major_ver == 1:
            if len(model_descr.inputs) != 1 or len(model_descr.outputs) != 1:
                raise NotImplementedError(
                    "Weight conversion for models with multiple inputs or outputs is not yet implemented."
                )
            return self._convert_tf1(
                weight_path,
                output_path,
                model_descr.inputs[0].id,
                model_descr.outputs[0].id,
                zip_weights,
            )
        else:
            return self._convert_tf2(weight_path, output_path, zip_weights)

    def _convert_tf2(
        self, keras_weight_path: Path, output_path: Path, zip_weights: bool
    ) -> v0_5.TensorflowSavedModelBundleWeightsDescr:
        try:
            # try to build the tf model with the keras import from tensorflow
            from tensorflow import keras
        except Exception:
            # if the above fails try to export with the standalone keras
            import keras

        model = keras.models.load_model(keras_weight_path)
        keras.models.save_model(model, output_path)

        if zip_weights:
            output_path = self._zip_model_bundle(output_path)
        print("TensorFlow model exported to", output_path)

        return v0_5.TensorflowSavedModelBundleWeightsDescr(
            source=output_path,
            parent="keras_hdf5",
            tensorflow_version=Version(tensorflow.__version__),
        )

    # adapted from
    # https://github.com/deepimagej/pydeepimagej/blob/master/pydeepimagej/yaml/create_config.py#L236
    def _convert_tf1(
        self,
        keras_weight_path: Path,
        output_path: Path,
        input_name: str,
        output_name: str,
        zip_weights: bool,
    ) -> v0_5.TensorflowSavedModelBundleWeightsDescr:
        try:
            # try to build the tf model with the keras import from tensorflow
            from tensorflow import (
                keras,  # type: ignore
            )

        except Exception:
            # if the above fails try to export with the standalone keras
            import keras

        @no_type_check
        def build_tf_model():
            keras_model = keras.models.load_model(keras_weight_path)
            assert tensorflow is not None
            builder = tensorflow.saved_model.builder.SavedModelBuilder(output_path)
            signature = (
                tensorflow.saved_model.signature_def_utils.predict_signature_def(
                    inputs={input_name: keras_model.input},
                    outputs={output_name: keras_model.output},
                )
            )

            signature_def_map = {
                tensorflow.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }

            builder.add_meta_graph_and_variables(
                keras.backend.get_session(),
                [tensorflow.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
            )
            builder.save()

        build_tf_model()

        if zip_weights:
            output_path = self._zip_model_bundle(output_path)
        print("TensorFlow model exported to", output_path)

        return v0_5.TensorflowSavedModelBundleWeightsDescr(
            source=output_path,
            parent="keras_hdf5",
            tensorflow_version=Version(tensorflow.__version__),
        )

    def _zip_model_bundle(self, model_bundle_folder: Path):
        zipped_model_bundle = model_bundle_folder.with_suffix(".zip")

        with ZipFile(zipped_model_bundle, "w") as zip_obj:
            for root, _, files in os.walk(model_bundle_folder):
                for filename in files:
                    src = os.path.join(root, filename)
                    zip_obj.write(src, os.path.relpath(src, model_bundle_folder))

        try:
            shutil.rmtree(model_bundle_folder)
        except Exception:
            print("TensorFlow bundled model was not removed after compression")

        return zipped_model_bundle
