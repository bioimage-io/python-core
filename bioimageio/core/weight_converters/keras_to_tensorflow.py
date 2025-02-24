import os
import shutil
from pathlib import Path
from typing import Union, no_type_check
from zipfile import ZipFile

import tensorflow

from bioimageio.core.io import ensure_unzipped
from bioimageio.spec._internal.io import download
from bioimageio.spec._internal.version_type import Version
from bioimageio.spec.common import ZipPath
from bioimageio.spec.model import v0_4, v0_5

try:
    # try to build the tf model with the keras import from tensorflow
    from tensorflow import keras  # type: ignore
except Exception:
    # if the above fails try to export with the standalone keras
    import keras


def convert(
    model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr], *, output_path: Path
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

        input_name = str(
            d.id
            if isinstance((d := model_descr.inputs[0]), v0_5.InputTensorDescr)
            else d.name
        )
        output_name = str(
            d.id
            if isinstance((d := model_descr.outputs[0]), v0_5.OutputTensorDescr)
            else d.name
        )
        return _convert_tf1(
            ensure_unzipped(weight_path, Path("bioimageio_unzipped_tf_weights")),
            output_path,
            input_name,
            output_name,
            zip_weights,
        )
    else:
        return _convert_tf2(weight_path, output_path, zip_weights)


def _convert_tf2(
    keras_weight_path: Union[Path, ZipPath], output_path: Path, zip_weights: bool
) -> v0_5.TensorflowSavedModelBundleWeightsDescr:
    model = keras.models.load_model(keras_weight_path)  # type: ignore
    keras.models.save_model(model, output_path)  # type: ignore

    if zip_weights:
        output_path = _zip_model_bundle(output_path)
    print("TensorFlow model exported to", output_path)

    return v0_5.TensorflowSavedModelBundleWeightsDescr(
        source=output_path,
        parent="keras_hdf5",
        tensorflow_version=Version(tensorflow.__version__),
    )


# adapted from
# https://github.com/deepimagej/pydeepimagej/blob/master/pydeepimagej/yaml/create_config.py#L236
def _convert_tf1(
    keras_weight_path: Path,
    output_path: Path,
    input_name: str,
    output_name: str,
    zip_weights: bool,
) -> v0_5.TensorflowSavedModelBundleWeightsDescr:

    @no_type_check
    def build_tf_model():
        keras_model = keras.models.load_model(keras_weight_path)
        assert tensorflow is not None
        builder = tensorflow.saved_model.builder.SavedModelBuilder(output_path)
        signature = tensorflow.saved_model.signature_def_utils.predict_signature_def(
            inputs={input_name: keras_model.input},
            outputs={output_name: keras_model.output},
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
        output_path = _zip_model_bundle(output_path)
    print("TensorFlow model exported to", output_path)

    return v0_5.TensorflowSavedModelBundleWeightsDescr(
        source=output_path,
        parent="keras_hdf5",
        tensorflow_version=Version(tensorflow.__version__),
    )


def _zip_model_bundle(model_bundle_folder: Path):
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
