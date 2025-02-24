import os
from typing import Any, Optional, Sequence, Union

from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec._internal.io import download
from bioimageio.spec._internal.type_guards import is_list, is_tuple
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import Version

from .._settings import settings
from ..digest_spec import get_axes_infos
from ._model_adapter import ModelAdapter

os.environ["KERAS_BACKEND"] = settings.keras_backend

# by default, we use the keras integrated with tensorflow
# TODO: check if we should prefer keras
try:
    import tensorflow as tf
    from tensorflow import (
        keras,  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]
    )

    tf_version = Version(tf.__version__)
except Exception:
    import keras


class KerasModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(model_description=model_description)
        if model_description.weights.keras_hdf5 is None:
            raise ValueError("model has not keras_hdf5 weights specified")
        model_tf_version = model_description.weights.keras_hdf5.tensorflow_version

        if tf_version is None or model_tf_version is None:
            logger.warning("Could not check tensorflow versions.")
        elif model_tf_version > tf_version:
            logger.warning(
                "The model specifies a newer tensorflow version than installed: {} > {}.",
                model_tf_version,
                tf_version,
            )
        elif (model_tf_version.major, model_tf_version.minor) != (
            tf_version.major,
            tf_version.minor,
        ):
            logger.warning(
                "Model tensorflow version {} does not match {}.",
                model_tf_version,
                tf_version,
            )

        # TODO keras device management
        if devices is not None:
            logger.warning(
                "Device management is not implemented for keras yet, ignoring the devices {}",
                devices,
            )

        weight_path = download(model_description.weights.keras_hdf5.source).path

        self._network = keras.models.load_model(weight_path)
        self._output_axes = [
            tuple(a.id for a in get_axes_infos(out))
            for out in model_description.outputs
        ]

    def _forward_impl(  # pyright: ignore[reportUnknownParameterType]
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ):
        network_output = self._network.predict(*input_arrays)  # type: ignore
        if is_list(network_output) or is_tuple(network_output):
            return network_output
        else:
            return [network_output]  # pyright: ignore[reportUnknownVariableType]

    def unload(self) -> None:
        logger.warning(
            "Device management is not implemented for keras yet, cannot unload model"
        )
