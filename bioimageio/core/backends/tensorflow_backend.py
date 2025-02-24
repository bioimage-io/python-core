from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from loguru import logger
from numpy.typing import NDArray

from bioimageio.core.io import ensure_unzipped
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from ._model_adapter import ModelAdapter


class TensorflowModelAdapter(ModelAdapter):
    weight_format = "tensorflow_saved_model_bundle"

    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        super().__init__(model_description=model_description)

        weight_file = model_description.weights.tensorflow_saved_model_bundle
        if model_description.weights.tensorflow_saved_model_bundle is None:
            raise ValueError("No `tensorflow_saved_model_bundle` weights found")

        if devices is not None:
            logger.warning(
                f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}"
            )

        # TODO: check how to load tf weights without unzipping
        weight_file = ensure_unzipped(
            model_description.weights.tensorflow_saved_model_bundle.source,
            Path("bioimageio_unzipped_tf_weights"),
        )
        self._network = str(weight_file)

    # TODO currently we relaod the model every time. it would be better to keep the graph and session
    # alive in between of forward passes (but then the sessions need to be properly opened / closed)
    def _forward_impl(  # pyright: ignore[reportUnknownParameterType]
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ):
        # TODO read from spec
        tag = (  # pyright: ignore[reportUnknownVariableType]
            tf.saved_model.tag_constants.SERVING  # pyright: ignore[reportAttributeAccessIssue]
        )
        signature_key = (  # pyright: ignore[reportUnknownVariableType]
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY  # pyright: ignore[reportAttributeAccessIssue]
        )

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(  # pyright: ignore[reportAttributeAccessIssue]
                graph=graph
            ) as sess:  # pyright: ignore[reportUnknownVariableType]
                # load the model and the signature
                graph_def = tf.saved_model.loader.load(  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]
                    sess, [tag], self._network
                )
                signature = (  # pyright: ignore[reportUnknownVariableType]
                    graph_def.signature_def
                )

                # get the tensors into the graph
                in_names = [  # pyright: ignore[reportUnknownVariableType]
                    signature[signature_key].inputs[key].name for key in self._input_ids
                ]
                out_names = [  # pyright: ignore[reportUnknownVariableType]
                    signature[signature_key].outputs[key].name
                    for key in self._output_ids
                ]
                in_tf_tensors = [
                    graph.get_tensor_by_name(
                        name  # pyright: ignore[reportUnknownArgumentType]
                    )
                    for name in in_names  # pyright: ignore[reportUnknownVariableType]
                ]
                out_tf_tensors = [
                    graph.get_tensor_by_name(
                        name  # pyright: ignore[reportUnknownArgumentType]
                    )
                    for name in out_names  # pyright: ignore[reportUnknownVariableType]
                ]

                # run prediction
                res = sess.run(  # pyright: ignore[reportUnknownVariableType]
                    dict(
                        zip(
                            out_names,  # pyright: ignore[reportUnknownArgumentType]
                            out_tf_tensors,
                        )
                    ),
                    dict(zip(in_tf_tensors, input_arrays)),
                )
                # from dict to list of tensors
                res = [  # pyright: ignore[reportUnknownVariableType]
                    res[out]
                    for out in out_names  # pyright: ignore[reportUnknownVariableType]
                ]

        return res  # pyright: ignore[reportUnknownVariableType]

    def unload(self) -> None:
        logger.warning(
            "Device management is not implemented for tensorflow 1, cannot unload model"
        )


class KerasModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        if model_description.weights.tensorflow_saved_model_bundle is None:
            raise ValueError("No `tensorflow_saved_model_bundle` weights found")

        super().__init__(model_description=model_description)
        if devices is not None:
            logger.warning(
                f"Device management is not implemented for tensorflow yet, ignoring the devices {devices}"
            )

        # TODO: check how to load tf weights without unzipping
        weight_file = ensure_unzipped(
            model_description.weights.tensorflow_saved_model_bundle.source,
            Path("bioimageio_unzipped_tf_weights"),
        )

        try:
            self._network = tf.keras.layers.TFSMLayer(  # pyright: ignore[reportAttributeAccessIssue]
                weight_file,
                call_endpoint="serve",
            )
        except Exception as e:
            try:
                self._network = tf.keras.layers.TFSMLayer(  # pyright: ignore[reportAttributeAccessIssue]
                    weight_file, call_endpoint="serving_default"
                )
            except Exception as ee:
                logger.opt(exception=ee).info(
                    "keras.layers.TFSMLayer error for alternative call_endpoint='serving_default'"
                )
                raise e

    def _forward_impl(  # pyright: ignore[reportUnknownParameterType]
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ):
        assert tf is not None
        tf_tensor = [
            None if ipt is None else tf.convert_to_tensor(ipt) for ipt in input_arrays
        ]

        result = self._network(*tf_tensor)  # pyright: ignore[reportUnknownVariableType]

        assert isinstance(result, dict)

        # TODO: Use RDF's `outputs[i].id` here
        result = list(  # pyright: ignore[reportUnknownVariableType]
            result.values()  # pyright: ignore[reportUnknownArgumentType]
        )

        return [  # pyright: ignore[reportUnknownVariableType]
            (None if r is None else r if isinstance(r, np.ndarray) else r.numpy())
            for r in result  # pyright: ignore[reportUnknownVariableType]
        ]

    def unload(self) -> None:
        logger.warning(
            "Device management is not implemented for tensorflow>=2 models"
            + f" using `{self.__class__.__name__}`, cannot unload model"
        )


def create_tf_model_adapter(
    model_description: AnyModelDescr, devices: Optional[Sequence[str]]
):
    tf_version = v0_5.Version(tf.__version__)
    weights = model_description.weights.tensorflow_saved_model_bundle
    if weights is None:
        raise ValueError("No `tensorflow_saved_model_bundle` weights found")

    model_tf_version = weights.tensorflow_version
    if model_tf_version is None:
        logger.warning(
            "The model does not specify the tensorflow version."
            + f"Cannot check if it is compatible with intalled tensorflow {tf_version}."
        )
    elif model_tf_version > tf_version:
        logger.warning(
            f"The model specifies a newer tensorflow version than installed: {model_tf_version} > {tf_version}."
        )
    elif (model_tf_version.major, model_tf_version.minor) != (
        tf_version.major,
        tf_version.minor,
    ):
        logger.warning(
            "The tensorflow version specified by the model does not match the installed: "
            + f"{model_tf_version} != {tf_version}."
        )

    if tf_version.major <= 1:
        return TensorflowModelAdapter(
            model_description=model_description, devices=devices
        )
    else:
        return KerasModelAdapter(model_description=model_description, devices=devices)
