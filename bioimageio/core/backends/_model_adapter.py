import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    final,
)

from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from ..common import SupportedWeightsFormat
from ..digest_spec import get_axes_infos, get_member_ids
from ..sample import Sample, SampleBlock, SampleBlockWithOrigin
from ..tensor import Tensor

# Known weight formats in order of priority
# First match wins
DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER: Tuple[SupportedWeightsFormat, ...] = (
    "pytorch_state_dict",
    "tensorflow_saved_model_bundle",
    "torchscript",
    "onnx",
    "keras_hdf5",
)


class ModelAdapter(ABC):
    """
    Represents model *without* any preprocessing or postprocessing.

    ```
    from bioimageio.core import load_description

    model = load_description(...)

    # option 1:
    adapter = ModelAdapter.create(model)
    adapter.forward(...)
    adapter.unload()

    # option 2:
    with ModelAdapter.create(model) as adapter:
        adapter.forward(...)
    ```
    """

    def __init__(self, model_description: AnyModelDescr):
        super().__init__()
        self._model_descr = model_description
        self._input_ids = get_member_ids(model_description.inputs)
        self._output_ids = get_member_ids(model_description.outputs)
        self._input_axes = [
            tuple(a.id for a in get_axes_infos(t)) for t in model_description.inputs
        ]
        self._output_axes = [
            tuple(a.id for a in get_axes_infos(t)) for t in model_description.outputs
        ]
        if isinstance(model_description, v0_4.ModelDescr):
            self._input_is_optional = [False] * len(model_description.inputs)
        else:
            self._input_is_optional = [ipt.optional for ipt in model_description.inputs]

    @final
    @classmethod
    def create(
        cls,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        *,
        devices: Optional[Sequence[str]] = None,
        weight_format_priority_order: Optional[Sequence[SupportedWeightsFormat]] = None,
    ):
        """
        Creates model adapter based on the passed spec
        Note: All specific adapters should happen inside this function to prevent different framework
        initializations interfering with each other
        """
        if not isinstance(model_description, (v0_4.ModelDescr, v0_5.ModelDescr)):
            raise TypeError(
                f"expected v0_4.ModelDescr or v0_5.ModelDescr, but got {type(model_description)}"
            )

        weights = model_description.weights
        errors: List[Tuple[SupportedWeightsFormat, Exception]] = []
        weight_format_priority_order = (
            DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER
            if weight_format_priority_order is None
            else weight_format_priority_order
        )
        # limit weight formats to the ones present
        weight_format_priority_order_present: Sequence[SupportedWeightsFormat] = [
            w for w in weight_format_priority_order if getattr(weights, w) is not None
        ]
        if not weight_format_priority_order_present:
            raise ValueError(
                f"None of the specified weight formats ({weight_format_priority_order}) is present ({weight_format_priority_order_present})"
            )

        for wf in weight_format_priority_order_present:
            if wf == "pytorch_state_dict":
                assert weights.pytorch_state_dict is not None
                try:
                    from .pytorch_backend import PytorchModelAdapter

                    return PytorchModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "tensorflow_saved_model_bundle":
                assert weights.tensorflow_saved_model_bundle is not None
                try:
                    from .tensorflow_backend import create_tf_model_adapter

                    return create_tf_model_adapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "onnx":
                assert weights.onnx is not None
                try:
                    from .onnx_backend import ONNXModelAdapter

                    return ONNXModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "torchscript":
                assert weights.torchscript is not None
                try:
                    from .torchscript_backend import TorchscriptModelAdapter

                    return TorchscriptModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            elif wf == "keras_hdf5":
                assert weights.keras_hdf5 is not None
                # keras can either be installed as a separate package or used as part of tensorflow
                # we try to first import the keras model adapter using the separate package and,
                # if it is not available, try to load the one using tf
                try:
                    try:
                        from .keras_backend import KerasModelAdapter
                    except Exception:
                        from .tensorflow_backend import KerasModelAdapter

                    return KerasModelAdapter(
                        model_description=model_description, devices=devices
                    )
                except Exception as e:
                    errors.append((wf, e))
            else:
                assert_never(wf)

        assert errors
        if len(weight_format_priority_order) == 1:
            assert len(errors) == 1
            wf, e = errors[0]
            raise ValueError(
                f"The '{wf}' model adapter could not be created"
                + f" in this environment:\n{e.__class__.__name__}({e}).\n\n"
            ) from e

        else:
            error_list = "\n - ".join(
                f"{wf}: {e.__class__.__name__}({e})" for wf, e in errors
            )
            raise ValueError(
                "None of the weight format specific model adapters could be created"
                + f" in this environment. Errors are:\n\n{error_list}.\n\n"
            )

    @final
    def load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        warnings.warn("Deprecated. ModelAdapter is loaded on initialization")

    def forward(
        self, input_sample: Union[Sample, SampleBlock, SampleBlockWithOrigin]
    ) -> Sample:
        """
        Run forward pass of model to get model predictions

        Note: sample id and stample stat attributes are passed through
        """
        unexpected = [mid for mid in input_sample.members if mid not in self._input_ids]
        if unexpected:
            warnings.warn(f"Got unexpected input tensor IDs: {unexpected}")

        input_arrays = [
            (
                None
                if (a := input_sample.members.get(in_id)) is None
                else a.transpose(in_order).data.data
            )
            for in_id, in_order in zip(self._input_ids, self._input_axes)
        ]
        output_arrays = self._forward_impl(input_arrays)
        assert len(output_arrays) <= len(self._output_ids)
        output_tensors = [
            None if a is None else Tensor(a, dims=d)
            for a, d in zip(output_arrays, self._output_axes)
        ]
        return Sample(
            members={
                tid: out
                for tid, out in zip(
                    self._output_ids,
                    output_tensors,
                )
                if out is not None
            },
            stat=input_sample.stat,
            id=(
                input_sample.id
                if isinstance(input_sample, Sample)
                else input_sample.sample_id
            ),
        )

    @abstractmethod
    def _forward_impl(
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ) -> Union[List[Optional[NDArray[Any]]], Tuple[Optional[NDArray[Any]]]]:
        """framework specific forward implementation"""

    @abstractmethod
    def unload(self):
        """
        Unload model from any devices, freeing their memory.
        The moder adapter should be considered unusable afterwards.
        """

    def _get_input_args_numpy(self, input_sample: Sample):
        """helper to extract tensor args as transposed numpy arrays"""


create_model_adapter = ModelAdapter.create
