from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from torch.jit import ScriptModule
from typing_extensions import assert_never

from bioimageio.spec._internal.version_type import Version
from bioimageio.spec.model import v0_4, v0_5

from .. import __version__
from ..backends.pytorch_backend import load_torch_model


def convert(
    model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    *,
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
        input_data = [torch.from_numpy(inp) for inp in input_data]
        model = load_torch_model(state_dict_weights_descr, load_state=True)
        scripted_model: Union[  # pyright: ignore[reportUnknownVariableType]
            ScriptModule, Tuple[Any, ...]
        ] = (
            torch.jit.trace(model, input_data)
            if use_tracing
            else torch.jit.script(model)
        )
        assert not isinstance(scripted_model, tuple), scripted_model
        _check_predictions(  # TODO: remove
            model=model,
            scripted_model=scripted_model,
            model_spec=model_descr,
            input_data=input_data,
        )

    scripted_model.save(output_path)

    return v0_5.TorchscriptWeightsDescr(
        source=output_path,
        pytorch_version=Version(torch.__version__),
        parent="pytorch_state_dict",
        comment=(
            f"Converted with bioimageio.core {__version__}"
            + f" with use_tracing={use_tracing}."
        ),
    )


def _check_predictions(
    model: Any,
    scripted_model: Any,
    model_spec: v0_4.ModelDescr | v0_5.ModelDescr,
    input_data: Sequence[torch.Tensor],
):
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
        outputs: List[np.ndarray[Any, Any]] = [out.numpy() for out in output_tensors]

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

    input_tensor = input_data[0]
    max_shape = input_tensor.shape
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
        sliced_input = input_tensor[slice_]
        if any(
            sliced_dim < min_dim
            for sliced_dim, min_dim in zip(sliced_input.shape, min_shape)
        ):
            return
        _check([sliced_input])
