import gc
import warnings
from typing import Any, List, Optional, Sequence, Union

import torch
from numpy.typing import NDArray

from bioimageio.spec._internal.type_guards import is_list, is_tuple
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.utils import download

from ..model_adapters import ModelAdapter


class TorchscriptModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
        devices: Optional[Sequence[str]] = None,
    ):
        super().__init__(model_description=model_description)
        if model_description.weights.torchscript is None:
            raise ValueError(
                f"No torchscript weights found for model {model_description.name}"
            )

        weight_path = download(model_description.weights.torchscript.source).path
        if devices is None:
            self.devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        else:
            self.devices = [torch.device(d) for d in devices]

        if len(self.devices) > 1:
            warnings.warn(
                "Multiple devices for single torchscript model not yet implemented"
            )

        self._model = torch.jit.load(weight_path)
        self._model.to(self.devices[0])
        self._model = self._model.eval()

    def _forward_impl(
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ) -> List[Optional[NDArray[Any]]]:

        with torch.no_grad():
            torch_tensor = [
                None if a is None else torch.from_numpy(a).to(self.devices[0])
                for a in input_arrays
            ]
            output: Any = self._model.forward(*torch_tensor)
            if is_list(output) or is_tuple(output):
                output_seq: Sequence[Any] = output
            else:
                output_seq = [output]

            return [
                (
                    None
                    if r is None
                    else r.cpu().numpy() if isinstance(r, torch.Tensor) else r
                )
                for r in output_seq
            ]

    def unload(self) -> None:
        self._devices = None
        del self._model
        _ = gc.collect()  # deallocate memory
        torch.cuda.empty_cache()  # release reserved memory
