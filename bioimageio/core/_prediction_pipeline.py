import warnings
from types import MappingProxyType
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from tqdm import tqdm

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from ._op_base import BlockedOperator
from .axis import AxisId, PerAxis
from .common import (
    BlocksizeParameter,
    Halo,
    MemberId,
    PerMember,
    SampleId,
    SupportedWeightsFormat,
)
from .digest_spec import (
    get_block_transform,
    get_input_halo,
    get_member_ids,
)
from .model_adapters import ModelAdapter, create_model_adapter
from .model_adapters import get_weight_formats as get_weight_formats
from .proc_ops import Processing
from .proc_setup import setup_pre_and_postprocessing
from .sample import Sample, SampleBlock, SampleBlockWithOrigin
from .stat_measures import DatasetMeasure, MeasureValue, Stat
from .tensor import Tensor

Predict_IO = TypeVar(
    "Predict_IO",
    Sample,
    Iterable[Sample],
)


class PredictionPipeline:
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the `PredictionPipeline` in a with statement
        (as a context manager).
    """

    def __init__(
        self,
        *,
        name: str,
        model_description: AnyModelDescr,
        preprocessing: List[Processing],
        postprocessing: List[Processing],
        model_adapter: ModelAdapter,
        default_ns: Optional[BlocksizeParameter] = None,
        default_blocksize_parameter: BlocksizeParameter = 10,
        default_batch_size: int = 1,
    ) -> None:
        """Use `create_prediction_pipeline` to create a `PredictionPipeline`"""
        super().__init__()
        default_blocksize_parameter = default_ns or default_blocksize_parameter
        if default_ns is not None:
            warnings.warn(
                "Argument `default_ns` is deprecated in favor of"
                + " `default_blocksize_paramter` and will be removed soon."
            )
        del default_ns

        if model_description.run_mode:
            warnings.warn(
                f"Not yet implemented inference for run mode '{model_description.run_mode.name}'"
            )

        self.name = name
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing

        self.model_description = model_description
        if isinstance(model_description, v0_4.ModelDescr):
            self._default_input_halo: PerMember[PerAxis[Halo]] = {}
            self._block_transform = None
        else:
            default_output_halo = {
                t.id: {
                    a.id: Halo(a.halo, a.halo)
                    for a in t.axes
                    if isinstance(a, v0_5.WithHalo)
                }
                for t in model_description.outputs
            }
            self._default_input_halo = get_input_halo(
                model_description, default_output_halo
            )
            self._block_transform = get_block_transform(model_description)

        self._default_blocksize_parameter = default_blocksize_parameter
        self._default_batch_size = default_batch_size

        self._input_ids = get_member_ids(model_description.inputs)
        self._output_ids = get_member_ids(model_description.outputs)

        self._adapter: ModelAdapter = model_adapter

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def predict_sample_block(
        self,
        sample_block: SampleBlockWithOrigin,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> SampleBlock:
        if isinstance(self.model_description, v0_4.ModelDescr):
            raise NotImplementedError(
                f"predict_sample_block not implemented for model {self.model_description.format_version}"
            )
        else:
            assert self._block_transform is not None

        if not skip_preprocessing:
            self.apply_preprocessing(sample_block)

        output_meta = sample_block.get_transformed_meta(self._block_transform)
        local_output = self._adapter.forward(sample_block)

        output = output_meta.with_data(local_output.members, stat=local_output.stat)
        if not skip_postprocessing:
            self.apply_postprocessing(output)

        return output

    def predict_sample_without_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> Sample:
        """predict a sample.
        The sample's tensor shapes have to match the model's input tensor description.
        If that is not the case, consider `predict_sample_with_blocking`"""

        if not skip_preprocessing:
            self.apply_preprocessing(sample)

        output = self._adapter.forward(sample)
        if not skip_postprocessing:
            self.apply_postprocessing(output)

        return output

    def get_output_sample_id(self, input_sample_id: SampleId):
        warnings.warn(
            "`PredictionPipeline.get_output_sample_id()` is deprecated and will be"
            + " removed soon. Output sample id is equal to input sample id, hence this"
            + " function is not needed."
        )
        return input_sample_id

    def predict_sample_with_fixed_blocking(
        self,
        sample: Sample,
        input_block_shape: Mapping[MemberId, Mapping[AxisId, int]],
        *,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> Sample:
        if not skip_preprocessing:
            self.apply_preprocessing(sample)

        n_blocks, input_blocks = sample.split_into_blocks(
            input_block_shape,
            halo=self._default_input_halo,
            pad_mode="reflect",
        )
        input_blocks = list(input_blocks)
        predicted_blocks: List[SampleBlock] = []
        for b in tqdm(
            input_blocks,
            desc=f"predict sample {sample.id or ''} with {self.model_description.id or self.model_description.name}",
            unit="block",
            unit_divisor=1,
            total=n_blocks,
        ):
            predicted_blocks.append(
                self.predict_sample_block(
                    b, skip_preprocessing=True, skip_postprocessing=True
                )
            )

        predicted_sample = Sample.from_blocks(predicted_blocks)
        if not skip_postprocessing:
            self.apply_postprocessing(predicted_sample)

        return predicted_sample

    def predict_sample_with_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        ns: Optional[
            Union[
                v0_5.ParameterizedSize_N,
                Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize_N],
            ]
        ] = None,
        batch_size: Optional[int] = None,
    ) -> Sample:
        """predict a sample by splitting it into blocks according to the model and the `ns` parameter"""

        if isinstance(self.model_description, v0_4.ModelDescr):
            raise NotImplementedError(
                "`predict_sample_with_blocking` not implemented for v0_4.ModelDescr"
                + f" {self.model_description.name}."
                + " Consider using `predict_sample_with_fixed_blocking`"
            )

        ns = ns or self._default_blocksize_parameter
        if isinstance(ns, int):
            ns = {
                (ipt.id, a.id): ns
                for ipt in self.model_description.inputs
                for a in ipt.axes
                if isinstance(a.size, v0_5.ParameterizedSize)
            }
        input_block_shape = self.model_description.get_tensor_sizes(
            ns, batch_size or self._default_batch_size
        ).inputs

        return self.predict_sample_with_fixed_blocking(
            sample,
            input_block_shape=input_block_shape,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )

    # def predict(
    #     self,
    #     inputs: Predict_IO,
    #     skip_preprocessing: bool = False,
    #     skip_postprocessing: bool = False,
    # ) -> Predict_IO:
    #     """Run model prediction **including** pre/postprocessing."""

    #     if isinstance(inputs, Sample):
    #         return self.predict_sample_with_blocking(
    #             inputs,
    #             skip_preprocessing=skip_preprocessing,
    #             skip_postprocessing=skip_postprocessing,
    #         )
    #     elif isinstance(inputs, collections.abc.Iterable):
    #         return (
    #             self.predict(
    #                 ipt,
    #                 skip_preprocessing=skip_preprocessing,
    #                 skip_postprocessing=skip_postprocessing,
    #             )
    #             for ipt in inputs
    #         )
    #     else:
    #         assert_never(inputs)

    def apply_preprocessing(self, sample: Union[Sample, SampleBlockWithOrigin]) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(sample)

    def apply_postprocessing(
        self, sample: Union[Sample, SampleBlock, SampleBlockWithOrigin]
    ) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        for op in self._postprocessing:
            if isinstance(sample, (Sample, SampleBlockWithOrigin)):
                op(sample)
            elif not isinstance(op, BlockedOperator):
                raise NotImplementedError(
                    "block wise update of output statistics not yet implemented"
                )
            else:
                op(sample)

    def load(self):
        """
        optional step: load model onto devices before calling forward if not using it as context manager
        """
        pass

    def unload(self):
        """
        free any device memory in use
        """
        self._adapter.unload()


def create_prediction_pipeline(
    bioimageio_model: AnyModelDescr,
    *,
    devices: Optional[Sequence[str]] = None,
    weight_format: Optional[SupportedWeightsFormat] = None,
    weights_format: Optional[SupportedWeightsFormat] = None,
    dataset_for_initial_statistics: Iterable[Union[Sample, Sequence[Tensor]]] = tuple(),
    keep_updating_initial_dataset_statistics: bool = False,
    fixed_dataset_statistics: Mapping[DatasetMeasure, MeasureValue] = MappingProxyType(
        {}
    ),
    model_adapter: Optional[ModelAdapter] = None,
    ns: Optional[BlocksizeParameter] = None,
    default_blocksize_parameter: BlocksizeParameter = 10,
    **deprecated_kwargs: Any,
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * computation of input statistics
    * preprocessing
    * model prediction
    * computation of output statistics
    * postprocessing

    Args:
        bioimageio_model: A bioimageio model description.
        devices: (optional)
        weight_format: deprecated in favor of **weights_format**
        weights_format: (optional) Use a specific **weights_format** rather than
            choosing one automatically.
            A corresponding `bioimageio.core.model_adapters.ModelAdapter` will be
            created to run inference with the **bioimageio_model**.
        dataset_for_initial_statistics: (optional) If preprocessing steps require input
            dataset statistics, **dataset_for_initial_statistics** allows you to
            specifcy a dataset from which these statistics are computed.
        keep_updating_initial_dataset_statistics: (optional) Set to `True` if you want
            to update dataset statistics with each processed sample.
        fixed_dataset_statistics: (optional) Allows you to specify a mapping of
            `DatasetMeasure`s to precomputed `MeasureValue`s.
        model_adapter: (optional) Allows you to use a custom **model_adapter** instead
            of creating one according to the present/selected **weights_format**.
        ns: deprecated in favor of **default_blocksize_parameter**
        default_blocksize_parameter: Allows to control the default block size for
            blockwise predictions, see `BlocksizeParameter`.

    """
    weights_format = weight_format or weights_format
    del weight_format
    default_blocksize_parameter = ns or default_blocksize_parameter
    del ns
    if deprecated_kwargs:
        warnings.warn(
            f"deprecated create_prediction_pipeline kwargs: {set(deprecated_kwargs)}"
        )

    model_adapter = model_adapter or create_model_adapter(
        model_description=bioimageio_model,
        devices=devices,
        weight_format_priority_order=weights_format and (weights_format,),
    )

    input_ids = get_member_ids(bioimageio_model.inputs)

    def dataset():
        common_stat: Stat = {}
        for i, x in enumerate(dataset_for_initial_statistics):
            if isinstance(x, Sample):
                yield x
            else:
                yield Sample(members=dict(zip(input_ids, x)), stat=common_stat, id=i)

    preprocessing, postprocessing = setup_pre_and_postprocessing(
        bioimageio_model,
        dataset(),
        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_statistics,
        fixed_dataset_stats=fixed_dataset_statistics,
    )

    return PredictionPipeline(
        name=bioimageio_model.name,
        model_description=bioimageio_model,
        model_adapter=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        default_blocksize_parameter=default_blocksize_parameter,
    )
