import collections.abc
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import (
    Collection,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr
from typing_extensions import Self, assert_never

from bioimageio.core.digest_spec import get_member_id
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import (
    _convert_proc,  # pyright: ignore [reportPrivateUsage]
)

from ._op_base import BlockedOperator, Operator
from .axis import AxisId, PerAxis
from .block import Block
from .common import DTypeStr, MemberId
from .sample import Sample, SampleBlock, SampleBlockWithOrigin
from .stat_calculators import StatsCalculator
from .stat_measures import (
    DatasetMean,
    DatasetMeasure,
    DatasetPercentile,
    DatasetStd,
    MeanMeasure,
    Measure,
    MeasureValue,
    SampleMean,
    SampleQuantile,
    SampleStd,
    Stat,
    StdMeasure,
)
from .tensor import Tensor


def _convert_axis_ids(
    axes: v0_4.AxesInCZYX,
    mode: Literal["per_sample", "per_dataset"],
) -> Tuple[AxisId, ...]:
    if not isinstance(axes, str):
        return tuple(axes)

    if mode == "per_sample":
        ret = []
    elif mode == "per_dataset":
        ret = [v0_5.BATCH_AXIS_ID]
    else:
        assert_never(mode)

    ret.extend(
        [
            AxisId(v0_5._AXIS_ID_MAP.get(a, a))  # pyright: ignore[reportPrivateUsage]
            for a in axes
        ]
    )
    return tuple(ret)


@dataclass
class _SimpleOperator(BlockedOperator, ABC):
    input: MemberId
    output: MemberId

    @property
    def required_measures(self) -> Collection[Measure]:
        return set()

    @abstractmethod
    def get_output_shape(self, input_shape: PerAxis[int]) -> PerAxis[int]: ...

    def __call__(self, sample: Union[Sample, SampleBlock]) -> None:
        if self.input not in sample.members:
            return

        input_tensor = sample.members[self.input]
        output_tensor = self._apply(input_tensor, sample.stat)

        if self.output in sample.members:
            assert (
                sample.members[self.output].tagged_shape == output_tensor.tagged_shape
            )

        if isinstance(sample, Sample):
            sample.members[self.output] = output_tensor
        elif isinstance(sample, SampleBlock):
            b = sample.blocks[self.input]
            sample.blocks[self.output] = Block(
                sample_shape=self.get_output_shape(sample.shape[self.input]),
                data=output_tensor,
                inner_slice=b.inner_slice,
                halo=b.halo,
                block_index=b.block_index,
                blocks_in_sample=b.blocks_in_sample,
            )
        else:
            assert_never(sample)

    @abstractmethod
    def _apply(self, input: Tensor, stat: Stat) -> Tensor: ...


@dataclass
class AddKnownDatasetStats(BlockedOperator):
    dataset_stats: Mapping[DatasetMeasure, MeasureValue]

    @property
    def required_measures(self) -> Set[Measure]:
        return set()

    def __call__(self, sample: Union[Sample, SampleBlock]) -> None:
        sample.stat.update(self.dataset_stats.items())


# @dataclass
# class UpdateStats(Operator):
#     """Calculates sample and/or dataset measures"""

#     measures: Union[Sequence[Measure], Set[Measure], Mapping[Measure, MeasureValue]]
#     """sample and dataset `measuers` to be calculated by this operator. Initial/fixed
#     dataset measure values may be given, see `keep_updating_dataset_stats` for details.
#     """
#     keep_updating_dataset_stats: Optional[bool] = None
#     """indicates if operator calls should keep updating dataset statistics or not

#     default (None): if `measures` is a `Mapping` (i.e. initial measure values are
#     given) no further updates to dataset statistics is conducted, otherwise (w.o.
#     initial measure values) dataset statistics are updated by each processed sample.
#     """
#     _keep_updating_dataset_stats: bool = field(init=False)
#     _stats_calculator: StatsCalculator = field(init=False)

#     @property
#     def required_measures(self) -> Set[Measure]:
#         return set()

#     def __post_init__(self):
#         self._stats_calculator = StatsCalculator(self.measures)
#         if self.keep_updating_dataset_stats is None:
#             self._keep_updating_dataset_stats = not isinstance(self.measures, collections.abc.Mapping)
#         else:
#             self._keep_updating_dataset_stats = self.keep_updating_dataset_stats

#     def __call__(self, sample_block: SampleBlockWithOrigin> None:
#         if self._keep_updating_dataset_stats:
#             sample.stat.update(self._stats_calculator.update_and_get_all(sample))
#         else:
#             sample.stat.update(self._stats_calculator.skip_update_and_get_all(sample))


@dataclass
class UpdateStats(Operator):
    """Calculates sample and/or dataset measures"""

    stats_calculator: StatsCalculator
    """`StatsCalculator` to be used by this operator."""
    keep_updating_initial_dataset_stats: bool = False
    """indicates if operator calls should keep updating initial dataset statistics or not;
    if the `stats_calculator` was not provided with any initial dataset statistics,
    these are always updated with every new sample.
    """
    _keep_updating_dataset_stats: bool = field(init=False)

    @property
    def required_measures(self) -> Set[Measure]:
        return set()

    def __post_init__(self):
        self._keep_updating_dataset_stats = (
            self.keep_updating_initial_dataset_stats
            or not self.stats_calculator.has_dataset_measures
        )

    def __call__(self, sample: Union[Sample, SampleBlockWithOrigin]) -> None:
        if isinstance(sample, SampleBlockWithOrigin):
            # update stats with whole sample on first block
            if sample.block_index != 0:
                return

            origin = sample.origin
        else:
            origin = sample

        if self._keep_updating_dataset_stats:
            sample.stat.update(self.stats_calculator.update_and_get_all(origin))
        else:
            sample.stat.update(self.stats_calculator.skip_update_and_get_all(origin))


@dataclass
class Binarize(_SimpleOperator):
    """'output = tensor > threshold'."""

    threshold: Union[float, Sequence[float]]
    axis: Optional[AxisId] = None

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input > self.threshold

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls, descr: Union[v0_4.BinarizeDescr, v0_5.BinarizeDescr], member_id: MemberId
    ) -> Self:
        if isinstance(descr.kwargs, (v0_4.BinarizeKwargs, v0_5.BinarizeKwargs)):
            return cls(
                input=member_id, output=member_id, threshold=descr.kwargs.threshold
            )
        elif isinstance(descr.kwargs, v0_5.BinarizeAlongAxisKwargs):
            return cls(
                input=member_id,
                output=member_id,
                threshold=descr.kwargs.threshold,
                axis=descr.kwargs.axis,
            )
        else:
            assert_never(descr.kwargs)


@dataclass
class Clip(_SimpleOperator):
    min: Optional[float] = None
    """minimum value for clipping"""
    max: Optional[float] = None
    """maximum value for clipping"""

    def __post_init__(self):
        assert self.min is not None or self.max is not None, "missing min or max value"
        assert (
            self.min is None or self.max is None or self.min < self.max
        ), f"expected min < max, but {self.min} !< {self.max}"

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input.clip(self.min, self.max)

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls, descr: Union[v0_4.ClipDescr, v0_5.ClipDescr], member_id: MemberId
    ) -> Self:
        return cls(
            input=member_id,
            output=member_id,
            min=descr.kwargs.min,
            max=descr.kwargs.max,
        )


@dataclass
class EnsureDtype(_SimpleOperator):
    dtype: DTypeStr

    @classmethod
    def from_proc_descr(cls, descr: v0_5.EnsureDtypeDescr, member_id: MemberId):
        return cls(input=member_id, output=member_id, dtype=descr.kwargs.dtype)

    def get_descr(self):
        return v0_5.EnsureDtypeDescr(kwargs=v0_5.EnsureDtypeKwargs(dtype=self.dtype))

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input.astype(self.dtype)


@dataclass
class ScaleLinear(_SimpleOperator):
    gain: Union[float, xr.DataArray] = 1.0
    """multiplicative factor"""

    offset: Union[float, xr.DataArray] = 0.0
    """additive term"""

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return input * self.gain + self.offset

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: Union[v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr],
        member_id: MemberId,
    ) -> Self:
        kwargs = descr.kwargs
        if isinstance(kwargs, v0_5.ScaleLinearKwargs):
            axis = None
        elif isinstance(kwargs, v0_5.ScaleLinearAlongAxisKwargs):
            axis = kwargs.axis
        elif isinstance(kwargs, v0_4.ScaleLinearKwargs):
            if kwargs.axes is not None:
                raise NotImplementedError(
                    "model.v0_4.ScaleLinearKwargs with axes not implemented, please consider updating the model to v0_5."
                )
            axis = None
        else:
            assert_never(kwargs)

        if axis:
            gain = xr.DataArray(np.atleast_1d(kwargs.gain), dims=axis)
            offset = xr.DataArray(np.atleast_1d(kwargs.offset), dims=axis)
        else:
            assert (
                isinstance(kwargs.gain, (float, int)) or len(kwargs.gain) == 1
            ), kwargs.gain
            gain = (
                kwargs.gain if isinstance(kwargs.gain, (float, int)) else kwargs.gain[0]
            )
            assert isinstance(kwargs.offset, (float, int)) or len(kwargs.offset) == 1
            offset = (
                kwargs.offset
                if isinstance(kwargs.offset, (float, int))
                else kwargs.offset[0]
            )

        return cls(input=member_id, output=member_id, gain=gain, offset=offset)


@dataclass
class ScaleMeanVariance(_SimpleOperator):
    axes: Optional[Sequence[AxisId]] = None
    reference_tensor: Optional[MemberId] = None
    eps: float = 1e-6
    mean: Union[SampleMean, DatasetMean] = field(init=False)
    std: Union[SampleStd, DatasetStd] = field(init=False)
    ref_mean: Union[SampleMean, DatasetMean] = field(init=False)
    ref_std: Union[SampleStd, DatasetStd] = field(init=False)

    @property
    def required_measures(self):
        return {self.mean, self.std, self.ref_mean, self.ref_std}

    def __post_init__(self):
        axes = None if self.axes is None else tuple(self.axes)
        ref_tensor = self.reference_tensor or self.input
        if axes is None or AxisId("batch") not in axes:
            Mean = SampleMean
            Std = SampleStd
        else:
            Mean = DatasetMean
            Std = DatasetStd

        self.mean = Mean(member_id=self.input, axes=axes)
        self.std = Std(member_id=self.input, axes=axes)
        self.ref_mean = Mean(member_id=ref_tensor, axes=axes)
        self.ref_std = Std(member_id=ref_tensor, axes=axes)

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        mean = stat[self.mean]
        std = stat[self.std] + self.eps
        ref_mean = stat[self.ref_mean]
        ref_std = stat[self.ref_std] + self.eps
        return (input - mean) / std * ref_std + ref_mean

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: Union[v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr],
        member_id: MemberId,
    ) -> Self:
        kwargs = descr.kwargs
        _, axes = _get_axes(descr.kwargs)

        return cls(
            input=member_id,
            output=member_id,
            reference_tensor=MemberId(str(kwargs.reference_tensor)),
            axes=axes,
            eps=kwargs.eps,
        )


def _get_axes(
    kwargs: Union[
        v0_4.ZeroMeanUnitVarianceKwargs,
        v0_5.ZeroMeanUnitVarianceKwargs,
        v0_4.ScaleRangeKwargs,
        v0_5.ScaleRangeKwargs,
        v0_4.ScaleMeanVarianceKwargs,
        v0_5.ScaleMeanVarianceKwargs,
    ],
) -> Tuple[bool, Optional[Tuple[AxisId, ...]]]:
    if kwargs.axes is None:
        return True, None
    elif isinstance(kwargs.axes, str):
        axes = _convert_axis_ids(kwargs.axes, kwargs["mode"])
        return AxisId("b") in axes, axes
    elif isinstance(kwargs.axes, collections.abc.Sequence):
        axes = tuple(kwargs.axes)
        return AxisId("batch") in axes, axes
    else:
        assert_never(kwargs.axes)


@dataclass
class ScaleRange(_SimpleOperator):
    lower_percentile: InitVar[Optional[Union[SampleQuantile, DatasetPercentile]]] = None
    upper_percentile: InitVar[Optional[Union[SampleQuantile, DatasetPercentile]]] = None
    lower: Union[SampleQuantile, DatasetPercentile] = field(init=False)
    upper: Union[SampleQuantile, DatasetPercentile] = field(init=False)

    eps: float = 1e-6

    def __post_init__(
        self,
        lower_percentile: Optional[Union[SampleQuantile, DatasetPercentile]],
        upper_percentile: Optional[Union[SampleQuantile, DatasetPercentile]],
    ):
        if lower_percentile is None:
            tid = self.input if upper_percentile is None else upper_percentile.member_id
            self.lower = DatasetPercentile(q=0.0, member_id=tid)
        else:
            self.lower = lower_percentile

        if upper_percentile is None:
            self.upper = DatasetPercentile(q=1.0, member_id=self.lower.member_id)
        else:
            self.upper = upper_percentile

        assert self.lower.member_id == self.upper.member_id
        assert self.lower.q < self.upper.q
        assert self.lower.axes == self.upper.axes

    @property
    def required_measures(self):
        return {self.lower, self.upper}

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: Union[v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr],
        member_id: MemberId,
    ):
        kwargs = descr.kwargs
        ref_tensor = (
            member_id
            if kwargs.reference_tensor is None
            else MemberId(str(kwargs.reference_tensor))
        )
        dataset_mode, axes = _get_axes(descr.kwargs)
        if dataset_mode:
            Percentile = DatasetPercentile
        else:
            Percentile = SampleQuantile

        return cls(
            input=member_id,
            output=member_id,
            lower_percentile=Percentile(
                q=kwargs.min_percentile / 100, axes=axes, member_id=ref_tensor
            ),
            upper_percentile=Percentile(
                q=kwargs.max_percentile / 100, axes=axes, member_id=ref_tensor
            ),
        )

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        lower = stat[self.lower]
        upper = stat[self.upper]
        return (input - lower) / (upper - lower + self.eps)

    def get_descr(self):
        assert self.lower.axes == self.upper.axes
        assert self.lower.member_id == self.upper.member_id

        return v0_5.ScaleRangeDescr(
            kwargs=v0_5.ScaleRangeKwargs(
                axes=self.lower.axes,
                min_percentile=self.lower.q * 100,
                max_percentile=self.upper.q * 100,
                eps=self.eps,
                reference_tensor=self.lower.member_id,
            )
        )


@dataclass
class Sigmoid(_SimpleOperator):
    """1 / (1 + e^(-input))."""

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return Tensor(1.0 / (1.0 + np.exp(-input)), dims=input.dims)

    @property
    def required_measures(self) -> Collection[Measure]:
        return {}

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls, descr: Union[v0_4.SigmoidDescr, v0_5.SigmoidDescr], member_id: MemberId
    ) -> Self:
        assert isinstance(descr, (v0_4.SigmoidDescr, v0_5.SigmoidDescr))
        return cls(input=member_id, output=member_id)

    def get_descr(self):
        return v0_5.SigmoidDescr()


@dataclass
class ZeroMeanUnitVariance(_SimpleOperator):
    """normalize to zero mean, unit variance."""

    mean: MeanMeasure
    std: StdMeasure

    eps: float = 1e-6

    def __post_init__(self):
        assert self.mean.axes == self.std.axes

    @property
    def required_measures(self) -> Set[Union[MeanMeasure, StdMeasure]]:
        return {self.mean, self.std}

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: Union[v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr],
        member_id: MemberId,
    ):
        dataset_mode, axes = _get_axes(descr.kwargs)

        if dataset_mode:
            Mean = DatasetMean
            Std = DatasetStd
        else:
            Mean = SampleMean
            Std = SampleStd

        return cls(
            input=member_id,
            output=member_id,
            mean=Mean(axes=axes, member_id=member_id),
            std=Std(axes=axes, member_id=member_id),
        )

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        mean = stat[self.mean]
        std = stat[self.std]
        return (input - mean) / (std + self.eps)

    def get_descr(self):
        return v0_5.ZeroMeanUnitVarianceDescr(
            kwargs=v0_5.ZeroMeanUnitVarianceKwargs(axes=self.mean.axes, eps=self.eps)
        )


@dataclass
class FixedZeroMeanUnitVariance(_SimpleOperator):
    """normalize to zero mean, unit variance with precomputed values."""

    mean: Union[float, xr.DataArray]
    std: Union[float, xr.DataArray]

    eps: float = 1e-6

    def __post_init__(self):
        assert (
            isinstance(self.mean, (int, float))
            or isinstance(self.std, (int, float))
            or self.mean.dims == self.std.dims
        )

    def get_output_shape(
        self, input_shape: Mapping[AxisId, int]
    ) -> Mapping[AxisId, int]:
        return input_shape

    @classmethod
    def from_proc_descr(
        cls,
        descr: v0_5.FixedZeroMeanUnitVarianceDescr,
        member_id: MemberId,
    ) -> Self:
        if isinstance(descr.kwargs, v0_5.FixedZeroMeanUnitVarianceKwargs):
            dims = None
        elif isinstance(descr.kwargs, v0_5.FixedZeroMeanUnitVarianceAlongAxisKwargs):
            dims = (descr.kwargs.axis,)
        else:
            assert_never(descr.kwargs)

        return cls(
            input=member_id,
            output=member_id,
            mean=xr.DataArray(descr.kwargs.mean, dims=dims),
            std=xr.DataArray(descr.kwargs.std, dims=dims),
        )

    def get_descr(self):
        if isinstance(self.mean, (int, float)):
            assert isinstance(self.std, (int, float))
            kwargs = v0_5.FixedZeroMeanUnitVarianceKwargs(mean=self.mean, std=self.std)
        else:
            assert isinstance(self.std, xr.DataArray)
            assert len(self.mean.dims) == 1
            kwargs = v0_5.FixedZeroMeanUnitVarianceAlongAxisKwargs(
                axis=AxisId(str(self.mean.dims[0])),
                mean=list(self.mean),
                std=list(self.std),
            )

        return v0_5.FixedZeroMeanUnitVarianceDescr(kwargs=kwargs)

    def _apply(self, input: Tensor, stat: Stat) -> Tensor:
        return (input - self.mean) / (self.std + self.eps)


ProcDescr = Union[
    v0_4.PreprocessingDescr,
    v0_4.PostprocessingDescr,
    v0_5.PreprocessingDescr,
    v0_5.PostprocessingDescr,
]

Processing = Union[
    AddKnownDatasetStats,
    Binarize,
    Clip,
    EnsureDtype,
    FixedZeroMeanUnitVariance,
    ScaleLinear,
    ScaleMeanVariance,
    ScaleRange,
    Sigmoid,
    UpdateStats,
    ZeroMeanUnitVariance,
]


def get_proc(
    proc_descr: ProcDescr,
    tensor_descr: Union[
        v0_4.InputTensorDescr,
        v0_4.OutputTensorDescr,
        v0_5.InputTensorDescr,
        v0_5.OutputTensorDescr,
    ],
) -> Processing:
    member_id = get_member_id(tensor_descr)

    if isinstance(proc_descr, (v0_4.BinarizeDescr, v0_5.BinarizeDescr)):
        return Binarize.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, (v0_4.ClipDescr, v0_5.ClipDescr)):
        return Clip.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.EnsureDtypeDescr):
        return EnsureDtype.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, v0_5.FixedZeroMeanUnitVarianceDescr):
        return FixedZeroMeanUnitVariance.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, (v0_4.ScaleLinearDescr, v0_5.ScaleLinearDescr)):
        return ScaleLinear.from_proc_descr(proc_descr, member_id)
    elif isinstance(
        proc_descr, (v0_4.ScaleMeanVarianceDescr, v0_5.ScaleMeanVarianceDescr)
    ):
        return ScaleMeanVariance.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, (v0_4.ScaleRangeDescr, v0_5.ScaleRangeDescr)):
        return ScaleRange.from_proc_descr(proc_descr, member_id)
    elif isinstance(proc_descr, (v0_4.SigmoidDescr, v0_5.SigmoidDescr)):
        return Sigmoid.from_proc_descr(proc_descr, member_id)
    elif (
        isinstance(proc_descr, v0_4.ZeroMeanUnitVarianceDescr)
        and proc_descr.kwargs.mode == "fixed"
    ):
        if not isinstance(
            tensor_descr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)
        ):
            raise TypeError(
                "Expected v0_4 tensor description for v0_4 processing description"
            )

        v5_proc_descr = _convert_proc(proc_descr, tensor_descr.axes)
        assert isinstance(v5_proc_descr, v0_5.FixedZeroMeanUnitVarianceDescr)
        return FixedZeroMeanUnitVariance.from_proc_descr(v5_proc_descr, member_id)
    elif isinstance(
        proc_descr,
        (v0_4.ZeroMeanUnitVarianceDescr, v0_5.ZeroMeanUnitVarianceDescr),
    ):
        return ZeroMeanUnitVariance.from_proc_descr(proc_descr, member_id)
    else:
        assert_never(proc_descr)
