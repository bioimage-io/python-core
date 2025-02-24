from __future__ import annotations

import collections.abc
import warnings
from itertools import product
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.spec.model.v0_5 import BATCH_AXIS_ID

from .axis import AxisId, PerAxis
from .common import MemberId
from .sample import Sample
from .stat_measures import (
    DatasetMean,
    DatasetMeasure,
    DatasetMeasureBase,
    DatasetPercentile,
    DatasetStd,
    DatasetVar,
    Measure,
    MeasureValue,
    SampleMean,
    SampleMeasure,
    SampleQuantile,
    SampleStd,
    SampleVar,
)
from .tensor import Tensor

try:
    import crick  # pyright: ignore[reportMissingImports]

except Exception:
    crick = None

    class TDigest:
        def update(self, obj: Any):
            pass

        def quantile(self, q: Any) -> Any:
            pass

else:
    TDigest = crick.TDigest  # type: ignore


class MeanCalculator:
    """to calculate sample and dataset mean for in-memory samples"""

    def __init__(self, member_id: MemberId, axes: Optional[Sequence[AxisId]]):
        super().__init__()
        self._n: int = 0
        self._mean: Optional[Tensor] = None
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id
        self._sample_mean = SampleMean(member_id=self._member_id, axes=self._axes)
        self._dataset_mean = DatasetMean(member_id=self._member_id, axes=self._axes)

    def compute(self, sample: Sample) -> Dict[SampleMean, MeasureValue]:
        return {self._sample_mean: self._compute_impl(sample)}

    def _compute_impl(self, sample: Sample) -> Tensor:
        tensor = sample.members[self._member_id].astype("float64", copy=False)
        return tensor.mean(dim=self._axes)

    def update(self, sample: Sample) -> None:
        mean = self._compute_impl(sample)
        self._update_impl(sample.members[self._member_id], mean)

    def compute_and_update(self, sample: Sample) -> Dict[SampleMean, MeasureValue]:
        mean = self._compute_impl(sample)
        self._update_impl(sample.members[self._member_id], mean)
        return {self._sample_mean: mean}

    def _update_impl(self, tensor: Tensor, tensor_mean: Tensor):
        assert tensor_mean.dtype == "float64"
        # reduced voxel count
        n_b = int(tensor.size / tensor_mean.size)

        if self._mean is None:
            assert self._n == 0
            self._n = n_b
            self._mean = tensor_mean
        else:
            assert self._n != 0
            n_a = self._n
            mean_old = self._mean
            self._n = n_a + n_b
            self._mean = (n_a * mean_old + n_b * tensor_mean) / self._n
            assert self._mean.dtype == "float64"

    def finalize(self) -> Dict[DatasetMean, MeasureValue]:
        if self._mean is None:
            return {}
        else:
            return {self._dataset_mean: self._mean}


class MeanVarStdCalculator:
    """to calculate sample and dataset mean, variance or standard deviation"""

    def __init__(self, member_id: MemberId, axes: Optional[Sequence[AxisId]]):
        super().__init__()
        self._axes = None if axes is None else tuple(map(AxisId, axes))
        self._member_id = member_id
        self._n: int = 0
        self._mean: Optional[Tensor] = None
        self._m2: Optional[Tensor] = None

    def compute(
        self, sample: Sample
    ) -> Dict[Union[SampleMean, SampleVar, SampleStd], MeasureValue]:
        tensor = sample.members[self._member_id]
        mean = tensor.mean(dim=self._axes)
        c = (tensor - mean).data
        if self._axes is None:
            n = tensor.size
        else:
            n = int(np.prod([tensor.sizes[d] for d in self._axes]))

        var = xr.dot(c, c, dim=self._axes) / n
        assert isinstance(var, xr.DataArray)
        std = np.sqrt(var)
        assert isinstance(std, xr.DataArray)
        return {
            SampleMean(axes=self._axes, member_id=self._member_id): mean,
            SampleVar(axes=self._axes, member_id=self._member_id): Tensor.from_xarray(
                var
            ),
            SampleStd(axes=self._axes, member_id=self._member_id): Tensor.from_xarray(
                std
            ),
        }

    def update(self, sample: Sample):
        if self._axes is not None and BATCH_AXIS_ID not in self._axes:
            return

        tensor = sample.members[self._member_id].astype("float64", copy=False)
        mean_b = tensor.mean(dim=self._axes)
        assert mean_b.dtype == "float64"
        # reduced voxel count
        n_b = int(tensor.size / mean_b.size)
        m2_b = ((tensor - mean_b) ** 2).sum(dim=self._axes)
        assert m2_b.dtype == "float64"
        if self._mean is None:
            assert self._m2 is None
            self._n = n_b
            self._mean = mean_b
            self._m2 = m2_b
        else:
            n_a = self._n
            mean_a = self._mean
            m2_a = self._m2
            self._n = n = n_a + n_b
            self._mean = (n_a * mean_a + n_b * mean_b) / n
            assert self._mean.dtype == "float64"
            d = mean_b - mean_a
            self._m2 = m2_a + m2_b + d**2 * n_a * n_b / n
            assert self._m2.dtype == "float64"

    def finalize(
        self,
    ) -> Dict[Union[DatasetMean, DatasetVar, DatasetStd], MeasureValue]:
        if (
            self._axes is not None
            and BATCH_AXIS_ID not in self._axes
            or self._mean is None
        ):
            return {}
        else:
            assert self._m2 is not None
            var = self._m2 / self._n
            sqrt = var**0.5
            if isinstance(sqrt, (int, float)):
                # var and mean are scalar tensors, let's keep it consistent
                sqrt = Tensor.from_xarray(xr.DataArray(sqrt))

            assert isinstance(sqrt, Tensor), type(sqrt)
            return {
                DatasetMean(member_id=self._member_id, axes=self._axes): self._mean,
                DatasetVar(member_id=self._member_id, axes=self._axes): var,
                DatasetStd(member_id=self._member_id, axes=self._axes): sqrt,
            }


class SamplePercentilesCalculator:
    """to calculate sample percentiles"""

    def __init__(
        self,
        member_id: MemberId,
        axes: Optional[Sequence[AxisId]],
        qs: Collection[float],
    ):
        super().__init__()
        assert all(0.0 <= q <= 1.0 for q in qs)
        self._qs = sorted(set(qs))
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id

    def compute(self, sample: Sample) -> Dict[SampleQuantile, MeasureValue]:
        tensor = sample.members[self._member_id]
        ps = tensor.quantile(self._qs, dim=self._axes)
        return {
            SampleQuantile(q=q, axes=self._axes, member_id=self._member_id): p
            for q, p in zip(self._qs, ps)
        }


class MeanPercentilesCalculator:
    """to calculate dataset percentiles heuristically by averaging across samples
    **note**: the returned dataset percentiles are an estiamte and **not mathematically correct**
    """

    def __init__(
        self,
        member_id: MemberId,
        axes: Optional[Sequence[AxisId]],
        qs: Collection[float],
    ):
        super().__init__()
        assert all(0.0 <= q <= 1.0 for q in qs)
        self._qs = sorted(set(qs))
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id
        self._n: int = 0
        self._estimates: Optional[Tensor] = None

    def update(self, sample: Sample):
        tensor = sample.members[self._member_id]
        sample_estimates = tensor.quantile(self._qs, dim=self._axes).astype(
            "float64", copy=False
        )

        # reduced voxel count
        n = int(tensor.size / np.prod(sample_estimates.shape_tuple[1:]))

        if self._estimates is None:
            assert self._n == 0
            self._estimates = sample_estimates
        else:
            self._estimates = (self._n * self._estimates + n * sample_estimates) / (
                self._n + n
            )
            assert self._estimates.dtype == "float64"

        self._n += n

    def finalize(self) -> Dict[DatasetPercentile, MeasureValue]:
        if self._estimates is None:
            return {}
        else:
            warnings.warn(
                "Computed dataset percentiles naively by averaging percentiles of samples."
            )
            return {
                DatasetPercentile(q=q, axes=self._axes, member_id=self._member_id): e
                for q, e in zip(self._qs, self._estimates)
            }


class CrickPercentilesCalculator:
    """to calculate dataset percentiles with the experimental [crick libray](https://github.com/dask/crick)"""

    def __init__(
        self,
        member_id: MemberId,
        axes: Optional[Sequence[AxisId]],
        qs: Collection[float],
    ):
        warnings.warn(
            "Computing dataset percentiles with experimental 'crick' library."
        )
        super().__init__()
        assert all(0.0 <= q <= 1.0 for q in qs)
        assert axes is None or "_percentiles" not in axes
        self._qs = sorted(set(qs))
        self._axes = None if axes is None else tuple(axes)
        self._member_id = member_id
        self._digest: Optional[List[TDigest]] = None
        self._dims: Optional[Tuple[AxisId, ...]] = None
        self._indices: Optional[Iterator[Tuple[int, ...]]] = None
        self._shape: Optional[Tuple[int, ...]] = None

    def _initialize(self, tensor_sizes: PerAxis[int]):
        assert crick is not None
        out_sizes: OrderedDict[AxisId, int] = collections.OrderedDict(
            _percentiles=len(self._qs)
        )
        if self._axes is not None:
            for d, s in tensor_sizes.items():
                if d not in self._axes:
                    out_sizes[d] = s

        self._dims, self._shape = zip(*out_sizes.items())
        assert self._shape is not None
        d = int(np.prod(self._shape[1:]))
        self._digest = [TDigest() for _ in range(d)]
        self._indices = product(*map(range, self._shape[1:]))

    def update(self, part: Sample):
        tensor = (
            part.members[self._member_id]
            if isinstance(part, Sample)
            else part.members[self._member_id].data
        )
        assert "_percentiles" not in tensor.dims
        if self._digest is None:
            self._initialize(tensor.tagged_shape)

        assert self._digest is not None
        assert self._indices is not None
        assert self._dims is not None
        for i, idx in enumerate(self._indices):
            self._digest[i].update(tensor[dict(zip(self._dims[1:], idx))])

    def finalize(self) -> Dict[DatasetPercentile, MeasureValue]:
        if self._digest is None:
            return {}
        else:
            assert self._dims is not None
            assert self._shape is not None

            vs: NDArray[Any] = np.asarray(
                [[d.quantile(q) for d in self._digest] for q in self._qs]
            ).reshape(self._shape)
            return {
                DatasetPercentile(
                    q=q, axes=self._axes, member_id=self._member_id
                ): Tensor(v, dims=self._dims[1:])
                for q, v in zip(self._qs, vs)
            }


if crick is None:
    DatasetPercentilesCalculator: Type[
        Union[MeanPercentilesCalculator, CrickPercentilesCalculator]
    ] = MeanPercentilesCalculator
else:
    DatasetPercentilesCalculator = CrickPercentilesCalculator


class NaiveSampleMeasureCalculator:
    """wrapper for measures to match interface of other sample measure calculators"""

    def __init__(self, member_id: MemberId, measure: SampleMeasure):
        super().__init__()
        self.tensor_name = member_id
        self.measure = measure

    def compute(self, sample: Sample) -> Dict[SampleMeasure, MeasureValue]:
        return {self.measure: self.measure.compute(sample)}


SampleMeasureCalculator = Union[
    MeanCalculator,
    MeanVarStdCalculator,
    SamplePercentilesCalculator,
    NaiveSampleMeasureCalculator,
]
DatasetMeasureCalculator = Union[
    MeanCalculator, MeanVarStdCalculator, DatasetPercentilesCalculator
]


class StatsCalculator:
    """Estimates dataset statistics and computes sample statistics efficiently"""

    def __init__(
        self,
        measures: Collection[Measure],
        initial_dataset_measures: Optional[
            Mapping[DatasetMeasure, MeasureValue]
        ] = None,
    ):
        super().__init__()
        self.sample_count = 0
        self.sample_calculators, self.dataset_calculators = get_measure_calculators(
            measures
        )
        if not initial_dataset_measures:
            self._current_dataset_measures: Optional[
                Dict[DatasetMeasure, MeasureValue]
            ] = None
        else:
            missing_dataset_meas = {
                m
                for m in measures
                if isinstance(m, DatasetMeasureBase)
                and m not in initial_dataset_measures
            }
            if missing_dataset_meas:
                logger.debug(
                    f"ignoring `initial_dataset_measure` as it is missing {missing_dataset_meas}"
                )
                self._current_dataset_measures = None
            else:
                self._current_dataset_measures = dict(initial_dataset_measures)

    @property
    def has_dataset_measures(self):
        return self._current_dataset_measures is not None

    def update(
        self,
        sample: Union[Sample, Iterable[Sample]],
    ) -> None:
        _ = self._update(sample)

    def finalize(self) -> Dict[DatasetMeasure, MeasureValue]:
        """returns aggregated dataset statistics"""
        if self._current_dataset_measures is None:
            self._current_dataset_measures = {}
            for calc in self.dataset_calculators:
                values = calc.finalize()
                self._current_dataset_measures.update(values.items())

        return self._current_dataset_measures

    def update_and_get_all(
        self,
        sample: Union[Sample, Iterable[Sample]],
    ) -> Dict[Measure, MeasureValue]:
        """Returns sample as well as updated dataset statistics"""
        last_sample = self._update(sample)
        if last_sample is None:
            raise ValueError("`sample` was not a `Sample`, nor did it yield any.")

        return {**self._compute(last_sample), **self.finalize()}

    def skip_update_and_get_all(self, sample: Sample) -> Dict[Measure, MeasureValue]:
        """Returns sample as well as previously computed dataset statistics"""
        return {**self._compute(sample), **self.finalize()}

    def _compute(self, sample: Sample) -> Dict[SampleMeasure, MeasureValue]:
        ret: Dict[SampleMeasure, MeasureValue] = {}
        for calc in self.sample_calculators:
            values = calc.compute(sample)
            ret.update(values.items())

        return ret

    def _update(self, sample: Union[Sample, Iterable[Sample]]) -> Optional[Sample]:
        self.sample_count += 1
        samples = [sample] if isinstance(sample, Sample) else sample
        last_sample = None
        for el in samples:
            last_sample = el
            for calc in self.dataset_calculators:
                calc.update(el)

        self._current_dataset_measures = None
        return last_sample


def get_measure_calculators(
    required_measures: Iterable[Measure],
) -> Tuple[List[SampleMeasureCalculator], List[DatasetMeasureCalculator]]:
    """determines which calculators are needed to compute the required measures efficiently"""

    sample_calculators: List[SampleMeasureCalculator] = []
    dataset_calculators: List[DatasetMeasureCalculator] = []

    # split required measures into groups
    required_sample_means: Set[SampleMean] = set()
    required_dataset_means: Set[DatasetMean] = set()
    required_sample_mean_var_std: Set[Union[SampleMean, SampleVar, SampleStd]] = set()
    required_dataset_mean_var_std: Set[Union[DatasetMean, DatasetVar, DatasetStd]] = (
        set()
    )
    required_sample_percentiles: Dict[
        Tuple[MemberId, Optional[Tuple[AxisId, ...]]], Set[float]
    ] = {}
    required_dataset_percentiles: Dict[
        Tuple[MemberId, Optional[Tuple[AxisId, ...]]], Set[float]
    ] = {}

    for rm in required_measures:
        if isinstance(rm, SampleMean):
            required_sample_means.add(rm)
        elif isinstance(rm, DatasetMean):
            required_dataset_means.add(rm)
        elif isinstance(rm, (SampleVar, SampleStd)):
            required_sample_mean_var_std.update(
                {
                    msv(axes=rm.axes, member_id=rm.member_id)
                    for msv in (SampleMean, SampleStd, SampleVar)
                }
            )
            assert rm in required_sample_mean_var_std
        elif isinstance(rm, (DatasetVar, DatasetStd)):
            required_dataset_mean_var_std.update(
                {
                    msv(axes=rm.axes, member_id=rm.member_id)
                    for msv in (DatasetMean, DatasetStd, DatasetVar)
                }
            )
            assert rm in required_dataset_mean_var_std
        elif isinstance(rm, SampleQuantile):
            required_sample_percentiles.setdefault((rm.member_id, rm.axes), set()).add(
                rm.q
            )
        elif isinstance(rm, DatasetPercentile):
            required_dataset_percentiles.setdefault((rm.member_id, rm.axes), set()).add(
                rm.q
            )
        else:
            assert_never(rm)

    for rm in required_sample_means:
        if rm in required_sample_mean_var_std:
            # computed togehter with var and std
            continue

        sample_calculators.append(MeanCalculator(member_id=rm.member_id, axes=rm.axes))

    for rm in required_sample_mean_var_std:
        sample_calculators.append(
            MeanVarStdCalculator(member_id=rm.member_id, axes=rm.axes)
        )

    for rm in required_dataset_means:
        if rm in required_dataset_mean_var_std:
            # computed togehter with var and std
            continue

        dataset_calculators.append(MeanCalculator(member_id=rm.member_id, axes=rm.axes))

    for rm in required_dataset_mean_var_std:
        dataset_calculators.append(
            MeanVarStdCalculator(member_id=rm.member_id, axes=rm.axes)
        )

    for (tid, axes), qs in required_sample_percentiles.items():
        sample_calculators.append(
            SamplePercentilesCalculator(member_id=tid, axes=axes, qs=qs)
        )

    for (tid, axes), qs in required_dataset_percentiles.items():
        dataset_calculators.append(
            DatasetPercentilesCalculator(member_id=tid, axes=axes, qs=qs)
        )

    return sample_calculators, dataset_calculators


def compute_dataset_measures(
    measures: Iterable[DatasetMeasure], dataset: Iterable[Sample]
) -> Dict[DatasetMeasure, MeasureValue]:
    """compute all dataset `measures` for the given `dataset`"""
    sample_calculators, calculators = get_measure_calculators(measures)
    assert not sample_calculators

    ret: Dict[DatasetMeasure, MeasureValue] = {}

    for sample in dataset:
        for calc in calculators:
            calc.update(sample)

    for calc in calculators:
        ret.update(calc.finalize().items())

    return ret


def compute_sample_measures(
    measures: Iterable[SampleMeasure], sample: Sample
) -> Dict[SampleMeasure, MeasureValue]:
    """compute all sample `measures` for the given `sample`"""
    calculators, dataset_calculators = get_measure_calculators(measures)
    assert not dataset_calculators
    ret: Dict[SampleMeasure, MeasureValue] = {}

    for calc in calculators:
        ret.update(calc.compute(sample).items())

    return ret


def compute_measures(
    measures: Iterable[Measure], dataset: Iterable[Sample]
) -> Dict[Measure, MeasureValue]:
    """compute all `measures` for the given `dataset`
    sample measures are computed for the last sample in `dataset`"""
    sample_calculators, dataset_calculators = get_measure_calculators(measures)
    ret: Dict[Measure, MeasureValue] = {}
    sample = None
    for sample in dataset:
        for calc in dataset_calculators:
            calc.update(sample)
    if sample is None:
        raise ValueError("empty dataset")

    for calc in dataset_calculators:
        ret.update(calc.finalize().items())

    for calc in sample_calculators:
        ret.update(calc.compute(sample).items())

    return ret
