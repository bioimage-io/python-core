from typing import Tuple

import numpy as np
import pytest
from git import Optional
from xarray.testing import assert_allclose  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import MeanVarStdCalculator
from bioimageio.core.stat_measures import (
    DatasetMean,
    DatasetStd,
    DatasetVar,
    SampleMean,
    SampleStd,
    SampleVar,
)
from bioimageio.core.tensor import Tensor


def create_random_dataset(tid: MemberId, axes: Tuple[AxisId, ...]):
    n = 3
    sizes = list(range(n, len(axes) + n))
    data = np.asarray(np.random.rand(*sizes))
    ds = [
        Sample(members={tid: Tensor(data[i : i + 1], dims=axes)}, stat={}, id=None)
        for i in range(n)
    ]
    return Tensor(data, dims=axes), ds


@pytest.mark.parametrize(
    "axes",
    [
        (AxisId("x"), AxisId("y")),
        (AxisId("channel"), AxisId("y")),
    ],
)
def test_sample_mean_var_std_calculator(axes: Optional[Tuple[AxisId, ...]]):
    tid = MemberId("tensor")
    d_axes = tuple(map(AxisId, ("batch", "channel", "x", "y")))
    data, ds = create_random_dataset(tid, d_axes)
    expected_mean = data[0:1].mean(axes)
    expected_var = data[0:1].var(axes)
    expected_std = data[0:1].std(axes)

    calc = MeanVarStdCalculator(tid, axes=axes)

    actual = calc.compute(ds[0])
    actual_mean = actual[SampleMean(member_id=tid, axes=axes)]
    actual_var = actual[SampleVar(member_id=tid, axes=axes)]
    actual_std = actual[SampleStd(member_id=tid, axes=axes)]

    assert_allclose(
        actual_mean if isinstance(actual_mean, (int, float)) else actual_mean.data,
        expected_mean.data,
    )
    assert_allclose(
        actual_var if isinstance(actual_var, (int, float)) else actual_var.data,
        expected_var.data,
    )
    assert_allclose(
        actual_std if isinstance(actual_std, (int, float)) else actual_std.data,
        expected_std.data,
    )


@pytest.mark.parametrize(
    "axes",
    [
        None,
        (AxisId("batch"), AxisId("channel"), AxisId("x"), AxisId("y")),
    ],
)
def test_dataset_mean_var_std_calculator(axes: Optional[Tuple[AxisId, ...]]):
    tid = MemberId("tensor")
    d_axes = tuple(map(AxisId, ("batch", "channel", "x", "y")))
    data, ds = create_random_dataset(tid, d_axes)
    expected_mean = data.mean(axes)
    expected_var = data.var(axes)
    expected_std = data.std(axes)

    calc = MeanVarStdCalculator(tid, axes=axes)
    for s in ds:
        calc.update(s)

    actual = calc.finalize()
    actual_mean = actual[DatasetMean(member_id=tid, axes=axes)]
    actual_var = actual[DatasetVar(member_id=tid, axes=axes)]
    actual_std = actual[DatasetStd(member_id=tid, axes=axes)]

    assert_allclose(
        actual_mean if isinstance(actual_mean, (int, float)) else actual_mean.data,
        expected_mean.data,
    )
    assert_allclose(
        actual_var if isinstance(actual_var, (int, float)) else actual_var.data,
        expected_var.data,
    )
    assert_allclose(
        actual_std if isinstance(actual_std, (int, float)) else actual_std.data,
        expected_std.data,
    )
