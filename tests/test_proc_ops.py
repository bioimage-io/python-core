from typing import Iterable, Optional, Tuple, Type, TypeVar

import numpy as np
import pytest
import xarray as xr
from typing_extensions import TypeGuard

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import compute_measures
from bioimageio.core.stat_measures import SampleMean, SampleQuantile, SampleStd
from bioimageio.core.tensor import Tensor


@pytest.fixture(scope="module")
def tid():
    return MemberId("data123")


def test_scale_linear(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleLinear

    offset = xr.DataArray([1, 2, 42], dims=("c"))
    gain = xr.DataArray([1, 2, 3], dims=("c"))
    data = xr.DataArray(np.arange(6).reshape((1, 2, 3)), dims=("x", "y", "c"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)

    op = ScaleLinear(input=tid, output=tid, offset=offset, gain=gain)
    op(sample)

    expected = xr.DataArray(np.array([[[1, 4, 48], [4, 10, 57]]]), dims=("x", "y", "c"))
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_scale_linear_no_channel(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleLinear

    op = ScaleLinear(tid, tid, offset=1, gain=2)
    data = xr.DataArray(np.arange(6).reshape(2, 3), dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    op(sample)

    expected = xr.DataArray(np.array([[1, 3, 5], [7, 9, 11]]), dims=("x", "y"))
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


T = TypeVar("T")


def is_iterable(val: Iterable[T], inner: Type[T]) -> TypeGuard[Iterable[T]]:
    """Determines whether all objects in the list are strings"""
    return all(isinstance(x, inner) for x in val)


def test_zero_mean_unit_variance(tid: MemberId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    m = SampleMean(member_id=tid)
    std = SampleStd(member_id=tid)
    op = ZeroMeanUnitVariance(tid, tid, m, std)
    req = op.required_measures
    sample.stat = compute_measures(req, [sample])
    op(sample)

    expected = xr.DataArray(
        np.array(
            [
                [-1.54919274, -1.16189455, -0.77459637],
                [-0.38729818, 0.0, 0.38729818],
                [0.77459637, 1.16189455, 1.54919274],
            ]
        ),
        dims=("x", "y"),
    )
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_zero_mean_unit_variance_fixed(tid: MemberId):
    from bioimageio.core.proc_ops import FixedZeroMeanUnitVariance

    op = FixedZeroMeanUnitVariance(
        tid,
        tid,
        mean=xr.DataArray([3, 4, 5], dims=("c")),
        std=xr.DataArray([2.44948974, 2.44948974, 2.44948974], dims=("c")),
    )
    data = xr.DataArray(np.arange(9).reshape((1, 3, 3)), dims=("b", "c", "x"))
    expected = xr.DataArray(
        np.array(
            [
                [
                    [-1.22474487, -0.81649658, -0.40824829],
                    [-0.40824829, 0.0, 0.40824829],
                    [0.40824829, 0.81649658, 1.22474487],
                ]
            ]
        ),
        dims=("b", "c", "x"),
    )
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_zero_mean_unit_variance_fixed2(tid: MemberId):
    from bioimageio.core.proc_ops import FixedZeroMeanUnitVariance

    np_data = np.arange(9).reshape(3, 3)
    mean = float(np_data.mean())
    std = float(np_data.mean())
    eps = 1.0e-7
    op = FixedZeroMeanUnitVariance(tid, tid, mean=mean, std=std, eps=eps)

    data = xr.DataArray(np_data, dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    expected = xr.DataArray((np_data - mean) / (std + eps), dims=("x", "y"))
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_zero_mean_unit_across_axes(tid: MemberId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(18).reshape((2, 3, 3)), dims=("c", "x", "y"))

    op = ZeroMeanUnitVariance(
        tid,
        tid,
        SampleMean(member_id=tid, axes=(AxisId("x"), AxisId("y"))),
        SampleStd(member_id=tid, axes=(AxisId("x"), AxisId("y"))),
    )
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    sample.stat = compute_measures(op.required_measures, [sample])

    expected = xr.concat(
        [(data[i : i + 1] - data[i].mean()) / data[i].std() for i in range(2)], dim="c"
    )
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_binarize(tid: MemberId):
    from bioimageio.core.proc_ops import Binarize

    op = Binarize(tid, tid, threshold=14)
    data = xr.DataArray(np.arange(30).reshape((2, 3, 5)), dims=("x", "y", "c"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    expected = xr.zeros_like(data)
    expected[{"x": slice(1, None)}] = 1
    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data)


def test_binarize2(tid: MemberId):
    from bioimageio.core.proc_ops import Binarize

    shape = (3, 32, 32)
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)

    threshold = 0.5
    exp = xr.DataArray(np_data > threshold, dims=axes)

    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    binarize = Binarize(tid, tid, threshold=threshold)
    binarize(sample)
    xr.testing.assert_allclose(exp, sample.members[tid].data)


def test_clip(tid: MemberId):
    from bioimageio.core.proc_ops import Clip

    op = Clip(tid, tid, min=3, max=5)
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)

    expected = xr.DataArray(
        np.array([[3, 3, 3], [3, 4, 5], [5, 5, 5]]), dims=("x", "y")
    )
    op(sample)
    xr.testing.assert_equal(expected, sample.members[tid].data)


def test_combination_of_op_steps_with_dims_specified(tid: MemberId):
    from bioimageio.core.proc_ops import ZeroMeanUnitVariance

    data = xr.DataArray(np.arange(18).reshape((2, 3, 3)), dims=("c", "x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    op = ZeroMeanUnitVariance(
        tid,
        tid,
        SampleMean(
            member_id=tid,
            axes=(AxisId("x"), AxisId("y")),
        ),
        SampleStd(
            member_id=tid,
            axes=(AxisId("x"), AxisId("y")),
        ),
    )
    sample.stat = compute_measures(op.required_measures, [sample])

    expected = xr.DataArray(
        np.array(
            [
                [
                    [-1.54919274, -1.16189455, -0.77459637],
                    [-0.38729818, 0.0, 0.38729818],
                    [0.77459637, 1.16189455, 1.54919274],
                ],
                [
                    [-1.54919274, -1.16189455, -0.77459637],
                    [-0.38729818, 0.0, 0.38729818],
                    [0.77459637, 1.16189455, 1.54919274],
                ],
            ]
        ),
        dims=("c", "x", "y"),
    )

    op(sample)
    xr.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "axes",
    [
        None,
        tuple(map(AxisId, "cy")),
        tuple(map(AxisId, "cyx")),
        tuple(map(AxisId, "x")),
    ],
)
def test_scale_mean_variance(tid: MemberId, axes: Optional[Tuple[AxisId, ...]]):
    from bioimageio.core.proc_ops import ScaleMeanVariance

    shape = (3, 32, 46)
    ipt_axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)
    ref_data = xr.DataArray((np_data * 2) + 3, dims=ipt_axes)

    op = ScaleMeanVariance(tid, tid, reference_tensor=MemberId("ref_name"), axes=axes)
    sample = Sample(
        members={
            tid: Tensor.from_xarray(ipt_data),
            MemberId("ref_name"): Tensor.from_xarray(ref_data),
        },
        stat={},
        id=None,
    )
    sample.stat = compute_measures(op.required_measures, [sample])
    op(sample)
    xr.testing.assert_allclose(ref_data, sample.members[tid].data, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "axes_str",
    [None, "cy", "y", "yx"],
)
def test_scale_mean_variance_per_channel(tid: MemberId, axes_str: Optional[str]):
    from bioimageio.core.proc_ops import ScaleMeanVariance

    axes = None if axes_str is None else tuple(map(AxisId, axes_str))

    shape = (3, 32, 46)
    ipt_axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    ipt_data = xr.DataArray(np_data, dims=ipt_axes)

    # set different mean, std per channel
    np_ref_data = np.stack([d * i + i for i, d in enumerate(np_data, start=2)])
    ref_data = xr.DataArray(np_ref_data, dims=ipt_axes)

    op = ScaleMeanVariance(tid, tid, reference_tensor=MemberId("ref_name"), axes=axes)
    sample = Sample(
        members={
            tid: Tensor.from_xarray(ipt_data),
            MemberId("ref_name"): Tensor.from_xarray(ref_data),
        },
        stat={},
        id=None,
    )
    sample.stat = compute_measures(op.required_measures, [sample])
    op(sample)

    if axes is not None and AxisId("c") not in axes:
        # mean,std per channel should match exactly
        xr.testing.assert_allclose(
            ref_data, sample.members[tid].data, rtol=1e-5, atol=1e-7
        )
    else:
        # mean,std across channels should not match
        with pytest.raises(AssertionError):
            xr.testing.assert_allclose(
                ref_data, sample.members[tid].data, rtol=1e-5, atol=1e-7
            )


def test_scale_range(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleRange

    op = ScaleRange(tid, tid)
    np_data = np.arange(9).reshape(3, 3).astype("float32")
    data = xr.DataArray(np_data, dims=("x", "y"))
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    sample.stat = compute_measures(op.required_measures, [sample])

    eps = 1.0e-6
    mi, ma = np_data.min(), np_data.max()
    exp_data = (np_data - mi) / (ma - mi + eps)
    expected = xr.DataArray(exp_data, dims=("x", "y"))

    op(sample)
    # NOTE xarray.testing.assert_allclose compares irrelavant properties here and fails although the result is correct
    np.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_scale_range_axes(tid: MemberId):
    from bioimageio.core.proc_ops import ScaleRange

    eps = 1.0e-6

    lower_quantile = SampleQuantile(
        member_id=tid, q=0.1, axes=(AxisId("x"), AxisId("y"))
    )
    upper_quantile = SampleQuantile(
        member_id=tid, q=0.9, axes=(AxisId("x"), AxisId("y"))
    )
    op = ScaleRange(tid, tid, lower_quantile, upper_quantile, eps=eps)

    np_data = np.arange(18).reshape((2, 3, 3)).astype("float32")
    data = Tensor.from_xarray(xr.DataArray(np_data, dims=("c", "x", "y")))
    sample = Sample(members={tid: data}, stat={}, id=None)

    p_low_direct = lower_quantile.compute(sample)
    p_up_direct = upper_quantile.compute(sample)

    p_low_expected = np.quantile(np_data, lower_quantile.q, axis=(1, 2), keepdims=True)
    p_up_expected = np.quantile(np_data, upper_quantile.q, axis=(1, 2), keepdims=True)

    np.testing.assert_allclose(p_low_expected.squeeze(), p_low_direct)
    np.testing.assert_allclose(p_up_expected.squeeze(), p_up_direct)

    sample.stat = compute_measures(op.required_measures, [sample])

    np.testing.assert_allclose(p_low_expected.squeeze(), sample.stat[lower_quantile])
    np.testing.assert_allclose(p_up_expected.squeeze(), sample.stat[upper_quantile])

    exp_data = (np_data - p_low_expected) / (p_up_expected - p_low_expected + eps)
    expected = xr.DataArray(exp_data, dims=("c", "x", "y"))

    op(sample)
    # NOTE xarray.testing.assert_allclose compares irrelavant properties here and fails although the result is correct
    np.testing.assert_allclose(expected, sample.members[tid].data, rtol=1e-5, atol=1e-7)


def test_sigmoid(tid: MemberId):
    from bioimageio.core.proc_ops import Sigmoid

    shape = (3, 32, 32)
    axes = ("c", "y", "x")
    np_data = np.random.rand(*shape)
    data = xr.DataArray(np_data, dims=axes)
    sample = Sample(members={tid: Tensor.from_xarray(data)}, stat={}, id=None)
    sigmoid = Sigmoid(tid, tid)
    sigmoid(sample)

    exp = xr.DataArray(1.0 / (1 + np.exp(-np_data)), dims=axes)
    xr.testing.assert_allclose(exp, sample.members[tid].data, rtol=1e-5, atol=1e-7)
