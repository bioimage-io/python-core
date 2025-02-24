from typing import Sequence

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core import AxisId, Tensor


@pytest.mark.parametrize(
    "axes",
    [
        "yx",
        "xy",
        "cyx",
        "yxc",
        "bczyx",
        "xyz",
        "xyzc",
        "bzyxc",
        ("batch", "channel", "x", "y"),
    ],
)
def test_transpose_tensor_2d(axes: Sequence[str]):

    tensor = Tensor.from_numpy(np.random.rand(256, 256), dims=None)
    transposed = tensor.transpose([AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


@pytest.mark.parametrize(
    "axes",
    [
        "zyx",
        "cyzx",
        "yzixc",
        "bczyx",
        "xyz",
        "xyzc",
        "bzyxtc",
        ("batch", "channel", "x", "y", "z"),
    ],
)
def test_transpose_tensor_3d(axes: Sequence[str]):
    tensor = Tensor.from_numpy(np.random.rand(64, 64, 64), dims=None)
    transposed = tensor.transpose([AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


def test_crop_and_pad():
    tensor = Tensor.from_xarray(
        xr.DataArray(np.random.rand(10, 20), dims=("x", "y"), name="id")
    )
    padded = tensor.pad({AxisId("x"): 7, AxisId("y"): (3, 3)})
    cropped = padded.crop_to(tensor.sizes)
    assert_equal(tensor.data, cropped.data)


def test_some_magic_ops():
    tensor = Tensor.from_numpy(np.random.rand(256, 256), dims=None)
    assert tensor + 2 == 2 + tensor
