from __future__ import annotations

import collections.abc
from itertools import permutations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
)

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Self, assert_never

from bioimageio.spec.model import v0_5

from ._magic_tensor_ops import MagicTensorOpsMixin
from .axis import Axis, AxisId, AxisInfo, AxisLike, PerAxis
from .common import (
    CropWhere,
    DTypeStr,
    PadMode,
    PadWhere,
    PadWidth,
    PadWidthLike,
    SliceInfo,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


_ScalarOrArray = Union["ArrayLike", np.generic, "NDArray[Any]"]  # TODO: add "DaskArray"


# TODO: complete docstrings
class Tensor(MagicTensorOpsMixin):
    """A wrapper around an xr.DataArray for better integration with bioimageio.spec
    and improved type annotations."""

    _Compatible = Union["Tensor", xr.DataArray, _ScalarOrArray]

    def __init__(
        self,
        array: NDArray[Any],
        dims: Sequence[Union[AxisId, AxisLike]],
    ) -> None:
        super().__init__()
        axes = tuple(
            a if isinstance(a, AxisId) else AxisInfo.create(a).id for a in dims
        )
        self._data = xr.DataArray(array, dims=axes)

    def __array__(self, dtype: DTypeLike = None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(
        self, key: Union[SliceInfo, slice, int, PerAxis[Union[SliceInfo, slice, int]]]
    ) -> Self:
        if isinstance(key, SliceInfo):
            key = slice(*key)
        elif isinstance(key, collections.abc.Mapping):
            key = {
                a: s if isinstance(s, int) else s if isinstance(s, slice) else slice(*s)
                for a, s in key.items()
            }
        return self.__class__.from_xarray(self._data[key])

    def __setitem__(self, key: PerAxis[Union[SliceInfo, slice]], value: Tensor) -> None:
        key = {a: s if isinstance(s, slice) else slice(*s) for a, s in key.items()}
        self._data[key] = value._data

    def __len__(self) -> int:
        return len(self.data)

    def _iter(self: Any) -> Iterator[Any]:
        for n in range(len(self)):
            yield self[n]

    def __iter__(self: Any) -> Iterator[Any]:
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return self._iter()

    def _binary_op(
        self,
        other: _Compatible,
        f: Callable[[Any, Any], Any],
        reflexive: bool = False,
    ) -> Self:
        data = self._data._binary_op(  # pyright: ignore[reportPrivateUsage]
            (other._data if isinstance(other, Tensor) else other),
            f,
            reflexive,
        )
        return self.__class__.from_xarray(data)

    def _inplace_binary_op(
        self,
        other: _Compatible,
        f: Callable[[Any, Any], Any],
    ) -> Self:
        _ = self._data._inplace_binary_op(  # pyright: ignore[reportPrivateUsage]
            (
                other_d
                if (other_d := getattr(other, "data")) is not None
                and isinstance(
                    other_d,
                    xr.DataArray,
                )
                else other
            ),
            f,
        )
        return self

    def _unary_op(self, f: Callable[[Any], Any], *args: Any, **kwargs: Any) -> Self:
        data = self._data._unary_op(  # pyright: ignore[reportPrivateUsage]
            f, *args, **kwargs
        )
        return self.__class__.from_xarray(data)

    @classmethod
    def from_xarray(cls, data_array: xr.DataArray) -> Self:
        """create a `Tensor` from an xarray data array

        note for internal use: this factory method is round-trip save
            for any `Tensor`'s  `data` property (an xarray.DataArray).
        """
        return cls(
            array=data_array.data, dims=tuple(AxisId(d) for d in data_array.dims)
        )

    @classmethod
    def from_numpy(
        cls,
        array: NDArray[Any],
        *,
        dims: Optional[Union[AxisLike, Sequence[AxisLike]]],
    ) -> Tensor:
        """create a `Tensor` from a numpy array

        Args:
            array: the nd numpy array
            axes: A description of the array's axes,
                if None axes are guessed (which might fail and raise a ValueError.)

        Raises:
            ValueError: if `axes` is None and axes guessing fails.
        """

        if dims is None:
            return cls._interprete_array_wo_known_axes(array)
        elif isinstance(dims, (str, Axis, v0_5.AxisBase)):
            dims = [dims]

        axis_infos = [AxisInfo.create(a) for a in dims]
        original_shape = tuple(array.shape)

        successful_view = _get_array_view(array, axis_infos)
        if successful_view is None:
            raise ValueError(
                f"Array shape {original_shape} does not map to axes {dims}"
            )

        return Tensor(successful_view, dims=tuple(a.id for a in axis_infos))

    @property
    def data(self):
        return self._data

    @property
    def dims(self):  # TODO: rename to `axes`?
        """Tuple of dimension names associated with this tensor."""
        return cast(Tuple[AxisId, ...], self._data.dims)

    @property
    def dtype(self) -> DTypeStr:
        dt = str(self.data.dtype)  # pyright: ignore[reportUnknownArgumentType]
        assert dt in get_args(DTypeStr)
        return dt  # pyright: ignore[reportReturnType]

    @property
    def ndim(self):
        """Number of tensor dimensions."""
        return self._data.ndim

    @property
    def shape_tuple(self):
        """Tuple of tensor axes lengths"""
        return self._data.shape

    @property
    def size(self):
        """Number of elements in the tensor.

        Equal to math.prod(tensor.shape), i.e., the product of the tensors’ dimensions.
        """
        return self._data.size

    @property
    def sizes(self):
        """Ordered, immutable mapping from axis ids to axis lengths."""
        return cast(Mapping[AxisId, int], self.data.sizes)

    @property
    def tagged_shape(self):
        """(alias for `sizes`) Ordered, immutable mapping from axis ids to lengths."""
        return self.sizes

    def argmax(self) -> Mapping[AxisId, int]:
        ret = self._data.argmax(...)
        assert isinstance(ret, dict)
        return {cast(AxisId, k): cast(int, v.item()) for k, v in ret.items()}

    def astype(self, dtype: DTypeStr, *, copy: bool = False):
        """Return tensor cast to `dtype`

        note: if dtype is already satisfied copy if `copy`"""
        return self.__class__.from_xarray(self._data.astype(dtype, copy=copy))

    def clip(self, min: Optional[float] = None, max: Optional[float] = None):
        """Return a tensor whose values are limited to [min, max].
        At least one of max or min must be given."""
        return self.__class__.from_xarray(self._data.clip(min, max))

    def crop_to(
        self,
        sizes: PerAxis[int],
        crop_where: Union[
            CropWhere,
            PerAxis[CropWhere],
        ] = "left_and_right",
    ) -> Self:
        """crop to match `sizes`"""
        if isinstance(crop_where, str):
            crop_axis_where: PerAxis[CropWhere] = {a: crop_where for a in self.dims}
        else:
            crop_axis_where = crop_where

        slices: Dict[AxisId, SliceInfo] = {}

        for a, s_is in self.sizes.items():
            if a not in sizes or sizes[a] == s_is:
                pass
            elif sizes[a] > s_is:
                logger.warning(
                    "Cannot crop axis {} of size {} to larger size {}",
                    a,
                    s_is,
                    sizes[a],
                )
            elif a not in crop_axis_where:
                raise ValueError(
                    f"Don't know where to crop axis {a}, `crop_where`={crop_where}"
                )
            else:
                crop_this_axis_where = crop_axis_where[a]
                if crop_this_axis_where == "left":
                    slices[a] = SliceInfo(s_is - sizes[a], s_is)
                elif crop_this_axis_where == "right":
                    slices[a] = SliceInfo(0, sizes[a])
                elif crop_this_axis_where == "left_and_right":
                    slices[a] = SliceInfo(
                        start := (s_is - sizes[a]) // 2, sizes[a] + start
                    )
                else:
                    assert_never(crop_this_axis_where)

        return self[slices]

    def expand_dims(self, dims: Union[Sequence[AxisId], PerAxis[int]]) -> Self:
        return self.__class__.from_xarray(self._data.expand_dims(dims=dims))

    def item(
        self,
        key: Union[
            None, SliceInfo, slice, int, PerAxis[Union[SliceInfo, slice, int]]
        ] = None,
    ):
        """Copy a tensor element to a standard Python scalar and return it."""
        if key is None:
            ret = self._data.item()
        else:
            ret = self[key]._data.item()

        assert isinstance(ret, (bool, float, int))
        return ret

    def mean(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.mean(dim=dim))

    def pad(
        self,
        pad_width: PerAxis[PadWidthLike],
        mode: PadMode = "symmetric",
    ) -> Self:
        pad_width = {a: PadWidth.create(p) for a, p in pad_width.items()}
        return self.__class__.from_xarray(
            self._data.pad(pad_width=pad_width, mode=mode)
        )

    def pad_to(
        self,
        sizes: PerAxis[int],
        pad_where: Union[PadWhere, PerAxis[PadWhere]] = "left_and_right",
        mode: PadMode = "symmetric",
    ) -> Self:
        """pad `tensor` to match `sizes`"""
        if isinstance(pad_where, str):
            pad_axis_where: PerAxis[PadWhere] = {a: pad_where for a in self.dims}
        else:
            pad_axis_where = pad_where

        pad_width: Dict[AxisId, PadWidth] = {}
        for a, s_is in self.sizes.items():
            if a not in sizes or sizes[a] == s_is:
                pad_width[a] = PadWidth(0, 0)
            elif s_is > sizes[a]:
                pad_width[a] = PadWidth(0, 0)
                logger.warning(
                    "Cannot pad axis {} of size {} to smaller size {}",
                    a,
                    s_is,
                    sizes[a],
                )
            elif a not in pad_axis_where:
                raise ValueError(
                    f"Don't know where to pad axis {a}, `pad_where`={pad_where}"
                )
            else:
                pad_this_axis_where = pad_axis_where[a]
                d = sizes[a] - s_is
                if pad_this_axis_where == "left":
                    pad_width[a] = PadWidth(d, 0)
                elif pad_this_axis_where == "right":
                    pad_width[a] = PadWidth(0, d)
                elif pad_this_axis_where == "left_and_right":
                    pad_width[a] = PadWidth(left := d // 2, d - left)
                else:
                    assert_never(pad_this_axis_where)

        return self.pad(pad_width, mode)

    def quantile(
        self,
        q: Union[float, Sequence[float]],
        dim: Optional[Union[AxisId, Sequence[AxisId]]] = None,
    ) -> Self:
        assert (
            isinstance(q, (float, int))
            and q >= 0.0
            or not isinstance(q, (float, int))
            and all(qq >= 0.0 for qq in q)
        )
        assert (
            isinstance(q, (float, int))
            and q <= 1.0
            or not isinstance(q, (float, int))
            and all(qq <= 1.0 for qq in q)
        )
        assert dim is None or (
            (quantile_dim := AxisId("quantile")) != dim and quantile_dim not in set(dim)
        )
        return self.__class__.from_xarray(self._data.quantile(q, dim=dim))

    def resize_to(
        self,
        sizes: PerAxis[int],
        *,
        pad_where: Union[
            PadWhere,
            PerAxis[PadWhere],
        ] = "left_and_right",
        crop_where: Union[
            CropWhere,
            PerAxis[CropWhere],
        ] = "left_and_right",
        pad_mode: PadMode = "symmetric",
    ):
        """return cropped/padded tensor with `sizes`"""
        crop_to_sizes: Dict[AxisId, int] = {}
        pad_to_sizes: Dict[AxisId, int] = {}
        new_axes = dict(sizes)
        for a, s_is in self.sizes.items():
            a = AxisId(str(a))
            _ = new_axes.pop(a, None)
            if a not in sizes or sizes[a] == s_is:
                pass
            elif s_is > sizes[a]:
                crop_to_sizes[a] = sizes[a]
            else:
                pad_to_sizes[a] = sizes[a]

        tensor = self
        if crop_to_sizes:
            tensor = tensor.crop_to(crop_to_sizes, crop_where=crop_where)

        if pad_to_sizes:
            tensor = tensor.pad_to(pad_to_sizes, pad_where=pad_where, mode=pad_mode)

        if new_axes:
            tensor = tensor.expand_dims(new_axes)

        return tensor

    def std(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.std(dim=dim))

    def sum(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        """Reduce this Tensor's data by applying sum along some dimension(s)."""
        return self.__class__.from_xarray(self._data.sum(dim=dim))

    def transpose(
        self,
        axes: Sequence[AxisId],
    ) -> Self:
        """return a transposed tensor

        Args:
            axes: the desired tensor axes
        """
        # expand missing tensor axes
        missing_axes = tuple(a for a in axes if a not in self.dims)
        array = self._data
        if missing_axes:
            array = array.expand_dims(missing_axes)

        # transpose to the correct axis order
        return self.__class__.from_xarray(array.transpose(*axes))

    def var(self, dim: Optional[Union[AxisId, Sequence[AxisId]]] = None) -> Self:
        return self.__class__.from_xarray(self._data.var(dim=dim))

    @classmethod
    def _interprete_array_wo_known_axes(cls, array: NDArray[Any]):
        ndim = array.ndim
        if ndim == 2:
            current_axes = (
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[0]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[1]),
            )
        elif ndim == 3 and any(s <= 3 for s in array.shape):
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
            )
        elif ndim == 3:
            current_axes = (
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[0]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
            )
        elif ndim == 4:
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[2]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[3]),
            )
        elif ndim == 5:
            current_axes = (
                v0_5.BatchAxis(),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[1])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[2]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[3]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[4]),
            )
        else:
            raise ValueError(f"Could not guess an axis mapping for {array.shape}")

        return cls(array, dims=tuple(a.id for a in current_axes))


def _add_singletons(arr: NDArray[Any], axis_infos: Sequence[AxisInfo]):
    if len(arr.shape) > len(axis_infos):
        # remove singletons
        for i, s in enumerate(arr.shape):
            if s == 1:
                arr = np.take(arr, 0, axis=i)
                if len(arr.shape) == len(axis_infos):
                    break

    # add singletons if nececsary
    for i, a in enumerate(axis_infos):
        if len(arr.shape) >= len(axis_infos):
            break

        if a.maybe_singleton:
            arr = np.expand_dims(arr, i)

    return arr


def _get_array_view(
    original_array: NDArray[Any], axis_infos: Sequence[AxisInfo]
) -> Optional[NDArray[Any]]:
    perms = list(permutations(range(len(original_array.shape))))
    perms.insert(1, perms.pop())  # try A and A.T first

    for perm in perms:
        view = original_array.transpose(perm)
        view = _add_singletons(view, axis_infos)
        if len(view.shape) != len(axis_infos):
            return None

        for s, a in zip(view.shape, axis_infos):
            if s == 1 and not a.maybe_singleton:
                break
        else:
            return view

    return None
