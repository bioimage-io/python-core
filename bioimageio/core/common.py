from __future__ import annotations

from types import MappingProxyType
from typing import (
    Hashable,
    Literal,
    Mapping,
    NamedTuple,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Self, assert_never

from bioimageio.spec.model import v0_5

from .axis import AxisId

SupportedWeightsFormat = Literal[
    "keras_hdf5",
    "onnx",
    "pytorch_state_dict",
    "tensorflow_saved_model_bundle",
    "torchscript",
]


DTypeStr = Literal[
    "bool",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


_LeftRight_T = TypeVar("_LeftRight_T", bound="_LeftRight")
_LeftRightLike = Union[int, Tuple[int, int], _LeftRight_T]


class _LeftRight(NamedTuple):
    left: int
    right: int

    @classmethod
    def create(cls, like: _LeftRightLike[Self]) -> Self:
        if isinstance(like, cls):
            return like
        elif isinstance(like, tuple):
            return cls(*like)
        elif isinstance(like, int):
            return cls(like, like)
        else:
            assert_never(like)


_Where = Literal["left", "right", "left_and_right"]


class CropWidth(_LeftRight):
    pass


CropWidthLike = _LeftRightLike[CropWidth]
CropWhere = _Where


class Halo(_LeftRight):
    pass


HaloLike = _LeftRightLike[Halo]


class OverlapWidth(_LeftRight):
    pass


class PadWidth(_LeftRight):
    pass


PadWidthLike = _LeftRightLike[PadWidth]
PadMode = Literal["edge", "reflect", "symmetric"]
PadWhere = _Where


class SliceInfo(NamedTuple):
    start: int
    stop: int


SampleId = Hashable
"""ID of a sample, see `bioimageio.core.sample.Sample`"""
MemberId = v0_5.TensorId
"""ID of a `Sample` member, see `bioimageio.core.sample.Sample`"""

BlocksizeParameter = Union[
    v0_5.ParameterizedSize_N,
    Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize_N],
]
"""
Parameter to determine a concrete size for paramtrized axis sizes defined by
`bioimageio.spec.model.v0_5.ParameterizedSize`.
"""

T = TypeVar("T")
PerMember = Mapping[MemberId, T]

BlockIndex = int
TotalNumberOfBlocks = int


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

Frozen = MappingProxyType
# class Frozen(Mapping[K, V]):  # adapted from xarray.core.utils.Frozen
#     """Wrapper around an object implementing the mapping interface to make it
#     immutable."""

#     __slots__ = ("mapping",)

#     def __init__(self, mapping: Mapping[K, V]):
#         super().__init__()
#         self.mapping = deepcopy(
#             mapping
#         )  # added deepcopy (compared to xarray.core.utils.Frozen)

#     def __getitem__(self, key: K) -> V:
#         return self.mapping[key]

#     def __iter__(self) -> Iterator[K]:
#         return iter(self.mapping)

#     def __len__(self) -> int:
#         return len(self.mapping)

#     def __contains__(self, key: object) -> bool:
#         return key in self.mapping

#     def __repr__(self) -> str:
#         return f"{type(self).__name__}({self.mapping!r})"
