from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from .axis import AxisId, PerAxis
from .block import Block
from .block_meta import (
    BlockMeta,
    LinearAxisTransform,
    split_multiple_shapes_into_blocks,
)
from .common import (
    BlockIndex,
    Halo,
    HaloLike,
    MemberId,
    PadMode,
    PerMember,
    SampleId,
    SliceInfo,
    TotalNumberOfBlocks,
)
from .stat_measures import Stat
from .tensor import Tensor

# TODO: allow for lazy samples to read/write to disk


@dataclass
class Sample:
    """A dataset sample.

    A `Sample` has `members`, which allows to combine multiple tensors into a single
    sample.
    For example a `Sample` from a dataset with masked images may contain a
    `MemberId("raw")` and `MemberId("mask")` image.
    """

    members: Dict[MemberId, Tensor]
    """The sample's tensors"""

    stat: Stat
    """Sample and dataset statistics"""

    id: SampleId
    """Identifies the `Sample` within the dataset -- typically a number or a string."""

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.members.items()}

    def as_arrays(self) -> Dict[str, NDArray[Any]]:
        """Return sample as dictionary of arrays."""
        return {str(m): t.data.to_numpy() for m, t in self.members.items()}

    def split_into_blocks(
        self,
        block_shapes: PerMember[PerAxis[int]],
        halo: PerMember[PerAxis[HaloLike]],
        pad_mode: PadMode,
        broadcast: bool = False,
    ) -> Tuple[TotalNumberOfBlocks, Iterable[SampleBlockWithOrigin]]:
        assert not (
            missing := [m for m in block_shapes if m not in self.members]
        ), f"`block_shapes` specified for unknown members: {missing}"
        assert not (
            missing := [m for m in halo if m not in block_shapes]
        ), f"`halo` specified for members without `block_shape`: {missing}"

        n_blocks, blocks = split_multiple_shapes_into_blocks(
            shapes=self.shape,
            block_shapes=block_shapes,
            halo=halo,
            broadcast=broadcast,
        )
        return n_blocks, sample_block_generator(blocks, origin=self, pad_mode=pad_mode)

    def as_single_block(self, halo: Optional[PerMember[PerAxis[Halo]]] = None):
        if halo is None:
            halo = {}
        return SampleBlockWithOrigin(
            sample_shape=self.shape,
            sample_id=self.id,
            blocks={
                m: Block(
                    sample_shape=self.shape[m],
                    data=data,
                    inner_slice={
                        a: SliceInfo(0, s) for a, s in data.tagged_shape.items()
                    },
                    halo=halo.get(m, {}),
                    block_index=0,
                    blocks_in_sample=1,
                )
                for m, data in self.members.items()
            },
            stat=self.stat,
            origin=self,
            block_index=0,
            blocks_in_sample=1,
        )

    @classmethod
    def from_blocks(
        cls,
        sample_blocks: Iterable[SampleBlock],
        *,
        fill_value: float = float("nan"),
    ) -> Self:
        members: PerMember[Tensor] = {}
        stat: Stat = {}
        sample_id = None
        for sample_block in sample_blocks:
            assert sample_id is None or sample_id == sample_block.sample_id
            sample_id = sample_block.sample_id
            stat = sample_block.stat
            for m, block in sample_block.blocks.items():
                if m not in members:
                    if -1 in block.sample_shape.values():
                        raise NotImplementedError(
                            "merging blocks with data dependent axis not yet implemented"
                        )

                    members[m] = Tensor(
                        np.full(
                            tuple(block.sample_shape[a] for a in block.data.dims),
                            fill_value,
                            dtype=block.data.dtype,
                        ),
                        dims=block.data.dims,
                    )

                members[m][block.inner_slice] = block.inner_data

        return cls(members=members, stat=stat, id=sample_id)


BlockT = TypeVar("BlockT", Block, BlockMeta)


@dataclass
class SampleBlockBase(Generic[BlockT]):
    """base class for `SampleBlockMeta` and `SampleBlock`"""

    sample_shape: PerMember[PerAxis[int]]
    """the sample shape this block represents a part of"""

    sample_id: SampleId
    """identifier for the sample within its dataset"""

    blocks: Dict[MemberId, BlockT]
    """Individual tensor blocks comprising this sample block"""

    block_index: BlockIndex
    """the n-th block of the sample"""

    blocks_in_sample: TotalNumberOfBlocks
    """total number of blocks in the sample"""

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {mid: b.shape for mid, b in self.blocks.items()}

    @property
    def inner_shape(self) -> PerMember[PerAxis[int]]:
        return {mid: b.inner_shape for mid, b in self.blocks.items()}


@dataclass
class LinearSampleAxisTransform(LinearAxisTransform):
    member: MemberId


@dataclass
class SampleBlockMeta(SampleBlockBase[BlockMeta]):
    """Meta data of a dataset sample block"""

    def get_transformed(
        self, new_axes: PerMember[PerAxis[Union[LinearSampleAxisTransform, int]]]
    ) -> Self:
        sample_shape = {
            m: {
                a: (
                    trf
                    if isinstance(trf, int)
                    else trf.compute(self.sample_shape[trf.member][trf.axis])
                )
                for a, trf in new_axes[m].items()
            }
            for m in new_axes
        }

        def get_member_halo(m: MemberId, round: Callable[[float], int]):
            return {
                a: (
                    Halo(0, 0)
                    if isinstance(trf, int)
                    or trf.axis not in self.blocks[trf.member].halo
                    else Halo(
                        round(self.blocks[trf.member].halo[trf.axis].left * trf.scale),
                        round(self.blocks[trf.member].halo[trf.axis].right * trf.scale),
                    )
                )
                for a, trf in new_axes[m].items()
            }

        halo: Dict[MemberId, Dict[AxisId, Halo]] = {}
        for m in new_axes:
            halo[m] = get_member_halo(m, floor)
            if halo[m] != get_member_halo(m, ceil):
                raise ValueError(
                    f"failed to unambiguously scale halo {halo[m]} with {new_axes[m]}"
                    + f" for {m}."
                )

        inner_slice = {
            m: {
                a: (
                    SliceInfo(0, trf)
                    if isinstance(trf, int)
                    else SliceInfo(
                        trf.compute(
                            self.blocks[trf.member].inner_slice[trf.axis].start
                        ),
                        trf.compute(self.blocks[trf.member].inner_slice[trf.axis].stop),
                    )
                )
                for a, trf in new_axes[m].items()
            }
            for m in new_axes
        }
        return self.__class__(
            blocks={
                m: BlockMeta(
                    sample_shape=sample_shape[m],
                    inner_slice=inner_slice[m],
                    halo=halo[m],
                    block_index=self.block_index,
                    blocks_in_sample=self.blocks_in_sample,
                )
                for m in new_axes
            },
            sample_shape=sample_shape,
            sample_id=self.sample_id,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        )

    def with_data(self, data: PerMember[Tensor], *, stat: Stat) -> SampleBlock:
        return SampleBlock(
            sample_shape={
                m: {
                    a: data[m].tagged_shape[a] if s == -1 else s
                    for a, s in member_shape.items()
                }
                for m, member_shape in self.sample_shape.items()
            },
            sample_id=self.sample_id,
            blocks={
                m: Block.from_meta(b, data=data[m]) for m, b in self.blocks.items()
            },
            stat=stat,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        )


@dataclass
class SampleBlock(SampleBlockBase[Block]):
    """A block of a dataset sample"""

    stat: Stat
    """computed statistics"""

    @property
    def members(self) -> PerMember[Tensor]:
        """the sample block's tensors"""
        return {m: b.data for m, b in self.blocks.items()}

    def get_transformed_meta(
        self, new_axes: PerMember[PerAxis[Union[LinearSampleAxisTransform, int]]]
    ) -> SampleBlockMeta:
        return SampleBlockMeta(
            sample_id=self.sample_id,
            blocks=dict(self.blocks),
            sample_shape=self.sample_shape,
            block_index=self.block_index,
            blocks_in_sample=self.blocks_in_sample,
        ).get_transformed(new_axes)


@dataclass
class SampleBlockWithOrigin(SampleBlock):
    """A `SampleBlock` with a reference (`origin`) to the whole `Sample`"""

    origin: Sample
    """the sample this sample block was taken from"""


class _ConsolidatedMemberBlocks:
    def __init__(self, blocks: PerMember[BlockMeta]):
        super().__init__()
        block_indices = {b.block_index for b in blocks.values()}
        assert len(block_indices) == 1
        self.block_index = block_indices.pop()
        blocks_in_samples = {b.blocks_in_sample for b in blocks.values()}
        assert len(blocks_in_samples) == 1
        self.blocks_in_sample = blocks_in_samples.pop()


def sample_block_meta_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    sample_shape: PerMember[PerAxis[int]],
    sample_id: SampleId,
):
    for member_blocks in blocks:
        cons = _ConsolidatedMemberBlocks(member_blocks)
        yield SampleBlockMeta(
            blocks=dict(member_blocks),
            sample_shape=sample_shape,
            sample_id=sample_id,
            block_index=cons.block_index,
            blocks_in_sample=cons.blocks_in_sample,
        )


def sample_block_generator(
    blocks: Iterable[PerMember[BlockMeta]],
    *,
    origin: Sample,
    pad_mode: PadMode,
) -> Iterable[SampleBlockWithOrigin]:
    for member_blocks in blocks:
        cons = _ConsolidatedMemberBlocks(member_blocks)
        yield SampleBlockWithOrigin(
            blocks={
                m: Block.from_sample_member(
                    origin.members[m], block=member_blocks[m], pad_mode=pad_mode
                )
                for m in origin.members
            },
            sample_shape=origin.shape,
            origin=origin,
            stat=origin.stat,
            sample_id=origin.id,
            block_index=cons.block_index,
            blocks_in_sample=cons.blocks_in_sample,
        )
