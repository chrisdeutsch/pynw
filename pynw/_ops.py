"""Operation codes and index reconstruction for Needleman-Wunsch alignments."""

from collections.abc import Iterable, Iterator
from enum import IntEnum
from typing import TypeVar, assert_never

import numpy as np
import numpy.typing as npt

from pynw._native import OP_ALIGN, OP_DELETE, OP_INSERT


class EditOp(IntEnum):
    """
    Edit operation codes.

    Not guaranteed to be compatible between pynw version.
    """

    Align = OP_ALIGN
    Insert = OP_INSERT
    Delete = OP_DELETE


_STRIDE_TABLE = np.zeros((2, max(EditOp) + 1), dtype=np.intp)
_STRIDE_TABLE[:, EditOp.Align] = (1, 1)
_STRIDE_TABLE[:, EditOp.Insert] = (0, 1)
_STRIDE_TABLE[:, EditOp.Delete] = (1, 0)
_STRIDE_TABLE.flags.writeable = False


def indices_from_ops_stride_table(
    ops: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Reconstruct source and target indices from an ops array."""
    ops = np.asarray(ops, dtype=np.uint8)

    strides = _STRIDE_TABLE[:, ops]
    idx = np.cumsum(strides, axis=-1) - strides

    source_idx = np.ma.array(idx[0, :], mask=ops == EditOp.Insert)
    target_idx = np.ma.array(idx[1, :], mask=ops == EditOp.Delete)

    return source_idx, target_idx


def indices_from_ops_direct(
    ops: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Reconstruct source and target indices from an ops array."""
    ops = np.asarray(ops, dtype=np.uint8)

    insert_mask = ops == EditOp.Insert
    delete_mask = ops == EditOp.Delete

    source_advances = ~insert_mask
    target_advances = ~delete_mask

    source_idx = np.ma.array(
        np.cumsum(source_advances) - source_advances, mask=insert_mask
    )
    target_idx = np.ma.array(
        np.cumsum(target_advances) - target_advances, mask=delete_mask
    )

    return source_idx, target_idx


SourceType = TypeVar("SourceType")
TargetType = TypeVar("TargetType")


# TODO: Strict flag like zip?
def iter_alignment(
    ops: npt.NDArray[np.uint8],
    source_sequence: Iterable[SourceType],
    target_sequence: Iterable[TargetType],
) -> Iterator[tuple[EditOp, SourceType | None, TargetType | None]]:
    """Iterate over a Needleman-Wunsch alignment.

    Yields one ``(op, source_item, target_item)`` triple per alignment position.
    ``source_item`` is ``None`` for an insert (gap in the source sequence);
    ``target_item`` is ``None`` for a delete (gap in the target sequence).

    If the source or target sequence contains legitimate ``None`` values,
    use the ``op`` field to distinguish gaps from real elements: a ``None``
    source_item paired with ``EditOp.Insert`` is a gap, whereas a ``None``
    paired with ``EditOp.Align`` or ``EditOp.Delete`` is a real sequence
    element (and likewise for target_item).

    Parameters
    ----------
    ops : ndarray of uint8
        EditOp sequence returned by ``needleman_wunsch``.
    source_sequence : sequence
        The source sequence passed to the aligner.
    target_sequence : sequence
        The target sequence passed to the aligner.

    Yields
    ------
    op : Op
        The operation at this alignment position.
    source_item : element of source_seq, or None
        The source element consumed at this step, or ``None`` for an insert.
    target_item : element of target_seq, or None
        The target element consumed at this step, or ``None`` for a delete.
    """
    source_iter = iter(source_sequence)
    target_iter = iter(target_sequence)

    for op_uint in ops:
        op = EditOp(op_uint)
        try:
            match op:
                case EditOp.Align:
                    yield op, next(source_iter), next(target_iter)
                case EditOp.Insert:
                    yield op, None, next(target_iter)
                case EditOp.Delete:
                    yield op, next(source_iter), None
                case _:
                    assert_never(op)
        except StopIteration:
            raise ValueError(
                "Length of source and/or target iterable does not match edit operations"
            ) from None
