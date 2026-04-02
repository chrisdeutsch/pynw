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


_SOURCE_STRIDE = np.zeros(max(EditOp) + 1, dtype=np.intp)
_SOURCE_STRIDE[EditOp.Align] = 1
# INSERT = 0: no source element consumed
_SOURCE_STRIDE[EditOp.Delete] = 1
_SOURCE_STRIDE.flags.writeable = False

_TARGET_STRIDE = np.zeros(max(EditOp) + 1, dtype=np.intp)
_TARGET_STRIDE[EditOp.Align] = 1
_TARGET_STRIDE[EditOp.Insert] = 1
# DELETE = 0: no target element consumed
_TARGET_STRIDE.flags.writeable = False


def indices_from_ops(
    ops: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Reconstruct source and target indices from an ops array."""
    source_idx = np.cumsum(_SOURCE_STRIDE[ops]) - _SOURCE_STRIDE[ops]
    target_idx = np.cumsum(_TARGET_STRIDE[ops]) - _TARGET_STRIDE[ops]
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
