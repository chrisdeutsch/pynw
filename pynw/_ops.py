"""Operation codes and index reconstruction for Needleman-Wunsch alignments."""

from collections.abc import Iterable, Iterator
from enum import IntEnum
from typing import TypeAlias, TypeVar, assert_never

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


MaskedIndexArray: TypeAlias = np.ma.MaskedArray[tuple[int], np.dtype[np.intp]]


def indices_from_ops(
    ops: npt.NDArray[np.uint8],
) -> tuple[MaskedIndexArray, MaskedIndexArray]:
    """Reconstruct source and target indices from an ops array.

    Converts a sequence of edit operations into a pair of masked index arrays.
    Each array has one entry per alignment position.  Positions where the
    corresponding sequence has a gap are masked out.

    Parameters
    ----------
    ops : ndarray of uint8, shape (k,)
        Edit-operation sequence returned by ``needleman_wunsch``.

    Returns
    -------
    source_idx : masked array of intp, shape (k,)
        Index into the source sequence at each alignment position.
        Masked (invalid) at insert positions (gap in source).
    target_idx : masked array of intp, shape (k,)
        Index into the target sequence at each alignment position.
        Masked (invalid) at delete positions (gap in target).

    Examples
    --------
    >>> import numpy as np
    >>> from pynw import needleman_wunsch, indices_from_ops
    >>> source_seq = list("GAT")
    >>> target_seq = list("GT")
    >>> sm = np.where(
    ...     np.array(source_seq)[:, None] == np.array(target_seq)[None, :],
    ...     1.0, -1.0,
    ... )
    >>> _, ops = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> src, tgt = indices_from_ops(ops)
    >>> src.tolist()
    [0, 1, 2]
    >>> tgt.tolist()
    [0, None, 1]
    """
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
