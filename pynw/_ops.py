"""Operation codes and index reconstruction for `needleman_wunsch_merge_split`."""

from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from pynw._native import OP_ALIGN, OP_DELETE, OP_INSERT, OP_MERGE, OP_SPLIT

T = TypeVar("T")
S = TypeVar("S")


class Op(IntEnum):
    """Operation codes returned by `needleman_wunsch_merge_split`."""

    ALIGN = OP_ALIGN
    INSERT = OP_INSERT
    DELETE = OP_DELETE
    SPLIT = OP_SPLIT
    MERGE = OP_MERGE


_SOURCE_STRIDE = np.zeros(max(Op) + 1, dtype=np.intp)
_SOURCE_STRIDE[Op.ALIGN] = 1
# INSERT = 0: no source element consumed
_SOURCE_STRIDE[Op.DELETE] = 1
_SOURCE_STRIDE[Op.SPLIT] = 1
_SOURCE_STRIDE[Op.MERGE] = 2
_SOURCE_STRIDE.flags.writeable = False

_TARGET_STRIDE = np.zeros(max(Op) + 1, dtype=np.intp)
_TARGET_STRIDE[Op.ALIGN] = 1
_TARGET_STRIDE[Op.INSERT] = 1
# DELETE = 0: no target element consumed
_TARGET_STRIDE[Op.SPLIT] = 2
_TARGET_STRIDE[Op.MERGE] = 1
_TARGET_STRIDE.flags.writeable = False


def indices_from_ops(
    ops: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Reconstruct source and target indices from an ops array."""
    source_idx = np.cumsum(_SOURCE_STRIDE[ops]) - _SOURCE_STRIDE[ops]
    target_idx = np.cumsum(_TARGET_STRIDE[ops]) - _TARGET_STRIDE[ops]
    return source_idx, target_idx


def iter_alignment(
    ops: npt.NDArray[np.uint8],
    source_seq: Sequence[T],
    target_seq: Sequence[S],
) -> Iterator[tuple[Op, T | None, S | None]]:
    """Iterate over a Needleman-Wunsch alignment step by step.

    Yields one ``(op, source_item, target_item)`` triple per alignment position.
    ``source_item`` is ``None`` for an insert (gap in the source sequence);
    ``target_item`` is ``None`` for a delete (gap in the target sequence).

    Parameters
    ----------
    ops : ndarray of uint8
        Op sequence returned by ``needleman_wunsch``.
    source_seq : sequence
        The source sequence passed to the aligner.
    target_seq : sequence
        The target sequence passed to the aligner.

    Yields
    ------
    op : Op
        The operation at this alignment position.
    source_item : element of source_seq, or None
        The source element consumed at this step, or ``None`` for an insert.
    target_item : element of target_seq, or None
        The target element consumed at this step, or ``None`` for a delete.

    Examples
    --------
    >>> import numpy as np
    >>> from pynw import needleman_wunsch
    >>> from pynw._ops import iter_alignment
    >>> source_seq = list("GATTACA")
    >>> target_seq = list("GCATGCA")
    >>> sm = np.where(
    ...     np.array(source_seq)[:, None] == np.array(target_seq)[None, :], 1.0, -1.0
    ... )
    >>> score, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> aligned_source = "".join(source_seq[i] if i >= 0 else "-" for i in source_idx)
    >>> aligned_target = "".join(target_seq[i] if i >= 0 else "-" for i in target_idx)
    >>> aligned_source
    'G-ATTACA'
    >>> aligned_target
    'GCA-TGCA'
    """
    ri, ci = 0, 0
    for raw_op in ops:
        op = Op(raw_op)
        if op == Op.ALIGN:
            yield op, source_seq[ri], target_seq[ci]
            ri += 1
            ci += 1
        elif op == Op.INSERT:
            yield op, None, target_seq[ci]
            ci += 1
        else:  # Op.DELETE
            yield op, source_seq[ri], None
            ri += 1
