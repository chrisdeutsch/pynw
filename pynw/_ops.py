"""Operation codes and index reconstruction for Needleman-Wunsch alignments."""

from enum import IntEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from pynw._native import (
    OP_ALIGN,
    OP_DELETE,
    OP_INSERT,
)
from pynw._native import alignment_indices as _alignment_indices


class EditOp(IntEnum):
    """
    Edit operation codes.

    Not guaranteed to be compatible between pynw version.
    """

    Align = OP_ALIGN
    Insert = OP_INSERT
    Delete = OP_DELETE


MaskedIndexArray: TypeAlias = np.ma.MaskedArray[tuple[int], np.dtype[np.intp]]


# This is faster for extremely large k (k > 10000). Maybe due to hardware-level
# vectorization?
def _alignment_indices_numpy(
    editops: npt.ArrayLike,
) -> tuple[MaskedIndexArray, MaskedIndexArray]:
    editops = np.asarray(editops, dtype=np.uint8)

    insert_mask = editops == EditOp.Insert
    delete_mask = editops == EditOp.Delete

    source_advances = ~insert_mask
    target_advances = ~delete_mask

    source_idx = np.ma.array(
        np.cumsum(source_advances) - source_advances, mask=insert_mask
    )
    target_idx = np.ma.array(
        np.cumsum(target_advances) - target_advances, mask=delete_mask
    )

    return source_idx, target_idx


def alignment_indices(
    editops: npt.ArrayLike,
) -> tuple[MaskedIndexArray, MaskedIndexArray]:
    """Reconstruct source and target indices from an editops array.

    Converts a sequence of edit operations into a pair of masked index arrays.
    Each array has one entry per alignment position.  Positions where the
    corresponding sequence has a gap are masked out.

    Parameters
    ----------
    editops : array_like of uint8, shape (k,)
        Edit-operation sequence returned by ``needleman_wunsch``.

    Returns
    -------
    source_idx : masked array of intp, shape (k,)
        Index into the source sequence at each alignment position.
        Masked (invalid) at insert positions (gap in source).
    target_idx : masked array of intp, shape (k,)
        Index into the target sequence at each alignment position.
        Masked (invalid) at delete positions (gap in target).

    Raises
    ------
    ValueError
        If ``editops`` cannot be converted to a 1-D ``uint8`` array, if any
        element is out of the ``uint8`` range, or if any element is not a
        valid ``EditOp`` discriminant.

    Examples
    --------
    >>> import numpy as np
    >>> from pynw import needleman_wunsch, alignment_indices
    >>> source_seq = list("GAT")
    >>> target_seq = list("GT")
    >>> sm = np.where(
    ...     np.array(source_seq)[:, None] == np.array(target_seq)[None, :],
    ...     1.0, -1.0,
    ... )
    >>> _, editops = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> src, tgt = alignment_indices(editops)
    >>> src.tolist()
    [0, 1, 2]
    >>> tgt.tolist()
    [0, None, 1]
    """
    src_idx, src_mask, tgt_idx, tgt_mask = _alignment_indices(editops)
    return np.ma.array(src_idx, mask=src_mask), np.ma.array(tgt_idx, mask=tgt_mask)
