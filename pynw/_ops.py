"""Operation codes and index reconstruction for `needleman_wunsch_merge_split`."""

from enum import IntEnum

import numpy as np
import numpy.typing as npt

from pynw._native import OP_ALIGN, OP_DELETE, OP_INSERT, OP_MERGE, OP_SPLIT


class Op(IntEnum):
    """Operation codes returned by `needleman_wunsch_merge_split`."""

    ALIGN = OP_ALIGN
    INSERT = OP_INSERT
    DELETE = OP_DELETE
    SPLIT = OP_SPLIT
    MERGE = OP_MERGE


_ROW_STRIDE = np.zeros(max(Op) + 1, dtype=np.intp)
_ROW_STRIDE[Op.ALIGN] = 1
# INSERT = 0: no row element consumed
_ROW_STRIDE[Op.DELETE] = 1
_ROW_STRIDE[Op.SPLIT] = 1
_ROW_STRIDE[Op.MERGE] = 2
_ROW_STRIDE.flags.writeable = False

_COL_STRIDE = np.zeros(max(Op) + 1, dtype=np.intp)
_COL_STRIDE[Op.ALIGN] = 1
_COL_STRIDE[Op.INSERT] = 1
# DELETE = 0: no col element consumed
_COL_STRIDE[Op.SPLIT] = 2
_COL_STRIDE[Op.MERGE] = 1
_COL_STRIDE.flags.writeable = False


def indices_from_ops(
    ops: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Reconstruct row and column indices from an ops array."""
    row_idx = np.cumsum(_ROW_STRIDE[ops]) - _ROW_STRIDE[ops]
    col_idx = np.cumsum(_COL_STRIDE[ops]) - _COL_STRIDE[ops]
    return row_idx, col_idx
