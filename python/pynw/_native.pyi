# NOTE: This stub provides signatures for type checkers and IDE support.
# Docstrings live in src/lib.rs as Rust doc comments (runtime `help()`).

from typing import overload

import numpy as np
import numpy.typing as npt

OP_ALIGN: int
OP_INSERT: int
OP_DELETE: int

@overload
def needleman_wunsch(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float,
    insert_penalty: float | None = ...,
    delete_penalty: float | None = ...,
) -> tuple[float, npt.NDArray[np.uint8]]: ...
@overload
def needleman_wunsch(
    similarity_matrix: npt.ArrayLike,
    *,
    insert_penalty: float,
    delete_penalty: float,
) -> tuple[float, npt.NDArray[np.uint8]]: ...
@overload
def needleman_wunsch_score(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float,
    insert_penalty: float | None = ...,
    delete_penalty: float | None = ...,
) -> float: ...
@overload
def needleman_wunsch_score(
    similarity_matrix: npt.ArrayLike,
    *,
    insert_penalty: float,
    delete_penalty: float,
) -> float: ...
def alignment_indices(
    editops: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.bool_],
    npt.NDArray[np.intp],
    npt.NDArray[np.bool_],
]:
    """See ``pynw.alignment_indices`` for the public API."""
    ...
