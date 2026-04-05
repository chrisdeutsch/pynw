# NOTE: This stub provides the docstring for type checkers and IDE support.
# A copy exists as a Rust doc comment in src/lib.rs for runtime `help()`.
# Keep both copies in sync.

import numpy as np
import numpy.typing as npt

OP_ALIGN: int
OP_INSERT: int
OP_DELETE: int

def needleman_wunsch(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    insert_penalty: float | None = None,
    delete_penalty: float | None = None,
) -> tuple[float, npt.NDArray[np.uint8]]:
    """Align two ordered sequences given a precomputed similarity matrix.

    The total alignment score is the sum of similarity-matrix entries for
    matched positions and gap penalties for insertions/deletions.

    Parameters
    ----------
    similarity_matrix : array_like, shape (n, m)
        ``similarity_matrix[i, j]`` is the similarity score for aligning
        element *i* of the source sequence with element *j* of the target
        sequence.
    gap_penalty : float, default -1.0
        Penalty applied when a gap is inserted in either sequence.
        Use ``insert_penalty`` or ``delete_penalty`` to set them
        independently.
    insert_penalty : float, optional
        Penalty for advancing the target sequence without consuming a source
        element (gap in source).  Defaults to ``gap_penalty``.
    delete_penalty : float, optional
        Penalty for advancing the source sequence without consuming a target
        element (gap in target).  Defaults to ``gap_penalty``.

    Raises
    ------
    ValueError
        If ``similarity_matrix`` is not 2-dimensional, or if any value in
        ``similarity_matrix`` or the gap penalties is ``NaN`` or ``Inf``.

    Returns
    -------
    score : float
        The optimal alignment score.
    ops : ndarray of uint8, shape (k,)
        Sequence of edit operations describing the alignment.  Each element
        is of type ``EditOp``.  Use
        ``alignment_indices`` to reconstruct source and target index arrays.

    Examples
    --------
    Align two DNA sequences using a simple match/mismatch scoring scheme:

    >>> import numpy as np
    >>> from pynw import alignment_indices
    >>> seq1 = np.array(list("GATTACA"))
    >>> seq2 = np.array(list("GCATGCA"))
    >>> sm = np.where(seq1[:, None] == seq2[None, :], 1.0, -1.0)
    >>> score, ops = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> score
    2.0
    >>> src_idx, tgt_idx = alignment_indices(ops)
    >>> "".join(np.ma.array(seq1).take(src_idx).filled("-"))
    'G-ATTACA'
    >>> "".join(np.ma.array(seq2).take(tgt_idx).filled("-"))
    'GCA-TGCA'

    Notes
    -----
    When multiple alignments achieve the same optimal score, ties are
    broken deterministically: ``Align > Delete > Insert``.  This prefers
    substitutions over gaps, producing compact alignments.  Other tools
    may return different co-optimal alignments.

    All values in ``similarity_matrix`` and the gap penalties must be finite.
    """
    ...

def needleman_wunsch_score(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    insert_penalty: float | None = None,
    delete_penalty: float | None = None,
) -> float:
    """Compute the optimal Needleman-Wunsch alignment score without the traceback.

    Returns the same score as ``needleman_wunsch`` but uses O(m) memory instead
    of O(n*m) by retaining only two rows of the DP table at a time. The runtime
    difference between the two is minor. Use this function when you need the
    score but not the alignment itself.

    Parameters
    ----------
    similarity_matrix : array_like, shape (n, m)
        ``similarity_matrix[i, j]`` is the similarity score for aligning
        element *i* of the source sequence with element *j* of the target
        sequence.
    gap_penalty : float, default -1.0
        Penalty applied when a gap is inserted in either sequence.
        Use ``insert_penalty`` or ``delete_penalty`` to set them
        independently.
    insert_penalty : float, optional
        Penalty for advancing the target sequence without consuming a source
        element (gap in source).  Defaults to ``gap_penalty``.
    delete_penalty : float, optional
        Penalty for advancing the source sequence without consuming a target
        element (gap in target).  Defaults to ``gap_penalty``.

    Raises
    ------
    ValueError
        If ``similarity_matrix`` is not 2-dimensional, or if any value in
        ``similarity_matrix`` or the gap penalties is ``NaN`` or ``Inf``.

    Returns
    -------
    score : float
        The optimal alignment score.

    Examples
    --------
    >>> import numpy as np
    >>> seq1 = np.array(list("GATTACA"))
    >>> seq2 = np.array(list("GCATGCA"))
    >>> sm = np.where(seq1[:, None] == seq2[None, :], 1.0, -1.0)
    >>> needleman_wunsch_score(sm, gap_penalty=-1.0)
    2.0

    Notes
    -----
    All values in ``similarity_matrix`` and the gap penalties must be finite.
    """
    ...

def alignment_indices(
    ops: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.bool_],
    npt.NDArray[np.intp],
    npt.NDArray[np.bool_],
]:
    """Reconstruct source and target index arrays from an ops array.

    Parameters
    ----------
    ops : array_like of uint8, shape (k,)
        Edit-operation sequence returned by ``needleman_wunsch``.

    Returns
    -------
    src_idx : ndarray of intp, shape (k,)
        Source sequence index at each alignment position.
    src_mask : ndarray of bool, shape (k,)
        True where the source sequence has a gap (insert positions).
    tgt_idx : ndarray of intp, shape (k,)
        Target sequence index at each alignment position.
    tgt_mask : ndarray of bool, shape (k,)
        True where the target sequence has a gap (delete positions).
    """
    ...
