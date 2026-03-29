# NOTE: This stub provides the docstring for type checkers and IDE support.
# A copy exists as a Rust doc comment in src/lib.rs for runtime `help()`.
# Keep both copies in sync.

import numpy as np
import numpy.typing as npt

def needleman_wunsch(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    gap_penalty_row: float | None = None,
    gap_penalty_col: float | None = None,
    check_finite: bool = False,
) -> tuple[float, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Align two ordered sequences given a precomputed similarity matrix.

    The total alignment score is the sum of similarity-matrix entries for
    matched positions and gap penalties for insertions/deletions.

    Parameters
    ----------
    similarity_matrix : array_like, shape (n, m)
        ``similarity_matrix[i, j]`` is the similarity score for aligning
        element *i* of the row sequence with element *j* of the column
        sequence.
    gap_penalty : float, default -1.0
        Penalty applied when a gap is inserted in either sequence.
        Use ``gap_penalty_row`` or ``gap_penalty_col`` to specify
        different penalties for each sequence.
    gap_penalty_row : float, optional
        Penalty added when a gap is inserted in the row sequence
        (the column sequence advances). This can be thought of as the
        cost of a deletion from the row sequence or an insertion into the
        column sequence.
        Defaults to ``gap_penalty`` if not specified.
    gap_penalty_col : float, optional
        Penalty added when a gap is inserted in the column sequence
        (the row sequence advances). This can be thought of as the cost
        of a deletion from the column sequence or an insertion into the
        row sequence.
        Defaults to ``gap_penalty`` if not specified.
    check_finite : bool, default False
        If ``True``, raise a ``ValueError`` when ``similarity_matrix``
        or the gap penalties contain ``NaN`` or ``Inf``.

    Raises
    ------
    ValueError
        If ``similarity_matrix`` is not 2-dimensional, or if
        ``check_finite`` is ``True`` and any value in
        ``similarity_matrix`` or the gap penalties is ``NaN`` or ``Inf``.

    Returns
    -------
    score : float
        The optimal alignment score.
    row_idx : ndarray of intp
        Index into the row sequence at each alignment position, or ``-1``
        for a gap.
    col_idx : ndarray of intp
        Index into the column sequence at each alignment position, or ``-1``
        for a gap.

    Examples
    --------
    Align two DNA sequences using a simple match/mismatch scoring scheme:

    >>> import numpy as np
    >>> row_seq = list("GATTACA")
    >>> col_seq = list("GCATGCA")
    >>> match, mismatch = 1.0, -1.0
    >>> sm = np.where(
    ...     np.array(row_seq)[:, None] == np.array(col_seq)[None, :],
    ...     match, mismatch,
    ... )
    >>> score, row_idx, col_idx = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> score
    2.0
    >>> "".join(row_seq[i] if i >= 0 else "-" for i in row_idx)
    'G-ATTACA'
    >>> "".join(col_seq[i] if i >= 0 else "-" for i in col_idx)
    'GCA-TGCA'

    Notes
    -----
    When multiple alignments achieve the same optimal score, ties are
    broken deterministically: ``Diagonal > Up > Left``.  This prefers
    substitutions over gaps, producing compact alignments.  Other tools
    may return different co-optimal alignments.

    All values in ``similarity_matrix`` and the gap penalties must be finite.
    Passing ``NaN`` or ``Inf`` is undefined behavior — the output will be
    silently meaningless.
    """
    ...

def needleman_wunsch_2(
    similarity_matrix: npt.ArrayLike,
    similarity_matrix_split: npt.ArrayLike,
    similarity_matrix_merge: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    gap_penalty_row: float | None = None,
    gap_penalty_col: float | None = None,
    check_finite: bool = False,
) -> tuple[float, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
