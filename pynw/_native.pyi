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
    NeedlemanWunschResult
        A named tuple with fields ``score``, ``row_idx``, and ``col_idx``.
        ``row_idx`` and ``col_idx`` map each alignment position to an
        index in the original sequence, with ``-1`` indicating a gap.

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
    >>> result = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> result.score
    2.0
    >>> "".join(row_seq[i] if i >= 0 else "-" for i in result.row_idx)
    'G-ATTACA'
    >>> "".join(col_seq[i] if i >= 0 else "-" for i in result.col_idx)
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
