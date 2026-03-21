"""Global sequence alignment via Needleman-Wunsch dynamic programming.

``needleman_wunsch(similarity_matrix, ...)`` accepts a precomputed ``(n, m)``
similarity matrix and returns index arrays (similar to
``scipy.optimize.linear_sum_assignment``). The DP fill and traceback run
entirely in Rust.
"""

import math
from typing import NamedTuple

import numpy as np

from pynw._rust import nw_traceback_indices


class NeedlemanWunschResult(NamedTuple):
    """Result of the Needleman-Wunsch alignment."""

    score: float
    """The optimal alignment score."""
    row_idx: np.ndarray
    """Index into the row sequence at each alignment position, or ``-1``
    for a gap."""
    col_idx: np.ndarray
    """Index into the column sequence at each alignment position, or ``-1``
    for a gap."""


# Tie-breaking priority: DIAG > UP > LEFT.  This is enforced during the DP
# fill by choosing DIAG as the default and only overwriting on strict >.
# The effect is compact alignments that prefer substitutions over gaps when
# costs are equal.  The optimal *score* is unique regardless of tie-breaking,
# but the alignment path is not - different tools may return different
# co-optimal alignments.


def needleman_wunsch(
    similarity_matrix: np.ndarray,
    *,
    gap_penalty: float = -1.0,
    gap_penalty_row: float | None = None,
    gap_penalty_col: float | None = None,
    check_finite: bool = False,
) -> NeedlemanWunschResult:
    """Solve the global sequence alignment problem.

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

    Examples
    --------
    Align two DNA sequences (1.0 for match, -1.0 for mismatch):

    >>> import numpy as np
    >>> row_seq = np.array(list("GATTACA"))
    >>> col_seq = np.array(list("GCATGCA"))
    >>> sm = np.where(row_seq[:, None] == col_seq[None, :], 1.0, -1.0)
    >>> score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> score
    2.0
    >>> "".join(np.where(ri >= 0, row_seq[ri], "-"))
    'G-ATTACA'
    >>> "".join(np.where(ci >= 0, col_seq[ci], "-"))
    'GCA-TGCA'

    The match-score component is composable, like
    ``scipy.optimize.linear_sum_assignment``::

        matched = (ri >= 0) & (ci >= 0)
        sm[ri[matched], ci[matched]].sum()   # match contribution

    Notes
    -----
    Tie-breaking is deterministic: ``DIAG > UP > LEFT``.  The optimal
    *score* is unique, but multiple alignments may achieve it.  This
    rule prefers substitutions over gaps, producing compact alignments.
    Other tools may return different co-optimal alignments.

    All values in ``similarity_matrix`` and the gap penalties must be finite.
    Passing ``NaN`` or ``Inf`` is undefined behavior — the output will be
    silently meaningless.

    """
    sm = np.asarray(similarity_matrix, dtype=np.float64)
    if sm.ndim != 2:
        msg = (
            f"similarity_matrix must be 2-dimensional, got {sm.ndim}-dimensional array"
        )
        raise ValueError(msg)

    gap_row = gap_penalty if gap_penalty_row is None else gap_penalty_row
    gap_col = gap_penalty if gap_penalty_col is None else gap_penalty_col

    if check_finite:
        if not np.isfinite(sm).all():
            msg = "similarity_matrix contains non-finite values (NaN or Inf)"
            raise ValueError(msg)
        if not math.isfinite(gap_row):
            msg = f"gap_penalty_row is not finite: {gap_row}"
            raise ValueError(msg)
        if not math.isfinite(gap_col):
            msg = f"gap_penalty_col is not finite: {gap_col}"
            raise ValueError(msg)

    return NeedlemanWunschResult(*nw_traceback_indices(sm, gap_row, gap_col))
