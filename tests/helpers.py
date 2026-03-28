"""Shared test helpers for the pynw test suite."""

import numpy as np
import numpy.typing as npt

from pynw import needleman_wunsch


def char_score_matrix(
    s1: str,
    s2: str,
    match: float,
    mismatch: float,
) -> npt.NDArray[np.float64]:
    """
    Build a (len(s1), len(s2)) score matrix for two strings using match/mismatch
    scoring.
    """
    return np.where(
        np.array(list(s1))[:, None] == np.array(list(s2))[None, :], match, mismatch
    )


def nw_score(
    s1: str,
    s2: str,
    match: float,
    mismatch: float,
    gap: float,
) -> float:
    """Return the NW optimal alignment score for two strings."""
    sm = char_score_matrix(s1, s2, match, mismatch)
    score, _, _ = needleman_wunsch(sm, gap_penalty_row=gap, gap_penalty_col=gap)
    return score


def recompute_score_from_indices(
    row_idx: list[int] | npt.NDArray[np.intp],
    col_idx: list[int] | npt.NDArray[np.intp],
    score_matrix: npt.NDArray[np.float64],
    gap_in_row: float,
    gap_in_col: float,
) -> float:
    """Walk index arrays and sum up the alignment score."""
    total = 0.0
    for i, j in zip(row_idx, col_idx, strict=True):
        if i < 0:
            total += gap_in_row
        elif j < 0:
            total += gap_in_col
        else:
            total += float(score_matrix[i, j])
    return total


def assert_structural_invariants(
    row_idx: npt.NDArray[np.intp],
    col_idx: npt.NDArray[np.intp],
    n: int,
    m: int,
) -> None:
    """Assert all structural invariants that every valid alignment must satisfy."""
    # Same length
    assert len(row_idx) == len(col_idx), "Index arrays must have equal length"

    # No double gaps
    assert not ((row_idx == -1) & (col_idx == -1)).any(), "Double gap found"

    # Monotonicity: non-gap indices strictly increasing
    row_ng = row_idx[row_idx >= 0]
    col_ng = col_idx[col_idx >= 0]
    if len(row_ng) > 1:
        assert np.all(np.diff(row_ng) > 0), "row indices not monotonically increasing"
    if len(col_ng) > 1:
        assert np.all(np.diff(col_ng) > 0), "col indices not monotonically increasing"

    # Coverage: every element used exactly once
    np.testing.assert_array_equal(row_ng, np.arange(n))
    np.testing.assert_array_equal(col_ng, np.arange(m))

    # Length bounds: max(n,m) <= alignment_length <= n+m
    length = len(row_idx)
    assert length >= max(n, m, 0), f"Alignment too short: {length} < max({n},{m})"
    if n > 0 or m > 0:
        assert length <= n + m, f"Alignment too long: {length} > {n}+{m}"
