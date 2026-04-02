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
    score, _, _ = needleman_wunsch(sm, gap_penalty_source=gap, gap_penalty_target=gap)
    return score


def recompute_score_from_indices(
    source_idx: list[int] | npt.NDArray[np.intp],
    target_idx: list[int] | npt.NDArray[np.intp],
    score_matrix: npt.NDArray[np.float64],
    gap_in_source: float,
    gap_in_target: float,
) -> float:
    """Walk index arrays and sum up the alignment score."""
    total = 0.0
    for i, j in zip(source_idx, target_idx, strict=True):
        if i < 0:
            total += gap_in_source
        elif j < 0:
            total += gap_in_target
        else:
            total += float(score_matrix[i, j])
    return total


def assert_structural_invariants(
    source_idx: npt.NDArray[np.intp],
    target_idx: npt.NDArray[np.intp],
    n: int,
    m: int,
) -> None:
    """Assert all structural invariants that every valid alignment must satisfy."""
    # Same length
    assert len(source_idx) == len(target_idx), "Index arrays must have equal length"

    # No double gaps
    assert not ((source_idx == -1) & (target_idx == -1)).any(), "Double gap found"

    # Monotonicity: non-gap indices strictly increasing
    source_ng = source_idx[source_idx >= 0]
    target_ng = target_idx[target_idx >= 0]
    if len(source_ng) > 1:
        assert np.all(np.diff(source_ng) > 0), (
            "source indices not monotonically increasing"
        )
    if len(target_ng) > 1:
        assert np.all(np.diff(target_ng) > 0), (
            "target indices not monotonically increasing"
        )

    # Coverage: every element used exactly once
    np.testing.assert_array_equal(source_ng, np.arange(n))
    np.testing.assert_array_equal(target_ng, np.arange(m))

    # Length bounds: max(n,m) <= alignment_length <= n+m
    length = len(source_idx)
    assert length >= max(n, m, 0), f"Alignment too short: {length} < max({n},{m})"
    if n > 0 or m > 0:
        assert length <= n + m, f"Alignment too long: {length} > {n}+{m}"
