"""Tests for the low-level needleman_wunsch() public API."""

import numpy as np
import pytest

from pynw import needleman_wunsch
from tests.helpers import (
    assert_structural_invariants,
    ops_to_gap_indices,
    recompute_score_from_indices,
)

# ---------------------------------------------------------------------------
# Gap penalty behavior
# ---------------------------------------------------------------------------


class TestGapPenalties:
    def test_default_gap_penalty_is_negative_one(self) -> None:
        """Verify the default gap_penalty=-1.0 by comparing with explicit."""
        sm = np.array([[0.0, 0.0], [0.0, 0.0]])
        score_default, ops_d = needleman_wunsch(sm)
        score_explicit, ops_e = needleman_wunsch(sm, gap_penalty=-1.0)
        ri_d, ci_d = ops_to_gap_indices(ops_d)
        ri_e, ci_e = ops_to_gap_indices(ops_e)
        assert score_default == score_explicit
        np.testing.assert_array_equal(ri_d, ri_e)
        np.testing.assert_array_equal(ci_d, ci_e)

    def test_zero_gap_penalty(self) -> None:
        """Gaps are free — should still produce a valid alignment."""
        sm = np.array([[1.0, -1.0], [-1.0, 1.0]])
        score, ops = needleman_wunsch(sm, gap_penalty=0.0)
        ri, ci = ops_to_gap_indices(ops)
        assert_structural_invariants(ri, ci, 2, 2)
        assert score == 2.0  # Diagonal is preferred by tie-break

    def test_positive_gap_penalty_rewards_gaps(self) -> None:
        """Positive 'penalty' makes gaps profitable — should prefer all gaps."""
        sm = np.array([[0.0, 0.0], [0.0, 0.0]])
        score, ops = needleman_wunsch(sm, gap_penalty=1.0)
        ri, ci = ops_to_gap_indices(ops)
        # With +1 per gap, inserting gaps is beneficial.
        # All gaps: 2 row gaps + 2 col gaps = +4, better than 2 diags = 0.
        assert score == 4.0
        assert len(ri) == 4  # n + m when all gaps

    def test_very_large_negative_gap_penalty_forces_diagonal(self) -> None:
        """Extremely costly gaps force pure diagonal alignment on square matrices."""
        sm = np.zeros((3, 3))
        score, ops = needleman_wunsch(sm, gap_penalty=-1e10)
        ri, ci = ops_to_gap_indices(ops)
        # All diagonal — no gaps
        np.testing.assert_array_equal(ri, [0, 1, 2])
        np.testing.assert_array_equal(ci, [0, 1, 2])
        assert score == 0.0

    def test_asymmetric_gap_penalties(self) -> None:
        sm = np.array([[-1.0, 2.0], [2.0, -1.0]], dtype=np.float64)
        score, ops = needleman_wunsch(
            sm, gap_penalty_source=-0.5, gap_penalty_target=-3.0
        )
        ri, ci = ops_to_gap_indices(ops)
        assert score == -1.5
        np.testing.assert_array_equal(ri, [-1, 0, 1])
        np.testing.assert_array_equal(ci, [0, 1, -1])

    def test_gap_penalty_source_overrides_default(self) -> None:
        """gap_penalty_source takes precedence over gap_penalty for row gaps."""
        sm = np.empty((0, 2), dtype=np.float64)
        # gap_penalty=-100 should be ignored for row gaps
        score, _ = needleman_wunsch(sm, gap_penalty=-100.0, gap_penalty_source=-1.0)
        assert score == -2.0

    def test_gap_penalty_target_overrides_default(self) -> None:
        """gap_penalty_target takes precedence over gap_penalty for col gaps."""
        sm = np.empty((2, 0), dtype=np.float64)
        score, _ = needleman_wunsch(sm, gap_penalty=-100.0, gap_penalty_target=-1.0)
        assert score == -2.0


# ---------------------------------------------------------------------------
# Structural invariants on direct API calls
# ---------------------------------------------------------------------------


class TestStructuralInvariants:
    """Verify that structural invariants hold directly on needleman_wunsch output,
    independent of any edit-distance interpretation."""

    @pytest.mark.parametrize(
        "shape",
        [(0, 0), (1, 1), (1, 5), (5, 1), (3, 3), (3, 7), (7, 3), (10, 10)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_random_matrix_invariants(self, shape: tuple[int, int]) -> None:
        n, m = shape
        rng = np.random.default_rng(hash(shape) % 2**32)
        sm = rng.standard_normal((n, m)) if n > 0 and m > 0 else np.empty((n, m))
        score, ops = needleman_wunsch(sm, gap_penalty=-1.0)
        ri, ci = ops_to_gap_indices(ops)

        assert_structural_invariants(ri, ci, n, m)
        if n > 0 and m > 0:
            assert recompute_score_from_indices(
                ri, ci, sm, -1.0, -1.0
            ) == pytest.approx(score)

    @pytest.mark.parametrize("gap", [-10.0, -1.0, -0.1, 0.0, 0.5])
    def test_invariants_across_gap_penalties(self, gap: float) -> None:
        rng = np.random.default_rng(12345)
        sm = rng.standard_normal((5, 5))
        score, ops = needleman_wunsch(sm, gap_penalty=gap)
        ri, ci = ops_to_gap_indices(ops)

        assert_structural_invariants(ri, ci, 5, 5)
        assert recompute_score_from_indices(ri, ci, sm, gap, gap) == pytest.approx(
            score
        )

    def test_invariants_with_asymmetric_gaps(self) -> None:
        rng = np.random.default_rng(99)
        sm = rng.standard_normal((4, 6))
        score, ops = needleman_wunsch(
            sm, gap_penalty_source=-0.5, gap_penalty_target=-2.0
        )
        ri, ci = ops_to_gap_indices(ops)
        assert_structural_invariants(ri, ci, 4, 6)
        assert recompute_score_from_indices(ri, ci, sm, -0.5, -2.0) == pytest.approx(
            score
        )
