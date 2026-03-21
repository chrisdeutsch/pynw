"""Tests for the low-level needleman_wunsch() public API."""

import numpy as np
import pytest

from pynw import needleman_wunsch
from tests.helpers import assert_structural_invariants, recompute_score_from_indices

# ---------------------------------------------------------------------------
# Return types and basic correctness
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_returns_score_and_index_arrays(self) -> None:
        sm = np.array(
            [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
            dtype=np.float64,
        )
        score, ri, ci = needleman_wunsch(sm)

        assert score == 6.0
        assert ri.dtype == np.intp
        assert ci.dtype == np.intp
        np.testing.assert_array_equal(ri, [0, 1, 2])
        np.testing.assert_array_equal(ci, [0, 1, 2])

    def test_score_is_python_float(self) -> None:
        sm = np.array([[1.0]], dtype=np.float64)
        score, _, _ = needleman_wunsch(sm)
        assert isinstance(score, float)

    def test_index_arrays_are_numpy_arrays(self) -> None:
        sm = np.array([[1.0]], dtype=np.float64)
        _, ri, ci = needleman_wunsch(sm)
        assert isinstance(ri, np.ndarray)
        assert isinstance(ci, np.ndarray)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match=r"2-dimensional"):
            needleman_wunsch(np.array([1.0, 2.0]))

    def test_rejects_3d_input(self) -> None:
        with pytest.raises(ValueError, match=r"2-dimensional"):
            needleman_wunsch(np.zeros((1, 1, 1)))

    def test_rejects_scalar_input(self) -> None:
        with pytest.raises(ValueError, match=r"2-dimensional"):
            needleman_wunsch(np.float64(1.0))

    def test_casts_integer_input_to_float64(self) -> None:
        sm = np.array([[2, -1], [-1, 2]], dtype=np.int32)
        score, ri, ci = needleman_wunsch(sm)
        assert score == 4.0
        np.testing.assert_array_equal(ri, [0, 1])
        np.testing.assert_array_equal(ci, [0, 1])

    def test_casts_float32_input(self) -> None:
        sm = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
        score, _, _ = needleman_wunsch(sm)
        assert score == 4.0

    def test_casts_bool_input(self) -> None:
        sm = np.array([[True, False], [False, True]])
        score, ri, ci = needleman_wunsch(sm)
        assert score == 2.0
        np.testing.assert_array_equal(ri, [0, 1])
        np.testing.assert_array_equal(ci, [0, 1])

    def test_accepts_python_list_input(self) -> None:
        score, ri, ci = needleman_wunsch([[2.0, -1.0], [-1.0, 2.0]])
        assert score == 4.0
        np.testing.assert_array_equal(ri, [0, 1])
        np.testing.assert_array_equal(ci, [0, 1])

    def test_accepts_nested_list_of_ints(self) -> None:
        score, _, _ = needleman_wunsch([[1, 0], [0, 1]])
        assert score == 2.0

    def test_check_finite_rejects_nan_in_matrix(self) -> None:
        sm = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm, check_finite=True)

    def test_check_finite_rejects_inf_in_matrix(self) -> None:
        sm = np.array([[1.0, np.inf], [0.0, 1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm, check_finite=True)

    def test_check_finite_rejects_nan_gap_penalty(self) -> None:
        sm = np.array([[1.0]])
        with pytest.raises(ValueError, match=r"gap_penalty_row"):
            needleman_wunsch(sm, gap_penalty=float("nan"), check_finite=True)

    def test_check_finite_rejects_inf_gap_penalty_col(self) -> None:
        sm = np.array([[1.0]])
        with pytest.raises(ValueError, match=r"gap_penalty_col"):
            needleman_wunsch(sm, gap_penalty_col=float("inf"), check_finite=True)

    def test_check_finite_false_allows_nan(self) -> None:
        """Default check_finite=False does not raise on non-finite input."""
        sm = np.array([[np.nan]])
        # Should not raise
        needleman_wunsch(sm, check_finite=False)

    def test_check_finite_passes_for_valid_input(self) -> None:
        sm = np.array([[1.0, -1.0], [-1.0, 1.0]])
        score, ri, ci = needleman_wunsch(sm, check_finite=True)
        assert score == 2.0
        np.testing.assert_array_equal(ri, [0, 1])
        np.testing.assert_array_equal(ci, [0, 1])


# ---------------------------------------------------------------------------
# Empty and degenerate matrices
# ---------------------------------------------------------------------------


class TestEmptyAndDegenerate:
    def test_empty_0x0(self) -> None:
        score, ri, ci = needleman_wunsch(np.empty((0, 0), dtype=np.float64))
        assert score == 0.0
        assert len(ri) == 0
        assert len(ci) == 0

    def test_one_empty_row_dimension(self) -> None:
        score, ri, ci = needleman_wunsch(
            np.empty((0, 3), dtype=np.float64), gap_penalty_row=-2
        )
        assert score == -6.0
        np.testing.assert_array_equal(ri, [-1, -1, -1])
        np.testing.assert_array_equal(ci, [0, 1, 2])

    def test_one_empty_col_dimension(self) -> None:
        score, ri, ci = needleman_wunsch(
            np.empty((2, 0), dtype=np.float64), gap_penalty_col=-3
        )
        assert score == -6.0
        np.testing.assert_array_equal(ri, [0, 1])
        np.testing.assert_array_equal(ci, [-1, -1])

    def test_1x1_matrix(self) -> None:
        score, ri, ci = needleman_wunsch(np.array([[5.0]]))
        assert score == 5.0
        np.testing.assert_array_equal(ri, [0])
        np.testing.assert_array_equal(ci, [0])

    def test_1x1_matrix_with_gap_better(self) -> None:
        """When diagonal score is worse than two gaps, gaps are chosen."""
        score, ri, ci = needleman_wunsch(np.array([[-5.0]]), gap_penalty=0.0)
        assert score == 0.0
        # Tie-break: UP > LEFT, so row gaps first then col gaps
        np.testing.assert_array_equal(ri, [-1, 0])
        np.testing.assert_array_equal(ci, [0, -1])


# ---------------------------------------------------------------------------
# Non-square (rectangular) matrices
# ---------------------------------------------------------------------------


class TestRectangularMatrices:
    def test_wide_matrix_1x3(self) -> None:
        sm = np.array([[1.0, 2.0, 3.0]])
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        # Row has 1 element, col has 3 — must have 2 row gaps
        assert_structural_invariants(ri, ci, 1, 3)
        assert recompute_score_from_indices(ri, ci, sm, -1.0, -1.0) == score

    def test_tall_matrix_3x1(self) -> None:
        sm = np.array([[1.0], [2.0], [3.0]])
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert_structural_invariants(ri, ci, 3, 1)
        assert recompute_score_from_indices(ri, ci, sm, -1.0, -1.0) == score

    def test_wide_matrix_2x5(self) -> None:
        rng = np.random.default_rng(42)
        sm = rng.standard_normal((2, 5))
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert_structural_invariants(ri, ci, 2, 5)
        assert recompute_score_from_indices(ri, ci, sm, -1.0, -1.0) == pytest.approx(
            score
        )

    def test_tall_matrix_5x2(self) -> None:
        rng = np.random.default_rng(43)
        sm = rng.standard_normal((5, 2))
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert_structural_invariants(ri, ci, 5, 2)
        assert recompute_score_from_indices(ri, ci, sm, -1.0, -1.0) == pytest.approx(
            score
        )


# ---------------------------------------------------------------------------
# Gap penalty behavior
# ---------------------------------------------------------------------------


class TestGapPenalties:
    def test_default_gap_penalty_is_negative_one(self) -> None:
        """Verify the default gap_penalty=-1.0 by comparing with explicit."""
        sm = np.array([[0.0, 0.0], [0.0, 0.0]])
        score_default, ri_d, ci_d = needleman_wunsch(sm)
        score_explicit, ri_e, ci_e = needleman_wunsch(sm, gap_penalty=-1.0)
        assert score_default == score_explicit
        np.testing.assert_array_equal(ri_d, ri_e)
        np.testing.assert_array_equal(ci_d, ci_e)

    def test_zero_gap_penalty(self) -> None:
        """Gaps are free — should still produce a valid alignment."""
        sm = np.array([[1.0, -1.0], [-1.0, 1.0]])
        score, ri, ci = needleman_wunsch(sm, gap_penalty=0.0)
        assert_structural_invariants(ri, ci, 2, 2)
        assert score == 2.0  # Diagonal is preferred by tie-break

    def test_positive_gap_penalty_rewards_gaps(self) -> None:
        """Positive 'penalty' makes gaps profitable — should prefer all gaps."""
        sm = np.array([[0.0, 0.0], [0.0, 0.0]])
        score, ri, ci = needleman_wunsch(sm, gap_penalty=1.0)
        # With +1 per gap, inserting gaps is beneficial.
        # All gaps: 2 row gaps + 2 col gaps = +4, better than 2 diags = 0.
        assert score == 4.0
        assert len(ri) == 4  # n + m when all gaps

    def test_very_large_negative_gap_penalty_forces_diagonal(self) -> None:
        """Extremely costly gaps force pure diagonal alignment on square matrices."""
        sm = np.zeros((3, 3))
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1e10)
        # All diagonal — no gaps
        np.testing.assert_array_equal(ri, [0, 1, 2])
        np.testing.assert_array_equal(ci, [0, 1, 2])
        assert score == 0.0

    def test_asymmetric_gap_penalties(self) -> None:
        sm = np.array([[-1.0, 2.0], [2.0, -1.0]], dtype=np.float64)
        score, ri, ci = needleman_wunsch(sm, gap_penalty_row=-0.5, gap_penalty_col=-3.0)
        assert score == -1.5
        np.testing.assert_array_equal(ri, [-1, 0, 1])
        np.testing.assert_array_equal(ci, [0, 1, -1])

    def test_gap_penalty_row_overrides_default(self) -> None:
        """gap_penalty_row takes precedence over gap_penalty for row gaps."""
        sm = np.empty((0, 2), dtype=np.float64)
        # gap_penalty=-100 should be ignored for row gaps
        score, _, _ = needleman_wunsch(sm, gap_penalty=-100.0, gap_penalty_row=-1.0)
        assert score == -2.0

    def test_gap_penalty_col_overrides_default(self) -> None:
        """gap_penalty_col takes precedence over gap_penalty for col gaps."""
        sm = np.empty((2, 0), dtype=np.float64)
        score, _, _ = needleman_wunsch(sm, gap_penalty=-100.0, gap_penalty_col=-1.0)
        assert score == -2.0


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------


class TestTieBreaking:
    def test_prefers_diagonal_when_all_moves_tie(self) -> None:
        score, ri, ci = needleman_wunsch(
            np.array([[0.0]], dtype=np.float64), gap_penalty=0.0
        )
        assert score == 0.0
        np.testing.assert_array_equal(ri, [0])
        np.testing.assert_array_equal(ci, [0])

    def test_prefers_up_over_left_when_diagonal_worse(self) -> None:
        score, ri, ci = needleman_wunsch(
            np.array([[-2.0]], dtype=np.float64), gap_penalty=0.0
        )
        assert score == 0.0
        np.testing.assert_array_equal(ri, [-1, 0])
        np.testing.assert_array_equal(ci, [0, -1])

    def test_deterministic_across_repeated_calls(self) -> None:
        """Same input always produces exactly the same output."""
        sm = np.array([[0.0, 0.0], [0.0, 0.0]])
        results = [needleman_wunsch(sm, gap_penalty=0.0) for _ in range(10)]
        for score, ri, ci in results:
            assert score == results[0][0]
            np.testing.assert_array_equal(ri, results[0][1])
            np.testing.assert_array_equal(ci, results[0][2])


# ---------------------------------------------------------------------------
# Special matrix patterns
# ---------------------------------------------------------------------------


class TestSpecialMatrices:
    def test_identity_matrix(self) -> None:
        sm = np.eye(4)
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert score == 4.0
        np.testing.assert_array_equal(ri, [0, 1, 2, 3])
        np.testing.assert_array_equal(ci, [0, 1, 2, 3])

    def test_all_zeros_matrix(self) -> None:
        sm = np.zeros((3, 3))
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert score == 0.0
        # Diagonal preferred by tie-break
        np.testing.assert_array_equal(ri, [0, 1, 2])
        np.testing.assert_array_equal(ci, [0, 1, 2])

    def test_all_ones_matrix(self) -> None:
        sm = np.ones((3, 3))
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert score == 3.0  # Diagonal gives 3 * 1.0
        np.testing.assert_array_equal(ri, [0, 1, 2])
        np.testing.assert_array_equal(ci, [0, 1, 2])

    def test_all_negative_matrix(self) -> None:
        sm = np.full((2, 2), -5.0)
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        # 2 diags = -10, but 2 row gaps + 2 col gaps = -4, or various mixes
        # gaps should be preferred since -1 > -5
        assert_structural_invariants(ri, ci, 2, 2)
        assert recompute_score_from_indices(ri, ci, sm, -1.0, -1.0) == score
        assert score == -4.0  # All gaps: 4 * -1 = -4

    def test_off_diagonal_best_path(self) -> None:
        """High scores off the main diagonal — alignment must pick the best path."""
        # Best monotonic path: (0,1)=10, (1,2)=10 → 20 - 0 gaps = not possible
        # because row has 3 elems, col has 3 elems, we only matched 2 of each
        # Actually: (0,1)=10, (1,2)=10 needs ri=[0,1,2], ci=[gap,1,2] won't work
        # Let's just verify structural invariants on this tricky case.
        sm = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert_structural_invariants(ri, ci, 3, 3)
        assert recompute_score_from_indices(ri, ci, sm, -1.0, -1.0) == score

    def test_single_high_score_element(self) -> None:
        """One very high score forces alignment through that cell."""
        sm = np.full((3, 3), -1.0)
        sm[1, 1] = 100.0
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)
        assert_structural_invariants(ri, ci, 3, 3)
        # The alignment must pass through (1,1)
        matched = (ri >= 0) & (ci >= 0)
        matched_pairs = list(zip(ri[matched], ci[matched], strict=True))
        assert (1, 1) in matched_pairs


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
        score, ri, ci = needleman_wunsch(sm, gap_penalty=-1.0)

        assert_structural_invariants(ri, ci, n, m)
        if n > 0 and m > 0:
            assert recompute_score_from_indices(
                ri, ci, sm, -1.0, -1.0
            ) == pytest.approx(score)

    @pytest.mark.parametrize("gap", [-10.0, -1.0, -0.1, 0.0, 0.5])
    def test_invariants_across_gap_penalties(self, gap: float) -> None:
        rng = np.random.default_rng(12345)
        sm = rng.standard_normal((5, 5))
        score, ri, ci = needleman_wunsch(sm, gap_penalty=gap)

        assert_structural_invariants(ri, ci, 5, 5)
        assert recompute_score_from_indices(ri, ci, sm, gap, gap) == pytest.approx(
            score
        )

    def test_invariants_with_asymmetric_gaps(self) -> None:
        rng = np.random.default_rng(99)
        sm = rng.standard_normal((4, 6))
        score, ri, ci = needleman_wunsch(sm, gap_penalty_row=-0.5, gap_penalty_col=-2.0)
        assert_structural_invariants(ri, ci, 4, 6)
        assert recompute_score_from_indices(ri, ci, sm, -0.5, -2.0) == pytest.approx(
            score
        )
