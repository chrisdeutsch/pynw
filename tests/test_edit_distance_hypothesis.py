"""Property-based tests: Needleman-Wunsch vs rapidfuzz edit distances.

These tests use Hypothesis to generate random string pairs and verify that
the NW algorithm reproduces each edit distance metric exactly.  They are
**deselected by default** and can be run on demand with::

    pytest -m hypothesis
    pytest -m hypothesis -x --hypothesis-seed=0   # reproducible run

See ``test_edit_distance.py`` for the deterministic counterparts and the
mathematical rationale behind each parameterisation.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rapidfuzz.distance import Hamming, Indel, LCSseq, Levenshtein

from pynw import needleman_wunsch
from tests.helpers import (
    assert_structural_invariants as _assert_structural_invariants,
)
from tests.helpers import (
    char_score_matrix as _char_score_matrix,
)
from tests.helpers import (
    nw_score as _nw_score,
)
from tests.helpers import (
    recompute_score_from_indices as _recompute_score_from_indices,
)

pytestmark = pytest.mark.hypothesis


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Short ASCII strings - fast to score, good for volume.
ascii_text = st.text(st.characters(codec="ascii", categories=("L", "N")), max_size=30)

# Full Unicode strings — broader coverage.
unicode_text = st.text(max_size=20)

# Equal-length ASCII pairs for Hamming.
equal_length_pair = st.integers(min_value=0, max_value=30).flatmap(
    lambda n: st.tuples(
        st.text(
            st.characters(codec="ascii", categories=("L", "N")), min_size=n, max_size=n
        ),
        st.text(
            st.characters(codec="ascii", categories=("L", "N")), min_size=n, max_size=n
        ),
    )
)


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


class TestLevenshteinProperty:
    """NW(match=0, mismatch=-1, gap=-1) == -Levenshtein.distance."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_score_matches(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 0, -1, -1) == -Levenshtein.distance(s1, s2)

    @given(s1=unicode_text, s2=unicode_text)
    def test_score_matches_unicode(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 0, -1, -1) == -Levenshtein.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 0, -1, -1) == _nw_score(s2, s1, 0, -1, -1)

    @given(s=ascii_text)
    def test_identity(self, s: str) -> None:
        assert _nw_score(s, s, 0, -1, -1) == 0

    @given(s=ascii_text)
    def test_distance_to_empty(self, s: str) -> None:
        assert _nw_score(s, "", 0, -1, -1) == -len(s)
        assert _nw_score("", s, 0, -1, -1) == -len(s)

    @given(s1=ascii_text, s2=ascii_text)
    def test_bounded_by_max_length(self, s1: str, s2: str) -> None:
        dist = -_nw_score(s1, s2, 0, -1, -1)
        assert 0 <= dist <= max(len(s1), len(s2), 0)

    @given(a=ascii_text, b=ascii_text, c=ascii_text)
    @settings(max_examples=50)
    def test_triangle_inequality(self, a: str, b: str, c: str) -> None:
        d_ab = -_nw_score(a, b, 0, -1, -1)
        d_bc = -_nw_score(b, c, 0, -1, -1)
        d_ac = -_nw_score(a, c, 0, -1, -1)
        assert d_ac <= d_ab + d_bc


# ---------------------------------------------------------------------------
# Indel distance
# ---------------------------------------------------------------------------


class TestIndelProperty:
    """NW(match=0, mismatch=-2, gap=-1) == -Indel.distance."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_score_matches(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 0, -2, -1) == -Indel.distance(s1, s2)

    @given(s1=unicode_text, s2=unicode_text)
    def test_score_matches_unicode(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 0, -2, -1) == -Indel.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 0, -2, -1) == _nw_score(s2, s1, 0, -2, -1)

    @given(s1=ascii_text, s2=ascii_text)
    def test_geq_levenshtein(self, s1: str, s2: str) -> None:
        indel = -_nw_score(s1, s2, 0, -2, -1)
        lev = -_nw_score(s1, s2, 0, -1, -1)
        assert indel >= lev

    @given(s1=ascii_text, s2=ascii_text)
    def test_relation_to_lcs(self, s1: str, s2: str) -> None:
        """Indel.distance == len(s1) + len(s2) - 2 * LCS_length."""
        indel_dist = Indel.distance(s1, s2)
        lcs_len = LCSseq.similarity(s1, s2)
        assert indel_dist == len(s1) + len(s2) - 2 * lcs_len


# ---------------------------------------------------------------------------
# LCS length (longest common subsequence)
# ---------------------------------------------------------------------------


class TestLCSProperty:
    """NW(match=1, mismatch=0, gap=0) == LCSseq.similarity (= LCS length)."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_score_matches(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 1, 0, 0) == LCSseq.similarity(s1, s2)

    @given(s1=unicode_text, s2=unicode_text)
    def test_score_matches_unicode(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 1, 0, 0) == LCSseq.similarity(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert _nw_score(s1, s2, 1, 0, 0) == _nw_score(s2, s1, 1, 0, 0)

    @given(s=ascii_text)
    def test_identity(self, s: str) -> None:
        assert _nw_score(s, s, 1, 0, 0) == len(s)

    @given(s1=ascii_text, s2=ascii_text)
    def test_bounded_by_min_length(self, s1: str, s2: str) -> None:
        lcs = _nw_score(s1, s2, 1, 0, 0)
        assert 0 <= lcs <= min(len(s1), len(s2), float("inf"))

    @given(s1=ascii_text, s2=ascii_text)
    def test_consistency_with_indel(self, s1: str, s2: str) -> None:
        """2 * LCS_length + indel_distance == len(s1) + len(s2)."""
        lcs = _nw_score(s1, s2, 1, 0, 0)
        indel = -_nw_score(s1, s2, 0, -2, -1)
        assert 2 * lcs + indel == len(s1) + len(s2)


# ---------------------------------------------------------------------------
# Hamming distance (equal-length strings)
# ---------------------------------------------------------------------------


class TestHammingProperty:
    """NW(match=0, mismatch=-1, gap=-(n+1)) == -Hamming.distance
    for equal-length strings."""

    @given(pair=equal_length_pair)
    def test_score_matches(self, pair: tuple[str, str]) -> None:
        s1, s2 = pair
        n = len(s1)
        gap = -(n + 1) if n > 0 else -1.0
        assert _nw_score(s1, s2, 0, -1, gap) == -Hamming.distance(s1, s2)

    @given(pair=equal_length_pair)
    def test_hamming_geq_levenshtein(self, pair: tuple[str, str]) -> None:
        s1, s2 = pair
        if not s1:
            return
        gap = -(len(s1) + 1)
        hamming = -_nw_score(s1, s2, 0, -1, gap)
        lev = -_nw_score(s1, s2, 0, -1, -1)
        assert hamming >= lev


# ---------------------------------------------------------------------------
# Cross-metric consistency
# ---------------------------------------------------------------------------


class TestCrossMetricProperty:
    """Relationships that must hold across all metrics simultaneously."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_levenshtein_leq_indel(self, s1: str, s2: str) -> None:
        lev = -_nw_score(s1, s2, 0, -1, -1)
        indel = -_nw_score(s1, s2, 0, -2, -1)
        assert lev <= indel

    @given(s1=ascii_text, s2=ascii_text)
    def test_lcs_plus_indel_equals_total_length(self, s1: str, s2: str) -> None:
        lcs = _nw_score(s1, s2, 1, 0, 0)
        indel = -_nw_score(s1, s2, 0, -2, -1)
        assert 2 * lcs + indel == len(s1) + len(s2)


# ---------------------------------------------------------------------------
# Weighted Levenshtein
# ---------------------------------------------------------------------------


class TestWeightedLevenshteinProperty:
    """NW with asymmetric gap penalties reproduces weighted Levenshtein."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_insertion_weight_2(self, s1: str, s2: str) -> None:
        weights = (2, 1, 1)  # (insertion, deletion, substitution)
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        score, _, _ = needleman_wunsch(
            sm, gap_penalty_source=-weights[0], gap_penalty_target=-weights[1]
        )
        assert score == expected

    @given(s1=ascii_text, s2=ascii_text)
    def test_deletion_weight_3(self, s1: str, s2: str) -> None:
        weights = (1, 3, 1)
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        score, _, _ = needleman_wunsch(
            sm, gap_penalty_source=-weights[0], gap_penalty_target=-weights[1]
        )
        assert score == expected

    @given(s1=ascii_text, s2=ascii_text)
    def test_substitution_weight_2(self, s1: str, s2: str) -> None:
        weights = (1, 1, 2)
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        score, _, _ = needleman_wunsch(
            sm, gap_penalty_source=-weights[0], gap_penalty_target=-weights[1]
        )
        assert score == expected

    @given(s1=ascii_text, s2=ascii_text)
    def test_uniform_scale(self, s1: str, s2: str) -> None:
        """Scaling all costs by k scales the score by k."""
        base = _nw_score(s1, s2, 0, -1, -1)
        scaled = _nw_score(s1, s2, 0, -5, -5)
        assert scaled == pytest.approx(5 * base)


# ---------------------------------------------------------------------------
# Index structural properties (property-based)
# ---------------------------------------------------------------------------


class TestIndexStructuralPropertiesProperty:
    """Hypothesis-driven structural invariants for returned alignment indices."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_levenshtein_structural(self, s1: str, s2: str) -> None:
        """
        All structural invariants + score recomputation + optimality (Levenshtein).
        """
        sm = _char_score_matrix(s1, s2, 0, -1)
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=-1, gap_penalty_target=-1
        )

        _assert_structural_invariants(src_idx, tgt_idx, len(s1), len(s2))
        assert _recompute_score_from_indices(src_idx, tgt_idx, sm, -1, -1) == score
        assert score == -Levenshtein.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_indel_structural(self, s1: str, s2: str) -> None:
        """All structural invariants + score recomputation + optimality (Indel)."""
        sm = _char_score_matrix(s1, s2, 0, -2)
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=-1, gap_penalty_target=-1
        )

        _assert_structural_invariants(src_idx, tgt_idx, len(s1), len(s2))
        assert _recompute_score_from_indices(src_idx, tgt_idx, sm, -1, -1) == score
        assert score == -Indel.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_lcs_structural(self, s1: str, s2: str) -> None:
        """All structural invariants + score recomputation + optimality (LCS)."""
        sm = _char_score_matrix(s1, s2, 1, 0)
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=0, gap_penalty_target=0
        )

        _assert_structural_invariants(src_idx, tgt_idx, len(s1), len(s2))
        assert _recompute_score_from_indices(src_idx, tgt_idx, sm, 0, 0) == score
        assert score == LCSseq.similarity(s1, s2)

    @given(pair=equal_length_pair)
    def test_hamming_structural(self, pair: tuple[str, str]) -> None:
        """Hamming: purely diagonal alignment, no gaps."""
        s1, s2 = pair
        n = len(s1)
        gap = -(n + 1) if n > 0 else -1.0
        sm = _char_score_matrix(s1, s2, 0, -1)
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=gap, gap_penalty_target=gap
        )

        # No gaps at all
        assert not (src_idx == -1).any()
        assert not (tgt_idx == -1).any()
        # Length == n
        assert len(src_idx) == n
        # Indices are exactly 0..n-1
        if n > 0:
            np.testing.assert_array_equal(src_idx, np.arange(n))
            np.testing.assert_array_equal(tgt_idx, np.arange(n))
        # Score recomputation + optimality
        assert _recompute_score_from_indices(src_idx, tgt_idx, sm, gap, gap) == score
        assert score == -Hamming.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_weighted_levenshtein_structural(self, s1: str, s2: str) -> None:
        """Structural invariants + optimality for asymmetric weighted Levenshtein."""
        weights = (2, 1, 1)  # (insertion, deletion, substitution)
        sm = _char_score_matrix(s1, s2, 0, -weights[2])
        gap1 = -float(weights[0])
        gap2 = -float(weights[1])
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=gap1, gap_penalty_target=gap2
        )

        _assert_structural_invariants(src_idx, tgt_idx, len(s1), len(s2))
        assert _recompute_score_from_indices(src_idx, tgt_idx, sm, gap1, gap2) == score
        assert score == -Levenshtein.distance(s1, s2, weights=weights)

    @given(s1=unicode_text, s2=unicode_text)
    def test_levenshtein_structural_unicode(self, s1: str, s2: str) -> None:
        """Structural invariants for Unicode strings (Levenshtein)."""
        sm = _char_score_matrix(s1, s2, 0, -1)
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=-1, gap_penalty_target=-1
        )

        _assert_structural_invariants(src_idx, tgt_idx, len(s1), len(s2))
        assert _recompute_score_from_indices(src_idx, tgt_idx, sm, -1, -1) == score
        assert score == -Levenshtein.distance(s1, s2)

    @given(
        n=st.integers(min_value=0, max_value=12),
        m=st.integers(min_value=0, max_value=12),
        data=st.data(),
    )
    def test_random_score_matrix_structural(
        self, n: int, m: int, data: st.DataObject
    ) -> None:
        """Structural invariants hold for arbitrary score matrices."""
        sm = (
            np.array(
                data.draw(
                    st.lists(
                        st.lists(
                            st.floats(min_value=-10, max_value=10, allow_nan=False),
                            min_size=m,
                            max_size=m,
                        ),
                        min_size=n,
                        max_size=n,
                    )
                )
            ).reshape(n, m)
            if n > 0 and m > 0
            else np.empty((n, m))
        )
        gap = -1.0
        score, src_idx, tgt_idx = needleman_wunsch(
            sm, gap_penalty_source=gap, gap_penalty_target=gap
        )

        _assert_structural_invariants(src_idx, tgt_idx, n, m)
        assert _recompute_score_from_indices(
            src_idx, tgt_idx, sm, gap, gap
        ) == pytest.approx(score)
