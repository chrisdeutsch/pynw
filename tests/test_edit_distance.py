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
from hypothesis import given
from hypothesis import strategies as st
from pynw import needleman_wunsch
from rapidfuzz.distance import Hamming, Indel, LCSseq, Levenshtein

pytestmark = pytest.mark.hypothesis


def char_score_matrix(s1: str, s2: str, match: float, mismatch: float):
    a1 = np.array(list(s1), dtype="U1").reshape(-1, 1)
    a2 = np.array(list(s2), dtype="U1").reshape(1, -1)
    return np.where(a1 == a2, match, mismatch)


def nw_score(
    s1: str,
    s2: str,
    match: float,
    mismatch: float,
    gap: float,
) -> float:
    """Return the NW optimal alignment score for two strings."""
    sm = char_score_matrix(s1, s2, match, mismatch)
    score, _ = needleman_wunsch(sm, gap_penalty=gap)
    return score


# Strategies

# Short ASCII strings - fast to score, good for volume.
ascii_text = st.text(st.characters(codec="ascii", categories=("L", "N")), max_size=10)

# Equal-length ASCII pairs for Hamming.
equal_length_pair = st.integers(min_value=0, max_value=10).flatmap(
    lambda n: st.tuples(
        st.text(
            st.characters(codec="ascii", categories=("L", "N")), min_size=n, max_size=n
        ),
        st.text(
            st.characters(codec="ascii", categories=("L", "N")), min_size=n, max_size=n
        ),
    )
)


class TestLevenshteinProperty:
    """NW(match=0, mismatch=-1, gap=-1) == -Levenshtein.distance."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_score_matches(self, s1: str, s2: str) -> None:
        assert nw_score(s1, s2, 0, -1, -1) == -Levenshtein.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert nw_score(s1, s2, 0, -1, -1) == nw_score(s2, s1, 0, -1, -1)

    @given(s=ascii_text)
    def test_identity(self, s: str) -> None:
        assert nw_score(s, s, 0, -1, -1) == 0

    @given(s=ascii_text)
    def test_distance_to_empty(self, s: str) -> None:
        assert nw_score(s, "", 0, -1, -1) == -len(s)
        assert nw_score("", s, 0, -1, -1) == -len(s)


class TestIndelProperty:
    """NW(match=0, mismatch=-2, gap=-1) == -Indel.distance."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_score_matches(self, s1: str, s2: str) -> None:
        assert nw_score(s1, s2, 0, -2, -1) == -Indel.distance(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert nw_score(s1, s2, 0, -2, -1) == nw_score(s2, s1, 0, -2, -1)


class TestLCSProperty:
    """NW(match=1, mismatch=0, gap=0) == LCSseq.similarity (= LCS length)."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_score_matches(self, s1: str, s2: str) -> None:
        assert nw_score(s1, s2, 1, 0, 0) == LCSseq.similarity(s1, s2)

    @given(s1=ascii_text, s2=ascii_text)
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert nw_score(s1, s2, 1, 0, 0) == nw_score(s2, s1, 1, 0, 0)

    @given(s=ascii_text)
    def test_identity(self, s: str) -> None:
        assert nw_score(s, s, 1, 0, 0) == len(s)


class TestHammingProperty:
    """NW(match=0, mismatch=-1, gap=-(n+1)) == -Hamming.distance
    for equal-length strings."""

    @given(pair=equal_length_pair)
    def test_score_matches(self, pair: tuple[str, str]) -> None:
        s1, s2 = pair
        n = len(s1)
        gap = -(n + 1) if n > 0 else -1.0
        assert nw_score(s1, s2, 0, -1, gap) == -Hamming.distance(s1, s2)


class TestWeightedLevenshteinProperty:
    """NW with asymmetric gap penalties reproduces weighted Levenshtein."""

    @given(s1=ascii_text, s2=ascii_text)
    def test_insertion_weight_2(self, s1: str, s2: str) -> None:
        weights = (2, 1, 1)  # (insertion, deletion, substitution)
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        sm = char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        score, _ = needleman_wunsch(
            sm, insert_penalty=-weights[0], delete_penalty=-weights[1]
        )
        assert score == expected

    @given(s1=ascii_text, s2=ascii_text)
    def test_deletion_weight_3(self, s1: str, s2: str) -> None:
        weights = (1, 3, 1)
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        sm = char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        score, _ = needleman_wunsch(
            sm, insert_penalty=-weights[0], delete_penalty=-weights[1]
        )
        assert score == expected

    @given(s1=ascii_text, s2=ascii_text)
    def test_substitution_weight_2(self, s1: str, s2: str) -> None:
        weights = (1, 1, 2)
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        sm = char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        score, _ = needleman_wunsch(
            sm, insert_penalty=-weights[0], delete_penalty=-weights[1]
        )
        assert score == expected

    @given(s1=ascii_text, s2=ascii_text)
    def test_uniform_scale(self, s1: str, s2: str) -> None:
        """Scaling all costs by k scales the score by k."""
        base = nw_score(s1, s2, 0, -1, -1)
        scaled = nw_score(s1, s2, 0, -5, -5)
        assert scaled == pytest.approx(5 * base)
