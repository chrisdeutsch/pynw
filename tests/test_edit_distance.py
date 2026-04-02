"""Compare Needleman-Wunsch alignment scores against rapidfuzz edit distances.

The Needleman-Wunsch algorithm generalises several classic string edit distance
metrics.  With the right scoring parameters it reproduces each metric exactly:

Levenshtein distance
    ``similarity(a, b) = 0 if a == b else -1``  with  ``gap = -1``
    NW score = -levenshtein_distance

Indel distance  (insertions + deletions only, no substitutions)
    ``similarity(a, b) = 0 if a == b else -2``  with  ``gap = -1``
    A mismatch (-2) costs exactly as much as delete + insert (-1 + -1), so the
    algorithm never prefers a substitution over two indels.
    NW score = -indel_distance

LCS length  (longest common subsequence)
    ``similarity(a, b) = 1 if a == b else 0``  with  ``gap = 0``
    Only matches contribute positively; gaps and mismatches are free.
    NW score = LCS_length

Hamming distance  (equal-length strings, no gaps allowed)
    ``similarity(a, b) = 0 if a == b else -1``  with  ``gap = -(n + 1)``
    The gap penalty is large enough that introducing any gap is always worse
    than the worst-case all-mismatch diagonal alignment.
    NW score = -hamming_distance
"""

import numpy as np
import pytest
from rapidfuzz.distance import Hamming, Indel, LCSseq, Levenshtein

from pynw import needleman_wunsch
from tests.helpers import (
    char_score_matrix as _char_score_matrix,
)
from tests.helpers import (
    nw_score as _nw_score,
)
from tests.helpers import (
    recompute_score_from_indices as _recompute_score_from_indices,
)

# ---------------------------------------------------------------------------
# String-pair fixtures
# ---------------------------------------------------------------------------

# Hand-picked pairs covering edge cases and typical patterns.
BASIC_PAIRS: list[tuple[str, str]] = [
    ("", ""),
    ("a", ""),
    ("", "b"),
    ("a", "a"),
    ("a", "b"),
    ("abc", "abc"),
    ("abc", "axc"),
    ("kitten", "sitting"),
    ("saturday", "sunday"),
    ("flaw", "lawn"),
    ("intention", "execution"),
    ("abcdef", "azced"),
    ("abcde", "ace"),
    ("aaa", "a"),
    ("a", "aaa"),
    ("abc", "def"),
    ("abcdefgh", "aecfg"),
]

# Pairs with Unicode / multibyte characters.
UNICODE_PAIRS: list[tuple[str, str]] = [
    ("cafe\u0301", "caf\u00e9"),  # combining accent vs precomposed
    ("\u00fc\u00f6\u00e4", "\u00fc\u00e4"),  # umlaut chars
    ("\u4f60\u597d\u4e16\u754c", "\u4f60\u4e16\u754c"),  # Chinese characters
    ("\U0001f600\U0001f601", "\U0001f601\U0001f600"),  # emoji
]

# Pairs where both strings have equal length (for Hamming).
EQUAL_LENGTH_PAIRS: list[tuple[str, str]] = [
    ("abc", "abc"),
    ("abc", "axc"),
    ("abc", "xyz"),
    ("kitten", "mitten"),
    ("abcdef", "abcfed"),
    ("hello", "hullo"),
    ("", ""),
    ("a", "b"),
    ("a", "a"),
    ("\U0001f600\U0001f601", "\U0001f601\U0001f600"),
]

ALL_PAIRS = BASIC_PAIRS + UNICODE_PAIRS


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


class TestLevenshtein:
    """NW(match=0, mismatch=-1, gap=-1) == -Levenshtein.distance."""

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_score_equals_negative_distance(self, s1: str, s2: str) -> None:
        expected = -Levenshtein.distance(s1, s2)
        assert _nw_score(s1, s2, match=0, mismatch=-1, gap=-1) == expected

    def test_symmetry(self) -> None:
        """Levenshtein is symmetric; NW scores should agree both ways."""
        for s1, s2 in ALL_PAIRS:
            fwd = _nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
            rev = _nw_score(s2, s1, match=0, mismatch=-1, gap=-1)
            assert fwd == rev, f"Asymmetric for ({s1!r}, {s2!r})"

    def test_identity_gives_zero(self) -> None:
        """Distance from a string to itself is 0."""
        for s in ["", "a", "abc", "hello world"]:
            assert _nw_score(s, s, match=0, mismatch=-1, gap=-1) == 0

    def test_distance_to_empty_equals_length(self) -> None:
        """Distance from s to '' equals len(s)."""
        for s in ["a", "abc", "hello"]:
            score = _nw_score(s, "", match=0, mismatch=-1, gap=-1)
            assert score == -len(s)

    def test_triangle_inequality(self) -> None:
        """Levenshtein satisfies the triangle inequality."""
        triples = [
            ("kitten", "sitting", "kitchen"),
            ("abc", "axc", "axy"),
            ("flaw", "lawn", "law"),
        ]
        for a, b, c in triples:
            d_ab = -_nw_score(a, b, match=0, mismatch=-1, gap=-1)
            d_bc = -_nw_score(b, c, match=0, mismatch=-1, gap=-1)
            d_ac = -_nw_score(a, c, match=0, mismatch=-1, gap=-1)
            assert d_ac <= d_ab + d_bc, (
                f"Triangle inequality violated: d({a!r},{c!r})={d_ac} > "
                f"d({a!r},{b!r})={d_ab} + d({b!r},{c!r})={d_bc}"
            )

    def test_bounded_by_max_length(self) -> None:
        """Levenshtein distance <= max(len(s1), len(s2))."""
        for s1, s2 in ALL_PAIRS:
            dist = -_nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
            assert dist <= max(len(s1), len(s2))

    def test_random_strings(self) -> None:
        """Fuzz test with randomly generated ASCII strings."""
        rng = np.random.default_rng(12345)
        alphabet = list("abcdefghij")
        for _ in range(100):
            n1 = int(rng.integers(0, 20))
            n2 = int(rng.integers(0, 20))
            s1 = "".join(rng.choice(alphabet, size=n1).tolist())
            s2 = "".join(rng.choice(alphabet, size=n2).tolist())
            expected = -Levenshtein.distance(s1, s2)
            actual = _nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
            assert actual == expected, f"Failed for ({s1!r}, {s2!r})"


# ---------------------------------------------------------------------------
# Indel distance
# ---------------------------------------------------------------------------


class TestIndel:
    """NW(match=0, mismatch=-2, gap=-1) == -Indel.distance.

    The mismatch penalty of -2 equals two gap penalties (-1 + -1), so
    the algorithm never prefers a substitution over insert + delete.
    """

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_score_equals_negative_distance(self, s1: str, s2: str) -> None:
        expected = -Indel.distance(s1, s2)
        assert _nw_score(s1, s2, match=0, mismatch=-2, gap=-1) == expected

    def test_indel_distance_relation_to_lcs(self) -> None:
        """Indel.distance == len(s1) + len(s2) - 2 * LCS_length."""
        for s1, s2 in ALL_PAIRS:
            indel_dist = Indel.distance(s1, s2)
            lcs_len = LCSseq.similarity(s1, s2)
            assert indel_dist == len(s1) + len(s2) - 2 * lcs_len

    def test_symmetry(self) -> None:
        for s1, s2 in ALL_PAIRS:
            fwd = _nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
            rev = _nw_score(s2, s1, match=0, mismatch=-2, gap=-1)
            assert fwd == rev

    def test_indel_geq_levenshtein(self) -> None:
        """Indel distance >= Levenshtein distance (more restricted operations)."""
        for s1, s2 in ALL_PAIRS:
            indel = -_nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
            lev = -_nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
            assert indel >= lev, f"Failed for ({s1!r}, {s2!r})"

    def test_random_strings(self) -> None:
        rng = np.random.default_rng(67890)
        alphabet = list("abcdefghij")
        for _ in range(100):
            n1 = int(rng.integers(0, 20))
            n2 = int(rng.integers(0, 20))
            s1 = "".join(rng.choice(alphabet, size=n1).tolist())
            s2 = "".join(rng.choice(alphabet, size=n2).tolist())
            expected = -Indel.distance(s1, s2)
            actual = _nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
            assert actual == expected, f"Failed for ({s1!r}, {s2!r})"


# ---------------------------------------------------------------------------
# LCS length (longest common subsequence)
# ---------------------------------------------------------------------------


class TestLCS:
    """NW(match=1, mismatch=0, gap=0) == LCSseq.similarity.

    Only character matches contribute +1 to the score; mismatches and
    gaps contribute 0.  The optimal score equals the LCS length.
    """

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_score_equals_lcs_length(self, s1: str, s2: str) -> None:
        expected = LCSseq.similarity(s1, s2)
        actual = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
        assert actual == expected, (
            f"LCS mismatch for ({s1!r}, {s2!r}): NW={actual}, expected={expected}"
        )

    def test_lcs_bounded_by_min_length(self) -> None:
        """LCS length <= min(len(s1), len(s2))."""
        for s1, s2 in ALL_PAIRS:
            lcs_len = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
            assert lcs_len <= min(len(s1), len(s2))

    def test_identical_strings_lcs_is_full_length(self) -> None:
        for s in ["", "a", "abc", "hello world"]:
            assert _nw_score(s, s, match=1, mismatch=0, gap=0) == len(s)

    def test_no_common_characters(self) -> None:
        assert _nw_score("abc", "xyz", match=1, mismatch=0, gap=0) == 0

    def test_symmetry(self) -> None:
        for s1, s2 in ALL_PAIRS:
            fwd = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
            rev = _nw_score(s2, s1, match=1, mismatch=0, gap=0)
            assert fwd == rev

    def test_lcs_consistency_with_indel(self) -> None:
        """LCS_length == (len(s1) + len(s2) - indel_distance) / 2."""
        for s1, s2 in ALL_PAIRS:
            lcs_from_nw = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
            indel_from_nw = -_nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
            derived = (len(s1) + len(s2) - indel_from_nw) / 2
            assert lcs_from_nw == derived, f"Failed for ({s1!r}, {s2!r})"

    def test_subsequence_gives_full_match(self) -> None:
        """When s2 is a subsequence of s1, LCS == len(s2)."""
        cases = [
            ("abcdef", "ace"),
            ("abcdef", "abcdef"),
            ("abcdef", ""),
            ("hello world", "hlwrd"),
        ]
        for s1, s2 in cases:
            score = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
            assert score == len(s2), f"Failed for ({s1!r}, {s2!r})"

    def test_random_strings(self) -> None:
        rng = np.random.default_rng(24680)
        alphabet = list("abcdefghij")
        for _ in range(100):
            n1 = int(rng.integers(0, 20))
            n2 = int(rng.integers(0, 20))
            s1 = "".join(rng.choice(alphabet, size=n1).tolist())
            s2 = "".join(rng.choice(alphabet, size=n2).tolist())
            expected = LCSseq.similarity(s1, s2)
            actual = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
            assert actual == expected, f"Failed for ({s1!r}, {s2!r})"


# ---------------------------------------------------------------------------
# Hamming distance (equal-length strings only)
# ---------------------------------------------------------------------------


class TestHamming:
    """NW(match=0, mismatch=-1, gap=-(n+1)) == -Hamming.distance.

    The gap penalty is so large that the algorithm never introduces a
    gap, forcing a pure diagonal alignment for equal-length strings.
    """

    @pytest.mark.parametrize("s1, s2", EQUAL_LENGTH_PAIRS)
    def test_score_equals_negative_distance(self, s1: str, s2: str) -> None:
        n = len(s1)
        gap_penalty = -(n + 1) if n > 0 else -1.0
        expected = -Hamming.distance(s1, s2)
        actual = _nw_score(s1, s2, match=0, mismatch=-1, gap=gap_penalty)
        assert actual == expected, (
            f"Hamming mismatch for ({s1!r}, {s2!r}): NW={actual}, expected={expected}"
        )

    def test_identical_strings_give_zero(self) -> None:
        for s in ["", "a", "abc", "hello"]:
            gap = -(len(s) + 1) if s else -1.0
            assert _nw_score(s, s, match=0, mismatch=-1, gap=gap) == 0

    def test_completely_different_gives_negative_length(self) -> None:
        """When no characters match, score = -len."""
        pairs = [("abc", "xyz"), ("aa", "bb")]
        for s1, s2 in pairs:
            gap = -(len(s1) + 1)
            assert _nw_score(s1, s2, match=0, mismatch=-1, gap=gap) == -len(s1)

    def test_hamming_leq_levenshtein(self) -> None:
        """For equal-length strings, Hamming >= Levenshtein (more restrictive)."""
        for s1, s2 in EQUAL_LENGTH_PAIRS:
            if not s1:
                continue
            gap = -(len(s1) + 1)
            hamming = -_nw_score(s1, s2, match=0, mismatch=-1, gap=gap)
            lev = -_nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
            assert hamming >= lev

    def test_random_equal_length_strings(self) -> None:
        rng = np.random.default_rng(13579)
        alphabet = list("abcdefghij")
        for _ in range(100):
            n = int(rng.integers(0, 20))
            s1 = "".join(rng.choice(alphabet, size=n).tolist())
            s2 = "".join(rng.choice(alphabet, size=n).tolist())
            gap = -(n + 1) if n > 0 else -1.0
            expected = -Hamming.distance(s1, s2)
            actual = _nw_score(s1, s2, match=0, mismatch=-1, gap=gap)
            assert actual == expected, f"Failed for ({s1!r}, {s2!r})"


# ---------------------------------------------------------------------------
# Cross-metric consistency
# ---------------------------------------------------------------------------


class TestCrossMetricConsistency:
    """Verify relationships between the different metrics hold when
    all are computed through NW with different parameterisations."""

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_levenshtein_leq_indel(self, s1: str, s2: str) -> None:
        """Levenshtein allows substitutions, so distance <= indel distance."""
        lev = -_nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
        indel = -_nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
        assert lev <= indel

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lcs_plus_indel_equals_total_length(self, s1: str, s2: str) -> None:
        """2 * LCS_length + indel_distance == len(s1) + len(s2)."""
        lcs = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
        indel = -_nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
        assert 2 * lcs + indel == len(s1) + len(s2)

    @pytest.mark.parametrize("s1, s2", EQUAL_LENGTH_PAIRS)
    def test_hamming_geq_levenshtein(self, s1: str, s2: str) -> None:
        """Hamming (no gaps) >= Levenshtein (gaps allowed)."""
        if not s1:
            return
        gap = -(len(s1) + 1)
        hamming = -_nw_score(s1, s2, match=0, mismatch=-1, gap=gap)
        lev = -_nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
        assert hamming >= lev


# ---------------------------------------------------------------------------
# Weighted / scaled scoring
# ---------------------------------------------------------------------------


class TestWeightedScoring:
    """Verify that uniformly scaling scores preserves the metric relationship."""

    @pytest.mark.parametrize("s1, s2", BASIC_PAIRS)
    def test_scaled_levenshtein(self, s1: str, s2: str) -> None:
        """Doubling all scores doubles the NW score."""
        base = _nw_score(s1, s2, match=0, mismatch=-1, gap=-1)
        scaled = _nw_score(s1, s2, match=0, mismatch=-2, gap=-2)
        assert scaled == pytest.approx(2 * base), f"Failed for ({s1!r}, {s2!r})"

    @pytest.mark.parametrize("s1, s2", BASIC_PAIRS)
    def test_scaled_lcs(self, s1: str, s2: str) -> None:
        """Scaling the match reward scales the LCS score proportionally."""
        base = _nw_score(s1, s2, match=1, mismatch=0, gap=0)
        scaled = _nw_score(s1, s2, match=3, mismatch=0, gap=0)
        assert scaled == pytest.approx(3 * base), f"Failed for ({s1!r}, {s2!r})"


# ---------------------------------------------------------------------------
# Levenshtein with custom weights
# ---------------------------------------------------------------------------


class TestLevenshteinWeighted:
    """Rapidfuzz supports weighted Levenshtein with custom insert/delete/replace
    costs.  Verify NW can reproduce these by adjusting gap and mismatch penalties."""

    def test_insertion_weight_2(self) -> None:
        """Insertion cost = 2, deletion cost = 1, replace cost = 1.

        In NW terms:
        - gap_in_source = -2  (gap in the source sequence = insertion into s1 = deletion
                              from s2 ... but from the edit-distance perspective,
                              inserting a character into s1 to match s2 means the
                              target sequence advances)
        - gap_in_target = -1  (deletion from s1)
        - mismatch = -1  (replacement)

        Actually, the rapidfuzz weight convention:
        - weights=(insertion, deletion, substitution)
        - insertion = cost to insert a char into s1 (i.e., s2 has a char s1 doesn't)
        - deletion = cost to delete a char from s1 (i.e., s1 has a char s2 doesn't)

        In NW terms:
        - gap_in_source = cost when the target sequence advances without the source
          sequence = insertion cost (negated)
        - gap_in_target = cost when the source sequence advances without the target
          sequence = deletion cost (negated)
        """
        pairs = [
            ("kitten", "sitting"),
            ("abc", "axc"),
            ("abcdef", "azced"),
            ("flaw", "lawn"),
        ]
        weights = (2, 1, 1)  # (insertion, deletion, substitution)
        for s1, s2 in pairs:
            expected = -Levenshtein.distance(s1, s2, weights=weights)
            sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
            score, _, _ = needleman_wunsch(
                sm,
                gap_penalty_source=-weights[0],  # insertion into s1
                gap_penalty_target=-weights[1],  # deletion from s1
            )
            assert score == expected, (
                f"Weighted Levenshtein mismatch for ({s1!r}, {s2!r}): "
                f"NW={score}, expected={expected}"
            )

    def test_substitution_weight_2(self) -> None:
        """Substitution cost = 2, insertion = deletion = 1.

        When substitution costs twice as much as an insertion + deletion pair,
        this is equivalent to the indel-only distance.
        """
        pairs = [("abc", "axc"), ("kitten", "sitting"), ("flaw", "lawn")]
        weights = (1, 1, 2)
        for s1, s2 in pairs:
            expected = -Levenshtein.distance(s1, s2, weights=weights)
            sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
            score, _, _ = needleman_wunsch(
                sm, gap_penalty_source=-weights[0], gap_penalty_target=-weights[1]
            )
            assert score == expected

    def test_all_weights_3(self) -> None:
        """All operations cost 3.  Score = -3 * levenshtein_distance."""
        pairs = [("abc", "axc"), ("kitten", "sitting"), ("abcde", "ace")]
        weights = (3, 3, 3)
        for s1, s2 in pairs:
            expected = -Levenshtein.distance(s1, s2, weights=weights)
            sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
            score, _, _ = needleman_wunsch(
                sm, gap_penalty_source=-weights[0], gap_penalty_target=-weights[1]
            )
            assert score == expected


# ---------------------------------------------------------------------------
# Indel tie-handling validation
# ---------------------------------------------------------------------------


class TestIndelTieHandling:
    """Validate Indel scoring in the presence of co-optimal paths."""

    @pytest.mark.parametrize("s1, s2", BASIC_PAIRS)
    def test_indel_score_matches_distance_under_ties(self, s1: str, s2: str) -> None:
        """When mismatch and two gaps tie, score still matches Indel distance.

        For mismatch=-2 and gap=-1, multiple paths can be co-optimal.
        We assert score optimality instead of a specific path shape.
        """
        expected_score = -Indel.distance(s1, s2)
        actual_score = _nw_score(s1, s2, match=0, mismatch=-2, gap=-1)
        assert actual_score == expected_score


# ---------------------------------------------------------------------------
# Index verification via rapidfuzz editops / opcodes
# ---------------------------------------------------------------------------


def _opcodes_to_indices(
    opcodes: object,
) -> tuple[list[int], list[int]]:
    """Convert rapidfuzz Opcodes to (row_idx, col_idx) lists.

    Each opcode has .tag ('equal', 'replace', 'delete', 'insert') and
    .src_start, .src_end, .dest_start, .dest_end ranges.
    """
    row_idx: list[int] = []
    col_idx: list[int] = []
    for op in opcodes:  # type: ignore[union-attr]
        if op.tag in ("equal", "replace"):
            for i, j in zip(
                range(op.src_start, op.src_end),
                range(op.dest_start, op.dest_end),
                strict=True,
            ):
                row_idx.append(i)
                col_idx.append(j)
        elif op.tag == "delete":
            for i in range(op.src_start, op.src_end):
                row_idx.append(i)
                col_idx.append(-1)
        elif op.tag == "insert":
            for j in range(op.dest_start, op.dest_end):
                row_idx.append(-1)
                col_idx.append(j)
    return row_idx, col_idx


class TestEditopsIndexComparison:
    """Compare NW-returned indices against rapidfuzz opcodes.

    Since tie-breaking may differ, the index arrays won't always match.
    Instead we verify that both alignments produce the same optimal score.
    """

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_levenshtein_indices_score_matches_opcodes(self, s1: str, s2: str) -> None:
        sm = _char_score_matrix(s1, s2, match=0, mismatch=-1)
        gap = -1.0
        nw_score, nw_ri, nw_ci = needleman_wunsch(
            sm, gap_penalty_source=gap, gap_penalty_target=gap
        )

        rf_ri, rf_ci = _opcodes_to_indices(Levenshtein.opcodes(s1, s2))
        rf_score = _recompute_score_from_indices(rf_ri, rf_ci, sm, gap, gap)

        # Both must achieve the same optimal score.
        assert nw_score == rf_score, (
            f"Score mismatch for ({s1!r}, {s2!r}): NW={nw_score}, RF={rf_score}"
        )
        # Score recomputed from NW indices must equal reported score.
        recomputed = _recompute_score_from_indices(nw_ri, nw_ci, sm, gap, gap)
        assert recomputed == nw_score

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_indel_indices_score_matches_opcodes(self, s1: str, s2: str) -> None:
        sm = _char_score_matrix(s1, s2, match=0, mismatch=-2)
        gap = -1.0
        nw_score, nw_ri, nw_ci = needleman_wunsch(
            sm, gap_penalty_source=gap, gap_penalty_target=gap
        )

        rf_ri, rf_ci = _opcodes_to_indices(Indel.opcodes(s1, s2))
        rf_score = _recompute_score_from_indices(rf_ri, rf_ci, sm, gap, gap)

        assert nw_score == rf_score
        recomputed = _recompute_score_from_indices(nw_ri, nw_ci, sm, gap, gap)
        assert recomputed == nw_score

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_weighted_levenshtein_indices_structural(self, s1: str, s2: str) -> None:
        """Asymmetric weights: insertion=2, deletion=1, substitution=1.

        rapidfuzz.opcodes() does not support ``weights``, so we verify
        against the known distance and check structural invariants only.
        """
        weights = (2, 1, 1)
        sm = _char_score_matrix(s1, s2, match=0, mismatch=-weights[2])
        gap_in_row = -float(weights[0])
        gap_in_col = -float(weights[1])

        nw_score, nw_ri, nw_ci = needleman_wunsch(
            sm, gap_penalty_source=gap_in_row, gap_penalty_target=gap_in_col
        )

        # Score must match the known weighted Levenshtein distance.
        expected = -Levenshtein.distance(s1, s2, weights=weights)
        assert nw_score == expected, (
            f"Score mismatch for ({s1!r}, {s2!r}): NW={nw_score}, expected={expected}"
        )
        # Recomputation from indices must agree.
        recomputed = _recompute_score_from_indices(
            nw_ri, nw_ci, sm, gap_in_row, gap_in_col
        )
        assert recomputed == nw_score
        # No double gaps.
        assert not ((nw_ri == -1) & (nw_ci == -1)).any()
        # Monotonicity.
        row_ng = nw_ri[nw_ri >= 0]
        col_ng = nw_ci[nw_ci >= 0]
        if len(row_ng) > 1:
            assert np.all(np.diff(row_ng) > 0)
        if len(col_ng) > 1:
            assert np.all(np.diff(col_ng) > 0)
        # Coverage.
        np.testing.assert_array_equal(row_ng, np.arange(len(s1)))
        np.testing.assert_array_equal(col_ng, np.arange(len(s2)))


# ---------------------------------------------------------------------------
# Exhaustive structural properties of returned indices
# ---------------------------------------------------------------------------


class TestIndexStructuralProperties:
    """Verify invariants that every valid optimal alignment must satisfy,
    independent of tie-breaking."""

    @staticmethod
    def _get_indices(
        s1: str,
        s2: str,
        match: float,
        mismatch: float,
        gap: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return (score, row_idx, col_idx, score_matrix)."""
        sm = _char_score_matrix(s1, s2, match, mismatch)
        score, ri, ci = needleman_wunsch(
            sm, gap_penalty_source=gap, gap_penalty_target=gap
        )
        return score, ri, ci, sm

    # -- Levenshtein parameterisation -----------------------------------------

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lev_no_double_gaps(self, s1: str, s2: str) -> None:
        """No position may have gaps in both sequences simultaneously."""
        _, ri, ci, _ = self._get_indices(s1, s2, 0, -1, -1)
        double = (ri == -1) & (ci == -1)
        assert not double.any()

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lev_monotonicity(self, s1: str, s2: str) -> None:
        """Non-gap indices must be strictly increasing."""
        _, ri, ci, _ = self._get_indices(s1, s2, 0, -1, -1)
        row_nongap = ri[ri >= 0]
        col_nongap = ci[ci >= 0]
        assert np.all(np.diff(row_nongap) > 0) if len(row_nongap) > 1 else True
        assert np.all(np.diff(col_nongap) > 0) if len(col_nongap) > 1 else True

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lev_coverage(self, s1: str, s2: str) -> None:
        """Non-gap indices must cover every element exactly once."""
        _, ri, ci, _ = self._get_indices(s1, s2, 0, -1, -1)
        np.testing.assert_array_equal(ri[ri >= 0], np.arange(len(s1)))
        np.testing.assert_array_equal(ci[ci >= 0], np.arange(len(s2)))

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lev_alignment_length_bounds(self, s1: str, s2: str) -> None:
        """Alignment length must be in [max(n,m), n+m]."""
        _, ri, ci, _ = self._get_indices(s1, s2, 0, -1, -1)
        length = len(ri)
        n, m = len(s1), len(s2)
        assert length >= max(n, m, 0)
        assert length <= n + m or (n == 0 and m == 0 and length == 0)

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lev_score_recomputation(self, s1: str, s2: str) -> None:
        """Score recomputed from indices must equal the reported score."""
        score, ri, ci, sm = self._get_indices(s1, s2, 0, -1, -1)
        recomputed = _recompute_score_from_indices(ri, ci, sm, -1, -1)
        assert recomputed == score

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lev_score_is_optimal(self, s1: str, s2: str) -> None:
        """Score must match the known optimal from rapidfuzz."""
        score, _, _, _ = self._get_indices(s1, s2, 0, -1, -1)
        assert score == -Levenshtein.distance(s1, s2)

    # -- Indel parameterisation -----------------------------------------------

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_indel_structural_invariants(self, s1: str, s2: str) -> None:
        """All structural invariants for indel parameterisation."""
        score, ri, ci, sm = self._get_indices(s1, s2, 0, -2, -1)
        n, m = len(s1), len(s2)

        # No double gaps
        assert not ((ri == -1) & (ci == -1)).any()
        # Monotonicity
        row_ng, col_ng = ri[ri >= 0], ci[ci >= 0]
        if len(row_ng) > 1:
            assert np.all(np.diff(row_ng) > 0)
        if len(col_ng) > 1:
            assert np.all(np.diff(col_ng) > 0)
        # Coverage
        np.testing.assert_array_equal(row_ng, np.arange(n))
        np.testing.assert_array_equal(col_ng, np.arange(m))
        # Length bounds
        length = len(ri)
        assert length >= max(n, m, 0)
        assert length <= n + m or (n == 0 and m == 0)
        # Score recomputation
        assert _recompute_score_from_indices(ri, ci, sm, -1, -1) == score
        # Optimality
        assert score == -Indel.distance(s1, s2)

    # -- LCS parameterisation -------------------------------------------------

    @pytest.mark.parametrize("s1, s2", ALL_PAIRS)
    def test_lcs_structural_invariants(self, s1: str, s2: str) -> None:
        """All structural invariants for LCS parameterisation."""
        score, ri, ci, sm = self._get_indices(s1, s2, 1, 0, 0)
        n, m = len(s1), len(s2)

        assert not ((ri == -1) & (ci == -1)).any()
        row_ng, col_ng = ri[ri >= 0], ci[ci >= 0]
        if len(row_ng) > 1:
            assert np.all(np.diff(row_ng) > 0)
        if len(col_ng) > 1:
            assert np.all(np.diff(col_ng) > 0)
        np.testing.assert_array_equal(row_ng, np.arange(n))
        np.testing.assert_array_equal(col_ng, np.arange(m))
        length = len(ri)
        assert length >= max(n, m, 0)
        assert length <= n + m or (n == 0 and m == 0)
        assert _recompute_score_from_indices(ri, ci, sm, 0, 0) == score
        assert score == LCSseq.similarity(s1, s2)

    # -- Hamming parameterisation ---------------------------------------------

    @pytest.mark.parametrize("s1, s2", EQUAL_LENGTH_PAIRS)
    def test_hamming_structural_invariants(self, s1: str, s2: str) -> None:
        """Hamming: all-diagonal alignment, no gaps at all."""
        n = len(s1)
        gap = -(n + 1) if n > 0 else -1.0
        score, ri, ci, sm = self._get_indices(s1, s2, 0, -1, gap)

        # No gaps at all (pure diagonal alignment)
        assert not (ri == -1).any(), "Unexpected gap in row indices"
        assert not (ci == -1).any(), "Unexpected gap in col indices"
        # Length == n (purely diagonal)
        assert len(ri) == n
        # Indices are exactly 0..n-1
        np.testing.assert_array_equal(ri, np.arange(n))
        np.testing.assert_array_equal(ci, np.arange(n))
        # Score recomputation
        assert _recompute_score_from_indices(ri, ci, sm, gap, gap) == score
        assert score == -Hamming.distance(s1, s2)

    # -- Random score matrices ------------------------------------------------

    def test_random_matrices_structural_invariants(self) -> None:
        """Structural invariants hold for arbitrary score matrices."""
        rng = np.random.default_rng(99999)
        gap = -1.0
        for _ in range(50):
            n = int(rng.integers(0, 15))
            m = int(rng.integers(0, 15))
            sm = rng.standard_normal((n, m))
            score, ri, ci = needleman_wunsch(
                sm, gap_penalty_source=gap, gap_penalty_target=gap
            )

            # No double gaps
            assert not ((ri == -1) & (ci == -1)).any()
            # Monotonicity
            row_ng, col_ng = ri[ri >= 0], ci[ci >= 0]
            if len(row_ng) > 1:
                assert np.all(np.diff(row_ng) > 0)
            if len(col_ng) > 1:
                assert np.all(np.diff(col_ng) > 0)
            # Coverage
            np.testing.assert_array_equal(row_ng, np.arange(n))
            np.testing.assert_array_equal(col_ng, np.arange(m))
            # Length bounds
            length = len(ri)
            assert length >= max(n, m, 0)
            assert length <= n + m or (n == 0 and m == 0)
            # Score recomputation
            recomputed = _recompute_score_from_indices(ri, ci, sm, gap, gap)
            assert recomputed == pytest.approx(score)
