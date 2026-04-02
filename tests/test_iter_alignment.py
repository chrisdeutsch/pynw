"""Tests for iter_alignment."""

import numpy as np

from pynw import needleman_wunsch
from pynw._ops import Op, iter_alignment
from tests.helpers import char_score_matrix


def _indices_to_ops(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
) -> np.ndarray:
    """Convert (source_idx, target_idx) arrays to an ops array for iter_alignment."""
    return np.where(
        source_idx < 0,
        Op.INSERT,
        np.where(target_idx < 0, Op.DELETE, Op.ALIGN),
    ).astype(np.uint8)


def ops_from_seqs(
    s1: str,
    s2: str,
    match: float = 1.0,
    mismatch: float = -1.0,
    gap: float = -1.0,
) -> tuple[float, list[Op]]:
    sm = char_score_matrix(s1, s2, match, mismatch)
    score, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=gap)
    ops = _indices_to_ops(source_idx, target_idx)
    return score, list(Op(o) for o in ops)


def run_iter(s1: str, s2: str, **kwargs) -> list[tuple[Op, str | None, str | None]]:
    sm = (
        char_score_matrix(s1, s2, **kwargs)
        if kwargs
        else char_score_matrix(s1, s2, 1.0, -1.0)
    )
    score, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
    ops = _indices_to_ops(source_idx, target_idx)
    return list(iter_alignment(ops, list(s1), list(s2)))


# ---------------------------------------------------------------------------
# Basic op semantics
# ---------------------------------------------------------------------------


class TestOpSemantics:
    def test_pure_align(self) -> None:
        sm = np.array([[1.0]])
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        result = list(iter_alignment(ops, ["A"], ["A"]))
        assert result == [(Op.ALIGN, "A", "A")]

    def test_pure_insert(self) -> None:
        """Source is empty — all steps are inserts."""
        sm = np.empty((0, 3), dtype=np.float64)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        result = list(iter_alignment(ops, [], ["X", "Y", "Z"]))
        assert result == [
            (Op.INSERT, None, "X"),
            (Op.INSERT, None, "Y"),
            (Op.INSERT, None, "Z"),
        ]

    def test_pure_delete(self) -> None:
        """Target is empty — all steps are deletes."""
        sm = np.empty((3, 0), dtype=np.float64)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        result = list(iter_alignment(ops, ["X", "Y", "Z"], []))
        assert result == [
            (Op.DELETE, "X", None),
            (Op.DELETE, "Y", None),
            (Op.DELETE, "Z", None),
        ]

    def test_insert_has_none_row_item(self) -> None:
        sm = np.array([[-5.0, 1.0]])  # align[0,1] is best, so [0,0] gets a gap
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, ["A"], ["X", "Y"]))
        insert_steps = [(op, r, c) for op, r, c in steps if op == Op.INSERT]
        for _, r, c in insert_steps:
            assert r is None
            assert c is not None

    def test_delete_has_none_col_item(self) -> None:
        sm = np.array([[-5.0], [1.0]])  # align[1,0] is best, so [0,0] gets a gap
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, ["A", "B"], ["X"]))
        delete_steps = [(op, r, c) for op, r, c in steps if op == Op.DELETE]
        for _, r, c in delete_steps:
            assert r is not None
            assert c is None


# ---------------------------------------------------------------------------
# Empty sequences
# ---------------------------------------------------------------------------


class TestEmpty:
    def test_both_empty(self) -> None:
        sm = np.empty((0, 0), dtype=np.float64)
        _, source_idx, target_idx = needleman_wunsch(sm)
        ops = _indices_to_ops(source_idx, target_idx)
        result = list(iter_alignment(ops, [], []))
        assert result == []

    def test_empty_row_seq(self) -> None:
        sm = np.empty((0, 2), dtype=np.float64)
        _, source_idx, target_idx = needleman_wunsch(sm)
        ops = _indices_to_ops(source_idx, target_idx)
        result = list(iter_alignment(ops, [], ["A", "B"]))
        assert len(result) == 2
        assert all(op == Op.INSERT for op, _, _ in result)

    def test_empty_col_seq(self) -> None:
        sm = np.empty((2, 0), dtype=np.float64)
        _, source_idx, target_idx = needleman_wunsch(sm)
        ops = _indices_to_ops(source_idx, target_idx)
        result = list(iter_alignment(ops, ["A", "B"], []))
        assert len(result) == 2
        assert all(op == Op.DELETE for op, _, _ in result)


# ---------------------------------------------------------------------------
# Coverage: every element appears exactly once
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_all_row_elements_appear(self) -> None:
        sm = char_score_matrix("GATTACA", "GCATGCA", 1.0, -1.0)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, list("GATTACA"), list("GCATGCA")))
        row_items = [r for _, r, _ in steps if r is not None]
        assert row_items == list("GATTACA")

    def test_all_col_elements_appear(self) -> None:
        sm = char_score_matrix("GATTACA", "GCATGCA", 1.0, -1.0)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, list("GATTACA"), list("GCATGCA")))
        col_items = [c for _, _, c in steps if c is not None]
        assert col_items == list("GCATGCA")

    def test_no_double_gaps(self) -> None:
        sm = char_score_matrix("ABCD", "EFGH", 1.0, -1.0)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, list("ABCD"), list("EFGH")))
        assert not any(r is None and c is None for _, r, c in steps)


# ---------------------------------------------------------------------------
# Alignment strings match expected output
# ---------------------------------------------------------------------------


class TestAlignmentStrings:
    def test_gattaca_gcatgca(self) -> None:
        sm = char_score_matrix("GATTACA", "GCATGCA", 1.0, -1.0)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, list("GATTACA"), list("GCATGCA")))
        aligned_row = "".join(r if r is not None else "-" for _, r, _ in steps)
        aligned_col = "".join(c if c is not None else "-" for _, _, c in steps)
        assert aligned_row == "G-ATTACA"
        assert aligned_col == "GCA-TGCA"

    def test_identical_sequences(self) -> None:
        sm = char_score_matrix("ABC", "ABC", 1.0, -1.0)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, list("ABC"), list("ABC")))
        assert all(op == Op.ALIGN for op, _, _ in steps)
        assert all(r == c for _, r, c in steps)

    def test_completely_different_sequences(self) -> None:
        """With sufficiently negative gap penalty, forces diagonal alignment."""
        sm = char_score_matrix("ABC", "XYZ", 0.0, -1.0)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-100.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, list("ABC"), list("XYZ")))
        assert all(op == Op.ALIGN for op, _, _ in steps)


# ---------------------------------------------------------------------------
# Works on non-string sequences
# ---------------------------------------------------------------------------


class TestSequenceTypes:
    def test_integer_sequences(self) -> None:
        row = [1, 2, 3]
        col = [1, 2, 3]
        sm = np.eye(3)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, row, col))
        assert steps == [(Op.ALIGN, 1, 1), (Op.ALIGN, 2, 2), (Op.ALIGN, 3, 3)]

    def test_tuple_sequences(self) -> None:
        row = ("a", "b")
        col = ("a", "b")
        sm = np.eye(2)
        _, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
        ops = _indices_to_ops(source_idx, target_idx)
        steps = list(iter_alignment(ops, row, col))
        assert len(steps) == 2
        assert all(op == Op.ALIGN for op, _, _ in steps)
