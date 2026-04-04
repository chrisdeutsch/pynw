import pytest

from pynw import EditOp, alignment_indices


@pytest.mark.parametrize(
    "ops, expected_src, expected_tgt",
    [
        pytest.param([], [], [], id="empty"),
        # Single op
        pytest.param([EditOp.Align], [0], [0], id="align"),
        pytest.param([EditOp.Insert], [None], [0], id="insert"),
        pytest.param([EditOp.Delete], [0], [None], id="delete"),
        # Permutations of two ops
        pytest.param(
            [EditOp.Align, EditOp.Align],
            [0, 1],
            [0, 1],
            id="align-align",
        ),
        pytest.param(
            [EditOp.Align, EditOp.Insert],
            [0, None],
            [0, 1],
            id="align-insert",
        ),
        pytest.param(
            [EditOp.Align, EditOp.Delete],
            [0, 1],
            [0, None],
            id="align-delete",
        ),
        pytest.param(
            [EditOp.Insert, EditOp.Align],
            [None, 0],
            [0, 1],
            id="insert-align",
        ),
        pytest.param(
            [EditOp.Insert, EditOp.Insert],
            [None, None],
            [0, 1],
            id="insert-insert",
        ),
        pytest.param(
            [EditOp.Insert, EditOp.Delete],
            [None, 0],
            [0, None],
            id="insert-delete",
        ),
        pytest.param(
            [EditOp.Delete, EditOp.Align],
            [0, 1],
            [None, 0],
            id="delete-align",
        ),
        pytest.param(
            [EditOp.Delete, EditOp.Insert],
            [0, None],
            [None, 0],
            id="delete-insert",
        ),
        pytest.param(
            [EditOp.Delete, EditOp.Delete],
            [0, 1],
            [None, None],
            id="delete-delete",
        ),
    ],
)
def test_alignment_indices(
    ops: list[EditOp],
    expected_src: list[int | None],
    expected_tgt: list[int | None],
) -> None:
    src_idx, tgt_idx = alignment_indices(ops)
    assert src_idx.tolist() == expected_src
    assert tgt_idx.tolist() == expected_tgt


def test_raises_on_unknown_op():
    ops = [EditOp.Align, EditOp.Delete, EditOp.Insert, 99]
    with pytest.raises(ValueError, match="EditOp"):
        alignment_indices(ops)


@pytest.mark.parametrize("editop", [-1, 999])
def test_raises_on_overflow(editop):
    with pytest.raises(OverflowError):
        alignment_indices([editop])
