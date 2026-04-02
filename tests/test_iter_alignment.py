from itertools import count

import pytest

from pynw import EditOp, iter_alignment


@pytest.mark.parametrize(
    "ops, expected",
    [
        pytest.param([], [], id="empty"),
        # Single op
        pytest.param([EditOp.Align], [(0, 0)], id="align"),
        pytest.param([EditOp.Insert], [(None, 0)], id="insert"),
        pytest.param([EditOp.Delete], [(0, None)], id="delete"),
        # Permutations of two ops
        pytest.param(
            [EditOp.Align, EditOp.Align],
            [(0, 0), (1, 1)],
            id="align-align",
        ),
        pytest.param(
            [EditOp.Align, EditOp.Insert],
            [(0, 0), (None, 1)],
            id="align-insert",
        ),
        pytest.param(
            [EditOp.Align, EditOp.Delete],
            [(0, 0), (1, None)],
            id="align-delete",
        ),
        pytest.param(
            [EditOp.Insert, EditOp.Align],
            [(None, 0), (0, 1)],
            id="insert-align",
        ),
        pytest.param(
            [EditOp.Insert, EditOp.Insert],
            [(None, 0), (None, 1)],
            id="insert-insert",
        ),
        pytest.param(
            [EditOp.Insert, EditOp.Delete],
            [(None, 0), (0, None)],
            id="insert-delete",
        ),
        pytest.param(
            [EditOp.Delete, EditOp.Align],
            [(0, None), (1, 0)],
            id="delete-align",
        ),
        pytest.param(
            [EditOp.Delete, EditOp.Insert],
            [(0, None), (None, 0)],
            id="delete-insert",
        ),
        pytest.param(
            [EditOp.Delete, EditOp.Delete],
            [(0, None), (1, None)],
            id="delete-delete",
        ),
    ],
)
def test_iter_alignment(
    ops: list[EditOp], expected: list[tuple[int | None, int | None]]
) -> None:
    source_sequence = count()
    target_sequence = count()

    result = list(
        (source, target)
        for (_, source, target) in iter_alignment(ops, source_sequence, target_sequence)
    )
    assert result == expected


@pytest.mark.parametrize(
    "ops, source, target",
    [
        pytest.param(
            [],
            [0],
            [0],
            marks=pytest.mark.xfail(reason="Not implemented"),
            id="short-ops",
        ),
        pytest.param([EditOp.Align], [], [0], id="short-source"),
        pytest.param([EditOp.Align], [0], [], id="short-target"),
    ],
)
def test_iter_alignment_strict(
    ops: list[EditOp], source: list[int], target: list[int]
) -> None:
    with pytest.raises(ValueError):
        list(iter_alignment(ops, source, target))
