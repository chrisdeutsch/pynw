import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pynw import needleman_wunsch
from pynw._ops import EditOp


def test_return_types() -> None:
    score, ops = needleman_wunsch(np.eye(3, dtype=np.float64))
    assert isinstance(score, float)
    assert isinstance(ops, np.ndarray)
    assert ops.dtype == np.uint8

    assert score == 3.0
    np.testing.assert_equal(ops, [EditOp.Align, EditOp.Align, EditOp.Align])


class TestInputValidation:
    @pytest.mark.parametrize(
        "similarity_matrix",
        [
            pytest.param(np.array([1.0, 2.0]), id="1d"),
            pytest.param(np.zeros((1, 1, 1)), id="3d"),
            pytest.param(np.float64(1.0), id="scalar"),
        ],
    )
    def test_rejects_wrong_dim(self, similarity_matrix):
        with pytest.raises(ValueError, match="2-dimensional"):
            needleman_wunsch(similarity_matrix)

    @pytest.mark.parametrize(
        "similarity_matrix, expected_score",
        [
            pytest.param(
                np.array([[2, -1], [-1, 2]], dtype=np.int32),
                4.0,
                id="array-int",
            ),
            pytest.param(
                np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32),
                4.0,
                id="array-f32",
            ),
            pytest.param(
                np.array([[True, False], [False, True]]),
                2.0,
                id="array-bool",
            ),
            pytest.param(
                [[2.0, -1.0], [-1.0, 2.0]],
                4.0,
                id="list-float",
            ),
            pytest.param(
                [[2, -1], [-1, 2]],
                4.0,
                id="list-int",
            ),
            pytest.param(
                [[True, False], [False, True]],
                2.0,
                id="list-bool",
            ),
        ],
    )
    def test_casts(self, similarity_matrix, expected_score) -> None:
        score, _ = needleman_wunsch(similarity_matrix)
        assert isinstance(score, float)
        assert score == expected_score

    def test_rejects_nan_in_matrix(self) -> None:
        sm = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm)

    def test_rejects_inf_in_matrix(self) -> None:
        sm = np.array([[1.0, np.inf], [0.0, 1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm)

    def test_rejects_nan_gap_penalty(self) -> None:
        sm = np.array([[1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm, gap_penalty=float("nan"))

    def test_rejects_inf_gap_penalty_source(self) -> None:
        sm = np.array([[1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm, gap_penalty_source=float("inf"))

    def test_rejects_inf_gap_penalty_target(self) -> None:
        sm = np.array([[1.0]])
        with pytest.raises(ValueError, match=r"non-finite"):
            needleman_wunsch(sm, gap_penalty_target=float("inf"))


class TestEmptyAndDegenerateMatrices:
    @pytest.mark.parametrize(
        "similarity_matrix, expected_score, expected_ops",
        [
            pytest.param(np.empty((0, 0)), 0.0, [], id="0x0"),
            pytest.param(np.empty((0, 1)), -1.0, [EditOp.Insert], id="0x1"),
            pytest.param(np.empty((1, 0)), -1.0, [EditOp.Delete], id="1x0"),
        ],
    )
    def test_empty_dim(self, similarity_matrix, expected_score, expected_ops) -> None:
        score, ops = needleman_wunsch(similarity_matrix)
        assert score == expected_score
        np.testing.assert_equal(ops, expected_ops)

    def test_1x1(self) -> None:
        score, ops = needleman_wunsch([[5.0]])
        assert score == 5.0
        np.testing.assert_equal(ops, [EditOp.Align])

    def test_1x1_no_align(self) -> None:
        score, ops = needleman_wunsch([[-5.0]], gap_penalty=0.0)
        assert score == 0.0
        # The DP fill at [1, 1] is a tie between insert and delete. Therefore,
        # it is deterministically broken and filled with delete. Then the
        # traceback yields [delete, insert], which has to be reversed to give
        # the result of [insert, delete].
        np.testing.assert_equal(ops, [EditOp.Insert, EditOp.Delete])


@pytest.mark.parametrize(
    "matrix, expected_score, expected_ops",
    [
        pytest.param(
            [[3.0, 2.0, 1.0]],
            1.0,
            [EditOp.Align, EditOp.Insert, EditOp.Insert],
            id="wide-1",
        ),
        pytest.param(
            [[3.0], [2.0], [1.0]],
            1.0,
            [EditOp.Align, EditOp.Delete, EditOp.Delete],
            id="tall-1",
        ),
        pytest.param(
            [[1.0, 2.0, 3.0]],
            1.0,
            [EditOp.Insert, EditOp.Insert, EditOp.Align],
            id="wide-2",
        ),
        pytest.param(
            [[1.0], [2.0], [3.0]],
            1.0,
            [EditOp.Delete, EditOp.Delete, EditOp.Align],
            id="tall-2",
        ),
    ],
)
def test_rectangular_matrices(matrix, expected_score, expected_ops) -> None:
    score, ops = needleman_wunsch(matrix)
    assert score == expected_score
    np.testing.assert_equal(ops, expected_ops)


@pytest.mark.parametrize(
    "matrix, expected_ops",
    [
        pytest.param(
            [[0.0]],
            [EditOp.Align],
            id="all-tied",
        ),
        pytest.param(
            [[-1.0]],
            [EditOp.Insert, EditOp.Delete],
            id="insert-delete-tied",
        ),
    ],
)
def test_tie_breaking(matrix, expected_ops) -> None:
    _, ops = needleman_wunsch(matrix, gap_penalty=0.0)
    np.testing.assert_equal(ops, expected_ops)


@pytest.mark.parametrize(
    "matrix, expected_score, expected_ops",
    [
        pytest.param(
            np.eye(3),
            3.0,
            3 * [EditOp.Align],
            id="identity",
        ),
        pytest.param(
            np.zeros((3, 3)),
            0.0,
            3 * [EditOp.Align],
            id="zeros",
        ),
        pytest.param(
            np.ones((3, 3)),
            3.0,
            3 * [EditOp.Align],
            id="ones",
        ),
        pytest.param(
            np.full((3, 3), -5.0),
            -6.0,
            3 * [EditOp.Insert] + 3 * [EditOp.Delete],
            id="negative",
        ),
        pytest.param(
            [[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]],
            18.0,
            [EditOp.Insert, EditOp.Align, EditOp.Align, EditOp.Delete],
            id="off-diagonal",
        ),
        pytest.param(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            96.0,
            [EditOp.Delete, EditOp.Delete, EditOp.Align, EditOp.Insert, EditOp.Insert],
            id="large-bottom-left",
        ),
        pytest.param(
            [[0.0, 0.0, 100.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            96.0,
            [EditOp.Insert, EditOp.Insert, EditOp.Align, EditOp.Delete, EditOp.Delete],
            id="large-top-right",
        ),
    ],
)
def test_special_matrices(matrix, expected_score, expected_ops) -> None:
    score, ops = needleman_wunsch(matrix)
    assert score == expected_score
    np.testing.assert_equal(ops, expected_ops)


@pytest.mark.hypothesis
class TestStructuralInvariants:
    def assert_structural_invariants(self, ops, n, m):
        num_align = np.count_nonzero(ops == EditOp.Align)
        num_delete = np.count_nonzero(ops == EditOp.Delete)
        num_insert = np.count_nonzero(ops == EditOp.Insert)

        assert num_align + num_delete == n
        assert num_align + num_insert == m
        assert len(ops) == num_align + num_delete + num_insert
        assert num_align <= min(n, m)

    @given(
        n=st.integers(min_value=0, max_value=20),
        m=st.integers(min_value=0, max_value=20),
        data=st.data(),
    )
    def test_random_matrix(self, n: int, m: int, data: st.DataObject) -> None:
        matrix = (
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
            if n > 0 and m > 0
            else np.empty((n, m))
        )

        _, ops = needleman_wunsch(matrix)
        self.assert_structural_invariants(ops, n, m)
