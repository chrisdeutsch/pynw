import numpy as np
import pytest

from pynw import needleman_wunsch


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
    def test_casts(self, similarity_matrix, expected_score):
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
