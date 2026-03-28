import numpy as np
import numpy.typing as npt

def needleman_wunsch(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    gap_penalty_row: float | None = None,
    gap_penalty_col: float | None = None,
    check_finite: bool = False,
) -> tuple[float, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
