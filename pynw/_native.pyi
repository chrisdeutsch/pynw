import numpy as np
import numpy.typing as npt

def nw_traceback_indices(
    similarity_matrix: npt.NDArray[np.float64],
    gap_penalty_row: float,
    gap_penalty_col: float,
) -> tuple[float, npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
