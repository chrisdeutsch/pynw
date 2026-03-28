//! PyO3 binding layer for the `pynw._native` extension module.
//!
//! Safe `Array2` indexing is used throughout.  Benchmarking showed that
//! `unsafe get_unchecked` offers no measurable improvement.

use numpy::{IntoPyArray, PyArray1, PyArrayLikeDyn, PyReadonlyArray2};
use pyo3::prelude::*;

mod nw_core;

#[pymodule(name = "_native")]
mod pynw_native {
    use numpy::AllowTypeChange;

    use super::*;

    type NwTracebackIndicesOutput<'py> = (
        f64,
        Bound<'py, PyArray1<isize>>,
        Bound<'py, PyArray1<isize>>,
    );

    #[pyfunction]
    fn nw_traceback_indices<'py>(
        py: Python<'py>,
        similarity_matrix: PyReadonlyArray2<'py, f64>,
        gap_penalty_row: f64,
        gap_penalty_col: f64,
    ) -> PyResult<NwTracebackIndicesOutput<'py>> {
        let sm = similarity_matrix.as_array();
        let (score, row_idx, col_idx) =
            nw_core::nw_traceback_indices_core(&sm, gap_penalty_row, gap_penalty_col);
        Ok((score, row_idx.into_pyarray(py), col_idx.into_pyarray(py)))
    }

    #[pyfunction]
    #[pyo3(
        signature = (similarity_matrix, *, gap_penalty=-1.0, gap_penalty_row=None, gap_penalty_col=None, check_finite=false),
        text_signature = "(similarity_matrix, *, gap_penalty=-1.0, gap_penalty_row=None, gap_penalty_col=None, check_finite=False)",
    )]
    fn needleman_wunsch<'py>(
        py: Python<'py>,
        similarity_matrix: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        gap_penalty: f64,
        gap_penalty_row: Option<f64>,
        gap_penalty_col: Option<f64>,
        check_finite: bool,
    ) -> PyResult<NwTracebackIndicesOutput<'py>> {
        let similarity_matrix =
            similarity_matrix
                .as_array()
                .into_dimensionality()
                .map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "similarity_matrix must be 2-dimensional",
                    )
                })?;

        let gap_penalty_row = gap_penalty_row.unwrap_or(gap_penalty);
        let gap_penalty_col = gap_penalty_col.unwrap_or(gap_penalty);

        if check_finite {
            if !gap_penalty_row.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "gap_penalty_row is non-finite",
                ));
            }
            if !gap_penalty_col.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "gap_penalty_col is non-finite",
                ));
            }
            if !similarity_matrix.iter().all(|v: &f64| v.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "similarity_matrix contains non-finite values (NaN or Inf)",
                ));
            }
        }

        let (score, row_idx, col_idx) = nw_core::nw_traceback_indices_core(
            &similarity_matrix,
            gap_penalty_row,
            gap_penalty_col,
        );

        Ok((score, row_idx.into_pyarray(py), col_idx.into_pyarray(py)))
    }
}
