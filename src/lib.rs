//! PyO3 binding layer for the `pynw._native` extension module.
//!
//! Safe `Array2` indexing is used throughout.  Benchmarking showed that
//! `unsafe get_unchecked` offers no measurable improvement.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod nw_core;

#[pymodule(name = "_native")]
mod pynw_native {
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
}
