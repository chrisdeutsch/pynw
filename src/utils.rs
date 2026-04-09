use ndarray::prelude::*;
use numpy::{Element, PyArray, PyArrayMethods, PyReadonlyArray, get_array_module};
use pyo3::{intern, prelude::*, sync::PyOnceLock, types::PyDict};

// Modified version of the PyArrayLike extract method. Calls numpy.asarray(obj, dtype=dtype).
pub fn numpy_asarray<'py>(
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
    dtype: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    static AS_ARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let as_array = AS_ARRAY
        .get_or_try_init(py, || {
            get_array_module(py)?.getattr("asarray").map(Into::into)
        })?
        .bind(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "dtype"), dtype)?;
    as_array.call((obj,), Some(&kwargs))
}

pub fn to_pyreadonly<'py, A, D>(
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray<'py, A, D>>
where
    A: Element,
    D: Dimension,
{
    if let Ok(array) = obj.cast::<PyArray<A, D>>() {
        return Ok(array.readonly());
    }

    numpy_asarray(py, obj, A::get_dtype(py).into_any())?
        .extract()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into the expected array type",
            )
        })
}

pub fn validate_inputs(
    similarity_matrix: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> PyResult<()> {
    if !similarity_matrix.iter().all(|v: &f64| v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "similarity_matrix contains non-finite values (NaN or Inf)",
        ));
    }
    if !insert_penalty.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "insert_penalty is non-finite",
        ));
    }
    if !delete_penalty.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "delete_penalty is non-finite",
        ));
    }
    Ok(())
}
