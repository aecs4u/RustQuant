// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Python bindings for RustQuant math module.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use pyo3::prelude::*;
use RustQuant::math::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Numerical Integration
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Numerically integrate a Python callable f(x) -> float over [a, b].
///
/// Uses the Tanh-Sinh (double exponential) quadrature method.
///
/// Args:
///     f: A callable taking a float and returning a float.
///     a: Lower bound of integration.
///     b: Upper bound of integration.
///
/// Returns:
///     The integral value.
#[pyfunction]
fn integrate_func(f: Py<pyo3::types::PyAny>, a: f64, b: f64) -> PyResult<f64> {
    let result = integrate(
        |x: f64| -> f64 {
            Python::attach(|py| {
                f.call1(py, (x,))
                    .and_then(|r| r.extract::<f64>(py))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
    );
    Ok(result)
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Sequences
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Generate a linearly spaced sequence.
///
/// Args:
///     start: Start value.
///     end: End value (must be > start).
///     n: Number of points.
///
/// Returns:
///     List of evenly spaced values.
#[pyfunction]
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    f64::linspace(start, end, n)
}

/// Generate a sequence from start to end with given step.
///
/// Args:
///     start: Start value.
///     end: End value.
///     step: Step size.
///
/// Returns:
///     List of values.
#[pyfunction]
fn seq(start: f64, end: f64, step: f64) -> Vec<f64> {
    f64::seq(start, end, step)
}

/// Compute the cumulative sum of a list.
#[pyfunction]
fn cumsum(v: Vec<f64>) -> Vec<f64> {
    f64::cumsum(&v)
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module registration
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn register(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "math")?;

    m.add_function(wrap_pyfunction!(integrate_func, &m)?)?;
    m.add_function(wrap_pyfunction!(linspace, &m)?)?;
    m.add_function(wrap_pyfunction!(seq, &m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, &m)?)?;

    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("RustQuant.math", m)?;

    Ok(())
}
