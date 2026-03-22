// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Python bindings for RustQuant machine learning module.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use RustQuant::ml::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Linear Regression
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Ordinary Least Squares linear regression.
///
/// Supports QR and SVD decomposition methods.
#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    intercept: f64,
    coefficients: Vec<f64>,
}

#[pymethods]
impl PyLinearRegression {
    /// Fit a linear regression model.
    ///
    /// Args:
    ///     x: Feature matrix as list of lists (n_samples x n_features).
    ///     y: Response vector as list (n_samples).
    ///     method: Decomposition method - "qr" or "svd" (default: "qr").
    ///
    /// Returns:
    ///     A fitted LinearRegression model.
    #[new]
    #[pyo3(signature = (x, y, method="qr"))]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, method: &str) -> PyResult<Self> {
        let n_rows = x.len();
        let n_cols = if n_rows > 0 { x[0].len() } else { 0 };

        let flat: Vec<f64> = x.into_iter().flatten().collect();
        let x_mat = DMatrix::from_row_slice(n_rows, n_cols, &flat);
        let y_vec = DVector::from_vec(y);

        let input = LinearRegressionInput { x: x_mat, y: y_vec };

        let decomp = match method.to_lowercase().as_str() {
            "svd" => Decomposition::SVD,
            _ => Decomposition::QR,
        };

        let output = input
            .fit(decomp)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            intercept: output.intercept,
            coefficients: output.coefficients.as_slice().to_vec(),
        })
    }

    /// The intercept (bias) term.
    #[getter]
    fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The fitted coefficients.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    /// Predict response for new data.
    ///
    /// Args:
    ///     x: Feature matrix as list of lists.
    ///
    /// Returns:
    ///     List of predicted values.
    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let n_rows = x.len();
        let n_cols = if n_rows > 0 { x[0].len() } else { 0 };
        let flat: Vec<f64> = x.into_iter().flatten().collect();
        let x_mat = DMatrix::from_row_slice(n_rows, n_cols, &flat);

        let output = LinearRegressionOutput {
            intercept: self.intercept,
            coefficients: DVector::from_vec(self.coefficients.clone()),
        };

        let preds = output
            .predict(x_mat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(preds.as_slice().to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearRegression(intercept={:.4}, n_features={})",
            self.intercept,
            self.coefficients.len()
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Logistic Regression
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Logistic regression for binary classification (IRLS method).
#[pyclass(name = "LogisticRegression")]
pub struct PyLogisticRegression {
    coefficients: Vec<f64>,
    iterations: usize,
}

#[pymethods]
impl PyLogisticRegression {
    /// Fit a logistic regression model using IRLS.
    ///
    /// Args:
    ///     x: Feature matrix as list of lists (n_samples x n_features).
    ///     y: Binary response vector (0s and 1s).
    ///     tolerance: Convergence tolerance (default: 1e-8).
    #[new]
    #[pyo3(signature = (x, y, tolerance=1e-8))]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, tolerance: f64) -> PyResult<Self> {
        let n_rows = x.len();
        let n_cols = if n_rows > 0 { x[0].len() } else { 0 };

        let flat: Vec<f64> = x.into_iter().flatten().collect();
        let x_mat = DMatrix::from_row_slice(n_rows, n_cols, &flat);
        let y_vec = DVector::from_vec(y);

        let input = LogisticRegressionInput { x: x_mat, y: y_vec };

        let output = input
            .fit(LogisticRegressionAlgorithm::IRLS, tolerance)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            coefficients: output.coefficients.as_slice().to_vec(),
            iterations: output.iterations,
        })
    }

    /// The fitted coefficients (first element is the intercept).
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    /// Number of IRLS iterations to converge.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Predict class labels (0 or 1) for new data.
    fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let output = self._output();
        let x_mat = self._to_matrix(x);
        output.predict(&x_mat).as_slice().to_vec()
    }

    /// Predict class probabilities P(y=1) for new data.
    fn predict_proba(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let output = self._output();
        let x_mat = self._to_matrix(x);
        output.predict_proba(&x_mat).as_slice().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "LogisticRegression(n_params={}, iterations={})",
            self.coefficients.len(),
            self.iterations
        )
    }
}

impl PyLogisticRegression {
    fn _output(&self) -> LogisticRegressionOutput<f64> {
        LogisticRegressionOutput {
            coefficients: DVector::from_vec(self.coefficients.clone()),
            iterations: self.iterations,
        }
    }

    fn _to_matrix(&self, x: Vec<Vec<f64>>) -> DMatrix<f64> {
        let n_rows = x.len();
        let n_cols = if n_rows > 0 { x[0].len() } else { 0 };
        let flat: Vec<f64> = x.into_iter().flatten().collect();
        DMatrix::from_row_slice(n_rows, n_cols, &flat)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module registration
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn register(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "ml")?;

    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyLogisticRegression>()?;

    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("RustQuant.ml", m)?;

    Ok(())
}
