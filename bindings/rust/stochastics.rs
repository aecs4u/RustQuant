// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Python bindings for RustQuant stochastic processes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use pyo3::prelude::*;
use RustQuant::stochastics::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Trajectories output wrapper
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Output of a stochastic process simulation.
#[pyclass(name = "Trajectories")]
pub struct PyTrajectories {
    /// Time points of the simulation.
    #[pyo3(get)]
    pub times: Vec<f64>,
    /// Simulated paths (list of lists).
    #[pyo3(get)]
    pub paths: Vec<Vec<f64>>,
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Helper to run any StochasticProcess and return PyTrajectories
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fn run_process<P: StochasticProcess>(
    process: &P,
    x0: f64,
    t0: f64,
    t_end: f64,
    n_steps: usize,
    n_paths: usize,
    parallel: bool,
    seed: Option<u64>,
) -> PyTrajectories {
    let config = StochasticProcessConfig::new(
        x0,
        t0,
        t_end,
        n_steps,
        StochasticScheme::EulerMaruyama,
        n_paths,
        parallel,
        seed,
    );
    let out = process.generate(&config);
    PyTrajectories {
        times: out.times,
        paths: out.paths,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Geometric Brownian Motion
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
#[pyclass(name = "GeometricBrownianMotion")]
pub struct PyGBM {
    mu: f64,
    sigma: f64,
}

#[pymethods]
impl PyGBM {
    #[new]
    fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    /// Simulate paths. Returns a Trajectories object.
    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = GeometricBrownianMotion::new(self.mu, self.sigma);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!("GeometricBrownianMotion(mu={}, sigma={})", self.mu, self.sigma)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Arithmetic Brownian Motion
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Arithmetic Brownian Motion: dX = mu*dt + sigma*dW
#[pyclass(name = "ArithmeticBrownianMotion")]
pub struct PyABM {
    mu: f64,
    sigma: f64,
}

#[pymethods]
impl PyABM {
    #[new]
    fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = ArithmeticBrownianMotion::new(self.mu, self.sigma);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!("ArithmeticBrownianMotion(mu={}, sigma={})", self.mu, self.sigma)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Standard Brownian Motion
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Standard Brownian Motion: dW
#[pyclass(name = "BrownianMotion")]
pub struct PyBM;

#[pymethods]
impl PyBM {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = BrownianMotion::new();
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        "BrownianMotion()".to_string()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Ornstein-Uhlenbeck
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Ornstein-Uhlenbeck process: dX = theta*(mu - X)*dt + sigma*dW
#[pyclass(name = "OrnsteinUhlenbeck")]
pub struct PyOU {
    mu: f64,
    sigma: f64,
    theta: f64,
}

#[pymethods]
impl PyOU {
    #[new]
    fn new(mu: f64, sigma: f64, theta: f64) -> Self {
        Self { mu, sigma, theta }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = OrnsteinUhlenbeck::new(self.mu, self.sigma, self.theta);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!(
            "OrnsteinUhlenbeck(mu={}, sigma={}, theta={})",
            self.mu, self.sigma, self.theta
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Cox-Ingersoll-Ross
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Cox-Ingersoll-Ross process: dr = theta*(mu - r)*dt + sigma*sqrt(r)*dW
#[pyclass(name = "CoxIngersollRoss")]
pub struct PyCIR {
    mu: f64,
    sigma: f64,
    theta: f64,
}

#[pymethods]
impl PyCIR {
    #[new]
    fn new(mu: f64, sigma: f64, theta: f64) -> Self {
        Self { mu, sigma, theta }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = CoxIngersollRoss::new(self.mu, self.sigma, self.theta);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!(
            "CoxIngersollRoss(mu={}, sigma={}, theta={})",
            self.mu, self.sigma, self.theta
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hull-White
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Hull-White short-rate model: dr = (theta - alpha*r)*dt + sigma*dW
#[pyclass(name = "HullWhite")]
pub struct PyHW {
    alpha: f64,
    sigma: f64,
    theta: f64,
}

#[pymethods]
impl PyHW {
    #[new]
    fn new(alpha: f64, sigma: f64, theta: f64) -> Self {
        Self { alpha, sigma, theta }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = HullWhite::new(self.alpha, self.sigma, self.theta);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!(
            "HullWhite(alpha={}, sigma={}, theta={})",
            self.alpha, self.sigma, self.theta
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Extended Vasicek
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Extended Vasicek model: dr = (theta - alpha*r)*dt + sigma*dW
#[pyclass(name = "ExtendedVasicek")]
pub struct PyEV {
    alpha: f64,
    sigma: f64,
    theta: f64,
}

#[pymethods]
impl PyEV {
    #[new]
    fn new(alpha: f64, sigma: f64, theta: f64) -> Self {
        Self { alpha, sigma, theta }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = ExtendedVasicek::new(self.alpha, self.sigma, self.theta);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtendedVasicek(alpha={}, sigma={}, theta={})",
            self.alpha, self.sigma, self.theta
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Ho-Lee
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Ho-Lee short-rate model: dr = theta*dt + sigma*dW
#[pyclass(name = "HoLee")]
pub struct PyHL {
    sigma: f64,
    theta: f64,
}

#[pymethods]
impl PyHL {
    #[new]
    fn new(sigma: f64, theta: f64) -> Self {
        Self { sigma, theta }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = HoLee::new(self.sigma, self.theta);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!("HoLee(sigma={}, theta={})", self.sigma, self.theta)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Black-Derman-Toy
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Black-Derman-Toy short-rate model.
#[pyclass(name = "BlackDermanToy")]
pub struct PyBDT {
    sigma: f64,
    theta: f64,
}

#[pymethods]
impl PyBDT {
    #[new]
    fn new(sigma: f64, theta: f64) -> Self {
        Self { sigma, theta }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = BlackDermanToy::new(self.sigma, self.theta);
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!("BlackDermanToy(sigma={}, theta={})", self.sigma, self.theta)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Merton Jump Diffusion
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Merton Jump Diffusion: GBM + Poisson jumps.
#[pyclass(name = "MertonJumpDiffusion")]
pub struct PyMJD {
    mu: f64,
    sigma: f64,
    lambda: f64,
    jump_mean: f64,
    jump_volatility: f64,
}

#[pymethods]
impl PyMJD {
    #[new]
    fn new(mu: f64, sigma: f64, lambda: f64, jump_mean: f64, jump_volatility: f64) -> Self {
        Self {
            mu,
            sigma,
            lambda,
            jump_mean,
            jump_volatility,
        }
    }

    #[pyo3(signature = (x0, t_end, n_steps, n_paths, *, t0=0.0, parallel=false, seed=None))]
    fn simulate(
        &self,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        t0: f64,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyTrajectories {
        let process = MertonJumpDiffusion::new(
            self.mu,
            self.sigma,
            self.lambda,
            self.jump_mean,
            self.jump_volatility,
        );
        run_process(&process, x0, t0, t_end, n_steps, n_paths, parallel, seed)
    }

    fn __repr__(&self) -> String {
        format!(
            "MertonJumpDiffusion(mu={}, sigma={}, lambda={}, jump_mean={}, jump_vol={})",
            self.mu, self.sigma, self.lambda, self.jump_mean, self.jump_volatility
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module registration
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn register(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "stochastics")?;

    m.add_class::<PyTrajectories>()?;
    m.add_class::<PyGBM>()?;
    m.add_class::<PyABM>()?;
    m.add_class::<PyBM>()?;
    m.add_class::<PyOU>()?;
    m.add_class::<PyCIR>()?;
    m.add_class::<PyHW>()?;
    m.add_class::<PyEV>()?;
    m.add_class::<PyHL>()?;
    m.add_class::<PyBDT>()?;
    m.add_class::<PyMJD>()?;

    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("RustQuant.stochastics", m)?;

    Ok(())
}
