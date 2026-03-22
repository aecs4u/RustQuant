// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Python bindings for RustQuant instruments (option pricing).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use pyo3::prelude::*;
use time::Date;
use RustQuant::instruments::options::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TypeFlag enum
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Option type: Call or Put.
#[pyclass(name = "OptionType", eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyOptionType {
    Call,
    Put,
}

impl From<PyOptionType> for TypeFlag {
    fn from(val: PyOptionType) -> Self {
        match val {
            PyOptionType::Call => TypeFlag::Call,
            PyOptionType::Put => TypeFlag::Put,
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Black-Scholes-Merton European Option
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Generalised Black-Scholes-Merton European Option.
///
/// Supports multiple models via the cost_of_carry parameter:
/// - b = r: Black-Scholes 1973
/// - b = r - q: Merton 1973 (continuous dividend yield)
/// - b = 0: Black 1976 (futures)
/// - b = r_d - r_f: Garman-Kohlhagen 1983 (FX)
#[pyclass(name = "BlackScholesMerton")]
pub struct PyBSM {
    underlying_price: f64,
    strike_price: f64,
    volatility: f64,
    risk_free_rate: f64,
    cost_of_carry: f64,
    expiration_date: Date,
    option_type: TypeFlag,
}

impl PyBSM {
    fn bsm(&self) -> BlackScholesMerton {
        BlackScholesMerton::new(
            self.cost_of_carry,
            self.underlying_price,
            self.strike_price,
            self.volatility,
            self.risk_free_rate,
            None,
            self.expiration_date,
            self.option_type,
        )
    }
}

#[pymethods]
impl PyBSM {
    /// Create a new Black-Scholes-Merton option.
    ///
    /// Args:
    ///     underlying_price: Current price of the underlying (S).
    ///     strike_price: Strike price (K).
    ///     volatility: Annualized volatility (sigma).
    ///     risk_free_rate: Risk-free interest rate (r).
    ///     cost_of_carry: Cost of carry (b). Set b=r for vanilla BS.
    ///     expiry_year: Expiration year.
    ///     expiry_month: Expiration month (1-12).
    ///     expiry_day: Expiration day.
    ///     option_type: OptionType.Call or OptionType.Put.
    #[new]
    #[pyo3(signature = (underlying_price, strike_price, volatility, risk_free_rate, cost_of_carry, expiry_year, expiry_month, expiry_day, option_type))]
    fn new(
        underlying_price: f64,
        strike_price: f64,
        volatility: f64,
        risk_free_rate: f64,
        cost_of_carry: f64,
        expiry_year: i32,
        expiry_month: u8,
        expiry_day: u8,
        option_type: PyOptionType,
    ) -> PyResult<Self> {
        let month = time::Month::try_from(expiry_month)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let expiration_date = Date::from_calendar_date(expiry_year, month, expiry_day)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self {
            underlying_price,
            strike_price,
            volatility,
            risk_free_rate,
            cost_of_carry,
            expiration_date,
            option_type: option_type.into(),
        })
    }

    /// Option price.
    fn price(&self) -> f64 {
        self.bsm().price()
    }

    /// Delta: dPrice/dSpot.
    fn delta(&self) -> f64 {
        self.bsm().delta()
    }

    /// Gamma: d²Price/dSpot².
    fn gamma(&self) -> f64 {
        self.bsm().gamma()
    }

    /// Vega: dPrice/dVol.
    fn vega(&self) -> f64 {
        self.bsm().vega()
    }

    /// Theta: dPrice/dTime.
    fn theta(&self) -> f64 {
        self.bsm().theta()
    }

    /// Rho: dPrice/dRate.
    fn rho(&self) -> f64 {
        self.bsm().rho()
    }

    /// Vanna: d²Price/(dSpot dVol).
    fn vanna(&self) -> f64 {
        self.bsm().vanna()
    }

    /// Charm: dDelta/dTime.
    fn charm(&self) -> f64 {
        self.bsm().charm()
    }

    /// Vomma: d²Price/dVol².
    fn vomma(&self) -> f64 {
        self.bsm().vomma()
    }

    /// Zomma: dGamma/dVol.
    fn zomma(&self) -> f64 {
        self.bsm().zomma()
    }

    /// Speed: dGamma/dSpot.
    fn speed(&self) -> f64 {
        self.bsm().speed()
    }

    /// Colour: dGamma/dTime.
    fn colour(&self) -> f64 {
        self.bsm().colour()
    }

    /// Lambda (elasticity).
    fn lambda(&self) -> f64 {
        self.bsm().lambda()
    }

    /// Implied volatility from a market price.
    fn implied_volatility(&self, market_price: f64) -> f64 {
        self.bsm().implied_volatility(market_price)
    }

    /// Return all Greeks as a dict.
    fn greeks(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        let bsm = self.bsm();
        dict.set_item("price", bsm.price())?;
        dict.set_item("delta", bsm.delta())?;
        dict.set_item("gamma", bsm.gamma())?;
        dict.set_item("vega", bsm.vega())?;
        dict.set_item("theta", bsm.theta())?;
        dict.set_item("rho", bsm.rho())?;
        dict.set_item("vanna", bsm.vanna())?;
        dict.set_item("vomma", bsm.vomma())?;
        dict.set_item("charm", bsm.charm())?;
        dict.set_item("zomma", bsm.zomma())?;
        dict.set_item("speed", bsm.speed())?;
        dict.set_item("colour", bsm.colour())?;
        dict.set_item("lambda", bsm.lambda())?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        let flag = match self.option_type {
            TypeFlag::Call => "Call",
            TypeFlag::Put => "Put",
        };
        format!(
            "BlackScholesMerton(S={}, K={}, vol={}, r={}, b={}, type={})",
            self.underlying_price, self.strike_price, self.volatility,
            self.risk_free_rate, self.cost_of_carry, flag
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module registration
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn register(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "instruments")?;

    m.add_class::<PyOptionType>()?;
    m.add_class::<PyBSM>()?;

    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("RustQuant.instruments", m)?;

    Ok(())
}
