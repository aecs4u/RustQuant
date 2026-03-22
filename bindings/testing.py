"""Test script for RustQuant Python bindings."""

from datetime import date
from pprint import pprint

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data module (existing)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RustQuant.data import Curve, CurveType, InterpolationMethod

dates = [date(2026, 1, 1), date(2027, 1, 2), date(2028, 1, 3), date(2029, 1, 4), date(2030, 1, 5)]
rates = [0.01, 0.015, 0.012, 0.014, 0.013]

crv = Curve(dates, rates, CurveType.Spot, InterpolationMethod.Linear)
print("=== Data Module ===")
print(f"Rate at 2026-06-01: {crv.get_rate(date(2026, 6, 1))}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Time module (existing)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RustQuant.time import Calendar, Market

cal = Calendar(Market.Australia)
print("\n=== Time Module ===")
print(f"Is 2026-01-03 a business day? {cal.is_business_day(date(2026, 1, 3))}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instruments module (NEW)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RustQuant.instruments import BlackScholesMerton, OptionType

print("\n=== Instruments Module ===")
call = BlackScholesMerton(
    underlying_price=100.0,
    strike_price=100.0,
    volatility=0.20,
    risk_free_rate=0.05,
    cost_of_carry=0.05,
    expiry_year=2027,
    expiry_month=3,
    expiry_day=21,
    option_type=OptionType.Call,
)

print(f"Call price: {call.price():.4f}")
print(f"Delta:      {call.delta():.4f}")
print(f"Gamma:      {call.gamma():.6f}")
print(f"Vega:       {call.vega():.4f}")
print(f"Theta:      {call.theta():.4f}")
print(f"Rho:        {call.rho():.4f}")

# Implied volatility roundtrip
iv = call.implied_volatility(call.price())
print(f"IV roundtrip: {iv:.6f} (expected ~0.20)")

# All Greeks at once
print("All Greeks:")
pprint(call.greeks())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stochastics module (NEW)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RustQuant.stochastics import (
    GeometricBrownianMotion,
    ArithmeticBrownianMotion,
    BrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HullWhite,
    MertonJumpDiffusion,
)

print("\n=== Stochastics Module ===")

gbm = GeometricBrownianMotion(mu=0.05, sigma=0.20)
traj = gbm.simulate(x0=100.0, t_end=1.0, n_steps=252, n_paths=5)
print(f"GBM: {len(traj.paths)} paths, {len(traj.times)} time steps")
for i, path in enumerate(traj.paths):
    print(f"  Path {i+1}: start={path[0]:.2f}, end={path[-1]:.2f}")

ou = OrnsteinUhlenbeck(mu=0.5, sigma=0.1, theta=2.0)
traj = ou.simulate(x0=0.5, t_end=5.0, n_steps=1000, n_paths=3)
print(f"\nOU (mean-reverting to 0.5): {len(traj.paths)} paths")
for i, path in enumerate(traj.paths):
    avg = sum(path) / len(path)
    print(f"  Path {i+1}: mean={avg:.4f}, final={path[-1]:.4f}")

mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lambda_=5.0, jump_mean=-0.02, jump_volatility=0.10)
traj = mjd.simulate(x0=100.0, t_end=1.0, n_steps=252, n_paths=3)
print(f"\nMerton Jump Diffusion: {len(traj.paths)} paths")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ML module (NEW)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RustQuant.ml import LinearRegression, LogisticRegression

print("\n=== ML Module ===")

# Linear regression
x_train = [
    [-0.084, -0.633, -0.399],
    [-0.983,  1.091, -0.468],
    [-1.875, -0.914,  0.327],
    [-0.186,  1.002, -0.413],
]
y_train = [-0.445, -1.848, -0.629, -0.861]

model = LinearRegression(x_train, y_train, method="qr")
print(f"Linear Regression: intercept={model.intercept:.4f}")
print(f"  Coefficients: {[f'{c:.4f}' for c in model.coefficients]}")

x_test = [
    [0.562, 0.596, -0.412],
    [0.663, 0.452, -0.294],
]
preds = model.predict(x_test)
print(f"  Predictions: {[f'{p:.4f}' for p in preds]}")

# Logistic regression
x_train = [
    [-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5], [-0.5, -0.8],
    [ 1.0,  0.5], [ 1.5,  1.0], [ 2.0,  1.5], [ 0.5,  1.2],
]
y_train = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

log_model = LogisticRegression(x_train, y_train)
print(f"\nLogistic Regression: {log_model.iterations} iterations")
print(f"  Coefficients: {[f'{c:.4f}' for c in log_model.coefficients]}")

x_test = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
probs = log_model.predict_proba(x_test)
labels = log_model.predict(x_test)
for i, (p, l) in enumerate(zip(probs, labels)):
    print(f"  Sample {i+1}: P(1)={p:.4f}, label={l:.0f}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Math module (NEW)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RustQuant.math import integrate_func, linspace, seq, cumsum
import math as pymath

print("\n=== Math Module ===")

# Numerical integration
result = integrate_func(lambda x: pymath.exp(-x**2 / 2) / pymath.sqrt(2 * pymath.pi), -5.0, 5.0)
print(f"Integral of N(0,1) PDF from -5 to 5: {result:.10f}")

# Sequences
print(f"linspace(0, 1, 5): {linspace(0.0, 1.0, 5)}")
print(f"seq(0, 2, 0.5):    {seq(0.0, 2.0, 0.5)}")
print(f"cumsum([1..5]):     {cumsum([1.0, 2.0, 3.0, 4.0, 5.0])}")

print("\n=== All tests passed! ===")
