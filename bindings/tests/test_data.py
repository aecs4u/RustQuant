"""Tests for the RustQuant.data module."""

from datetime import date

from RustQuant.data import Curve, CurveType, InterpolationMethod


def test_curve_creation():
    dates = [date(2026, 1, 1), date(2027, 1, 1), date(2028, 1, 1)]
    rates = [0.01, 0.015, 0.02]
    curve = Curve(dates, rates, CurveType.Spot, InterpolationMethod.Linear)
    assert curve.len() == 3
    assert not curve.is_empty()


def test_curve_interpolation():
    dates = [date(2026, 1, 1), date(2028, 1, 1)]
    rates = [0.01, 0.03]
    curve = Curve(dates, rates, CurveType.Spot, InterpolationMethod.Linear)
    mid = curve.get_rate(date(2027, 1, 1))
    assert mid is not None
    assert abs(mid - 0.02) < 0.01
