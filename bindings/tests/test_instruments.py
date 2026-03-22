"""Tests for the RustQuant.instruments module."""

import math

from RustQuant.instruments import BlackScholesMerton, OptionType


def _make_call(**kwargs):
    defaults = dict(
        underlying_price=100.0, strike_price=100.0, volatility=0.20,
        risk_free_rate=0.05, cost_of_carry=0.05,
        expiry_year=2027, expiry_month=6, expiry_day=1,
        option_type=OptionType.Call,
    )
    defaults.update(kwargs)
    return BlackScholesMerton(**defaults)


def test_call_price_positive():
    assert _make_call().price() > 0


def test_put_price_positive():
    assert _make_call(option_type=OptionType.Put).price() > 0


def test_put_call_parity():
    call = _make_call()
    put = _make_call(option_type=OptionType.Put)
    lhs = call.price() - put.price()
    rhs = 100.0 - 100.0 * math.exp(-0.05 * 1.0)
    assert abs(lhs - rhs) < 1.0  # approximate (date-based T)


def test_delta_range():
    delta = _make_call().delta()
    assert 0 < delta < 1


def test_greeks_dict():
    greeks = _make_call().greeks()
    assert "price" in greeks
    assert "delta" in greeks
    assert "gamma" in greeks
    assert "vega" in greeks


def test_implied_volatility_roundtrip():
    call = _make_call()
    price = call.price()
    iv = call.implied_volatility(price)
    assert abs(iv - 0.20) < 0.01
