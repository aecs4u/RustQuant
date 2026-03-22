"""Tests for the RustQuant.math module."""

import math

from RustQuant.math import integrate_func, linspace, seq, cumsum


def test_integrate_constant():
    result = integrate_func(lambda x: 1.0, 0.0, 1.0)
    assert abs(result - 1.0) < 1e-6


def test_integrate_polynomial():
    result = integrate_func(lambda x: x * x, 0.0, 1.0)
    assert abs(result - 1.0 / 3.0) < 1e-6


def test_integrate_normal_pdf():
    def normal_pdf(x):
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    result = integrate_func(normal_pdf, -5.0, 5.0)
    assert abs(result - 1.0) < 1e-4


def test_linspace():
    xs = linspace(0.0, 1.0, 5)
    assert len(xs) == 5
    assert abs(xs[0] - 0.0) < 1e-10
    assert abs(xs[-1] - 1.0) < 1e-10


def test_seq():
    xs = seq(0.0, 1.0, 0.25)
    assert len(xs) == 5
    assert abs(xs[0] - 0.0) < 1e-10


def test_cumsum():
    result = cumsum([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == [1.0, 3.0, 6.0, 10.0, 15.0]
