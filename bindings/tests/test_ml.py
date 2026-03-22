"""Tests for the RustQuant.ml module."""

from RustQuant.ml import LinearRegression, LogisticRegression


def test_linear_regression_fit():
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    model = LinearRegression(x, y)
    assert model.intercept is not None
    assert len(model.coefficients) >= 2


def test_linear_regression_predict():
    x = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.0, 4.0, 6.0, 8.0]
    model = LinearRegression(x, y)
    preds = model.predict([[5.0], [6.0]])
    assert len(preds) == 2
    assert abs(preds[0] - 10.0) < 1.0


def test_linear_regression_svd():
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    y = [1.0, 1.0, 2.0]
    model = LinearRegression(x, y, method="svd")
    assert len(model.coefficients) >= 2


def test_logistic_regression_fit():
    x = [
        [-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5], [-0.5, -0.8],
        [1.0, 0.5], [1.5, 1.0], [2.0, 1.5], [0.5, 1.2],
    ]
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    model = LogisticRegression(x, y)
    assert model.iterations >= 0
    assert len(model.coefficients) == 3  # intercept + 2 features


def test_logistic_regression_predict():
    x = [
        [-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5], [-0.5, -0.8],
        [1.0, 0.5], [1.5, 1.0], [2.0, 1.5], [0.5, 1.2],
    ]
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    model = LogisticRegression(x, y)
    probs = model.predict_proba([[-3.0, -2.0], [3.0, 2.0]])
    assert probs[0] < 0.5
    assert probs[1] > 0.5
