import pandas as pd
import pytest

from app.analytics import correlation_matrix, daily_returns, max_drawdown, sharpe_ratio, sma
from app.portfolio import simulate_portfolio, validate_weights


def test_returns_and_sma():
    prices = pd.Series([100, 110, 121], index=pd.date_range("2024-01-01", periods=3))
    assert daily_returns(prices).iloc[-1] == pytest.approx(0.1)
    assert sma(prices, 2).iloc[-1] == pytest.approx(115.5)


def test_risk_metrics():
    values = pd.Series([100, 110, 99])
    assert max_drawdown(values) == pytest.approx(-0.1)
    assert sharpe_ratio(pd.Series([0.01, 0.01])) == 0


def test_portfolio_validation_and_costs():
    prices = pd.DataFrame({"Gold": [100, 110], "Bitcoin": [100, 90]}, index=pd.date_range("2024-01-01", periods=2))
    try:
        validate_weights({"Gold": 0.8, "Bitcoin": 0.1}, ["Gold", "Bitcoin"])
        assert False
    except ValueError:
        pass
    result = simulate_portfolio(prices, {"Gold": 0.5, "Bitcoin": 0.5}, 1000, 0.01)
    assert result["transaction_cost"] == 10
    assert result["ending_value"] > 0


def test_correlation_shape():
    prices = pd.DataFrame({"Gold": [1, 2, 3], "Bitcoin": [3, 2, 1]})
    assert correlation_matrix(prices).loc["Gold", "Bitcoin"] == pytest.approx(1)
