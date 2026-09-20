import pandas as pd
import pytest

from app.analytics import correlation_matrix, daily_returns, max_drawdown, sharpe_ratio, sma
from app.portfolio import simulate_portfolio, validate_weights
from app.reporting import build_report


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


def test_report_generation():
    report = build_report(
        event_date="2024-01-02",
        nearest_market_date="2024-01-02",
        window=5,
        from_date="2024-01-01",
        to_date="2024-01-15",
        metrics={
            "Gold": {"return": 0.02, "volatility": 0.12, "max_drawdown": -0.04},
            "Bitcoin": {"return": -0.01, "volatility": 0.18, "max_drawdown": -0.07},
            "NVIDIA": {"return": 0.07, "volatility": 0.22, "max_drawdown": -0.09},
        },
        scenario_name="User decision",
        portfolio={
            "starting_capital": 10000,
            "ending_value": 11000,
            "return": 0.1,
            "volatility": 0.18,
            "max_drawdown": -0.08,
            "transaction_cost": 10,
        },
    )
    assert report["title"] == "FEICDE Market Event Report"
    assert "Gold" in report["summary"]
    assert report["portfolio"]["ending_value"] == pytest.approx(11000)
    assert "2024-01-02" in report["body"]
    assert "2024-01-15" in report["body"]
