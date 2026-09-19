from __future__ import annotations

import pandas as pd

from .analytics import annualized_volatility, max_drawdown, sharpe_ratio


def validate_weights(weights: dict[str, float], assets: list[str]) -> dict[str, float]:
    if set(weights) != set(assets):
        raise ValueError("Weights must include exactly the selected assets")
    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative")
    total = sum(weights.values())
    if abs(total - 1) > 1e-6:
        raise ValueError("Weights must sum to 1")
    return weights


def simulate_portfolio(prices: pd.DataFrame, weights: dict[str, float], capital: float = 10000, transaction_cost: float = 0.001) -> dict:
    validate_weights(weights, list(prices.columns))
    if capital <= 0 or transaction_cost < 0:
        raise ValueError("Capital must be positive and transaction cost cannot be negative")
    prices = prices[list(weights)].dropna(how="all").ffill().dropna()
    returns = prices.pct_change().fillna(0)
    portfolio_returns = returns.mul(pd.Series(weights)).sum(axis=1)
    equity = capital * (1 + portfolio_returns).cumprod()
    cost = capital * transaction_cost if len(prices) else 0
    equity.iloc[0] -= cost
    contributions = {asset: float(weights[asset] * (prices[asset].iloc[-1] / prices[asset].iloc[0] - 1)) for asset in weights}
    return {
        "equity": [{"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)} for date, value in equity.items()],
        "starting_capital": capital,
        "ending_value": round(float(equity.iloc[-1]), 2),
        "return": round(float(equity.iloc[-1] / capital - 1), 6),
        "volatility": round(annualized_volatility(portfolio_returns), 6),
        "sharpe": round(sharpe_ratio(portfolio_returns), 6),
        "max_drawdown": round(max_drawdown(equity), 6),
        "transaction_cost": round(cost, 2),
        "contributions": contributions,
    }
