from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.sort_index().pct_change().dropna(how="all")


def sma(prices: pd.Series, window: int = 20) -> pd.Series:
    _validate_window(window)
    return prices.rolling(window).mean()


def ema(prices: pd.Series, window: int = 20) -> pd.Series:
    _validate_window(window)
    return prices.ewm(span=window, adjust=False).mean()


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    clean = returns.dropna()
    return float(clean.std(ddof=1) * np.sqrt(periods)) if len(clean) > 1 else 0.0


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    clean = returns.dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return 0.0
    return float(clean.mean() / clean.std(ddof=1) * np.sqrt(periods))


def max_drawdown(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return 0.0
    drawdown = clean / clean.cummax() - 1
    return float(drawdown.min())


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return daily_returns(prices).corr().fillna(0)


def window_metrics(prices: pd.DataFrame, start: str, end: str) -> dict:
    window = prices.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna(how="all")
    if window.empty:
        raise ValueError("No market observations exist in the selected window")
    returns = daily_returns(window)
    result = {}
    for asset in window.columns:
        series = window[asset].dropna()
        asset_returns = returns[asset].dropna() if asset in returns else pd.Series(dtype=float)
        result[asset] = {
            "start_price": float(series.iloc[0]),
            "end_price": float(series.iloc[-1]),
            "return": float(series.iloc[-1] / series.iloc[0] - 1),
            "volatility": annualized_volatility(asset_returns),
            "max_drawdown": max_drawdown(series),
        }
    return result


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
