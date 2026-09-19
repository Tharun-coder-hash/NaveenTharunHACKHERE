from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ASSETS = {"Gold": "GC=F", "Bitcoin": "BTC-USD", "NVIDIA": "NVDA"}
CACHE = Path(".cache")


def market_data(period: str = "5y", force: bool = False) -> tuple[pd.DataFrame, str]:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"market-{period}.csv"
    if not force and path.exists() and time.time() - path.stat().st_mtime < 3600:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if _is_complete(cached):
            return cached, "cache"

    series = []
    for name, ticker in ASSETS.items():
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False, group_by="column", threads=False)
        if raw.empty or "Close" not in raw:
            raise RuntimeError(f"Market data provider returned no observations for {name}")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        series.append(close.rename(name))
    close = pd.concat(series, axis=1).dropna(how="all")
    if not _is_complete(close):
        raise RuntimeError("Market data provider returned an incomplete asset set")
    close.to_csv(path)
    return close, "Yahoo Finance via yfinance"


def _is_complete(prices: pd.DataFrame) -> bool:
    return all(asset in prices.columns and prices[asset].dropna().size > 50 for asset in ASSETS)


def fred_events(limit: int = 20) -> list[dict]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    observations = pd.read_csv(io.StringIO(response.text), parse_dates=["observation_date"])
    observations = observations.dropna(subset=["CPIAUCSL"]).tail(limit)
    return [{"date": row.observation_date.strftime("%Y-%m-%d"), "category": "Inflation", "title": "CPI observation release", "description": f"CPI index observation: {row.CPIAUCSL:.1f}", "source": "FRED CPIAUCSL"} for row in observations.itertuples()]
