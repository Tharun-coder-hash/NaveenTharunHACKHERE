from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analytics import correlation_matrix, daily_returns, ema, max_drawdown, sma, window_metrics
from .data import ASSETS, fred_events, market_data
from .portfolio import simulate_portfolio

load_dotenv()
app = FastAPI(title="FEICDE", description="Financial Event Impact & Counterfactual Decision Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")


class PortfolioRequest(BaseModel):
    weights: dict[str, float]
    capital: float = Field(default=10000, gt=0)
    transaction_cost: float = Field(default=0.001, ge=0)
    start: str | None = None
    end: str | None = None


def _prices(start: str | None = None, end: str | None = None):
    prices, source = market_data()
    if start:
        prices = prices.loc[pd.Timestamp(start):]
    if end:
        prices = prices.loc[:pd.Timestamp(end)]
    if prices.empty:
        raise HTTPException(404, "No observations in the selected date range")
    return prices, source


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/api/market")
def market(refresh: bool = False):
    prices, source = market_data(force=refresh)
    return {"assets": list(ASSETS), "source": source, "retrieved_at": datetime.utcnow().isoformat() + "Z", "series": {asset: [{"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)} for date, value in prices[asset].dropna().items()] for asset in ASSETS}}


@app.get("/api/events")
def events():
    try:
        return {"events": fred_events(), "source": "FRED CPIAUCSL"}
    except Exception as exc:
        raise HTTPException(502, f"Event source unavailable: {exc}") from exc


@app.get("/api/event-impact")
def event_impact(date: str, window: int = Query(default=5, ge=1, le=30)):
    prices, source = _prices()
    event_date = pd.Timestamp(date)
    index = prices.index[prices.index.get_indexer([event_date], method="nearest")[0]]
    position = prices.index.get_loc(index)
    start = prices.index[max(0, position - window)]
    end = prices.index[min(len(prices.index) - 1, position + window)]
    metrics = window_metrics(prices, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return {"event_date": date, "nearest_market_date": index.strftime("%Y-%m-%d"), "window": window, "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "metrics": metrics, "source": source, "interpretation": "Observed association around the selected date; this does not establish causation."}


@app.get("/api/quant")
def quant():
    prices, source = _prices()
    returns = daily_returns(prices)
    corr = correlation_matrix(prices)
    return {"source": source, "correlation": corr.round(3).to_dict(), "latest": {asset: {"price": round(float(prices[asset].dropna().iloc[-1]), 2), "daily_return": round(float(returns[asset].dropna().iloc[-1]), 5), "sma20": round(float(sma(prices[asset].dropna(), 20).iloc[-1]), 2), "ema20": round(float(ema(prices[asset].dropna(), 20).iloc[-1]), 2)} for asset in ASSETS}}


@app.post("/api/replay")
def replay(request: PortfolioRequest):
    prices, source = _prices(request.start, request.end)
    assets = list(ASSETS)
    scenarios = {"User decision": request.weights, "Equal weight": {asset: 1 / len(assets) for asset in assets}, "Gold tilt": {"Gold": 0.5, "Bitcoin": 0.25, "NVIDIA": 0.25}}
    try:
        results = {name: simulate_portfolio(prices, weights, request.capital, request.transaction_cost) for name, weights in scenarios.items()}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"scenarios": results, "source": source, "assumption": "Buy-and-hold weights, close-to-close execution, one initial transaction cost; historical comparison only."}


@app.get("/api/backtest")
def backtest(strategy: str = Query(default="sma", pattern="^(sma|ema|momentum|mean-reversion)$")):
    prices, source = _prices()
    asset = prices["NVIDIA"].dropna()
    fast = sma(asset, 20) if strategy == "sma" else ema(asset, 20)
    slow = sma(asset, 50) if strategy == "sma" else ema(asset, 50)
    if strategy == "momentum":
        signal = asset.pct_change(20) > 0
    elif strategy == "mean-reversion":
        signal = asset < sma(asset, 20)
    else:
        signal = fast > slow
    position = signal.shift(1).fillna(False).astype(int)
    returns = asset.pct_change().fillna(0) * position
    equity = 10000 * (1 + returns).cumprod()
    return {"strategy": strategy, "asset": "NVIDIA", "source": source, "ending_value": round(float(equity.iloc[-1]), 2), "return": round(float(equity.iloc[-1] / 10000 - 1), 5), "max_drawdown": round(max_drawdown(equity), 5), "equity": [{"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)} for date, value in equity.tail(500).items()]}


@app.get("/api/insights")
def insights(date: str, window: int = Query(default=5, ge=1, le=30)):
    impact = event_impact(date, window)
    metrics = impact["metrics"]
    leader = max(metrics, key=lambda asset: abs(metrics[asset]["return"]))
    direction = "gained" if metrics[leader]["return"] >= 0 else "declined"
    return {
        "mode": "deterministic fallback",
        "summary": f"Across the {window}-trading-day window, {leader} had the largest observed absolute return and {direction} {abs(metrics[leader]['return']) * 100:.2f}%.",
        "evidence": metrics,
        "limitations": "This is a historical association around an event date. It is not causal evidence, investment advice, or a forecast, and the backend calculations are not independently invented by an AI model.",
    }
