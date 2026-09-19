# FEICDE

Financial Event Impact & Counterfactual Decision Engine: a historical, multi-asset analytics dashboard for Gold, Bitcoin, and NVIDIA.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

The service retrieves real market observations from Yahoo Finance through `yfinance` and event observations from FRED (`CPIAUCSL`). Data is cached for one hour in `.cache/`. No API key was present in the supplied empty workspace. An optional `.env` can contain an authorized OpenAI-compatible key for a future explanation adapter; the current dashboard remains fully deterministic without one.

## Tests

```powershell
python -m pytest -q
```

## API surface

- `GET /api/market`: cached OHLC-derived close series and retrieval metadata
- `GET /api/events`: FRED CPI observation-release proxies
- `GET /api/event-impact?date=YYYY-MM-DD&window=5`: trading-day event-window metrics
- `GET /api/quant`: moving averages and return correlation matrix
- `POST /api/replay`: user, equal-weight, and Gold-tilt historical portfolio scenarios
- `GET /api/backtest?strategy=sma|ema|momentum|mean-reversion`: NVDA strategy replay

## Assumptions and limitations

The engine uses adjusted close data, close-to-close returns, buy-and-hold weights for replay, and a single initial transaction cost. Event dates are aligned to the nearest market observation. FRED CPI observation dates are event proxies rather than a complete official release calendar. Association around an event is not proof of causality, and historical results do not guarantee future returns. AI explanations should receive only validated backend metrics and must preserve those limitations.

## Demo flow

1. Load the dashboard and confirm the live-data timestamp.
2. Select a FRED inflation observation and change the event window.
3. Compare the event-window returns across Gold, Bitcoin, and NVIDIA.
4. Change replay weights and compare ending value, volatility, drawdown, and costs.
5. Inspect the return correlation matrix and run each NVDA strategy.
6. Use the Methodology section to explain the data and no-look-ahead execution assumptions.
