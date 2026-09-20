from __future__ import annotations

from typing import Any


def build_report(
    event_date: str,
    nearest_market_date: str,
    window: int,
    metrics: dict[str, dict[str, float]],
    scenario_name: str = "User decision",
    portfolio: dict[str, Any] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, Any]:
    if not metrics:
        metrics = {"Gold": {"return": 0.0, "volatility": 0.0, "max_drawdown": 0.0}}

    leader = max(metrics, key=lambda asset: abs(float(metrics[asset].get("return", 0.0))))
    leader_return = float(metrics[leader].get("return", 0.0))
    leader_volatility = float(metrics[leader].get("volatility", 0.0))
    leader_drawdown = float(metrics[leader].get("max_drawdown", 0.0))
    portfolio = portfolio or {}

    summary = (
        f"Across the {window}-trading-day window around {event_date}, Gold, Bitcoin, and NVIDIA were compared; "
        f"{leader} had the largest observed absolute return at {leader_return * 100:.2f}% with volatility of "
        f"{leader_volatility * 100:.2f}% and max drawdown of {leader_drawdown * 100:.2f}%."
    )
    body_lines = [
        "FEICDE Market Event Report",
        f"Event date: {event_date}",
        f"Nearest market date: {nearest_market_date}",
        f"Window: {window} trading days",
        f"From: {from_date or event_date}",
        f"To: {to_date or event_date}",
        "",
        "Observed event-window metrics:",
    ]
    for asset, values in metrics.items():
        body_lines.append(
            f"- {asset}: return {float(values.get('return', 0.0)) * 100:.2f}%, volatility {float(values.get('volatility', 0.0)) * 100:.2f}%, "
            f"max drawdown {float(values.get('max_drawdown', 0.0)) * 100:.2f}%"
        )

    body_lines.extend([
        "",
        f"Scenario checked: {scenario_name}",
    ])
    if portfolio:
        body_lines.extend([
            f"Starting capital: {portfolio.get('starting_capital', 0):,.2f}",
            f"Ending value: {portfolio.get('ending_value', 0):,.2f}",
            f"Total return: {portfolio.get('return', 0.0) * 100:.2f}%",
            f"Volatility: {portfolio.get('volatility', 0.0) * 100:.2f}%",
            f"Max drawdown: {portfolio.get('max_drawdown', 0.0) * 100:.2f}%",
            f"Transaction cost: {portfolio.get('transaction_cost', 0):,.2f}",
        ])

    return {
        "title": "FEICDE Market Event Report",
        "summary": summary,
        "body": "\n".join(body_lines),
        "metrics": metrics,
        "portfolio": portfolio,
        "scenario_name": scenario_name,
    }


def report_to_text(report: dict[str, Any]) -> str:
    return f"{report['title']}\n\n{report['summary']}\n\n{report['body']}\n"
