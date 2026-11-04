"""Build the data of the static online demo in its own environment (never the owner's data).

    DATABASE_URL=sqlite:///data/demo/demo.db DATA_DIR=data/demo FDA_JOB_RUNNER=inline \
    FDA_PROVIDERS=yfinance uv run python -m fda_api.demo_seed data/demo/subjects.json

Public market data only: a dozen liquid assets, markets, factors and capital-market
assumptions, then demo objects (two portfolios, a comparison, a universe, two strategies, a paper
account with history, an investor profile). Writes the subject list the exporter reads.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ASSETS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TLT", "IEF", "GLD", "BTC-USD"]
COMPANIES = ["AAPL", "MSFT", "NVDA", "AMZN"]
HEADERS = {"X-FDA-Client": "web"}
HYPOTHESIS = (
    "Trend persistence: medium-term momentum in broad equity indices (behavioural "
    "under-reaction), sized by volatility."
)
PROFILE = {
    "horizon": "10-20y",
    "income_stability": "stable",
    "emergency_fund": "3-6m",
    "savings_rate": "5-15%",
    "debt": "low",
    "liquidity_needs": "none",
    "drawdown_reaction": "hold",
    "max_loss": "20%",
    "preference": "balanced",
    "knowledge": "good",
    "experience": "3-10",
    "goal": {"amount": 300000, "years": 15, "current": 40000, "monthly_saving": 600},
}


def _ok(response: Any, what: str) -> Any:
    if response.status_code >= 400:
        print(f"  {what}: {response.status_code} {response.text[:200]}", file=sys.stderr)
        return None
    return response.json() if response.content else None


def seed(client: TestClient) -> dict[str, list[str]]:
    post = lambda path, **kw: client.post(path, headers=HEADERS, **kw)  # noqa: E731
    for s in ASSETS:
        print("prices", s)
        _ok(post(f"/v1/ingest/{s}"), f"ingest {s}")
    for s in COMPANIES:
        print("company", s)
        _ok(post(f"/v1/fundamentals/{s}/ingest"), f"fundamentals {s}")
        _ok(post(f"/v1/company-data/{s}/ingest"), f"company data {s}")
    for path in ["/v1/markets/ingest", "/v1/factors/US/ingest", "/v1/invest/ingest"]:
        print(path)
        _ok(post(path), path)
    for code in ["DTB3"]:
        _ok(post(f"/v1/series/{code}/ingest"), f"series {code}")

    portfolios = [
        _ok(
            post(
                "/v1/portfolios",
                json={
                    "name": "Demo · 60/40",
                    "holdings": [
                        {"symbol": "SPY", "weight": 0.6},
                        {"symbol": "IEF", "weight": 0.4},
                    ],
                },
            ),
            "portfolio 60/40",
        ),
        _ok(
            post(
                "/v1/portfolios",
                json={
                    "name": "Demo · Tech & gold",
                    "holdings": [
                        {"symbol": "AAPL", "weight": 0.2},
                        {"symbol": "MSFT", "weight": 0.2},
                        {"symbol": "NVDA", "weight": 0.2},
                        {"symbol": "AMZN", "weight": 0.2},
                        {"symbol": "GLD", "weight": 0.2},
                    ],
                },
            ),
            "portfolio tech",
        ),
    ]
    comparison = _ok(
        post(
            "/v1/comparisons",
            json={"name": "Demo · Growth vs market vs gold", "members": "QQQ, SPY, GLD"},
        ),
        "comparison",
    )
    universe = _ok(
        post("/v1/universes", json={"name": "Demo · Liquid assets", "symbols": "\n".join(ASSETS)}),
        "universe",
    )
    if universe:
        uid = universe["id"]
        _ok(post(f"/v1/universes/{uid}/ingest"), "universe ingest")
        _ok(post(f"/v1/universes/{uid}/snapshot"), "universe snapshot")
    strategies = [
        _ok(
            post(
                "/v1/strategies",
                json={
                    "name": "Demo · Carver trend on QQQ",
                    "hypothesis": HYPOTHESIS,
                    "spec": {"symbol": "QQQ", "template": "carver_trend"},
                },
            ),
            "strategy carver",
        ),
        _ok(
            post(
                "/v1/strategies",
                json={
                    "name": "Demo · EMA cross on SPY",
                    "hypothesis": HYPOTHESIS,
                    "spec": {
                        "symbol": "SPY",
                        "template": "ema_cross",
                        "grid": {"fast": [10, 20, 30], "slow": [100, 150, 200]},
                    },
                },
            ),
            "strategy ema",
        ),
    ]
    paper = _ok(
        post("/v1/paper", json={"name": "Demo · Paper account", "starting_cash": 100000}), "paper"
    )
    if paper:
        _paper_history(client, int(paper["id"]))
    profile = _ok(
        post("/v1/invest/profiles", json={"name": "Demo · Balanced saver", "answers": PROFILE}),
        "profile",
    )

    ids = lambda items: [str(x["id"]) for x in items if x]  # noqa: E731
    return {
        "asset": ASSETS,
        "portfolio": ids(portfolios),
        "comparison": ids([comparison]),
        "universe": ids([universe]),
        "strategy": ids(strategies),
        "paper": ids([paper]),
        "profile": ids([profile]),
        "market": ["global"],
    }


def _paper_history(client: TestClient, aid: int) -> None:
    """Back-dated orders so the account has fills, P&L and a statement (as in the API tests)."""
    from fda_data.paper import place_order, validate_order

    engine = client.app.state.engine  # type: ignore[attr-defined]
    start = date.today() - timedelta(days=200)
    orders = [
        ({"symbol": "SPY", "side": "buy", "quantity": 60}, start, 500.0),
        ({"symbol": "GLD", "side": "buy", "quantity": 80}, start + timedelta(days=20), 200.0),
        ({"symbol": "QQQ", "side": "buy", "quantity": 30}, start + timedelta(days=60), 450.0),
        ({"symbol": "GLD", "side": "sell", "quantity": 30}, start + timedelta(days=120), 220.0),
        (
            {
                "symbol": "SPY",
                "side": "sell",
                "quantity": 20,
                "order_type": "trailing_stop",
                "trail_pct": 0.05,
            },
            start + timedelta(days=90),
            520.0,
        ),
    ]
    for body, decided, price in orders:
        place_order(engine, aid, validate_order(body), decided, price)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/demo/subjects.json")
    from fda_api.main import app

    with TestClient(app) as client:
        subjects = seed(client)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subjects, indent=2), encoding="utf-8")
    print(json.dumps(subjects))


if __name__ == "__main__":
    main()
