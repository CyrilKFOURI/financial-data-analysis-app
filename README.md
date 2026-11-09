# Holdout

*by Cyfouri.* In quantitative finance the **holdout** is the data a model never saw while it was
built. It is where a result has to hold before anyone trusts it, and it is the rule of this app:
measure, validate out of sample, then conclude. Live demo: https://holdout-app.vercel.app

An institutional-grade ("Zero-Slop") analysis workbench for every kind of listed asset:
stocks, ETFs, funds, indices, crypto, FX, futures, and rates. One analytics engine (≈ 100
analyses, every formula from `docs/ANALYTICS_CATALOG.md`) serves a docked, keyboard-first web
UI. Built for one owner on a Windows laptop, portable to a home server.

**Status: V1 (core analysis, milestones J0–J6) complete.** Next: V2 company & portfolio
analysis (valuation, peers, portfolios). See `PLAN.md`.

## What V1 does

| Page | Content |
|---|---|
| Overview | Price, profile & caveats (type, why, conventions, one-click correction), key figures, risk card, health scorecard, data status |
| Price & Trend | Candles/OHLC/Heikin-Ashi, 7 moving-average kinds, Bollinger/Keltner/Donchian, period returns, base-100, MA structure |
| Returns & Statistics | Means & volatility drag, dispersion, up/down, distribution with Normal/Student-t/KDE, moments & normality tests, QQ, tails, calendar heatmap, quantiles, rolling stats, Sharpe stability (bootstrap), seasonality (BH-FDR), OHLC volatility estimators |
| Time Series Lab | 23 AFML-grade analyses: stationarity battery, serial dependence, long memory, fractional differentiation (min d*), GARCH family, OU half-life, ARIMA, regimes, Kalman, CUSUM/Bai–Perron breaks, SADF/GSADF bubbles, CUSUM events, entropy, microstructure, pairs, lead–lag, placebo |
| Risk | Drawdowns & episodes, Sharpe/Sortino/Calmar/Omega/PSR, VaR & ES (4 models), VaR backtests (Kupiec, Christoffersen, traffic light), FHS & Monte Carlo VaR, EVT, CAPM & rolling beta, benchmark-relative set, stress windows, leveraged decay, CDaR |
| Financials | Point-in-time statements (SEC XBRL for US filers, yfinance elsewhere): income, balance, cash flow (FY / Q / TTM), Sankey, waterfalls, cash bridge, balance structure, debt ladder, QC & coverage |
| Analysis | Growth, margins, ROIC/ROE (DuPont), liquidity, solvency, efficiency, cash quality, capital allocation, per-share, sector KPIs; health scorecard; Piotroski, Altman, Beneish, Montier, Ohlson, dividend safety, refinancing exposure |

V1–V3 *measure and explain*: no buy/sell verdicts or price targets.

## Quick start (Windows, PowerShell or Git Bash)

```bash
uv sync
corepack pnpm --dir apps/web install
cp .env.example .env          # set SEC_USER_AGENT="Your Name your@email" (SEC requirement)
docker compose up -d          # Postgres on 127.0.0.1:5433, Redis on 127.0.0.1:6379
set -a; . ./.env; set +a; uv run alembic upgrade head
uv run fastapi dev apps/api/src/fda_api/main.py      # API on :8000
uv run arq fda_worker.settings.WorkerSettings         # ingestion worker (optional: FDA_JOB_RUNNER=inline)
corepack pnpm --dir apps/web dev                      # UI on :5173
```

Open the UI, press **Ctrl K**, type a ticker (`AAPL`, `MC.PA`, `BTC-USD`, `^GSPC`…) and press
Enter: prices (and, for companies, statements) are ingested automatically.

## AI analyst (any LLM)
1. Put any key you have in `.env`: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `XAI_API_KEY` or `OPENROUTER_API_KEY`. You can also skip keys and run Ollama or LM Studio locally.
2. Open **AI · Analyst console** (`/ai`) and pick the provider and model.

Every number in an answer comes from the app's tools and is checked against the result it cites; unverified numbers are shown in red. Details: `docs/features/251-252-agent-runtime-console/spec.md`.

## Run it as a service (laptop, home server, cloud)
`docker compose --profile app up -d --build` runs the whole app in containers on http://127.0.0.1:8080.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- the home-server settings: LAN exposure, HTTPS from a domain name, shared password;
- the cloud override;
- nightly backups and the restore drill: `python -m fda_data.backup drill`.

## Checks

```bash
uv run pytest                      # engine, data, API, worker (offline: recorded provider fixtures)
uv run pytest -m live              # opt-in smoke tests against real providers
uv run ruff check . && uv run mypy
corepack pnpm --dir apps/web test && corepack pnpm --dir apps/web e2e
```

## Use the workbench from Claude (MCP)

Every analysis, plus the data, navigation and paper-trading actions, is exposed as a tool over the Model Context Protocol. The protocol is implemented on the standard library, so it needs no extra dependency.

1. In `.env`, set `FDA_MCP_TOKEN` to a random value:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   Keep `FDA_MCP_READ_ONLY=1` for read-only access.
2. Start the API: `uv run fastapi dev apps/api/src/fda_api/main.py`. It listens on 127.0.0.1:8000.
3. Add the stdio bridge to Claude Desktop (`claude_desktop_config.json`), replacing the path and the token:

```json
{
  "mcpServers": {
    "holdout": {
      "command": "uv",
      "args": ["run", "--directory", "C:/path/to/4_Financial_Data_Analysis_App",
               "python", "-m", "fda_api.mcp_stdio"],
      "env": {"FDA_API_URL": "http://127.0.0.1:8000", "FDA_MCP_TOKEN": "<token>"}
    }
  }
}
```

   For Claude Code, run this instead:

   ```bash
   claude mcp add holdout -e FDA_MCP_TOKEN=<token> -- uv run --directory <repo> python -m fda_api.mcp_stdio
   ```

   HTTP clients can also post JSON-RPC to `http://127.0.0.1:8000/mcp` with `Authorization: Bearer <token>`.

Results are compact summaries with a `result_id`. Cite numbers as `[r:result_id]`; the evaluation verifier checks each number against the result it cites.

With `FDA_MCP_READ_ONLY=0`, write tools such as paper orders and new strategies are exposed. They only *propose* an action. You approve or reject it under **Agent actions** (`/ai/actions`), and every decision stays in the audit log. There is no live-order tool.

## Layout

`packages/fda_core` engine (pure) · `packages/fda_data` providers, lake, ingestion, mappings ·
`packages/fda_contracts` wire models · `apps/api` FastAPI · `apps/worker` arq · `apps/web` React
SPA · `infra/` migrations · `docs/` specs (start at `docs/00_INDEX.md`) · `PLAN.md` milestones.

User guide: `docs/USER_GUIDE.md`.
