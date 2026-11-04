"""Portfolio engine (module M15, BACKEND_SCHEMA §6.2): pure path construction from a definition.

Weights mode simulates target weights with a rebalancing assumption (D-PF-2) and costs;
transactions mode replays a ledger with multi-currency cash and lots (D-PF-3). Both yield a
`PortfolioPath`: the base-currency value, the TWR NAV (P-003, flows removed), start-of-day
weights and the holdings' base-currency total returns — the inputs of every P-/C- analysis and,
via the NAV, of every price/risk/time-series analysis (the Subject abstraction).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from fda_core.registry import InsufficientDataError

CASH = "CASH"
LotMethod = Literal["fifo", "lifo", "average"]
CASH_TYPES = {"DEPOSIT", "WITHDRAWAL", "DIVIDEND", "INTEREST", "FEE"}
TX_TYPES = CASH_TYPES | {"BUY", "SELL", "SPLIT_ADJ", "TRANSFER_IN", "TRANSFER_OUT"}


@dataclass(frozen=True, slots=True)
class Holding:
    symbol: str
    weight: float
    is_cash: bool = False


@dataclass(frozen=True, slots=True)
class Rebalance:
    """D-PF-2: `calendar` (M/Q/A), `threshold` (absolute band) or `none` (drift)."""

    type: Literal["calendar", "threshold", "none"] = "calendar"
    freq: Literal["M", "Q", "A"] = "M"
    band: float = 0.05

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | None) -> "Rebalance":
        raw = raw or {}
        return cls(
            type=raw.get("type", "calendar"),
            freq=raw.get("freq", "M"),
            band=float(raw.get("band", 0.05)),
        )


@dataclass(slots=True)
class PortfolioPath:
    value: pd.Series  # base-currency market value, cash included
    nav: pd.Series  # time-weighted index (100 at start): flows removed
    weights: pd.DataFrame  # start-of-day weights (previous close, after trades)
    returns: pd.DataFrame  # holdings' daily total returns in base currency (cash 0)
    local_returns: pd.DataFrame  # same in listing currency (FX contribution, P-020)
    flows: pd.Series  # external flows in base currency (+ in, − out), on their date
    turnover: pd.Series  # one-way traded fraction of value
    costs: pd.Series  # costs in base currency
    holdings_value: pd.DataFrame | None = None  # end-of-day base value per holding (+ CASH)
    close_base: pd.DataFrame | None = None  # split-adjusted close in base (ledger mode)
    quantities: pd.DataFrame | None = None
    cash: pd.Series | None = None  # base-currency cash
    lots: list[dict[str, Any]] = field(default_factory=list)  # open lots
    realised: list[dict[str, Any]] = field(default_factory=list)  # closed lot slices
    income: list[dict[str, Any]] = field(default_factory=list)  # dividends & interest
    warnings: list[str] = field(default_factory=list)


def calendar_of(prices: Mapping[str, pd.DataFrame], seven_day: set[str]) -> pd.DatetimeIndex:
    """Union of the sessions of non-7-day holdings (crypto is sampled on them); 7-day only
    portfolios use every day."""
    pool = [f.index for s, f in prices.items() if s not in seven_day] or [
        f.index for f in prices.values()
    ]
    index = pool[0]
    for other in pool[1:]:
        index = index.union(other)
    return pd.DatetimeIndex(index).sort_values()


def base_frames(
    prices: Mapping[str, pd.DataFrame],
    currencies: Mapping[str, str],
    fx: Mapping[str, pd.Series],
    base: str,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(total-return price in base, close in base, total-return price in local) on `index`,
    carried forward over holidays of a market."""
    adj_base, close_base, adj_local = {}, {}, {}
    for symbol, frame in prices.items():
        adj = frame["adj_close"].reindex(index.union(frame.index)).ffill().reindex(index)
        close = frame["close"].reindex(index.union(frame.index)).ffill().reindex(index)
        ccy = currencies.get(symbol, base)
        rate = fx_rate(fx, ccy, base, index)
        adj_base[symbol], close_base[symbol], adj_local[symbol] = adj * rate, close * rate, adj
    return pd.DataFrame(adj_base), pd.DataFrame(close_base), pd.DataFrame(adj_local)


def fx_rate(fx: Mapping[str, pd.Series], ccy: str, base: str, index: pd.DatetimeIndex) -> pd.Series:
    """Units of base per unit of `ccy` on `index` (last known rate, never a future one)."""
    if ccy == base:
        return pd.Series(1.0, index=index)
    if ccy not in fx:
        raise InsufficientDataError(f"Ingest {ccy}{base}=X to convert {ccy} holdings to {base}")
    series = fx[ccy]
    return series.reindex(index.union(series.index)).ffill().reindex(index)


# ── Weights mode (P-001) ─────────────────────────────────────────────────────


def _is_rebalance_day(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Last session of each month / quarter / year (rebalance at that close)."""
    periods = {"M": index.to_period("M"), "Q": index.to_period("Q"), "A": index.to_period("Y")}[
        freq
    ]
    p = np.asarray(periods.astype(str))
    return np.asarray(np.r_[p[1:] != p[:-1], False], dtype=bool)


def simulate_weights(
    holdings: list[Holding],
    prices_base: pd.DataFrame,
    local: pd.DataFrame,
    policy: Rebalance,
    cost_bps: float = 0.0,
    initial_value: float = 100_000.0,
    start: date | None = None,
) -> PortfolioPath:
    total = sum(h.weight for h in holdings)
    if abs(total - 1) > 1e-6:
        raise InsufficientDataError(f"Weights sum to {total:.6f}, not 1")
    assets = [h.symbol for h in holdings if not h.is_cash]
    frame = prices_base[assets]
    first_full = frame.dropna().index
    if first_full.empty:
        raise InsufficientDataError("The holdings have no common price history")
    begin = max(first_full[0], pd.Timestamp(start)) if start else first_full[0]
    frame = frame.loc[frame.index >= begin].ffill()
    warnings = []
    if start and first_full[0] > pd.Timestamp(start):
        late = max(assets, key=lambda s: prices_base[s].first_valid_index())
        warnings.append(f"Starts {begin.date()} (first common price, {late} history begins then)")
    columns = assets + ([CASH] if any(h.is_cash for h in holdings) else [])
    target = np.array(
        [
            sum(h.weight for h in holdings if (h.symbol if not h.is_cash else CASH) == c)
            for c in columns
        ]
    )
    returns = frame.pct_change().fillna(0.0)
    if CASH in columns:
        returns[CASH] = 0.0
    r = returns[columns].to_numpy()
    n_days = len(frame)
    rebalance = (
        _is_rebalance_day(frame.index, policy.freq)
        if policy.type == "calendar"
        else np.zeros(n_days, dtype=bool)
    )
    h = target * initial_value
    values = np.empty(n_days)
    held = np.empty((n_days, len(columns)))
    weights = np.empty((n_days, len(columns)))
    turnover, costs = np.zeros(n_days), np.zeros(n_days)
    values[0], weights[0], held[0] = initial_value, target, target * initial_value
    for t in range(1, n_days):
        weights[t] = h / h.sum()
        h = h * (1 + r[t])
        v = h.sum()
        drift = np.abs(h / v - target).max()
        if rebalance[t] or (policy.type == "threshold" and drift > policy.band):
            traded = np.abs(target * v - h).sum()
            cost = traded * cost_bps / 1e4
            turnover[t], costs[t] = traded / (2 * v), cost
            h = target * (v - cost)
        values[t], held[t] = h.sum(), h
    value = pd.Series(values, index=frame.index)
    loc = local.reindex(frame.index)[assets].ffill().pct_change().fillna(0.0)
    if CASH in columns:
        loc[CASH] = 0.0
    return PortfolioPath(
        value=value,
        nav=100 * value / initial_value,
        weights=pd.DataFrame(weights, index=frame.index, columns=columns),
        returns=returns[columns],
        local_returns=loc[columns],
        flows=pd.Series(0.0, index=frame.index),
        turnover=pd.Series(turnover, index=frame.index),
        costs=pd.Series(costs, index=frame.index),
        holdings_value=pd.DataFrame(held, index=frame.index, columns=columns),
        warnings=warnings,
    )


# ── Transactions mode (P-002, P-005) ─────────────────────────────────────────


def split_factor_after(splits: pd.DataFrame | None, day: pd.Timestamp) -> float:
    """Product of split ratios with ex-date after `day`: trade quantities × factor (and prices
    ÷ factor) are in today's split-adjusted units, like stored prices."""
    if splits is None or splits.empty:
        return 1.0
    later = splits.loc[pd.to_datetime(splits["ex_date"]) > day, "ratio"]
    return float(np.prod(later)) if len(later) else 1.0


def normalise_transactions(
    tx: pd.DataFrame, actions: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """Sorted ledger with split-adjusted quantities/prices; voided rows dropped."""
    warnings: list[str] = []
    frame = tx.copy()
    if "voided" in frame:
        frame = frame.loc[~frame["voided"].fillna(False).astype(bool)]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["type"] = frame["type"].str.upper()
    unknown = set(frame["type"]) - TX_TYPES
    if unknown:
        raise InsufficientDataError(f"Unknown transaction types: {sorted(unknown)}")
    if (frame["type"] == "SPLIT_ADJ").any():
        warnings.append("SPLIT_ADJ rows ignored: splits are applied from stored corporate actions")
        frame = frame.loc[frame["type"] != "SPLIT_ADJ"]
    for col, default in (("fees", 0.0), ("price", 0.0), ("fx_rate_to_base", np.nan)):
        if col not in frame:
            frame[col] = default
        frame[col] = (
            frame[col].astype(float).fillna(default)
            if col != "fx_rate_to_base"
            else frame[col].astype(float)
        )
    trades = frame["type"].isin({"BUY", "SELL", "TRANSFER_IN", "TRANSFER_OUT"})
    for i in frame.index[trades]:
        splits = actions.get(str(frame.at[i, "symbol"]))
        if splits is not None and not splits.empty:
            splits = splits.loc[splits["action"] == "split"]
        k = split_factor_after(splits, frame.at[i, "trade_date"])
        frame.at[i, "quantity"] = float(frame.at[i, "quantity"]) * k
        frame.at[i, "price"] = float(frame.at[i, "price"]) / k
    order = {
        "DEPOSIT": 0,
        "TRANSFER_IN": 1,
        "BUY": 2,
        "SELL": 3,
        "DIVIDEND": 4,
        "INTEREST": 4,
        "FEE": 5,
        "TRANSFER_OUT": 6,
        "WITHDRAWAL": 7,
    }
    frame = frame.assign(_o=frame["type"].map(order)).sort_values(
        ["trade_date", "_o"], kind="stable"
    )
    return frame.drop(columns="_o").reset_index(drop=True), warnings


class LotBook:
    """Open lots per symbol; realised P&L by FIFO, LIFO or average cost (PAMP)."""

    def __init__(self, method: LotMethod) -> None:
        self.method = method
        self.lots: dict[str, list[list[Any]]] = {}  # symbol -> [[date, qty, unit cost (base)]]

    def buy(self, symbol: str, day: pd.Timestamp, qty: float, unit_cost: float) -> None:
        book = self.lots.setdefault(symbol, [])
        if self.method == "average" and book:
            held = sum(lot[1] for lot in book)
            cost = sum(lot[1] * lot[2] for lot in book) + qty * unit_cost
            self.lots[symbol] = [[book[0][0], held + qty, cost / (held + qty)]]
        else:
            book.append([day, qty, unit_cost])

    def sell(
        self, symbol: str, day: pd.Timestamp, qty: float, unit_proceeds: float
    ) -> list[dict[str, Any]]:
        book = self.lots.get(symbol, [])
        held = sum(lot[1] for lot in book)
        if qty > held + 1e-9:
            raise InsufficientDataError(
                f"Sell of {qty:g} {symbol} on {day.date()} exceeds the {held:g} held"
            )
        out, remaining = [], qty
        while remaining > 1e-12:
            lot = book[-1] if self.method == "lifo" else book[0]
            take = min(lot[1], remaining)
            out.append(
                {
                    "symbol": symbol,
                    "opened": lot[0].date().isoformat(),
                    "closed": day.date().isoformat(),
                    "quantity": take,
                    "cost": take * lot[2],
                    "proceeds": take * unit_proceeds,
                    "pnl": take * (unit_proceeds - lot[2]),
                    "days": int((day - lot[0]).days),
                }
            )
            lot[1] -= take
            remaining -= take
            if lot[1] <= 1e-12:
                book.remove(lot)
        return out

    def open_lots(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": s,
                "opened": lot[0].date().isoformat(),
                "quantity": lot[1],
                "unit_cost": lot[2],
            }
            for s, book in self.lots.items()
            for lot in book
        ]


def replay_transactions(
    tx: pd.DataFrame,
    close_local: pd.DataFrame,
    adj_local: pd.DataFrame,
    currencies: Mapping[str, str],
    fx: Mapping[str, pd.Series],
    base: str,
    actions: Mapping[str, pd.DataFrame],
    lot_method: LotMethod = "fifo",
) -> PortfolioPath:
    """Daily replay of the ledger: quantities and cash per currency are carried, then valued
    in base (positions × split-adjusted close × FX + cash × FX); TWR_t = (V_t − CF_t)/V_{t−1} − 1
    with external flows at end of day (P-003)."""
    ledger, warnings = normalise_transactions(tx, actions)
    if ledger.empty:
        raise InsufficientDataError("The ledger is empty: add a deposit and trades")
    start = ledger["trade_date"].iloc[0]
    index = close_local.index[close_local.index >= start]
    index = pd.DatetimeIndex([start]).union(index)
    symbols = list(close_local.columns)
    cash_ccys = sorted(set(ledger["currency"].dropna()) | {base})
    all_ccys = sorted(set(cash_ccys) | {currencies.get(s, base) for s in symbols})
    rates = pd.DataFrame({c: fx_rate(fx, c, base, index) for c in all_ccys})
    closes = close_local.reindex(close_local.index.union(index)).ffill().reindex(index)
    qty = dict.fromkeys(symbols, 0.0)
    cash = dict.fromkeys(cash_ccys, 0.0)
    book = LotBook(lot_method)
    q_rows, c_rows = np.zeros((len(index), len(symbols))), np.zeros((len(index), len(cash_ccys)))
    flows = np.zeros(len(index))
    realised: list[dict[str, Any]] = []
    income: list[dict[str, Any]] = []
    days = ledger["trade_date"].searchsorted(index, side="right")  # rows dated ≤ each day
    records = list(ledger.itertuples(index=False))  # plain rows: no per-day pandas slicing
    rate_cols = {c: i for i, c in enumerate(rates.columns)}
    rate_values = rates.to_numpy()
    done = 0
    for t in range(len(index)):
        for row in records[done : days[t]]:
            kind, ccy, q, when = row.type, row.currency, float(row.quantity), row.trade_date
            given = row.fx_rate_to_base
            rate = given if np.isfinite(given) else rate_values[t, rate_cols[ccy]]
            if kind in ("BUY", "TRANSFER_IN"):
                qty[row.symbol] += q
                gross = q * row.price + row.fees
                if kind == "BUY":
                    cash[ccy] -= gross
                else:
                    flows[t] += q * row.price * rate
                book.buy(row.symbol, when, q, gross * rate / q)
            elif kind in ("SELL", "TRANSFER_OUT"):
                qty[row.symbol] -= q
                net = q * row.price - row.fees
                if kind == "SELL":
                    cash[ccy] += net
                else:
                    flows[t] -= q * row.price * rate
                realised += book.sell(row.symbol, when, q, net * rate / q)
            elif kind in ("DEPOSIT", "WITHDRAWAL"):
                sign = 1 if kind == "DEPOSIT" else -1
                cash[ccy] += sign * q
                flows[t] += sign * q * rate
            elif kind in ("DIVIDEND", "INTEREST"):
                cash[ccy] += q
                income.append(
                    {
                        "date": when.date().isoformat(),
                        "type": kind,
                        "symbol": row.symbol if isinstance(row.symbol, str) else None,
                        "amount": q * rate,
                        "currency": ccy,
                    }
                )
            else:  # FEE
                cash[ccy] -= q
        done = days[t]
        q_rows[t] = [qty[s] for s in symbols]
        c_rows[t] = [cash[c] for c in cash_ccys]
    quantities = pd.DataFrame(q_rows, index=index, columns=symbols)
    to_base = pd.DataFrame({s: rates[currencies.get(s, base)] for s in symbols}, index=index)
    close_base = closes * to_base
    mv = quantities * close_base.fillna(0.0)
    cash_base = (pd.DataFrame(c_rows, index=index, columns=cash_ccys) * rates[cash_ccys]).sum(
        axis=1
    )
    mv[CASH] = cash_base
    value = mv.sum(axis=1)
    flow = pd.Series(flows, index=index)
    local = adj_local.reindex(adj_local.index.union(index)).ffill().reindex(index)
    local_returns = local.pct_change().fillna(0.0)
    returns = ((1 + local_returns) * (to_base / to_base.shift(1)).fillna(1.0) - 1).fillna(0.0)
    returns[CASH] = local_returns[CASH] = 0.0
    weights = mv.div(value.where(value != 0), axis=0).shift(1).fillna(0.0)
    return PortfolioPath(
        value=value,
        nav=_twr_index(value, flow),
        weights=weights,
        returns=returns,
        local_returns=local_returns,
        flows=flow,
        turnover=pd.Series(0.0, index=index),
        costs=pd.Series(0.0, index=index),
        quantities=quantities,
        holdings_value=mv,
        close_base=close_base,
        cash=cash_base,
        lots=book.open_lots(),
        realised=realised,
        income=income,
        warnings=warnings,
    )


def _twr_index(value: pd.Series, flows: pd.Series) -> pd.Series:
    prev = value.shift(1)
    r = ((value - flows) / prev - 1).where(prev > 0, 0.0).fillna(0.0)
    return 100 * (1 + r).cumprod()


@dataclass(slots=True)
class PortfolioInputs:
    """What the resolver hands to P-/C- analyses under the `portfolio` key."""

    path: PortfolioPath
    definition: dict[str, Any]  # id, name, mode, base_currency, holdings, rebalance, benchmark…
    meta: dict[str, dict[str, Any]]  # per symbol: asset_class, sector, country, currency, info…
    prices: dict[str, pd.DataFrame]  # stored frames (listing currency): volume, closes
    actions: dict[str, pd.DataFrame] = field(default_factory=dict)  # dividends & splits
    fx_last: dict[str, float] = field(default_factory=dict)  # latest base per unit of ccy
    fx: dict[str, pd.Series] = field(default_factory=dict)  # base per unit of ccy

    def end_weights(self) -> pd.Series:
        """Weights at the last close (after the day's moves)."""
        hv = self.path.holdings_value
        if hv is None or hv.empty:
            return self.path.weights.iloc[-1]
        last = hv.iloc[-1]
        return last / last.sum() if last.sum() else last


# ── Return measures (P-003, P-004) ───────────────────────────────────────────


def xirr(dates: list[pd.Timestamp], amounts: list[float]) -> float | None:
    """Annual IRR of dated flows (investor sign: contributions < 0, proceeds > 0); Newton from
    0.1 with a bisection fallback on [−0.9999, 100]."""
    if len(amounts) < 2 or not (min(amounts) < 0 < max(amounts)):
        return None
    t0 = min(dates)
    years = np.array([(d - t0).days / 365.0 for d in dates])
    cf = np.array(amounts, dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(cf / (1 + rate) ** years))

    rate = 0.1
    for _ in range(50):
        d = float(np.sum(-years * cf / (1 + rate) ** (years + 1)))
        if d == 0:
            break
        step = npv(rate) / d
        rate -= step
        if rate <= -0.9999:
            break
        if abs(step) < 1e-12:
            return rate
    lo, hi = -0.9999, 100.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(300):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ── Covariance estimators (P-012) ────────────────────────────────────────────

Estimator = Literal["sample", "ledoit_wolf", "oas", "ewma"]


def covariance(
    returns: pd.DataFrame, estimator: Estimator = "sample", halflife: int = 63
) -> np.ndarray:
    x = returns.to_numpy()
    n, p = x.shape
    if estimator == "ewma":
        w = 0.5 ** (np.arange(n)[::-1] / halflife)
        w /= w.sum()
        xc = x - w @ x
        return np.asarray((xc * w[:, None]).T @ xc)
    sample = np.cov(x, rowvar=False, ddof=1).reshape(p, p)
    if estimator == "sample" or p == 1:
        return sample
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / n  # MLE covariance for the shrinkage intensities
    mu = np.trace(s) / p
    target = mu * np.eye(p)
    if estimator == "ledoit_wolf":
        d2 = float(np.sum((s - target) ** 2))
        b2 = float(sum(np.sum((np.outer(row, row) - s) ** 2) for row in xc)) / n**2
        shrink = min(b2, d2) / d2 if d2 > 0 else 1.0
    else:  # OAS (Chen et al. 2010)
        tr_s2, tr2_s = float(np.trace(s @ s)), float(np.trace(s)) ** 2
        num = (1 - 2 / p) * tr_s2 + tr2_s
        den = (n + 1 - 2 / p) * (tr_s2 - tr2_s / p)
        shrink = 1.0 if den == 0 else min(num / den, 1.0)
    return np.asarray(((1 - shrink) * s + shrink * target) * n / (n - 1))


def risk_contributions(
    weights: np.ndarray, cov: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """σ_p, marginal contributions (Σw)_i/σ_p and Euler contributions w_i·MC_i (sum = σ_p)."""
    sigma = float(math.sqrt(max(weights @ cov @ weights, 0.0)))
    marginal = cov @ weights / sigma if sigma > 0 else np.zeros_like(weights)
    return sigma, marginal, weights * marginal


# ── Entry point ──────────────────────────────────────────────────────────────


def build_path(
    definition: Mapping[str, Any],
    prices: Mapping[str, pd.DataFrame],
    currencies: Mapping[str, str],
    fx: Mapping[str, pd.Series],
    actions: Mapping[str, pd.DataFrame] | None = None,
    seven_day: set[str] | None = None,
) -> PortfolioPath:
    """`definition`: mode, base_currency, holdings [{symbol, weight, is_cash}] or transactions
    (DataFrame), rebalance, cost_bps, start_date, initial_value, lot_method."""
    base = str(definition.get("base_currency", "USD"))
    index = calendar_of(prices, seven_day or set())
    adj_base, _, adj_local = base_frames(prices, currencies, fx, base, index)
    if definition.get("mode", "weights") == "weights":
        holdings = [
            Holding(str(h["symbol"]), float(h["weight"]), bool(h.get("is_cash")))
            for h in definition["holdings"]
        ]
        start = definition.get("start_date")
        return simulate_weights(
            holdings,
            adj_base,
            adj_local,
            Rebalance.parse(definition.get("rebalance")),
            float(definition.get("cost_bps") or 0.0),
            float(definition.get("initial_value") or 100_000),
            pd.Timestamp(start).date() if start else None,
        )
    close_local = pd.DataFrame(
        {
            s: f["close"].reindex(index.union(f.index)).ffill().reindex(index)
            for s, f in prices.items()
        }
    )
    return replay_transactions(
        definition["transactions"],
        close_local,
        adj_local,
        currencies,
        fx,
        base,
        actions or {},
        definition.get("lot_method", "fifo"),
    )
