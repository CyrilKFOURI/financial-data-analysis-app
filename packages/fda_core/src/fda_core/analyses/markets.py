"""Markets, macro & cross-asset (ANALYTICS_CATALOG MK-001…MK-011, MK-013, MK-014; module M19).

Subject `market` (id ignored, e.g. "global"). Each analysis declares in `MARKET_INPUTS` the
instruments and series it reads; the resolver passes those that are stored:
- `market_prices`: {symbol: level series (adjusted close, or close for levels such as ^VIX)};
- `market_series`: {code: level series} (FRED, ECB; percent levels stay levels).
Missing inputs are listed in the payload (`missing`) with the ingest hint, never invented.
Rates and spreads move in basis points; only prices have returns. Everything is descriptive (V3).
"""

import math
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.analyses.price import window_start
from fda_core.analyses.ts_common import clean, iso
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

MARKET: set[Any] = {"market"}
NEEDS = {"market"}

INDICES = {
    "^GSPC": "S&P 500", "^NDX": "Nasdaq-100", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
    "^STOXX50E": "Euro Stoxx 50", "^GDAXI": "DAX", "^FCHI": "CAC 40", "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225", "^HSI": "Hang Seng",
}  # fmt: skip
SECTORS = {
    "XLB": "Materials", "XLC": "Communication", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Staples", "XLRE": "Real estate",
    "XLU": "Utilities", "XLV": "Health care", "XLY": "Discretionary",
}  # fmt: skip
FX = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY", "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "EURCHF=X": "EUR/CHF", "DX-Y.NYB": "Dollar index",
}  # fmt: skip
COMMODITIES = {
    "CL=F": "WTI crude", "BZ=F": "Brent", "NG=F": "Natural gas", "GC=F": "Gold", "SI=F": "Silver",
    "HG=F": "Copper", "PL=F": "Platinum", "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
}  # fmt: skip
CRYPTO = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ether", "SOL-USD": "Solana", "XRP-USD": "XRP",
    "BNB-USD": "BNB", "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink", "DOT-USD": "Polkadot",
}  # fmt: skip
VOLATILITY = {
    "^VIX9D": "9D", "^VIX": "30D", "^VIX3M": "3M", "^VIX6M": "6M", "^VVIX": "VVIX",
    "^MOVE": "MOVE",
}  # fmt: skip
ASSET_CLASSES = {
    "SPY": "US equity", "VGK": "Europe equity", "EEM": "EM equity", "AGG": "US aggregate",
    "TLT": "Long Treasuries", "HYG": "High yield", "GLD": "Gold", "DBC": "Commodities",
    "VNQ": "REITs", "BTC-USD": "Bitcoin",
}  # fmt: skip
US_CURVE = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS3": 3, "DGS5": 5,
    "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30,
}  # fmt: skip
EU_CURVE = {
    "ECB_YC_3M": 0.25, "ECB_YC_6M": 0.5, "ECB_YC_1Y": 1, "ECB_YC_2Y": 2, "ECB_YC_3Y": 3,
    "ECB_YC_5Y": 5, "ECB_YC_7Y": 7, "ECB_YC_10Y": 10, "ECB_YC_20Y": 20, "ECB_YC_30Y": 30,
}  # fmt: skip
SPREADS = {
    "T10Y2Y": "10y − 2y", "T10Y3M": "10y − 3m", "DFII10": "10y real yield (TIPS)",
    "T10YIE": "10y breakeven inflation",
}  # fmt: skip
CREDIT = {
    "BAMLH0A0HYM2": "US high-yield OAS", "BAMLC0A0CM": "US investment-grade OAS",
    "BAMLC0A4CBBB": "US BBB OAS", "NFCI": "Chicago Fed financial conditions (NFCI)",
}  # fmt: skip
# code → (title, transform): index → YoY & 3m annualised; level → level & change; flow → Δ.
MACRO: dict[str, tuple[str, str]] = {
    "CPIAUCSL": ("US CPI", "index"), "CPILFESL": ("US core CPI", "index"),
    "PCEPI": ("US PCE prices", "index"), "PCEPILFE": ("US core PCE prices", "index"),
    "UNRATE": ("US unemployment rate", "level"), "PAYEMS": ("US nonfarm payrolls (k)", "flow"),
    "GDPC1": ("US real GDP", "gdp"), "RSAFS": ("US retail sales", "index"),
    "INDPRO": ("US industrial production", "index"), "M2SL": ("US M2", "index"),
    "DFF": ("Fed funds effective", "level"), "ESTR": ("€STR", "level"),
    "ECB_DFR": ("ECB deposit rate", "level"),
    "ECB_UNEMP": ("Euro area unemployment", "level"),
}  # fmt: skip
RATES_OVERVIEW = {"DGS2": "US 2y", "DGS10": "US 10y", "ECB_YC_10Y": "EUR AAA 10y"}

MARKET_INPUTS: dict[str, tuple[list[str], list[str]]] = {
    "MK-001": ([*INDICES, *SECTORS, *FX, *COMMODITIES, *CRYPTO], list(RATES_OVERVIEW)),
    "MK-002": (["SPY", *SECTORS], []),
    "MK-003": (["SPY", *VOLATILITY], []),
    "MK-004": ([], [*US_CURVE, *EU_CURVE]),
    "MK-005": ([], [*SPREADS, "USREC"]),
    "MK-006": ([], list(CREDIT)),
    "MK-007": ([], list(MACRO)),
    "MK-008": (["SPY", *FX], ["DTB3", "ESTR"]),
    "MK-009": (list(COMMODITIES), []),
    "MK-010": (["QQQ", "GLD", *CRYPTO], []),
    "MK-011": (
        ["^GSPC", "SPY", "DX-Y.NYB", "CL=F", "^VIX", *ASSET_CLASSES],
        ["DGS10", "BAMLH0A0HYM2", "T10Y2Y"],
    ),
    "MK-013": ([], list(MACRO)),
    "MK-014": (list(ASSET_CLASSES), []),
}  # fmt: skip


def all_market_inputs() -> tuple[list[str], list[str]]:
    """Every instrument and series used by the Markets pages (for one-click ingestion)."""
    symbols = sorted({s for syms, _ in MARKET_INPUTS.values() for s in syms})
    codes = sorted({c for _, cs in MARKET_INPUTS.values() for c in cs})
    return symbols, codes


class NoParams(BaseModel):
    pass


# ── helpers ──────────────────────────────────────────────────────────────────


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _stored(inputs: AnalysisInputs, key: str) -> dict[str, pd.Series]:
    data = inputs.data.get(key)
    if not isinstance(data, dict):
        return {}
    out = {}
    for name, series in data.items():
        s = pd.Series(series, dtype=float).dropna()
        s.index = pd.to_datetime(s.index)
        if not s.empty:
            out[str(name)] = s.sort_index()
    return out


def _prices(inputs: AnalysisInputs) -> dict[str, pd.Series]:
    return _stored(inputs, "market_prices")


def _series(inputs: AnalysisInputs) -> dict[str, pd.Series]:
    return _stored(inputs, "market_series")


def _missing(names: list[str] | dict[str, Any], have: dict[str, pd.Series]) -> list[str]:
    return [n for n in names if n not in have]


def _need(have: dict[str, pd.Series], names: list[str] | dict[str, Any], minimum: int) -> None:
    present = [n for n in names if n in have]
    if len(present) < minimum:
        raise InsufficientDataError(
            f"Ingest the market data first (POST /v1/markets/ingest): {len(present)} of "
            f"{len(names)} inputs stored, ≥ {minimum} needed"
        )


def _asof(s: pd.Series, when: pd.Timestamp) -> float | None:
    before = s.loc[:when]
    return None if before.empty else float(before.iloc[-1])


HORIZONS = {"1d": None, "1w": 7, "1m": 31, "3m": 92, "6m": 183, "ytd": "ytd", "1y": 365}


def _anchor(s: pd.Series, horizon: str) -> float | None:
    last = s.index[-1]
    days = HORIZONS[horizon]
    if days is None:
        return float(s.iloc[-2]) if len(s) > 1 else None
    if days == "ytd":
        when = pd.Timestamp(last.year, 1, 1) - timedelta(days=1)
    else:
        when = last - timedelta(days=int(days))
    if when < s.index[0]:
        return None
    return _asof(s, when)


def performance(
    s: pd.Series, horizons: tuple[str, ...] = ("1d", "1w", "1m", "ytd", "1y")
) -> dict[str, Any]:
    """Last level, simple returns over calendar horizons, position in the 52-week range."""
    last = float(s.iloc[-1])
    row: dict[str, Any] = {"last": last, "last_date": iso(s.index[-1:])[0]}
    for h in horizons:
        base = _anchor(s, h)
        row[f"return_{h}"] = None if not base else last / base - 1
    year = s.loc[s.index >= s.index[-1] - timedelta(days=365)]
    lo, hi = float(year.min()), float(year.max())
    row["position_52w"] = (last - lo) / (hi - lo) if hi > lo else None
    return row


def level_changes(
    s: pd.Series, horizons: tuple[str, ...] = ("1d", "1w", "1m", "ytd", "1y")
) -> dict[str, Any]:
    """Percent-level series (yields, spreads): last level and changes in basis points."""
    last = float(s.iloc[-1])
    row: dict[str, Any] = {"last": last, "last_date": iso(s.index[-1:])[0]}
    for h in horizons:
        base = _anchor(s, h)
        row[f"change_bp_{h}"] = None if base is None else (last - base) * 100
    return row


def percentile(s: pd.Series, years: int = 10) -> float | None:
    tail = s.loc[s.index >= s.index[-1] - timedelta(days=365 * years)]
    return float((tail <= tail.iloc[-1]).mean()) if len(tail) >= 20 else None


def _window(inputs: AnalysisInputs, s: pd.Series) -> pd.Series:
    start = window_start(
        str(inputs.conventions.get("range", "1Y")), s.index[-1].date(), s.index[0].date()
    )
    return s.loc[s.index >= pd.Timestamp(start)]


def _lines(series: dict[str, pd.Series]) -> tuple[list[str], dict[str, list[float | None]]]:
    if not series:
        return [], {}
    frame = pd.concat(series, axis=1).sort_index().ffill()
    return iso(frame.index), {k: clean(frame[k].to_numpy()) for k in frame.columns}


# ── MK-001 overview ──────────────────────────────────────────────────────────


@analysis(
    id="MK-001",
    version="1.0.0",
    title="Markets overview",
    description=(
        "Indices, US sector ETFs, FX, commodities and crypto: last, 1D/1W/1M/YTD/1Y simple "
        "returns (calendar anchors, last close on or before), position in the 52-week range; "
        "rates as levels with changes in bp."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def markets_overview(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    prices, series = _prices(inputs), _series(inputs)
    groups = {
        "index": INDICES,
        "sector": SECTORS,
        "fx": FX,
        "commodity": COMMODITIES,
        "crypto": CRYPTO,
    }
    _need(prices, [s for g in groups.values() for s in g], 1)
    rows = [
        {"group": g, "symbol": sym, "name": name, **performance(prices[sym])}
        for g, names in groups.items()
        for sym, name in names.items()
        if sym in prices and len(prices[sym]) > 1
    ]
    rates = [
        {"series": code, "name": name, **level_changes(series[code])}
        for code, name in RATES_OVERVIEW.items()
        if code in series and len(series[code]) > 1
    ]
    missing = _missing([s for g in groups.values() for s in g], prices) + _missing(
        list(RATES_OVERVIEW), series
    )
    return AnalysisOutput("table", {"rows": rows, "rates": rates, "missing": missing})


# ── MK-002 sector rotation ───────────────────────────────────────────────────


class RotationParams(BaseModel):
    ratio_weeks: int = Field(12, ge=4, le=52)
    momentum_weeks: int = Field(4, ge=1, le=26)
    tail_weeks: int = Field(10, ge=2, le=26)


def rrg(sector: pd.Series, bench: pd.Series, params: RotationParams) -> pd.DataFrame:
    """Weekly RS = P/P_bench; RS-ratio = 100·RS/SMA(RS, n); RS-momentum = 100·ratio/ratio(t−k)."""
    daily = pd.concat([sector, bench], axis=1, join="inner").dropna()
    weekly = daily.resample("W-FRI").last().dropna()
    if len(weekly) and weekly.index[-1] > daily.index[-1]:  # week in progress: its real date
        weekly = weekly.rename(index={weekly.index[-1]: daily.index[-1]})
    rs = weekly.iloc[:, 0] / weekly.iloc[:, 1]
    ratio = 100 * rs / rs.rolling(params.ratio_weeks).mean()
    momentum = 100 * ratio / ratio.shift(params.momentum_weeks)
    return pd.DataFrame({"ratio": ratio, "momentum": momentum}).dropna()


def quadrant(ratio: float, momentum: float) -> str:
    if ratio >= 100:
        return "leading" if momentum >= 100 else "weakening"
    return "improving" if momentum >= 100 else "lagging"


@analysis(
    id="MK-002",
    version="1.0.0",
    title="Sector rotation",
    description=(
        "US SPDR sectors: returns over 1W…1Y (grid, best per horizon) and relative-strength "
        "rank; relative rotation graph on weekly closes vs SPY: RS-ratio = 100·RS/SMA(RS, 12w), "
        "RS-momentum = 100·ratio/ratio 4 weeks earlier, 10-week tails; quadrant labels describe "
        "position, not a signal."
    ),
    applies_to=MARKET,
    params=RotationParams,
    result_kind="table",
    requires=NEEDS,
)
def sector_rotation(inputs: AnalysisInputs, params: RotationParams) -> AnalysisOutput:
    prices = _prices(inputs)
    if "SPY" not in prices:
        raise InsufficientDataError(
            "Ingest SPY (benchmark) and the sector ETFs (POST /v1/markets/ingest)"
        )
    _need(prices, SECTORS, 2)
    horizons = ("1w", "1m", "3m", "6m", "ytd", "1y")
    present = [s for s in SECTORS if s in prices]
    perf = {s: performance(prices[s], horizons) for s in present}
    values = [[_f(perf[s][f"return_{h}"]) for s in present] for h in horizons]
    best = [
        int(np.nanargmax([np.nan if v is None else v for v in row]))
        if any(v is not None for v in row)
        else None
        for row in values
    ]
    points = []
    for s in present:
        frame = rrg(prices[s], prices["SPY"], params)
        if frame.empty:
            continue
        tail = frame.iloc[-params.tail_weeks :]
        last = tail.iloc[-1]
        points.append(
            {
                "symbol": s,
                "sector": SECTORS[s],
                "rs_ratio": float(last["ratio"]),
                "rs_momentum": float(last["momentum"]),
                "quadrant": quadrant(float(last["ratio"]), float(last["momentum"])),
                "tail_ratio": clean(tail["ratio"].to_numpy()),
                "tail_momentum": clean(tail["momentum"].to_numpy()),
            }
        )
    rank_1m = sorted(present, key=lambda s: -(perf[s]["return_1m"] or -np.inf))
    return AnalysisOutput(
        "table",
        {
            "grid": {
                "rows": [h.upper() for h in horizons],
                "columns": [SECTORS[s] for s in present],
                "values": values,
                "unit": "percent",
                "best": best,
            },
            "rotation": points,
            "rank_1m": [
                {
                    "rank": i + 1,
                    "symbol": s,
                    "sector": SECTORS[s],
                    "return_1m": perf[s]["return_1m"],
                }
                for i, s in enumerate(rank_1m)
            ],
            "missing": _missing(SECTORS, prices),
        },
    )


# ── MK-003 volatility complex ────────────────────────────────────────────────


@analysis(
    id="MK-003",
    version="1.0.0",
    title="Volatility complex",
    description=(
        "VIX term structure (9D, 30D, 3M, 6M) and contango ratio VIX3M/VIX (> 1 = contango); "
        "VVIX, MOVE when stored; SPY 21-session realised volatility (√252, %) vs VIX; VIX "
        "percentile over 10 years."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def volatility_complex(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    prices = _prices(inputs)
    if "^VIX" not in prices:
        raise InsufficientDataError("Ingest ^VIX and its term structure (POST /v1/markets/ingest)")
    vix = prices["^VIX"]
    term = [
        {
            "tenor": tenor,
            "symbol": sym,
            "level": float(prices[sym].iloc[-1]),
            "date": iso(prices[sym].index[-1:])[0],
        }
        for sym, tenor in VOLATILITY.items()
        if sym in prices and sym not in ("^VVIX", "^MOVE")
    ]
    lines: dict[str, pd.Series] = {"vix": _window(inputs, vix)}
    if "^VIX3M" in prices:
        lines["vix3m"] = _window(inputs, prices["^VIX3M"])
    realised = None
    if "SPY" in prices:
        r = np.log(prices["SPY"]).diff()
        rv = r.rolling(21).std() * math.sqrt(252) * 100
        lines["spy_realised_21d"] = _window(inputs, rv.dropna())
        realised = _f(rv.iloc[-1])
    contango = float(prices["^VIX3M"].iloc[-1] / vix.iloc[-1]) if "^VIX3M" in prices else None
    dates, values = _lines(lines)
    return AnalysisOutput(
        "table",
        {
            "vix": float(vix.iloc[-1]),
            "vix_percentile_10y": percentile(vix),
            "contango_ratio": contango,
            "structure": None
            if contango is None
            else ("contango" if contango > 1 else "backwardation"),
            "vvix": _f(prices["^VVIX"].iloc[-1]) if "^VVIX" in prices else None,
            "move": _f(prices["^MOVE"].iloc[-1]) if "^MOVE" in prices else None,
            "spy_realised_21d": realised,
            "implied_minus_realised": None if realised is None else float(vix.iloc[-1]) - realised,
            "term_structure": term,
            "dates": dates,
            "lines": values,
            "missing": _missing(["SPY", *VOLATILITY], prices),
        },
    )


# ── MK-004 yield curves ──────────────────────────────────────────────────────


class CurveParams(BaseModel):
    curve: Literal["US", "EU"] = "US"
    as_of: str | None = Field(None, max_length=10, description="YYYY-MM-DD (default: latest)")


def curve_shape(curve: dict[float, float]) -> dict[str, Any]:
    """Slope (long − short), curvature 2·mid − short − long (5y vs 2y/10y), shape label."""
    tenors = sorted(curve)
    short, long_ = curve[tenors[0]], curve[tenors[-1]]

    def pick(t: float) -> float:
        return curve[min(tenors, key=lambda x: abs(x - t))]

    slope = pick(10) - pick(0.25)
    curvature = 2 * pick(5) - pick(2) - pick(10)
    hump = max(curve.values()) - max(short, long_)
    if hump > 0.25 and max(curve, key=lambda t: curve[t]) not in (tenors[0], tenors[-1]):
        shape = "humped"
    elif abs(slope) < 0.25:
        shape = "flat"
    else:
        shape = "normal" if slope > 0 else "inverted"
    return {"slope_10y_3m": slope, "curvature_2_5_10": curvature, "shape": shape}


@analysis(
    id="MK-004",
    version="1.0.0",
    title="Yield curves",
    description=(
        "US Treasury constant-maturity curve 1M–30Y (FRED) or euro-area AAA spot curve 3M–30Y "
        "(ECB): curve at a date vs 1M, 1Y and 5Y earlier (last observation on or before); shape "
        "(slope 10y−3m, curvature 2·5y−2y−10y); month-end history for animation; PCA of daily "
        "level changes over the range (level / slope / curvature shares)."
    ),
    applies_to=MARKET,
    params=CurveParams,
    result_kind="table",
    requires=NEEDS,
)
def yield_curves(inputs: AnalysisInputs, params: CurveParams) -> AnalysisOutput:
    series = _series(inputs)
    tenors = US_CURVE if params.curve == "US" else EU_CURVE
    _need(series, tenors, 4)
    frame = pd.concat({c: series[c] for c in tenors if c in series}, axis=1).sort_index().ffill()
    as_of = pd.Timestamp(params.as_of) if params.as_of else frame.index[-1]
    if as_of < frame.index[0]:
        raise InsufficientDataError(f"The stored curve starts {iso(frame.index[:1])[0]}")
    snapshots = {
        "as_of": as_of,
        "1m_before": as_of - timedelta(days=31),
        "1y_before": as_of - timedelta(days=365),
        "5y_before": as_of - timedelta(days=5 * 365),
    }
    curves: dict[str, dict[float, float]] = {}
    for name, when in snapshots.items():
        row = frame.loc[:when]
        if not row.empty:
            values = row.iloc[-1].dropna()
            curves[name] = {tenors[c]: float(v) for c, v in values.items()}
    rows = [
        {
            "tenor": c.split("_")[-1].replace("DGS", ""),
            "years": y,
            **{k: curves[k].get(y) for k in curves},
        }
        for c, y in tenors.items()
        if c in frame.columns
    ]
    month_end = frame.resample("ME").last().dropna(how="all").iloc[-120:]
    changes = _window(inputs, frame.index.to_series()).index
    diffs = frame.loc[frame.index.isin(changes)].diff().dropna()
    pca_out = None
    if len(diffs) > 60 and diffs.shape[1] >= 3:
        cov = np.cov(diffs.to_numpy(), rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        share = vals / vals.sum()
        pca_out = [
            {
                "component": name,
                "explained": float(share[i]),
                **{str(tenors[c]): float(vecs[j, i]) for j, c in enumerate(diffs.columns)},
            }
            for i, name in enumerate(("level", "slope", "curvature"))
        ]
    return AnalysisOutput(
        "table",
        {
            "curve": params.curve,
            "as_of": iso(frame.loc[:as_of].index[-1:])[0],
            "shape": curve_shape(curves["as_of"]),
            "rows": rows,
            "history": {
                "dates": iso(month_end.index),
                "tenors": [tenors[c] for c in month_end.columns],
                "values": [clean(r) for r in month_end.to_numpy()],
            },
            "pca": pca_out,
            "missing": _missing(tenors, series),
        },
    )


# ── MK-005 spreads / MK-006 credit ───────────────────────────────────────────


def recessions(usrec: pd.Series) -> list[dict[str, str]]:
    """NBER recession months (USREC = 1) as [start, end] periods."""
    flags = (usrec.astype(float) > 0.5).to_numpy()
    dates = iso(usrec.index)
    periods = []
    start: int | None = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            periods.append({"start": dates[start], "end": dates[i - 1]})
            start = None
    if start is not None:
        periods.append({"start": dates[start], "end": dates[-1]})
    return periods


@analysis(
    id="MK-005",
    version="1.0.0",
    title="Curve spreads & real yields",
    description=(
        "10y−2y and 10y−3m Treasury spreads, 10y real yield (TIPS) and 10y breakeven (percent "
        "levels; changes in bp); days inverted in the range; NBER recession periods (USREC) for "
        "shading; 10-year percentiles."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def curve_spreads(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    series = _series(inputs)
    _need(series, SPREADS, 1)
    rows, lines = [], {}
    for code, name in SPREADS.items():
        if code not in series:
            continue
        s = series[code]
        rows.append(
            {
                "series": code,
                "name": name,
                **level_changes(s, ("1m", "1y")),
                "percentile_10y": percentile(s),
            }
        )
        lines[code] = _window(inputs, s)
    inverted = {c: int((lines[c] < 0).sum()) for c in ("T10Y2Y", "T10Y3M") if c in lines}
    dates, values = _lines(lines)
    shading = recessions(series["USREC"]) if "USREC" in series else []
    if dates:
        shading = [p for p in shading if p["end"] >= dates[0]]
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "days_inverted": inverted,
            "recessions": shading,
            "dates": dates,
            "lines": values,
            "missing": _missing([*SPREADS, "USREC"], series),
        },
    )


@analysis(
    id="MK-006",
    version="1.0.0",
    title="Credit & financial conditions",
    description=(
        "ICE BofA US high-yield, investment-grade and BBB option-adjusted spreads (percent; "
        "changes in bp) and the Chicago Fed NFCI (index, > 0 = tighter than average); 10-year "
        "percentiles."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def credit(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    series = _series(inputs)
    _need(series, CREDIT, 1)
    rows, lines = [], {}
    for code, name in CREDIT.items():
        if code not in series:
            continue
        s = series[code]
        rows.append(
            {
                "series": code,
                "name": name,
                **level_changes(s, ("1w", "1m", "1y")),
                "percentile_10y": percentile(s),
            }
        )
        lines[code] = _window(inputs, s)
    dates, values = _lines(lines)
    return AnalysisOutput(
        "table",
        {"rows": rows, "dates": dates, "lines": values, "missing": _missing(CREDIT, series)},
    )


# ── MK-007 macro dashboard / MK-013 release history ──────────────────────────


def macro_row(code: str, s: pd.Series) -> dict[str, Any]:
    """Transforms by kind: index → YoY and 3m annualised; gdp → q/q annualised and YoY;
    flow → monthly change and 3-month average change; level → level and 1-year change (pp)."""
    title, kind = MACRO[code]
    last, last_date = float(s.iloc[-1]), s.index[-1]
    row: dict[str, Any] = {
        "series": code,
        "name": title,
        "kind": kind,
        "last": last,
        "last_date": iso([last_date])[0],
        "prior": _f(s.iloc[-2]) if len(s) > 1 else None,
    }
    year_ago = _asof(s, last_date - timedelta(days=360))
    if kind == "index":
        row["yoy"] = None if not year_ago else last / year_ago - 1
        three = _asof(s, last_date - timedelta(days=88))
        row["ann_3m"] = None if not three else (last / three) ** 4 - 1
    elif kind == "gdp":
        row["ann_qoq"] = (last / float(s.iloc[-2])) ** 4 - 1 if len(s) > 1 else None
        row["yoy"] = None if not year_ago else last / year_ago - 1
    elif kind == "flow":
        row["change"] = last - float(s.iloc[-2]) if len(s) > 1 else None
        row["change_3m_avg"] = (last - float(s.iloc[-4])) / 3 if len(s) > 3 else None
    else:
        row["change_1y_pp"] = None if year_ago is None else last - year_ago
    return row


@analysis(
    id="MK-007",
    version="1.0.0",
    title="Macro dashboard",
    description=(
        "US CPI/PCE headline & core (YoY, 3m annualised), unemployment, payrolls (monthly change, "
        "3-month average), real GDP (q/q annualised, YoY), retail sales, industrial production, "
        "M2 (YoY), Fed funds, €STR, ECB deposit rate, euro-area unemployment (HICP pending: "
        "the ECB series key changed in 2026). Values "
        "are the latest stored vintage (FRED/ECB revise history)."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def macro_dashboard(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    series = _series(inputs)
    _need(series, MACRO, 1)
    rows = [
        macro_row(code, series[code]) for code in MACRO if code in series and len(series[code]) > 1
    ]
    return AnalysisOutput("table", {"rows": rows, "missing": _missing(MACRO, series)})


class ReleaseParams(BaseModel):
    observations: int = Field(6, ge=2, le=24)


@analysis(
    id="MK-013",
    version="1.0.0",
    title="Macro release history",
    description=(
        "Last N observations of each stored macro series with the prior value and the change "
        "(levels; index series also as period % change). Scheduled release dates need the FRED "
        "API calendar (not used: keyless CSV)."
    ),
    applies_to=MARKET,
    params=ReleaseParams,
    result_kind="table",
    requires=NEEDS,
)
def release_history(inputs: AnalysisInputs, params: ReleaseParams) -> AnalysisOutput:
    series = _series(inputs)
    _need(series, MACRO, 1)
    rows = []
    for code, (title, kind) in MACRO.items():
        if code not in series:
            continue
        s = series[code].iloc[-(params.observations + 1) :]
        for i in range(len(s) - 1, 0, -1):
            value, prior = float(s.iloc[i]), float(s.iloc[i - 1])
            rows.append(
                {
                    "series": code,
                    "name": title,
                    "date": iso(s.index[i : i + 1])[0],
                    "value": value,
                    "prior": prior,
                    "change": value - prior,
                    "change_pct": value / prior - 1 if kind in ("index", "gdp") and prior else None,
                }
            )
    return AnalysisOutput("table", {"rows": rows, "missing": _missing(MACRO, series)})


# ── MK-008 FX ────────────────────────────────────────────────────────────────

SHORT_RATES = {"USD": "DTB3", "EUR": "ESTR"}


def _pair_ccys(symbol: str) -> tuple[str, str] | None:
    code = symbol.replace("=X", "")
    return (code[:3], code[3:6]) if len(code) == 6 else None


@analysis(
    id="MK-008",
    version="1.0.0",
    title="FX",
    description=(
        "Majors and EUR crosses: returns 1D…1Y, 63-session realised volatility (√252), 1-year "
        "correlation of daily returns with SPY; carry = short-rate differential base − quote "
        "where both rates are stored (USD 3m T-bill, €STR)."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def fx_matrix(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    prices, series = _prices(inputs), _series(inputs)
    _need(prices, FX, 1)
    spy = np.log(prices["SPY"]).diff() if "SPY" in prices else None
    rates = {
        ccy: float(series[code].iloc[-1]) for ccy, code in SHORT_RATES.items() if code in series
    }
    rows = []
    for sym, name in FX.items():
        if sym not in prices or len(prices[sym]) < 2:
            continue
        s = prices[sym]
        r = np.log(s).diff()
        row = {"symbol": sym, "name": name, **performance(s, ("1d", "1w", "1m", "3m", "ytd", "1y"))}
        row["volatility_63d"] = _f(r.iloc[-63:].std() * math.sqrt(252))
        if spy is not None:
            both = pd.concat([r, spy], axis=1, join="inner").dropna().iloc[-252:]
            row["corr_spy_1y"] = _f(both.corr().iloc[0, 1]) if len(both) > 60 else None
        ccys = _pair_ccys(sym)
        row["carry_pp"] = (
            rates[ccys[0]] - rates[ccys[1]]
            if ccys and ccys[0] in rates and ccys[1] in rates
            else None
        )
        rows.append(row)
    return AnalysisOutput(
        "table", {"rows": rows, "short_rates": rates, "missing": _missing(FX, prices)}
    )


# ── MK-009 commodities ───────────────────────────────────────────────────────


@analysis(
    id="MK-009",
    version="1.0.0",
    title="Commodities",
    description=(
        "Front-month continuous futures (Yahoo, roll gaps included): returns 1D…1Y, 52-week "
        "position; gold/oil and copper/gold ratios; average return by calendar month "
        "(seasonality over the stored history). Term structure needs contract data (not stored)."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def commodities(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    prices = _prices(inputs)
    _need(prices, COMMODITIES, 1)
    rows = [
        {"symbol": s, "name": n, **performance(prices[s])}
        for s, n in COMMODITIES.items()
        if s in prices and len(prices[s]) > 1
    ]
    ratios = {}
    if {"GC=F", "CL=F"} <= set(prices):
        ratios["gold_oil"] = _window(
            inputs, (prices["GC=F"] / prices["CL=F"].where(prices["CL=F"] > 0)).dropna()
        )
    if {"HG=F", "GC=F"} <= set(prices):
        ratios["copper_gold_x1000"] = _window(
            inputs, (1000 * prices["HG=F"] / prices["GC=F"]).dropna()
        )
    season_rows, season_values = [], []
    for s in ("CL=F", "NG=F", "GC=F", "HG=F", "ZC=F"):
        if s not in prices:
            continue
        monthly = prices[s].resample("ME").last().pct_change().dropna()
        means = monthly.groupby(monthly.index.month).mean()
        season_rows.append(COMMODITIES[s])
        season_values.append([_f(means.get(m)) for m in range(1, 13)])
    dates, values = _lines(ratios)
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "dates": dates,
            "lines": values,
            "grid": {
                "rows": season_rows,
                "columns": [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
                "values": season_values,
                "unit": "percent",
                "best": [],
            },
            "missing": _missing(COMMODITIES, prices),
        },
    )


# ── MK-010 crypto ────────────────────────────────────────────────────────────


@analysis(
    id="MK-010",
    version="1.0.0",
    title="Crypto market",
    description=(
        "Large-cap coins: returns 1D…1Y, 90-day realised volatility (√365), drawdown from the "
        "all-time high in the stored history; rolling 90-day correlation of BTC with QQQ and "
        "GLD (on common sessions). Total market cap and dominance need CoinGecko global data "
        "(not stored)."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def crypto_market(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    prices = _prices(inputs)
    _need(prices, CRYPTO, 1)
    rows = []
    for sym, name in CRYPTO.items():
        if sym not in prices or len(prices[sym]) < 2:
            continue
        s = prices[sym]
        r = np.log(s).diff()
        rows.append(
            {
                "symbol": sym,
                "name": name,
                **performance(s),
                "volatility_90d": _f(r.iloc[-90:].std() * math.sqrt(365)),
                "drawdown_from_ath": float(s.iloc[-1] / s.max() - 1),
            }
        )
    lines = {}
    if "BTC-USD" in prices:
        btc = np.log(prices["BTC-USD"]).diff()
        for other in ("QQQ", "GLD"):
            if other in prices:
                both = pd.concat([btc, np.log(prices[other]).diff()], axis=1, join="inner").dropna()
                corr = both.iloc[:, 0].rolling(90).corr(both.iloc[:, 1]).dropna()
                if not corr.empty:
                    lines[f"btc_corr_{other.lower()}_90d"] = _window(inputs, corr)
    dates, values = _lines(lines)
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "dates": dates,
            "lines": values,
            "missing": _missing(["QQQ", "GLD", *CRYPTO], prices),
        },
    )


# ── MK-011 cross-asset correlation regimes ───────────────────────────────────


class CrossAssetParams(BaseModel):
    window: int = Field(252, ge=63, le=756)
    stress_quantile: float = Field(0.1, gt=0.0, le=0.3)


def sign_regimes(corr: pd.Series, min_sessions: int = 1) -> list[dict[str, Any]]:
    """Consecutive periods where the rolling correlation keeps its sign; spells shorter than
    `min_sessions` are absorbed by the preceding regime (no flicker around zero)."""
    signs = np.sign(corr.dropna())
    dates = iso(signs.index)
    out: list[dict[str, Any]] = []
    for when, sign in zip(dates, signs.to_numpy(), strict=True):
        if out and out[-1]["sign"] == sign:
            out[-1]["end"], out[-1]["sessions"] = when, out[-1]["sessions"] + 1
        else:
            out.append({"start": when, "end": when, "sign": float(sign), "sessions": 1})
    merged: list[dict[str, Any]] = []
    for spell in out:
        if merged and (spell["sessions"] < min_sessions or merged[-1]["sign"] == spell["sign"]):
            merged[-1]["end"] = spell["end"]
            merged[-1]["sessions"] += spell["sessions"]
        else:
            merged.append(dict(spell))
    return [{**p, "regime": "positive" if p["sign"] > 0 else "negative"} for p in merged]


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    return both.iloc[:, 0].rolling(window).corr(both.iloc[:, 1]).dropna()


def _z(s: pd.Series, window: int = 756) -> pd.Series:
    """Trailing z-score (no look-ahead): (x − mean_{t−w..t}) / sd_{t−w..t}."""
    return (s - s.rolling(window, min_periods=252).mean()) / s.rolling(
        window, min_periods=252
    ).std()


@analysis(
    id="MK-011",
    version="1.0.0",
    title="Cross-asset correlation regimes",
    description=(
        "Rolling correlation (window sessions) of S&P 500 returns with Treasury returns proxied by "
        "−Δ10y yield (long history), with sign regimes (spells < 63 sessions absorbed); "
        "equity–USD and equity–oil; asset-class "
        "correlation matrix on all days vs stress days (SPY worst decile); risk appetite "
        "composite = mean of trailing 3-year z-scores of VIX, HY OAS, −(10y−2y) and the dollar "
        "index (higher = more risk-averse), descriptive only."
    ),
    applies_to=MARKET,
    params=CrossAssetParams,
    result_kind="table",
    requires=NEEDS,
)
def cross_asset(inputs: AnalysisInputs, params: CrossAssetParams) -> AnalysisOutput:
    prices, series = _prices(inputs), _series(inputs)
    equity = prices.get("^GSPC", prices.get("SPY"))
    if equity is None:
        raise InsufficientDataError("Ingest ^GSPC or SPY (POST /v1/markets/ingest)")
    eq = np.log(equity).diff()
    lines: dict[str, pd.Series] = {}
    regimes: list[dict[str, Any]] = []
    if "DGS10" in series:
        bond = -series["DGS10"].diff()
        stock_bond = _rolling_corr(eq, bond, params.window)
        lines["stock_bond"] = stock_bond
        regimes = sign_regimes(stock_bond, min_sessions=63)
    spy = np.log(prices["SPY"]).diff() if "SPY" in prices else eq
    for sym, key in (("DX-Y.NYB", "equity_usd"), ("CL=F", "equity_oil")):
        if sym in prices:
            lines[key] = _rolling_corr(
                spy, np.log(prices[sym].where(prices[sym] > 0)).diff(), params.window
            )
    grids = {}
    classes = [s for s in ASSET_CLASSES if s in prices]
    if len(classes) >= 3:
        r = pd.concat(
            {ASSET_CLASSES[s]: np.log(prices[s]).diff() for s in classes}, axis=1, join="inner"
        ).dropna()
        r = r.iloc[-5 * 252 :]
        stress = (
            r[ASSET_CLASSES["SPY"]] <= r[ASSET_CLASSES["SPY"]].quantile(params.stress_quantile)
            if "SPY" in classes
            else None
        )
        labels = list(r.columns)
        grids["all_days"] = {
            "rows": labels,
            "columns": labels,
            "values": r.corr().round(4).to_numpy().tolist(),
            "unit": "number",
            "best": [],
        }
        if stress is not None and stress.sum() > 20:
            grids["stress_days"] = {
                "rows": labels,
                "columns": labels,
                "values": r[stress].corr().round(4).to_numpy().tolist(),
                "unit": "number",
                "best": [],
            }
    composite = None
    parts = {}
    if "^VIX" in prices:
        parts["vix"] = _z(prices["^VIX"])
    if "BAMLH0A0HYM2" in series:
        parts["hy_oas"] = _z(series["BAMLH0A0HYM2"])
    if "T10Y2Y" in series:
        parts["inverted_slope"] = -_z(series["T10Y2Y"])
    if "DX-Y.NYB" in prices:
        parts["usd"] = _z(prices["DX-Y.NYB"])
    if len(parts) >= 2:
        frame = pd.concat(parts, axis=1).sort_index().ffill().dropna()
        composite = frame.mean(axis=1)
        lines["risk_aversion"] = composite
    shown = {k: _window(inputs, v) for k, v in lines.items() if not v.empty}
    dates, values = _lines(shown)
    return AnalysisOutput(
        "table",
        {
            "stock_bond_now": _f(lines["stock_bond"].iloc[-1])
            if "stock_bond" in lines and len(lines["stock_bond"])
            else None,
            "stock_bond_since": iso(lines["stock_bond"].index[:1])[0]
            if "stock_bond" in lines and len(lines["stock_bond"])
            else None,
            "risk_aversion_now": None
            if composite is None or composite.empty
            else float(composite.iloc[-1]),
            "risk_components": sorted(parts),
            "regimes": regimes[-12:],
            "dates": dates,
            "lines": values,
            **({"grid": grids["all_days"]} if "all_days" in grids else {}),
            "stress_grid": grids.get("stress_days"),
            "missing": _missing(
                ["^GSPC", "SPY", "DX-Y.NYB", "CL=F", "^VIX", *ASSET_CLASSES], prices
            )
            + _missing(["DGS10", "BAMLH0A0HYM2", "T10Y2Y"], series),
        },
    )


# ── MK-014 asset-class quilt ─────────────────────────────────────────────────


@analysis(
    id="MK-014",
    version="1.0.0",
    title="Asset-class returns quilt",
    description=(
        "Calendar-year total returns (adjusted close, USD) of asset-class proxies — SPY, VGK, EEM, "
        "AGG, TLT, HYG, GLD, DBC, VNQ, BTC-USD — ranked each year (the 'periodic table'); the "
        "current year is year-to-date. Years without a full year of data for a proxy are blank."
    ),
    applies_to=MARKET,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def returns_quilt(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    prices = _prices(inputs)
    _need(prices, ASSET_CLASSES, 2)
    classes = [s for s in ASSET_CLASSES if s in prices]
    annual: dict[str, pd.Series] = {}
    for s in classes:
        p = prices[s]
        ends = p.resample("YE").last()
        r = ends.pct_change()
        first_year = p.index[0].year
        if p.index[0] <= pd.Timestamp(first_year, 1, 10):  # first year complete enough
            r.iloc[0] = ends.iloc[0] / p.iloc[0] - 1
        annual[ASSET_CLASSES[s]] = r.dropna().set_axis(r.dropna().index.year)
    table = pd.DataFrame(annual).sort_index().iloc[-15:]
    years = [str(y) for y in table.index]
    values = [[_f(v) for v in row] for row in table.to_numpy()]
    best = [
        int(np.nanargmax(row)) if np.isfinite(row).any() else None
        for row in table.to_numpy(dtype=float)
    ]
    ranked = [
        {
            "year": str(y),
            **{
                f"rank_{i + 1}": name
                for i, name in enumerate(table.loc[y].dropna().sort_values(ascending=False).index)
            },
        }
        for y in table.index
    ]
    return AnalysisOutput(
        "table",
        {
            "grid": {
                "rows": years,
                "columns": list(table.columns),
                "values": values,
                "unit": "percent",
                "best": best,
            },
            "ranked": ranked,
            "current_year_is_ytd": True,
            "missing": _missing(ASSET_CLASSES, prices),
        },
    )
