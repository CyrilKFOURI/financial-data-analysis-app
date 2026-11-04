"""Factor & cross-sectional quant (ANALYTICS_CATALOG Q-001, Q-009, Q-010, Q-011, Q-015, Q-016;
module M18).

Inputs beyond the subject's prices:
- `regressors`: {symbol: level series} for the regression lab (default: the type's benchmark);
- `ff_factors`: {"US" | "EU": frame of daily Ken French factors as decimals};
- `macro`: stored macro proxies (as P-023 `factors`) plus the 5-year breakeven (T5YIE);
- `portfolio` / `members`: holdings or comparison members for PCA.
Every regression reports Newey–West (HAC) standard errors by default; the lag is
⌊4·(T/100)^(2/9)⌋ unless set.
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, Field
from scipy import optimize, stats
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from fda_core.analyses.comparison import common_window
from fda_core.analyses.portfolio import _pf, _risk_returns
from fda_core.analyses.price import PRICE_SUBJECTS, _field, _prices, _windowed
from fda_core.analyses.ts_common import clean, iso, make_test
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

Frequency = Literal["D", "W", "M"]
RULES = {"W": "W-FRI", "M": "ME"}
PERIODS = {"D": 252, "W": 52, "M": 12}
EUROPE = {"EUR", "GBP", "GBp", "CHF", "SEK", "NOK", "DKK", "PLN"}


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def nw_lags(n: int) -> int:
    """Newey–West (1994) plug-in lag ⌊4·(T/100)^(2/9)⌋."""
    return math.floor(4.0 * math.pow(n / 100, 2 / 9))


def _returns(levels: pd.Series, freq: Frequency) -> pd.Series:
    """Simple returns at D/W/M from a level series (period-end sampling)."""
    s = levels.astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    if freq != "D":
        s = s.resample(RULES[freq]).last().dropna()
    return s.pct_change().dropna()


def _compound(daily: pd.Series | pd.DataFrame, freq: Frequency) -> Any:
    """Compounds daily simple returns to W/M (factor returns are published as returns)."""
    if freq == "D":
        return daily
    return daily.add(1).resample(RULES[freq]).prod().dropna(how="all") - 1


def _subject(inputs: AnalysisInputs) -> pd.Series:
    frame = _windowed(inputs, _prices(inputs))
    return frame[_field(inputs)].astype(float)


def _subject_full(inputs: AnalysisInputs) -> pd.Series:
    return _prices(inputs)[_field(inputs)].astype(float)


def _coef_rows(fit: Any, names: list[str], se_ols: Any) -> list[dict[str, Any]]:
    """One row per term from a statsmodels result (HAC or robust SE) plus the plain OLS SE."""
    cols = [np.asarray(a, dtype=float) for a in (fit.params, fit.bse, fit.tvalues, fit.pvalues)]
    se_plain = np.asarray(se_ols, dtype=float)
    return [
        {
            "term": name,
            "coef": _f(cols[0][i]),
            "se": _f(cols[1][i]),
            "t": _f(cols[2][i]),
            "p_value": _f(cols[3][i]),
            "se_ols": _f(se_plain[i]),
        }
        for i, name in enumerate(names)
    ]


# ── Q-001 regression lab ─────────────────────────────────────────────────────


class RegressionParams(BaseModel):
    regressors: str = Field(
        "", max_length=200, description="Comma-separated symbols (default: the benchmark)"
    )
    method: Literal["ols", "wls", "huber"] = "ols"
    frequency: Frequency = "D"
    nw_lag: int | None = Field(None, ge=0, le=60)
    holdout: float = Field(0.2, ge=0.0, le=0.5)
    rolling: int = Field(63, ge=20, le=520)


def _design(inputs: AnalysisInputs, freq: Frequency) -> tuple[pd.Series, pd.DataFrame]:
    regressors = inputs.data.get("regressors")
    if not isinstance(regressors, dict) or not regressors:
        raise InsufficientDataError("Choose regressors with stored prices (param `regressors`)")
    y = _returns(_subject(inputs), freq).rename("y")
    xs = [_returns(s, freq).rename(name) for name, s in regressors.items()]
    data = pd.concat([y, *xs], axis=1, join="inner").dropna()
    return data["y"], data.drop(columns="y")


@analysis(
    id="Q-001",
    version="1.0.0",
    title="Regression lab",
    description=(
        "y = subject returns, design = returns of chosen series at D/W/M. OLS, feasible WLS "
        "(weights from a log-squared-residual auxiliary regression) or robust Huber (RLM). "
        "Newey–West SE by default; diagnostics: Breusch–Pagan, White, Durbin–Watson, "
        "Ljung–Box(10), Jarque–Bera, VIF, Cook's distance (> 4/n); Box–Cox λ of gross returns; "
        "rolling coefficients; out-of-sample R² on the last `holdout` share (vs train mean)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RegressionParams,
    result_kind="test",
    requires={"prices", "regressors"},
    min_obs=30,
    budget_ms=6_000,
)
def regression_lab(inputs: AnalysisInputs, params: RegressionParams) -> AnalysisOutput:
    y, x = _design(inputs, params.frequency)
    k = x.shape[1]
    if len(y) < max(30, 10 * (k + 1)):
        raise InsufficientDataError(
            f"Regression needs ≥ {max(30, 10 * (k + 1))} common observations, got {len(y)}"
        )
    names = ["const", *x.columns]
    design = sm.add_constant(x.to_numpy(), has_constant="add")
    yv = y.to_numpy()
    lag = params.nw_lag if params.nw_lag is not None else nw_lags(len(y))
    ols = sm.OLS(yv, design).fit()
    hac = {"cov_type": "HAC", "cov_kwds": {"maxlags": lag}}
    weights: np.ndarray | None = None
    if params.method == "ols":
        fit = sm.OLS(yv, design).fit(**hac)
    elif params.method == "wls":
        aux = sm.OLS(np.log(ols.resid**2 + 1e-12), design).fit()
        weights = 1 / np.exp(aux.fittedvalues)
        fit = sm.WLS(yv, design, weights=weights).fit(**hac)
    else:
        fit = sm.RLM(yv, design, M=sm.robust.norms.HuberT()).fit()
    resid = yv - design @ np.asarray(fit.params, dtype=float)

    bp = het_breuschpagan(ols.resid, design)
    tests = [
        make_test(
            "breusch_pagan",
            "homoskedastic residuals",
            float(bp[0]),
            float(bp[1]),
            reading_reject="heteroskedastic: read the Newey–West SE",
            reading_keep="no evidence of heteroskedasticity",
        ),
    ]
    try:
        white = het_white(ols.resid, design)
        tests.append(
            make_test(
                "white",
                "homoskedastic residuals",
                float(white[0]),
                float(white[1]),
                reading_reject="heteroskedastic",
                reading_keep="no evidence of heteroskedasticity",
            )
        )
    except (ValueError, np.linalg.LinAlgError, AssertionError):
        pass
    lb = acorr_ljungbox(ols.resid, lags=[10])
    jb = jarque_bera(ols.resid)
    tests += [
        make_test(
            "ljung_box_10",
            "no residual autocorrelation up to lag 10",
            float(lb["lb_stat"].iloc[0]),
            float(lb["lb_pvalue"].iloc[0]),
            reading_reject="autocorrelated residuals",
            reading_keep="white-noise residuals",
        ),
        make_test(
            "jarque_bera",
            "normal residuals",
            float(jb[0]),
            float(jb[1]),
            reading_reject="non-normal residuals (fat tails)",
            reading_keep="normal",
        ),
    ]
    vif = (
        [
            {"term": c, "vif": _f(variance_inflation_factor(design, i + 1))}
            for i, c in enumerate(x.columns)
        ]
        if k > 1
        else []
    )
    cooks = OLSInfluence(ols).cooks_distance[0]
    gross = 1 + yv
    box_cox = _f(stats.boxcox(gross)[1]) if (gross > 0).all() and np.ptp(gross) > 0 else None

    oos_r2 = None
    n_test = int(len(y) * params.holdout)
    if n_test >= 10 and len(y) - n_test >= 10 * (k + 1):
        train = sm.OLS(yv[:-n_test], design[:-n_test]).fit()
        pred = design[-n_test:] @ train.params
        sse = float(((yv[-n_test:] - pred) ** 2).sum())
        sst = float(((yv[-n_test:] - yv[:-n_test].mean()) ** 2).sum())
        oos_r2 = 1 - sse / sst if sst > 0 else None

    lines: dict[str, list[float | None]] = {}
    dates: list[str] = []
    if len(y) > params.rolling + 10:
        roll = RollingOLS(yv, design, window=params.rolling).fit()
        coefs = np.asarray(roll.params)
        dates = iso(y.index)
        for i, name in enumerate(names):
            lines[f"beta_{name}" if name != "const" else "alpha"] = clean(coefs[:, i])

    r2 = _f(ols.rsquared) if params.method != "huber" else _f(1 - resid.var() / yv.var())
    return AnalysisOutput(
        "test",
        {
            "method": params.method,
            "frequency": params.frequency,
            "n": len(y),
            "nw_lag": lag if params.method != "huber" else None,
            "se_kind": "Newey–West (HAC)" if params.method != "huber" else "robust (Huber)",
            "r2": r2,
            "adj_r2": _f(ols.rsquared_adj),
            "oos_r2": oos_r2,
            "durbin_watson": _f(durbin_watson(ols.resid)),
            "box_cox_lambda": box_cox,
            "influential_points": int((cooks > 4 / len(y)).sum()),
            "max_cooks_distance": _f(cooks.max()),
            "coefficients": _coef_rows(fit, names, ols.bse),
            "vif": vif,
            "tests": tests,
            "dates": dates,
            "lines": lines,
        },
        warnings=() if weights is None else ("WLS weights estimated from the OLS residuals",),
    )


# ── Q-010 factor exposures ───────────────────────────────────────────────────

MODELS: dict[str, list[str]] = {
    "capm": ["MKTRF"],
    "ff3": ["MKTRF", "SMB", "HML"],
    "carhart": ["MKTRF", "SMB", "HML", "MOM"],
    "ff5": ["MKTRF", "SMB", "HML", "RMW", "CMA"],
    "ff5_mom": ["MKTRF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


class FactorParams(BaseModel):
    model: Literal["capm", "ff3", "carhart", "ff5", "ff5_mom"] = "ff5"
    region: Literal["auto", "US", "EU"] = "auto"
    frequency: Literal["D", "M"] = "D"
    rolling_months: int = Field(36, ge=24, le=120)


def _region(inputs: AnalysisInputs, choice: str) -> str:
    if choice != "auto":
        return choice
    return "EU" if str(inputs.conventions.get("currency")) in EUROPE else "US"


def _factor_frame(inputs: AnalysisInputs, region: str, names: list[str]) -> pd.DataFrame:
    stored = inputs.data.get("ff_factors")
    frame = stored.get(region) if isinstance(stored, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise InsufficientDataError(
            f"Ingest the Ken French {region} factors (POST /v1/factors/{region}/ingest)"
        )
    missing = [c for c in [*names, "RF"] if c not in frame.columns]
    if missing:
        raise InsufficientDataError(f"Missing Ken French {region} series: {', '.join(missing)}")
    return frame[[*names, "RF"]].dropna()


def style_box(betas: dict[str, float]) -> dict[str, Any] | None:
    """Size × value cell from SMB/HML loadings (large ≤ 0 < mid ≤ 0.4 < small; growth < −0.2 ≤
    blend ≤ 0.2 < value)."""
    smb, hml = betas.get("SMB"), betas.get("HML")
    if smb is None or hml is None:
        return None
    size = "large" if smb <= 0 else "mid" if smb <= 0.4 else "small"
    value = "growth" if hml < -0.2 else "value" if hml > 0.2 else "blend"
    return {"size": size, "style": value, "smb": smb, "hml": hml}


@analysis(
    id="Q-010",
    version="1.0.0",
    title="Factor exposures",
    description=(
        "Excess returns (r − RF) regressed on Ken French factors: CAPM, FF3, Carhart, FF5, "
        "FF5 + momentum (US or developed Europe, auto by currency; FF3 uses the 2×3 five-factor "
        "SMB). β with Newey–West t, R², α annualised with its t. Attribution over the window: "
        "Σβ·Σf + α·T + Σε = Σ excess. Systematic share of variance β'Σβ / var(r). Rolling "
        "exposures on monthly returns over the full history. Style box from SMB/HML."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FactorParams,
    result_kind="table",
    requires={"prices", "ff_factors"},
    min_obs=60,
    budget_ms=5_000,
)
def factor_exposures(inputs: AnalysisInputs, params: FactorParams) -> AnalysisOutput:
    region = _region(inputs, params.region)
    names = MODELS[params.model]
    factors = _factor_frame(inputs, region, names)
    factors.index = pd.to_datetime(factors.index)
    r = _returns(_subject(inputs), "D").rename("r")
    data = pd.concat([r, factors], axis=1, join="inner").dropna()
    if params.frequency == "M":
        data = _compound(data, "M")
    minimum = 24 if params.frequency == "M" else 60
    if len(data) < minimum:
        raise InsufficientDataError(
            f"Factor regression needs ≥ {minimum} common observations with the {region} factors "
            f"(they end {iso(factors.index[-1:])[0]}), got {len(data)}"
        )
    excess = (data["r"] - data["RF"]).to_numpy()
    fac = data[names].to_numpy()
    design = sm.add_constant(fac, has_constant="add")
    lag = nw_lags(len(data))
    fit = sm.OLS(excess, design).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    periods = PERIODS[params.frequency]
    betas = {n: float(fit.params[i + 1]) for i, n in enumerate(names)}
    attribution = [
        {"source": n, "contribution": float(betas[n] * fac[:, i].sum())}
        for i, n in enumerate(names)
    ]
    attribution += [
        {"source": "alpha", "contribution": float(fit.params[0] * len(data))},
        {"source": "residual", "contribution": float(fit.resid.sum())},
        {"source": "total excess", "contribution": float(excess.sum())},
    ]
    beta_vec = np.array([betas[n] for n in names])
    cov = np.atleast_2d(np.cov(fac, rowvar=False, ddof=1))
    systematic = float(beta_vec @ cov @ beta_vec)
    total_var = float(np.var(excess, ddof=1))

    dates: list[str] = []
    lines: dict[str, list[float | None]] = {}
    full = pd.concat(
        [_returns(_subject_full(inputs), "D").rename("r"), factors], axis=1, join="inner"
    ).dropna()
    monthly = _compound(full, "M")
    if len(monthly) > params.rolling_months + 2:
        design_m = sm.add_constant(monthly[names].to_numpy(), has_constant="add")
        roll = RollingOLS(
            (monthly["r"] - monthly["RF"]).to_numpy(), design_m, window=params.rolling_months
        ).fit()
        coefs = np.asarray(roll.params)
        dates = iso(monthly.index)
        lines = {n: clean(coefs[:, i + 1]) for i, n in enumerate(names)}

    warnings = []
    if str(inputs.conventions.get("currency")) not in {"USD", "None"}:
        warnings.append(
            "Ken French factors are USD returns; the subject is in "
            f"{inputs.conventions.get('currency')}, so its FX return is part of α and ε"
        )
    return AnalysisOutput(
        "table",
        {
            "model": params.model,
            "region": region,
            "frequency": params.frequency,
            "n": len(data),
            "from": iso(data.index[:1])[0],
            "to": iso(data.index[-1:])[0],
            "nw_lag": lag,
            "r2": _f(fit.rsquared),
            "adj_r2": _f(fit.rsquared_adj),
            "alpha_annualised": _f(fit.params[0] * periods),
            "alpha_t": _f(fit.tvalues[0]),
            "alpha_p": _f(fit.pvalues[0]),
            "systematic_share": _f(systematic / total_var) if total_var > 0 else None,
            "rows": [
                {
                    "factor": n,
                    "beta": betas[n],
                    "t_nw": _f(fit.tvalues[i + 1]),
                    "p_value": _f(fit.pvalues[i + 1]),
                    "factor_return_annualised": _f(fac[:, i].mean() * periods),
                }
                for i, n in enumerate(names)
            ],
            "attribution": attribution,
            "style_box": style_box(betas),
            "dates": dates,
            "lines": lines,
        },
        warnings=tuple(warnings),
    )


# ── Q-011 macro factor sensitivity ───────────────────────────────────────────

MACRO_TRANSFORMS: dict[str, tuple[str, str]] = {
    "equity": ("return", "market (SPY)"),
    "rates_10y": ("change", "Δ 10y yield (^TNX, pp)"),
    "usd": ("return", "USD index"),
    "oil": ("return", "oil (WTI)"),
    "credit": ("change", "Δ HY OAS (pp)"),
    "inflation": ("change", "Δ 5y breakeven (pp)"),
}


CONTROL_MAX_CORR = 0.98


class MacroParams(BaseModel):
    frequency: Literal["W", "M"] = "W"
    lags: int = Field(1, ge=0, le=4)
    control_market: bool = True


def _macro_x(level: pd.Series, kind: str, freq: Frequency) -> pd.Series:
    s = level.astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.resample(RULES[freq]).last().dropna()
    return s.pct_change() if kind == "return" else s.diff()


@analysis(
    id="Q-011",
    version="1.0.0",
    title="Macro factor sensitivity",
    description=(
        "Weekly or monthly returns regressed on macro changes with 0…L lags: Δ10y yield, USD "
        "index return, oil return, ΔHY OAS, Δ5y breakeven (inflation-surprise proxy), with the "
        "market as control (dropped when |corr| > 0.98 with the subject, e.g. SPY itself). "
        "Per factor: Σβ over lags with its Newey–West t (Wald on the sum)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=MacroParams,
    result_kind="table",
    requires={"prices", "macro"},
    min_obs=40,
)
def macro_sensitivity(inputs: AnalysisInputs, params: MacroParams) -> AnalysisOutput:
    stored = inputs.data.get("macro")
    stored = stored if isinstance(stored, dict) else {}
    used = [n for n in MACRO_TRANSFORMS if n in stored and (params.control_market or n != "equity")]
    missing = [n for n in MACRO_TRANSFORMS if n not in stored]
    if len([n for n in used if n != "equity"]) < 1:
        raise InsufficientDataError(
            "Ingest macro series: ^TNX, DX-Y.NYB, CL=fac, FRED BAMLH0A0HYM2 and T5YIE"
        )
    y = _returns(_subject(inputs), params.frequency).rename("y")
    notes = []
    if "equity" in used:
        market = _macro_x(stored["equity"], "return", params.frequency)
        if abs(pd.concat([y, market], axis=1, join="inner").corr().iloc[0, 1]) > CONTROL_MAX_CORR:
            used.remove("equity")  # the subject is the market proxy: a control would explain all
            notes.append("Market control dropped: the subject tracks the market proxy")
    columns: dict[str, pd.Series] = {}
    for n in used:
        x = _macro_x(stored[n], MACRO_TRANSFORMS[n][0], params.frequency)
        for lag in range(params.lags + 1 if n != "equity" else 1):
            columns[f"{n}_l{lag}"] = x.shift(lag)
    data = pd.concat([y, pd.DataFrame(columns)], axis=1, join="inner").dropna()
    k = data.shape[1] - 1
    if len(data) < max(40, 5 * (k + 1)):
        raise InsufficientDataError(
            f"Macro regression needs ≥ {max(40, 5 * (k + 1))} periods, got {len(data)}"
        )
    names = list(columns)
    design = sm.add_constant(data[names].to_numpy(), has_constant="add")
    lag = nw_lags(len(data))
    fit = sm.OLS(data["y"].to_numpy(), design).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    rows = []
    for n in used:
        idx = [i + 1 for i, c in enumerate(names) if c.startswith(f"{n}_l")]
        weights = np.zeros(len(names) + 1)
        weights[idx] = 1.0
        test = fit.t_test(weights)
        rows.append(
            {
                "factor": MACRO_TRANSFORMS[n][1],
                "key": n,
                "beta_sum": _f(np.asarray(test.effect).ravel()[0]),
                "t_nw": _f(np.asarray(test.tvalue).ravel()[0]),
                "p_value": _f(np.asarray(test.pvalue).ravel()[0]),
                "beta_lag0": _f(fit.params[idx[0]]),
                "control": n == "equity",
            }
        )
    return AnalysisOutput(
        "table",
        {
            "frequency": params.frequency,
            "lags": params.lags,
            "n": len(data),
            "nw_lag": lag,
            "r2": _f(fit.rsquared),
            "adj_r2": _f(fit.rsquared_adj),
            "rows": rows,
            "missing": [MACRO_TRANSFORMS[n][1] for n in missing],
        },
        warnings=tuple(notes)
        + (
            (f"Not stored: {', '.join(MACRO_TRANSFORMS[n][1] for n in missing)}",)
            if missing
            else ()
        ),
    )


# ── Q-009 PCA & statistical factors ──────────────────────────────────────────


class PcaParams(BaseModel):
    standardise: bool = Field(True, description="Correlation (true) or covariance PCA")
    components: int = Field(5, ge=1, le=10)
    rolling: int = Field(252, ge=60, le=756)


def _panel(inputs: AnalysisInputs) -> pd.DataFrame:
    if isinstance(inputs.data.get("members"), list):
        members = inputs.data["members"]
        start, end = common_window(inputs, members)
        levels = pd.concat({m["label"]: m["series"] for m in members}, axis=1)
        levels = levels.loc[(levels.index >= start) & (levels.index <= end)]
        return levels.ffill().pct_change().iloc[1:].dropna()
    return _risk_returns(inputs, _pf(inputs), minimum=60).dropna(axis=1, how="all").dropna()


def pca(returns: np.ndarray, standardise: bool) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of the correlation or covariance
    matrix; each vector's largest-|loading| entry is made positive (sign is arbitrary)."""
    x = returns - returns.mean(axis=0)
    if standardise:
        x = x / x.std(axis=0, ddof=1)
    values, vectors = np.linalg.eigh(np.cov(x, rowvar=False, ddof=1))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
    return values, vectors * np.where(signs == 0, 1, signs)


@analysis(
    id="Q-009",
    version="1.0.0",
    title="PCA & statistical factors",
    description=(
        "Eigen-decomposition of the correlation (or covariance) matrix of holdings' or members' "
        "daily returns: scree and cumulative variance, loadings, eigen-portfolios (eigenvector "
        "/ Σ), PC1 correlation with the equal-weight average, rolling absorption ratio (share of "
        "variance in the top ⌈N/5⌉ components)."
    ),
    applies_to={"portfolio", "comparison"},
    params=PcaParams,
    result_kind="table",
    requires={"portfolio"},
    min_obs=60,
)
def principal_components(inputs: AnalysisInputs, params: PcaParams) -> AnalysisOutput:
    r = _panel(inputs)
    r = r.loc[:, r.std() > 0]
    n_assets = r.shape[1]
    if n_assets < 3:
        raise InsufficientDataError(f"PCA needs ≥ 3 series with varying returns, got {n_assets}")
    if len(r) < max(60, 2 * n_assets):
        raise InsufficientDataError(f"PCA needs ≥ {max(60, 2 * n_assets)} sessions, got {len(r)}")
    values, vectors = pca(r.to_numpy(), params.standardise)
    explained = values / values.sum()
    m = min(params.components, n_assets)
    pcs = [f"PC{i + 1}" for i in range(m)]
    scores = (r.to_numpy() - r.to_numpy().mean(axis=0)) @ vectors[:, 0]
    pc1_corr = float(np.corrcoef(scores, r.mean(axis=1).to_numpy())[0, 1])
    eigen = []
    for i in range(min(3, m)):
        total = vectors[:, i].sum()
        if abs(total) > 1e-9:
            eigen.append(
                {
                    "pc": pcs[i],
                    **{c: float(w) for c, w in zip(r.columns, vectors[:, i] / total, strict=True)},
                }
            )
    top = math.ceil(n_assets / 5)
    dates: list[str] = []
    absorption: list[float | None] = []
    if len(r) > params.rolling + 5:
        arr = r.to_numpy()
        for end in range(params.rolling, len(r) + 1):
            vals = np.linalg.eigvalsh(np.corrcoef(arr[end - params.rolling : end], rowvar=False))
            vals = np.sort(np.nan_to_num(vals))[::-1]
            absorption.append(float(vals[:top].sum() / vals.sum()))
        dates = iso(r.index[params.rolling - 1 :])
    return AnalysisOutput(
        "table",
        {
            "n_series": n_assets,
            "observations": len(r),
            "basis": "correlation" if params.standardise else "covariance",
            "pc1_share": float(explained[0]),
            "pc1_vs_average_corr": pc1_corr,
            "absorption_top": top,
            "scree": [
                {
                    "component": f"PC{i + 1}",
                    "eigenvalue": float(v),
                    "explained": float(e),
                    "cumulative": float(c),
                }
                for i, (v, e, c) in enumerate(
                    zip(values, explained, np.cumsum(explained), strict=True)
                )
            ],
            "grid": {
                "rows": list(r.columns),
                "columns": pcs,
                "values": [[float(v) for v in row[:m]] for row in vectors],
                "unit": "number",
                "best": [],
            },
            "eigen_portfolios": eigen,
            "dates": dates,
            "lines": {"absorption_ratio": absorption} if absorption else {},
        },
    )


# ── Q-015 copulas ────────────────────────────────────────────────────────────


class CopulaParams(BaseModel):
    other: str | None = Field(None, max_length=20)
    tail_quantile: float = Field(0.05, gt=0.0, le=0.2)


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    return np.asarray(stats.rankdata(x), dtype=float) / (len(x) + 1)


def gaussian_loglik(u: np.ndarray, v: np.ndarray, rho: float) -> float:
    a, b = stats.norm.ppf(u), stats.norm.ppf(v)
    det = 1 - rho**2
    return float(
        np.sum(-0.5 * np.log(det) - (rho**2 * (a**2 + b**2) - 2 * rho * a * b) / (2 * det))
    )


def t_loglik(u: np.ndarray, v: np.ndarray, rho: float, nu: float) -> float:
    a, b = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    det = 1 - rho**2
    q = (a**2 + b**2 - 2 * rho * a * b) / det
    joint = (
        math.lgamma((nu + 2) / 2)
        - math.lgamma(nu / 2)
        - math.log(nu * math.pi)
        - 0.5 * math.log(det)
        - (nu + 2) / 2 * np.log1p(q / nu)
    )
    return float(np.sum(joint - stats.t.logpdf(a, nu) - stats.t.logpdf(b, nu)))


def clayton_loglik(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    s = u**-theta + v**-theta - 1
    return float(
        np.sum(np.log1p(theta) - (1 + theta) * np.log(u * v) - (2 + 1 / theta) * np.log(s))
    )


def gumbel_loglik(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    lu, lv = -np.log(u), -np.log(v)
    a = lu**theta + lv**theta
    a_t = a ** (1 / theta)
    return float(
        np.sum(
            -a_t
            - np.log(u * v)
            + (theta - 1) * np.log(lu * lv)
            + (-2 + 2 / theta) * np.log(a)
            + np.log1p((theta - 1) * a ** (-1 / theta))
        )
    )


def fit_copulas(u: np.ndarray, v: np.ndarray) -> list[dict[str, Any]]:
    """Gaussian, t, Clayton, Gumbel by maximum likelihood on pseudo-observations, with AIC and
    the implied lower/upper tail-dependence coefficients."""
    tau = float(stats.kendalltau(u, v)[0])
    out: list[dict[str, Any]] = []
    g = optimize.minimize_scalar(
        lambda r: -gaussian_loglik(u, v, r), bounds=(-0.99, 0.99), method="bounded"
    )
    out.append(
        {
            "copula": "gaussian",
            "parameter": float(g.x),
            "nu": None,
            "loglik": -float(g.fun),
            "k": 1,
            "lambda_lower": 0.0,
            "lambda_upper": 0.0,
        }
    )
    rho_t = float(np.clip(math.sin(math.pi * tau / 2), -0.99, 0.99))
    nus = np.arange(2.5, 40.5, 0.5)
    lls = [t_loglik(u, v, rho_t, nu) for nu in nus]
    nu = float(nus[int(np.argmax(lls))])
    tf = optimize.minimize_scalar(
        lambda r: -t_loglik(u, v, r, nu), bounds=(-0.99, 0.99), method="bounded"
    )
    rho = float(tf.x)
    lam = float(2 * stats.t.cdf(-math.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))
    out.append(
        {
            "copula": "student_t",
            "parameter": rho,
            "nu": nu,
            "loglik": -float(tf.fun),
            "k": 2,
            "lambda_lower": lam,
            "lambda_upper": lam,
        }
    )
    if tau > 0.01:
        c = optimize.minimize_scalar(
            lambda t: -clayton_loglik(u, v, t), bounds=(1e-3, 30), method="bounded"
        )
        out.append(
            {
                "copula": "clayton",
                "parameter": float(c.x),
                "nu": None,
                "loglik": -float(c.fun),
                "k": 1,
                "lambda_lower": float(2 ** (-1 / c.x)),
                "lambda_upper": 0.0,
            }
        )
        gm = optimize.minimize_scalar(
            lambda t: -gumbel_loglik(u, v, t), bounds=(1.0001, 30), method="bounded"
        )
        out.append(
            {
                "copula": "gumbel",
                "parameter": float(gm.x),
                "nu": None,
                "loglik": -float(gm.fun),
                "k": 1,
                "lambda_lower": 0.0,
                "lambda_upper": float(2 - 2 ** (1 / gm.x)),
            }
        )
    for row in out:
        row["aic"] = 2 * row["k"] - 2 * row["loglik"]
    best = min(out, key=lambda r: r["aic"])
    for row in out:
        row["best"] = row is best
    return out


@analysis(
    id="Q-015",
    version="1.0.0",
    title="Copulas",
    description=(
        "Pseudo-observations (ranks / (n+1)) of the pair's daily log returns; Gaussian, Student t "
        "(ν by profile likelihood), Clayton and Gumbel fitted by maximum likelihood (the last two "
        "only for positive dependence); AIC comparison; implied tail-dependence λ_L / λ_U vs the "
        "empirical joint-exceedance ratio at the tail quantile."
    ),
    applies_to=PRICE_SUBJECTS,
    params=CopulaParams,
    result_kind="table",
    requires={"prices", "other_prices"},
    min_obs=250,
    budget_ms=6_000,
)
def copulas(inputs: AnalysisInputs, params: CopulaParams) -> AnalysisOutput:
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError("Choose a second asset with stored prices (param `other`)")
    field = _field(inputs)
    both = pd.concat([_subject(inputs), other[field].astype(float)], axis=1, join="inner").dropna()
    r = np.log(both).diff().dropna()
    if len(r) < 250:
        raise InsufficientDataError(f"Copula fits need ≥ 250 common sessions, got {len(r)}")
    u, v = pseudo_obs(r.iloc[:, 0].to_numpy()), pseudo_obs(r.iloc[:, 1].to_numpy())
    q = params.tail_quantile
    return AnalysisOutput(
        "table",
        {
            "other": inputs.conventions.get("other_symbol"),
            "n": len(r),
            "kendall_tau": _f(stats.kendalltau(u, v)[0]),
            "spearman_rho": _f(stats.spearmanr(u, v)[0]),
            "empirical_lower": float(np.mean((u < q) & (v < q)) / q),
            "empirical_upper": float(np.mean((u > 1 - q) & (v > 1 - q)) / q),
            "tail_quantile": q,
            "rows": fit_copulas(u, v),
        },
    )


# ── Q-016 event study ────────────────────────────────────────────────────────


class EventParams(BaseModel):
    events: str = Field("", max_length=2000, description="Comma-separated dates YYYY-MM-DD")
    other: str | None = Field(None, max_length=20)
    estimation: int = Field(120, ge=30, le=500)
    gap: int = Field(10, ge=0, le=60)
    before: int = Field(5, ge=0, le=30)
    after: int = Field(5, ge=0, le=60)


def event_study(
    r: pd.Series, m: pd.Series, events: list[pd.Timestamp], params: EventParams
) -> tuple[list[dict[str, Any]], np.ndarray, list[pd.Timestamp]]:
    """Market-model abnormal returns: per event (CAR, σ, SCAR) and the AR matrix (event × day)."""
    data = pd.concat([r, m], axis=1, join="inner").dropna().to_numpy()
    index = pd.concat([r, m], axis=1, join="inner").dropna().index
    rows, ars, used = [], [], []
    length = params.before + params.after + 1
    for event in events:
        t0 = int(index.searchsorted(event))
        lo, hi = t0 - params.before, t0 + params.after
        est_end = lo - params.gap
        est_start = est_end - params.estimation
        if est_start < 0 or hi >= len(index):
            continue
        est = data[est_start:est_end]
        beta, alpha = np.polyfit(est[:, 1], est[:, 0], 1)
        sigma = float(np.std(est[:, 0] - (alpha + beta * est[:, 1]), ddof=2))
        window = data[lo : hi + 1]
        ar = window[:, 0] - (alpha + beta * window[:, 1])
        car = float(ar.sum())
        scar = car / (sigma * math.sqrt(length)) if sigma > 0 else float("nan")
        rows.append(
            {
                "event": iso(index[t0 : t0 + 1])[0],
                "requested": event.date().isoformat(),
                "alpha": float(alpha),
                "beta": float(beta),
                "sigma": sigma,
                "car": car,
                "scar": _f(scar),
                "t": _f(scar),
            }
        )
        ars.append(ar)
        used.append(event)
    return rows, np.array(ars), used


@analysis(
    id="Q-016",
    version="1.0.0",
    title="Event study",
    description=(
        "Market-model abnormal returns around user dates (estimation window before a gap, event "
        "window [−before, +after] sessions). Per event CAR and SCAR = CAR / (σ_est √L); average "
        "AR and CAAR path with a 95 % cross-sectional band; cross-sectional t of CAR and BMP "
        "(Boehmer–Musumeci–Poulsen) t on SCARs."
    ),
    applies_to=PRICE_SUBJECTS,
    params=EventParams,
    result_kind="test",
    requires={"prices", "other_prices"},
    min_obs=60,
)
def generic_event_study(inputs: AnalysisInputs, params: EventParams) -> AnalysisOutput:
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError("Choose a market series with stored prices (param `other`)")
    try:
        events = sorted({pd.Timestamp(e.strip()) for e in params.events.split(",") if e.strip()})
    except ValueError as exc:
        raise InsufficientDataError(f"Events must be YYYY-MM-DD dates: {exc}") from exc
    if not events:
        raise InsufficientDataError(
            "List event dates (param `events`, YYYY-MM-DD, comma-separated)"
        )
    field = _field(inputs)
    r = _subject_full(inputs).pct_change().dropna()
    m = other[field].astype(float).pct_change().dropna()
    r.index, m.index = pd.to_datetime(r.index), pd.to_datetime(m.index)
    rows, ars, used = event_study(r, m, events, params)
    if not rows:
        raise InsufficientDataError(
            "No event has a full estimation and event window in the stored history"
        )
    n = len(rows)
    days = list(range(-params.before, params.after + 1))
    aar = ars.mean(axis=0)
    caar = np.cumsum(aar)
    car_paths = np.cumsum(ars, axis=1)
    band = (
        1.96 * car_paths.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.full(len(days), np.nan)
    )
    cars = np.array([row["car"] for row in rows])
    scars = np.array([row["scar"] if row["scar"] is not None else np.nan for row in rows])
    tests = []
    if n > 1:
        t_cs = cars.mean() / (cars.std(ddof=1) / math.sqrt(n))
        tests.append(
            make_test(
                "cross_sectional_t",
                "mean CAR = 0",
                float(t_cs),
                float(2 * stats.t.sf(abs(t_cs), n - 1)),
                reading_reject="abnormal reaction",
                reading_keep="no abnormal reaction",
            )
        )
        s = scars[np.isfinite(scars)]
        if len(s) > 1:
            t_bmp = s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))
            tests.append(
                make_test(
                    "bmp",
                    "mean SCAR = 0",
                    float(t_bmp),
                    float(2 * stats.t.sf(abs(t_bmp), len(s) - 1)),
                    reading_reject="abnormal reaction (variance-robust)",
                    reading_keep="no abnormal reaction",
                )
            )
    else:
        t1 = rows[0]["scar"]
        tests.append(
            make_test(
                "single_event_t",
                "CAR = 0",
                t1,
                None if t1 is None else float(2 * stats.norm.sf(abs(t1))),
                reading_reject="abnormal reaction",
                reading_keep="no abnormal reaction",
            )
        )
    return AnalysisOutput(
        "test",
        {
            "events_used": n,
            "events_skipped": len(events) - len(used),
            "market": inputs.conventions.get("other_symbol"),
            "window": [-params.before, params.after],
            "mean_car": float(cars.mean()),
            "tests": tests,
            "path": [
                {
                    "day": d,
                    "aar": float(a),
                    "caar": float(c),
                    "lower": _f(c - b),
                    "upper": _f(c + b),
                }
                for d, a, c, b in zip(days, aar, caar, band, strict=True)
            ],
            "events": rows,
        },
        warnings=(f"{len(events) - len(used)} event(s) lack a full window and were skipped",)
        if len(used) < len(events)
        else (),
    )
