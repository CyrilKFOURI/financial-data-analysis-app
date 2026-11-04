"""Golden tests for M18: Q-001, Q-009, Q-010, Q-011, Q-015, Q-016 on synthetic data with known
parameters."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fda_core.analyses.quant import (
    CopulaParams,
    EventParams,
    FactorParams,
    MacroParams,
    PcaParams,
    RegressionParams,
    copulas,
    factor_exposures,
    generic_event_study,
    macro_sensitivity,
    nw_lags,
    pca,
    principal_components,
    regression_lab,
    style_box,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

DAYS = pd.bdate_range("2019-01-02", periods=1500)


def prices(returns: np.ndarray, index: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
    level = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"adj_close": level, "close": level}, index=index)


def conv(**extra: object) -> dict[str, object]:
    return {
        "range": "MAX",
        "currency": "USD",
        "annualisation": 252,
        "price_field": "adj_close",
    } | extra


def factors(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0, [0.01, 0.005, 0.005, 0.004, 0.004, 0.006], (len(DAYS), 6))
    frame = pd.DataFrame(data, index=DAYS, columns=["MKTRF", "SMB", "HML", "RMW", "CMA", "MOM"])
    frame["RF"] = 0.0001
    return frame


def test_ff5_recovers_known_loadings_and_attribution_adds_up() -> None:
    f = factors()
    rng = np.random.default_rng(1)
    excess = 0.0002 + 1.1 * f["MKTRF"] + 0.6 * f["SMB"] - 0.4 * f["HML"] + 0.2 * f["RMW"]
    r = f["RF"] + excess + rng.normal(0, 0.001, len(f))
    inputs = AnalysisInputs({"prices": prices(r.to_numpy()), "ff_factors": {"US": f}}, conv())
    out = factor_exposures(inputs, FactorParams(model="ff5")).payload
    betas = {row["factor"]: row["beta"] for row in out["rows"]}
    # The level series starts at the first return, so returns are recomputed from day 2.
    assert betas["MKTRF"] == pytest.approx(1.1, abs=0.02)
    assert betas["SMB"] == pytest.approx(0.6, abs=0.03)
    assert betas["HML"] == pytest.approx(-0.4, abs=0.03)
    assert betas["CMA"] == pytest.approx(0.0, abs=0.03)
    assert out["r2"] > 0.95
    assert out["alpha_annualised"] == pytest.approx(0.0002 * 252, abs=0.02)
    parts = {a["source"]: a["contribution"] for a in out["attribution"]}
    total = parts.pop("total excess")
    assert sum(parts.values()) == pytest.approx(total, abs=1e-10)
    assert out["style_box"]["size"] == "small"
    assert out["style_box"]["style"] == "growth"
    assert 0.9 < out["systematic_share"] <= 1.0
    assert set(out["lines"]) == {"MKTRF", "SMB", "HML", "RMW", "CMA"}  # rolling 36 months


def test_factor_model_regions_and_missing_data() -> None:
    f = factors()
    r = f["RF"] + f["MKTRF"]
    eur = AnalysisInputs(
        {"prices": prices(r.to_numpy()), "ff_factors": {"US": f}}, conv(currency="EUR")
    )
    with pytest.raises(InsufficientDataError, match="EU factors"):
        factor_exposures(eur, FactorParams())
    out = factor_exposures(eur, FactorParams(region="US", model="capm"))
    assert out.payload["rows"][0]["beta"] == pytest.approx(1.0, abs=1e-6)
    assert any("USD returns" in w for w in out.warnings)
    assert style_box({"MKTRF": 1.0}) is None
    assert nw_lags(1000) == 6


def test_regression_lab_flags_heteroskedasticity_and_recovers_slope() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, len(DAYS))
    noise = rng.normal(0, 1, len(DAYS)) * (0.002 + 0.5 * np.abs(x))  # variance grows with |x|
    y = 0.0003 + 1.5 * x + noise
    inputs = AnalysisInputs(
        {"prices": prices(y), "regressors": {"X": prices(x)["adj_close"]}}, conv()
    )
    out = regression_lab(inputs, RegressionParams()).payload
    slope = out["coefficients"][1]
    assert slope["term"] == "X"
    assert slope["coef"] == pytest.approx(1.5, abs=0.05)
    assert slope["se"] != slope["se_ols"]  # Newey–West differs from the plain SE
    tests = {t["test"]: t for t in out["tests"]}
    assert tests["white"]["reject"] is True  # variance in |x|: White, not linear BP
    assert out["oos_r2"] is not None
    assert out["oos_r2"] > 0.5
    assert "beta_X" in out["lines"]
    assert len(out["lines"]["beta_X"]) == len(out["dates"])
    huber = regression_lab(inputs, RegressionParams(method="huber")).payload
    assert huber["coefficients"][1]["coef"] == pytest.approx(1.5, abs=0.05)
    wls = regression_lab(inputs, RegressionParams(method="wls")).payload
    assert wls["coefficients"][1]["coef"] == pytest.approx(1.5, abs=0.05)
    with pytest.raises(InsufficientDataError, match="regressors"):
        regression_lab(AnalysisInputs({"prices": prices(y)}, conv()), RegressionParams())


def test_pca_matches_the_svd_of_the_standardised_panel() -> None:
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, len(DAYS))
    members = []
    for i in range(6):
        r = (0.5 + 0.1 * i) * common + rng.normal(0, 0.006, len(DAYS))
        members.append({"label": f"S{i}", "series": prices(r)["adj_close"], "annualisation": 252})
    inputs = AnalysisInputs({"members": members}, conv())
    out = principal_components(inputs, PcaParams(rolling=252)).payload
    panel = pd.concat({m["label"]: m["series"] for m in members}, axis=1).pct_change().iloc[1:]
    z = (panel - panel.mean()) / panel.std(ddof=1)
    _, s, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    explained = s**2 / (s**2).sum()
    assert [row["explained"] for row in out["scree"]] == pytest.approx(explained, abs=1e-10)
    loadings = np.array(out["grid"]["values"])
    assert np.abs(loadings[:, 0]) == pytest.approx(np.abs(vt[0]), abs=1e-10)
    assert out["pc1_share"] > 0.5
    assert abs(out["pc1_vs_average_corr"]) > 0.95
    assert len(out["lines"]["absorption_ratio"]) == len(out["dates"])
    values, _ = pca(z.to_numpy(), standardise=False)
    assert values.sum() == pytest.approx(6.0)  # trace of a correlation matrix


def clayton_pair(theta: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    return u, v


def test_copulas_pick_clayton_for_lower_tail_dependence() -> None:
    u, v = clayton_pair(2.0, 1500, 4)
    a = stats.norm.ppf(u) * 0.01
    b = stats.t.ppf(v, 4) * 0.01  # margins do not matter: ranks only
    index = pd.bdate_range("2019-01-02", periods=len(a) + 1)
    inputs = AnalysisInputs(
        {"prices": prices(np.r_[0, a], index), "other_prices": prices(np.r_[0, b], index)}, conv()
    )
    out = copulas(inputs, CopulaParams()).payload
    rows = {r["copula"]: r for r in out["rows"]}
    assert rows["clayton"]["best"] is True
    assert rows["clayton"]["parameter"] == pytest.approx(2.0, rel=0.15)
    assert out["kendall_tau"] == pytest.approx(2.0 / (2.0 + 2), abs=0.05)  # τ = θ/(θ+2)
    assert rows["clayton"]["lambda_lower"] == pytest.approx(2 ** (-1 / 2.0), abs=0.05)
    assert out["empirical_lower"] > out["empirical_upper"]


def test_event_study_detects_an_injected_abnormal_return() -> None:
    rng = np.random.default_rng(5)
    m = rng.normal(0.0003, 0.01, len(DAYS))
    r = 0.0001 + 1.2 * m + rng.normal(0, 0.002, len(DAYS))
    events = [DAYS[200 + 150 * i] for i in range(8)]
    for e in events:
        r[DAYS.get_loc(e)] += 0.05
    inputs = AnalysisInputs(
        {"prices": prices(r), "other_prices": prices(m)}, conv(other_symbol="MKT")
    )
    dates = ",".join(e.date().isoformat() for e in events) + ",2019-01-03"
    out = generic_event_study(inputs, EventParams(events=dates))
    p = out.payload
    assert p["events_used"] == 8
    assert p["events_skipped"] == 1
    assert p["mean_car"] == pytest.approx(0.05, abs=0.01)
    day0 = next(row for row in p["path"] if row["day"] == 0)
    assert day0["aar"] == pytest.approx(0.05, abs=0.006)
    assert all(t["reject"] for t in p["tests"])
    assert p["events"][0]["beta"] == pytest.approx(1.2, abs=0.1)
    with pytest.raises(InsufficientDataError, match="events"):
        generic_event_study(inputs, EventParams())


def test_macro_sensitivity_sums_lags() -> None:
    rng = np.random.default_rng(6)
    weeks = pd.date_range("2015-01-02", periods=500, freq="W-FRI")
    rates = pd.Series(2 + np.cumsum(rng.normal(0, 0.08, len(weeks))), index=weeks)
    credit = pd.Series(4 + np.cumsum(rng.normal(0, 0.1, len(weeks))), index=weeks)
    d_rates = rates.diff().fillna(0).to_numpy()
    lagged = np.r_[0, d_rates[:-1]]
    r = -0.03 * d_rates - 0.02 * lagged + rng.normal(0, 0.002, len(weeks))
    inputs = AnalysisInputs(
        {"prices": prices(r, weeks), "macro": {"rates_10y": rates, "credit": credit}}, conv()
    )
    out = macro_sensitivity(inputs, MacroParams(lags=1)).payload
    rows = {row["key"]: row for row in out["rows"]}
    assert rows["rates_10y"]["beta_sum"] == pytest.approx(-0.05, abs=0.005)
    assert rows["rates_10y"]["p_value"] < 0.001
    assert rows["credit"]["p_value"] > 0.01
    assert "market (SPY)" in out["missing"]
