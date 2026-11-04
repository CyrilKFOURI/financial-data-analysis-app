"""Ken French data library: daily Fama–French 5 factors and momentum, US and developed Europe
(D-DA-1, M18).

Codes are `KF_<REGION>_<FACTOR>` with REGION ∈ {US, EU} and FACTOR ∈ {MKTRF, SMB, HML, RMW,
CMA, RF, MOM}. Values are daily simple returns as decimals (the library publishes percent).
Missing data (-99.99) is dropped. One zip per dataset is downloaded and parsed once per provider
instance (kept 12 h: the library updates monthly), so ingesting the seven codes of a region
costs two downloads.
"""

import io
import re
import time
import zipfile
from datetime import date

import httpx2 as httpx
import pandas as pd

from fda_data.providers.base import ProviderError, SeriesInfo, SymbolNotFoundError
from fda_data.providers.http import get, make_client

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    ("US", "5"): "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ("US", "MOM"): "F-F_Momentum_Factor_daily_CSV.zip",
    ("EU", "5"): "Europe_5_Factors_Daily_CSV.zip",
    ("EU", "MOM"): "Europe_Mom_Factor_Daily_CSV.zip",
}
COLUMNS = {"Mkt-RF": "MKTRF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA", "RF": "RF"}
FACTORS = ("MKTRF", "SMB", "HML", "RMW", "CMA", "RF", "MOM")
REGIONS = ("US", "EU")
TITLES = {
    "MKTRF": "market excess return",
    "SMB": "size (small minus big)",
    "HML": "value (high minus low B/M)",
    "RMW": "profitability (robust minus weak)",
    "CMA": "investment (conservative minus aggressive)",
    "RF": "risk-free rate (1-month T-bill, daily)",
    "MOM": "momentum (winners minus losers)",
}
_CODE_RE = re.compile(r"^KF_(US|EU)_(MKTRF|SMB|HML|RMW|CMA|RF|MOM)$")
TABLE_TTL_S = 12 * 3600
_ROW_RE = re.compile(r"^\s*(\d{8})\s*,")


def codes(region: str) -> list[str]:
    return [f"KF_{region}_{f}" for f in FACTORS]


def parse_daily(text: str) -> pd.DataFrame:
    """First table of a library CSV: the header row starts with a comma, rows with YYYYMMDD."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith(",")), None)
    if start is None:
        raise ProviderError("kenfrench: no table header")
    names = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for line in lines[start + 1 :]:
        if not _ROW_RE.match(line):
            break
        rows.append([c.strip() for c in line.split(",")])
    frame = pd.DataFrame(rows, columns=["date", *names])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d").dt.date
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        frame[name] = values.where(values > -99.0) / 100.0
    return frame.rename(columns={**COLUMNS, "Mom": "MOM", "MOM": "MOM", "WML": "MOM"})


class KenFrenchProvider:
    name = "kenfrench"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or make_client(timeout=60.0)
        self._tables: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}

    def supports_series(self, code: str) -> bool:
        return bool(_CODE_RE.match(code))

    def series_info(self, code: str) -> SeriesInfo:
        match = _CODE_RE.match(code)
        if match is None:
            raise SymbolNotFoundError(f"kenfrench: unknown series {code}")
        region, factor = match.groups()
        title = f"Fama–French {region} {TITLES[factor]}"
        return SeriesInfo(code, title, "return (decimal)", "D", code)

    def _table(self, region: str, dataset: str) -> pd.DataFrame:
        key = (region, dataset)
        hit = self._tables.get(key)
        if hit is None or time.monotonic() - hit[0] > TABLE_TTL_S:
            response = get(self._client, self.name, BASE + FILES[key])
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                text = archive.read(archive.namelist()[0]).decode("latin-1")
            hit = (time.monotonic(), parse_daily(text))
            self._tables[key] = hit
        return hit[1]

    def series(self, code: str, start: date, end: date) -> pd.DataFrame:
        match = _CODE_RE.match(code)
        if match is None:
            raise SymbolNotFoundError(f"kenfrench: unknown series {code}")
        region, factor = match.groups()
        table = self._table(region, "MOM" if factor == "MOM" else "5")
        frame = table[["date", factor]].rename(columns={factor: "value"}).dropna()
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        return frame.reset_index(drop=True)
