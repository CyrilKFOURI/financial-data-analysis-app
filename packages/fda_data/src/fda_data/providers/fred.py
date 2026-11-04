"""FRED series through the keyless `fredgraph.csv` export (TRD §5.2).

Missing observations (market holidays) come back empty and are dropped, never filled.
"""

import io
import re
from datetime import date

import httpx2 as httpx
import pandas as pd

from fda_data.providers.base import SeriesInfo, SymbolNotFoundError
from fda_data.providers.http import get, make_client

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_CODE_RE = re.compile(r"^[A-Z0-9]{2,30}$")

# Titles/units for the series the engine uses by default; any other FRED code still works.
KNOWN: dict[str, tuple[str, str, str]] = {
    "DTB3": ("3-Month Treasury Bill, secondary market rate", "percent", "D"),
    "DGS1MO": ("1-Month Treasury constant maturity", "percent", "D"),
    "DGS2": ("2-Year Treasury constant maturity", "percent", "D"),
    "DGS10": ("10-Year Treasury constant maturity", "percent", "D"),
    "T10Y2Y": ("10-Year minus 2-Year Treasury spread", "percent", "D"),
    "DFF": ("Effective federal funds rate", "percent", "D"),
    "SOFR": ("Secured overnight financing rate", "percent", "D"),
    "CPIAUCSL": ("CPI, all urban consumers (SA)", "index 1982-84=100", "M"),
    "VIXCLS": ("CBOE volatility index close", "index", "D"),
}


class FredProvider:
    name = "fred"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or make_client()

    def supports_series(self, code: str) -> bool:
        return bool(_CODE_RE.match(code))

    def series_info(self, code: str) -> SeriesInfo:
        title, unit, frequency = KNOWN.get(code, (f"FRED series {code}", "unknown", "unknown"))
        return SeriesInfo(code, title, unit, frequency, provider_key=code)

    def series(self, code: str, start: date, end: date) -> pd.DataFrame:
        response = get(
            self._client, self.name, URL, id=code, cosd=start.isoformat(), coed=end.isoformat()
        )
        if "observation_date" not in response.text[:200]:
            raise SymbolNotFoundError(f"fred: unknown series {code}")
        raw = pd.read_csv(io.StringIO(response.text))
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["observation_date"]).dt.date,
                "value": pd.to_numeric(raw[raw.columns[1]], errors="coerce"),
            }
        )
        return frame.dropna(subset=["value"]).reset_index(drop=True)
