"""ECB Data Portal (SDMX REST, `csvdata`): €STR, deposit rate, unemployment, the euro
area AAA spot curve and euro reference exchange rates."""

import io
import re
from datetime import date, timedelta

import httpx2 as httpx
import pandas as pd

from fda_data.providers.base import SeriesInfo, SymbolNotFoundError
from fda_data.providers.http import get, make_client

BASE_URL = "https://data-api.ecb.europa.eu/service/data"
_FX_RE = re.compile(r"^ECB_EUR([A-Z]{3})$")
# The portal is slow on long ranges (≈ 30 s for 25 years): ask 5 years at a time.
CHUNK_DAYS = 5 * 365
TIMEOUT_S = 60.0

# Our code → (dataflow, series key, title, unit, frequency).
SERIES: dict[str, tuple[str, str, str, str, str]] = {
    "ESTR": (
        "EST",
        "B.EU000A2X2A25.WT",
        "Euro short-term rate (€STR), volume-weighted trimmed mean",
        "percent",
        "B",
    ),
    "ECB_DFR": ("FM", "D.U2.EUR.4F.KR.DFR.LEV", "ECB deposit facility rate", "percent", "D"),
    "ECB_UNEMP": (
        "LFSI",
        "M.I9.S.UNEHRT.TOTAL0.15_74.T",
        "Euro area unemployment rate",
        "percent",
        "M",
    ),
    # Euro area AAA government spot curve (Svensson), ECB_YC_<tenor>.
    **{
        f"ECB_YC_{tenor}": (
            "YC",
            f"B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}",
            f"Euro area AAA government spot rate, {tenor}",
            "percent",
            "B",
        )
        for tenor in ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
    },
}


def _resolve(code: str) -> tuple[str, str, str, str, str] | None:
    if code in SERIES:
        return SERIES[code]
    match = _FX_RE.match(code)
    if match is None:
        return None
    quote = match.group(1)
    return (
        "EXR",
        f"D.{quote}.EUR.SP00.A",
        f"{quote} per EUR, ECB reference rate (14:15 CET)",
        f"{quote}/EUR",
        "D",
    )


class EcbProvider:
    name = "ecb"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or make_client(timeout=TIMEOUT_S)

    def supports_series(self, code: str) -> bool:
        return _resolve(code) is not None

    def series_info(self, code: str) -> SeriesInfo:
        resolved = _resolve(code)
        if resolved is None:
            raise SymbolNotFoundError(f"ecb: unknown series {code}")
        flow, key, title, unit, frequency = resolved
        return SeriesInfo(code, title, unit, frequency, provider_key=f"{flow}/{key}")

    def _chunk(self, provider_key: str, start: date, end: date) -> pd.DataFrame:
        response = get(
            self._client,
            self.name,
            f"{BASE_URL}/{provider_key}",
            startPeriod=start.isoformat(),
            endPeriod=end.isoformat(),
            format="csvdata",
            detail="dataonly",
        )
        if not response.text.strip():  # the ECB answers an empty body when no data in range
            return pd.DataFrame(columns=["date", "value"])
        raw = pd.read_csv(io.StringIO(response.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
        return pd.DataFrame(
            {
                "date": pd.to_datetime(raw["TIME_PERIOD"]).dt.date,
                "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
            }
        )

    def series(self, code: str, start: date, end: date) -> pd.DataFrame:
        info = self.series_info(code)
        chunks: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + timedelta(days=CHUNK_DAYS))
            chunks.append(self._chunk(info.provider_key, chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)
        frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if frame.empty:
            return pd.DataFrame(columns=["date", "value"])
        frame = frame.dropna(subset=["value"]).drop_duplicates(subset=["date"])
        return frame.sort_values("date", ignore_index=True)
