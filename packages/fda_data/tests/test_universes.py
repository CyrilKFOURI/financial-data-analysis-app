"""Universe sources and membership (SC-001), snapshot storage (SC-002) — offline."""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import Engine

from fda_data.lake import PriceLake
from fda_data.providers.base import ProviderError
from fda_data.universes import (
    SOURCES,
    Member,
    UniverseError,
    create_universe,
    list_universes,
    load_universe,
    parse_constituents,
    parse_pasted,
    read_snapshot,
    save_members,
    snapshot_dates,
    write_snapshot,
    yahoo_symbol,
)

PAGE = """
<table class="infobox"><tr><th>Founded</th><td>1957</td></tr></table>
<table class="wikitable" id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector<sup>[3]</sup></th></tr>
{rows}
</table>
<table class="navbox"><tr><td><table><tr><td>nested</td></tr></table></td></tr></table>
"""


def test_constituents_table_is_found_and_symbols_normalised() -> None:
    rows = "".join(
        f"<tr><td><a href='#'>T{i}</a></td><td>Company {i}</td><td>Tech</td></tr>"
        for i in range(478)
    )
    rows += "<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>"
    rows += "<tr><td>BF.B</td><td>Brown–Forman</td><td>Consumer Staples</td></tr>"
    members = parse_constituents(PAGE.format(rows=rows), SOURCES["sp500"])
    assert len(members) == 480
    assert members[0] == Member("T0", "Company 0", "Tech")
    assert members[-2].symbol == "BRK-B"
    assert yahoo_symbol("AI.PA") == "AI.PA"  # exchange suffixes stay
    with pytest.raises(ProviderError, match="≥ 480"):
        parse_constituents(PAGE.format(rows=rows[:500]), SOURCES["sp500"])


def test_pasted_lists() -> None:
    members = parse_pasted("aapl;Apple;Tech\nMSFT, nvda\n\nAAPL")
    assert [m.symbol for m in members] == ["AAPL", "MSFT", "NVDA"]
    assert members[0].sector is None  # the later bare AAPL replaces the first line
    with pytest.raises(UniverseError):
        parse_pasted("bad symbol!")
    with pytest.raises(UniverseError):
        parse_pasted(" \n ")


def test_membership_diff_tracks_entrants_and_leavers(engine: Engine) -> None:
    uid = create_universe(
        engine, "Test", "custom", None, [Member("A"), Member("B")], date(2026, 1, 2)
    )
    diff = save_members(engine, uid, [Member("B", "Bee"), Member("C")], date(2026, 2, 2))
    assert diff == {"entrants": ["C"], "leavers": ["A"]}
    loaded = load_universe(engine, uid)
    assert [(m["symbol"], m["name"]) for m in loaded["members"]] == [("B", "Bee"), ("C", None)]
    assert list_universes(engine)[0]["members"] == 2
    with pytest.raises(UniverseError):
        create_universe(engine, "Test", "custom", None, [Member("X")], date(2026, 1, 2))


def test_snapshots_round_trip_and_pick_the_latest(lake: PriceLake) -> None:
    frame = pd.DataFrame({"symbol": ["A", "B"], "roic": [0.1, None]})
    write_snapshot(lake, 7, date(2026, 10, 1), frame)
    write_snapshot(lake, 7, date(2026, 10, 8), frame.assign(roic=[0.2, 0.3]))
    assert snapshot_dates(lake, 7) == [date(2026, 10, 1), date(2026, 10, 8)]
    as_of, latest = read_snapshot(lake, 7) or (None, None)
    assert as_of == date(2026, 10, 8)
    assert list(latest["roic"]) == [0.2, 0.3]
    earlier = read_snapshot(lake, 7, date(2026, 10, 5))
    assert earlier is not None
    assert earlier[0] == date(2026, 10, 1)
    assert read_snapshot(lake, 8) is None
