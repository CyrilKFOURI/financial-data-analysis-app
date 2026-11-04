"""Filings reader: HTML → text, item sections despite the table of contents, risk blocks and the
year-on-year risk diff (new / removed / expanded)."""

from fda_core.filings import html_to_text, risk_blocks, risk_diff, split_sections

LOREM = "The company may be affected in ways that are hard to predict and could hurt results. "


def ten_k(risks: dict[str, int]) -> str:
    toc = "".join(
        f"<tr><td>Item {i}.</td><td>{t}</td><td>{p}</td></tr>"
        for i, t, p in [("1", "Business", 3), ("1A", "Risk Factors", 9), ("7", "MD&amp;A", 30)]
    )
    risk_html = "".join(f"<p><b>{h}</b></p><p>{LOREM * n}</p>" for h, n in risks.items())
    return (
        "<html><head><title>10-K</title><style>p{}</style></head><body>"
        f"<table>{toc}</table>"
        "<p>PART I</p><p>Item 1. Business</p>"
        f"<p>We make widgets.&nbsp;{LOREM * 5}</p>"
        "<p>Item 1A. Risk Factors</p><p>Investing involves risks, including those below.</p>"
        f"{risk_html}"
        "<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>"
        f"<p>Item 7. Management's Discussion and Analysis</p><p>{LOREM * 8}</p>"
        "<p>Item 8. Financial Statements</p><p>See below.</p></body></html>"
    )


RISKS_2024 = {
    "Competition in our markets could reduce our margins and market share.": 3,
    "We depend on a small number of suppliers for key components.": 2,
    "Changes in interest rates could increase our financing costs.": 2,
}
RISKS_2025 = {
    "Competition in our markets could reduce our margins and our market share.": 3,
    "We depend on a small number of suppliers for key components.": 6,
    "Cyberattacks could disrupt our operations and expose customer data.": 3,
}


def test_sections_skip_the_table_of_contents() -> None:
    text = html_to_text(ten_k(RISKS_2024))
    assert "p{}" not in text  # styles and titles are dropped
    assert "\xa0" not in text
    sections = split_sections(text, "10-K")
    assert sections["1"]["text"].startswith("We make widgets.")
    assert sections["1A"]["title"] == "Risk Factors"
    assert "suppliers" in sections["1A"]["text"]
    assert "Unresolved" not in sections["1A"]["text"]
    assert sections["7"]["words"] > 100
    assert "1C" not in sections


def test_risk_blocks_and_year_on_year_diff() -> None:
    old = risk_blocks(split_sections(html_to_text(ten_k(RISKS_2024)))["1A"]["text"])
    new = risk_blocks(split_sections(html_to_text(ten_k(RISKS_2025)))["1A"]["text"])
    assert [b["heading"] for b in old] == list(RISKS_2024)
    diff = risk_diff(old, new)
    assert [d["heading"] for d in diff["new"]] == [list(RISKS_2025)[2]]
    assert [d["heading"] for d in diff["removed"]] == [list(RISKS_2024)[2]]
    assert [d["heading"] for d in diff["changed"]] == [list(RISKS_2025)[1]]
    assert diff["changed"][0]["change"] > 1.5  # suppliers risk tripled in length
