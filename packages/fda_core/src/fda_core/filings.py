"""Filings reader (M30 §5, AI-004), deterministic part: EDGAR HTML → text → 10-K / 10-Q / 20-F
item sections → risk-factor blocks and their year-on-year diff. Pure functions, standard library.

Sections are located by their item headings ("Item 1A. Risk Factors"); the table of contents
repeats every heading, so for each item the occurrence followed by the longest body wins. Risk
factors are split into blocks at their headings (short sentence-like lines, usually the bold
lead-in of each risk); blocks are matched across years by text similarity (difflib):
new (no match ≥ 0.5), removed, and expanded/shortened (matched, length change ≥ 25 %).
"""

import re
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any

BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
SKIP_TAGS = {"script", "style", "head", "title", "ix:header"}

SECTIONS_10K = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for the Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}
SECTIONS_10Q = {
    "2": "Management's Discussion and Analysis",
    "1A": "Risk Factors",
    "3": "Market Risk",
}
SECTIONS_20F = {
    "3": "Key Information (incl. Risk Factors)",
    "4": "Information on the Company",
    "5": "Operating and Financial Review and Prospects",
    "11": "Market Risk",
}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


# Page furniture repeated on every page: "Apple Inc. | 2025 Form 10-K | 5", bare page numbers,
# "Table of Contents" back-links.
NOISE = re.compile(
    r"^(?:.{0,80}\|\s*\d{4}\s+form\s+\d{1,2}-[a-z]\s*\|\s*\d+|\d{1,3}|table of contents)$",
    re.IGNORECASE,
)


def html_to_text(html: str) -> str:
    """Visible text, one paragraph per line, whitespace normalised (non-breaking spaces too)."""
    parser = _Text()
    parser.feed(html)
    raw = "".join(parser.parts).replace("\xa0", " ").replace("​", "")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
    return "\n".join(line for line in lines if line and not NOISE.match(line))


def _heading(item: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*item\s*{re.escape(item)}\s*[.:\-—–]?\s*", re.IGNORECASE | re.MULTILINE)


def _drop_title(body: str, title: str) -> str:
    """Body after the heading: the rest of the heading line (or a title line of its own) goes."""
    lines = body.strip().split("\n")
    first = lines[0].strip().lower().rstrip(".") if lines else ""
    if first and len(first) <= 120 and (title.lower()[:12] in first or first in title.lower()):
        lines = lines[1:]
    return "\n".join(lines).strip()


def split_sections(text: str, form: str = "10-K") -> dict[str, dict[str, Any]]:
    """Item → {title, text, words}. For each item the heading occurrence with the longest body
    (up to the next item heading of any kind) wins, which skips the table of contents."""
    catalog = (
        SECTIONS_10Q
        if form.startswith("10-Q")
        else SECTIONS_20F
        if form.startswith("20-F")
        else SECTIONS_10K
    )
    any_item = re.compile(r"^\s*item\s*\d{1,2}[A-C]?\s*[.:\-—–]?\s", re.IGNORECASE | re.MULTILINE)
    starts = sorted(m.start() for m in any_item.finditer(text))
    out: dict[str, dict[str, Any]] = {}
    for item, title in catalog.items():
        best = ""
        for m in _heading(item).finditer(text):
            # the item number must not continue (Item 1 ≠ Item 1A ≠ Item 10)
            if re.match(
                rf"^\s*item\s*{re.escape(item)}[0-9A-C]", text[m.start() : m.end() + 1], re.I
            ):
                continue
            later = [s for s in starts if s > m.start()]
            end = later[0] if later else len(text)
            body = _drop_title(text[m.end() : end], title)
            if len(body) > len(best):
                best = body
        if best:
            out[item] = {"title": title, "text": best, "words": len(best.split())}
    return out


_LEADIN = re.compile(r"^[A-Z][^\n]{49,400}?[.:]?$")  # category titles are shorter


def risk_blocks(section: str) -> list[dict[str, str]]:
    """Risk factors as {heading, body}: a heading is a line of 25–400 characters that is followed
    by a longer paragraph (the explanation). Lines before the first heading form the preamble."""
    lines = [line for line in section.split("\n") if line.strip() and not NOISE.match(line.strip())]
    blocks: list[dict[str, str]] = []
    current: dict[str, Any] | None = None
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        is_heading = bool(_LEADIN.match(line)) and len(nxt) >= max(120, len(line) * 1.3)
        if is_heading:
            current = {"heading": line.strip(), "body": []}
            blocks.append(current)
        elif current is not None:
            current["body"].append(line)
    return [{"heading": b["heading"], "body": " ".join(b["body"])} for b in blocks]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower())


def risk_diff(
    previous: list[dict[str, str]],
    current: list[dict[str, str]],
    match_at: float = 0.5,
    change_at: float = 0.25,
) -> dict[str, list[dict[str, Any]]]:
    """Year-on-year comparison of risk blocks (headings weigh most, bodies break ties)."""
    prev = [(_norm(b["heading"]), b) for b in previous]
    used: set[int] = set()
    new, changed = [], []
    for block in current:
        key = _norm(block["heading"])
        best, best_i = 0.0, -1
        for i, (pk, _) in enumerate(prev):
            if i in used:
                continue
            score = SequenceMatcher(None, key, pk, autojunk=False).ratio()
            if score > best:
                best, best_i = score, i
        if best < match_at:
            new.append({"heading": block["heading"], "words": len(block["body"].split())})
            continue
        used.add(best_i)
        old = prev[best_i][1]
        a, b = len(old["body"].split()), len(block["body"].split())
        if a and abs(b - a) / a >= change_at:
            changed.append(
                {
                    "heading": block["heading"],
                    "previous_words": a,
                    "words": b,
                    "change": (b - a) / a,
                    "similarity": round(best, 3),
                }
            )
    removed = [
        {"heading": b["heading"], "words": len(b["body"].split())}
        for i, (_, b) in enumerate(prev)
        if i not in used
    ]
    return {"new": new, "removed": removed, "changed": changed}
