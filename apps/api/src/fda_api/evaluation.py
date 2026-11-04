"""Agent evaluation (M30 §7, AI-007): gold set runner and the citations verifier.

The gold set (`apps/api/tests/gold/gold_set.json`) pins questions to the tool path that answers
them and to a value path inside the tool's LLM summary; expected values come from a frozen
snapshot of the offline fake provider (fixed "today"), so any change to a formula, the tool layer
or result shaping shows up as a regression. The verifier is the trust layer of the console: every
number in an answer must appear, after unit and rounding normalisation, in the tool result it
cites with `[r:result_id]`.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOLD_DIR = Path(__file__).resolve().parents[2] / "tests" / "gold"
CITATION = re.compile(r"\[r:([0-9a-f]{6,16})\]")
NUMBER = re.compile(
    r"(?<![\w.])[-−+]?(?:\d{1,3}(?:[ ,\u202f]\d{3})+|\d+)(?:\.\d+)?\s*(?:%|bp|bps|k|K|m|M|bn|x|×)?"
)
IN_TEXT = re.compile(r"-?\d+(?:\.\d+)?")
SUFFIX = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "bn": 1e9}


def load_gold() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = json.loads((GOLD_DIR / "gold_set.json").read_text("utf-8"))
    return items


def load_answers() -> dict[str, float]:
    path = GOLD_DIR / "gold_answers.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def pick(content: Any, path: str) -> Any:
    node = content
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(f"{path!r}: no {part!r}")
    return node


def _label_numbers(text: str) -> list[float]:
    """Numbers written inside labels, as written and as a percentage (95 → 95 and 0.95)."""
    values = [float(x) for x in IN_TEXT.findall(text)]
    return values + [v / 100 for v in values]


def numbers_in(content: Any) -> list[float]:
    out: list[float] = []
    if isinstance(content, bool):
        return out
    if isinstance(content, int | float):
        if math.isfinite(float(content)):
            out.append(float(content))
    elif isinstance(content, str):
        if len(content) <= 200:  # labels such as "3Y", "95 %", "30 d" carry context numbers
            out.extend(_label_numbers(content))
    elif isinstance(content, dict):
        for k, v in content.items():
            out.extend(_label_numbers(str(k)))  # e.g. var95_1d → 95 (0.95), 1
            out.extend(numbers_in(v))
    elif isinstance(content, list):
        for v in content:
            out.extend(numbers_in(v))
    return out


@dataclass(slots=True)
class Mention:
    text: str
    value: float
    decimals: int
    percent: bool
    citation: str | None
    verified: bool = False


@dataclass(slots=True)
class Verification:
    mentions: list[Mention] = field(default_factory=list)

    @property
    def numbers(self) -> int:
        return len(self.mentions)

    @property
    def cited(self) -> int:
        return sum(1 for m in self.mentions if m.citation)

    @property
    def unverified(self) -> list[str]:
        return [m.text for m in self.mentions if not m.verified]

    @property
    def citation_coverage(self) -> float:
        return self.cited / self.numbers if self.numbers else 1.0


def _parse(token: str) -> tuple[float, int, bool]:
    raw = token.strip().replace("−", "-").replace("\u202f", "").replace(" ", "")
    suffix = ""
    for s in ("bps", "bp", "bn", "%", "k", "K", "m", "M", "x", "×"):
        if raw.endswith(s):
            suffix, raw = s, raw[: -len(s)]
            break
    raw = raw.replace(",", "")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    value = float(raw) * SUFFIX.get(suffix, 1.0)
    if suffix in SUFFIX:
        decimals = 0  # magnitudes are rounded loosely: compare at 3 significant digits
    return value, decimals, suffix == "%"


def _matches(m: Mention, candidates: list[float]) -> bool:
    step = 10.0 ** (-m.decimals)
    scales = [1.0]
    if m.percent:
        scales = [0.01]
    for v in candidates:
        for scale in scales:
            target = m.value * scale
            tol = 0.5 * step * scale + 1e-12
            if abs(m.value) >= 1e3 and m.decimals == 0:
                tol = max(tol, abs(target) * 5e-3)
            if abs(v - target) <= tol or abs(abs(v) - abs(target)) <= tol:
                return True
    return False


def verify(
    answer: str, results: dict[str, Any], ignore_years: bool = True, given: str = ""
) -> Verification:
    """Every number of the answer → its nearest following citation (same sentence) → is it in
    that tool result? Years (1900–2100 without decimals) and list ordinals are ignored. Numbers
    written exactly as in `given` (the user's question) restate an input, not a result."""
    out = Verification()
    inputs = {m.group(0).strip() for m in NUMBER.finditer(given)}
    sentences = re.split(r"(?<=[.;!?])\s+|\n", answer)
    for sentence in sentences:
        cites = CITATION.findall(sentence)
        cleaned = CITATION.sub(" ", sentence)
        for match in NUMBER.finditer(cleaned):
            token = match.group(0).strip()
            if not any(ch.isdigit() for ch in token):
                continue
            value, decimals, percent = _parse(token)
            if ignore_years and not percent and decimals == 0 and 1900 <= value <= 2100:
                continue
            if token in inputs or token.replace(",", " ") in inputs:
                continue
            mention = Mention(token, value, decimals, percent, cites[0] if cites else None)
            candidates = [n for c in cites if c in results for n in numbers_in(results[c])]
            mention.verified = bool(candidates) and _matches(mention, candidates)
            out.mentions.append(mention)
    return out


def score(value: Any, expected: float, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    return math.isclose(float(value), expected, rel_tol=rel, abs_tol=abs_tol)
