"""Five-field cron expressions (minute hour day-of-month month day-of-week) for M31 schedules.

Supports `*`, numbers, ranges `a-b`, lists `a,b`, steps `*/n` and `a-b/n`; day-of-week 0–6 with
0 = Sunday (7 also Sunday); when both day fields are restricted, either may match (Vixie cron).
Pure functions on timezone-aware datetimes: callers convert to the schedule's zone.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
NAMES = ("minute", "hour", "day of month", "month", "day of week")


class CronError(ValueError):
    pass


def _field(text: str, low: int, high: int, name: str, dow: bool = False) -> frozenset[int]:
    values: set[int] = set()
    for part in text.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            if not step_text.isdigit() or int(step_text) < 1:
                raise CronError(f"Bad step in {name}: {step_text!r}")
            step = int(step_text)
        if part == "*":
            start, end = low, high
        elif "-" in part:
            a, b = part.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise CronError(f"Bad range in {name}: {part!r}")
            start, end = int(a), int(b)
        elif part.isdigit():
            start = end = int(part)
            if step > 1:
                end = high
        else:
            raise CronError(f"Bad value in {name}: {part!r}")
        top = 7 if dow else high
        if start < low or end > top or start > end:
            raise CronError(f"{name} out of range {low}-{high}: {part!r}")
        values.update(v % 7 if dow else v for v in range(start, end + 1, step))
    return frozenset(values)


@dataclass(frozen=True, slots=True)
class Cron:
    minute: frozenset[int]
    hour: frozenset[int]
    dom: frozenset[int]
    month: frozenset[int]
    dow: frozenset[int]
    dom_star: bool
    dow_star: bool

    def matches(self, t: datetime) -> bool:
        if t.minute not in self.minute or t.hour not in self.hour or t.month not in self.month:
            return False
        dom_ok = t.day in self.dom
        dow_ok = (t.isoweekday() % 7) in self.dow
        if self.dom_star or self.dow_star:
            return dom_ok and dow_ok
        return dom_ok or dow_ok

    def next_after(self, t: datetime) -> datetime:
        """First matching minute strictly after `t` (searches up to 5 years)."""
        current = t.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = current + timedelta(days=366 * 5)
        while current < limit:
            if current.month not in self.month:
                year, month = (
                    (current.year + 1, 1)
                    if current.month == 12
                    else (current.year, current.month + 1)
                )
                current = current.replace(year=year, month=month, day=1, hour=0, minute=0)
                continue
            if not self._day_ok(current):
                current = (current + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if current.hour not in self.hour:
                current = (current + timedelta(hours=1)).replace(minute=0)
                continue
            if current.minute in self.minute:
                return current
            current += timedelta(minutes=1)
        raise CronError("No run time within 5 years")

    def _day_ok(self, t: datetime) -> bool:
        dom_ok = t.day in self.dom
        dow_ok = (t.isoweekday() % 7) in self.dow
        if self.dom_star or self.dow_star:
            return dom_ok and dow_ok
        return dom_ok or dow_ok


def parse(expr: str) -> Cron:
    parts = expr.split()
    if len(parts) != 5:
        raise CronError("A cron expression has 5 fields: minute hour day month weekday")
    minute, hour, dom, month, dow = (
        _field(p, lo, hi, n, i == 4)
        for i, (p, (lo, hi), n) in enumerate(zip(parts, BOUNDS, NAMES, strict=True))
    )
    return Cron(minute, hour, dom, month, dow, dom_star=parts[2] == "*", dow_star=parts[4] == "*")
