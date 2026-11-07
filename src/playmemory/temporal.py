"""Natural-language date ranges ("last Tuesday", "in March 2023", "3 days ago").

parse_time_range(text, now) -> (t0, t1) in epoch seconds (UTC, inclusive start,
exclusive end) or None. Stdlib only; English. Intended for query filters:
Memory.search(..., when="auto").
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())} | {"sept": 9})
_DAYS = {d: i for i, d in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
_DAYS.update({k[:3]: v for k, v in list(_DAYS.items())})
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DOW = "|".join(sorted(_DAYS, key=len, reverse=True))
_UNITS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _ts(d: datetime) -> float:
    return d.replace(tzinfo=timezone.utc).timestamp()


def _day(d: datetime) -> tuple[float, float]:
    d0 = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return _ts(d0), _ts(d0 + timedelta(days=1))


def _month(y: int, m: int) -> tuple[float, float]:
    a = datetime(y, m, 1)
    b = datetime(y + (m == 12), m % 12 + 1, 1)
    return _ts(a), _ts(b)


def parse_time_range(text: str, now: float) -> tuple[float, float] | None:
    t = text.lower()
    today = datetime.fromtimestamp(now, tz=timezone.utc).replace(tzinfo=None)
    d0 = today.replace(hour=0, minute=0, second=0, microsecond=0)

    if re.search(r"\byesterday\b", t):
        return _day(d0 - timedelta(days=1))
    if re.search(r"\btoday\b", t):
        return _day(d0)
    if re.search(r"\btomorrow\b", t):
        return _day(d0 + timedelta(days=1))
    if re.search(r"\blast week\b", t):
        mon = d0 - timedelta(days=d0.weekday() + 7)
        return _ts(mon), _ts(mon + timedelta(days=7))
    if re.search(r"\bthis week\b", t):
        mon = d0 - timedelta(days=d0.weekday())
        return _ts(mon), _ts(mon + timedelta(days=7))
    if re.search(r"\blast month\b", t):
        y, m = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return _month(y, m)
    if re.search(r"\bthis month\b", t):
        return _month(today.year, today.month)
    if re.search(r"\blast year\b", t):
        return _ts(datetime(today.year - 1, 1, 1)), _ts(datetime(today.year, 1, 1))
    if re.search(r"\bthis year\b", t):
        return _ts(datetime(today.year, 1, 1)), _ts(datetime(today.year + 1, 1, 1))
    m = re.search(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+ago\b", t)
    if m:
        words = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10)
        n = int(m[1]) if m[1].isdigit() else words[m[1]]
        if m[2] == "month":  # calendar months
            mm = today.month - n
            y, mo = today.year + (mm - 1) // 12, (mm - 1) % 12 + 1
            return _month(y, mo)
        if m[2] == "year":
            return _ts(datetime(today.year - n, 1, 1)), _ts(datetime(today.year - n + 1, 1, 1))
        if m[2] == "week":  # the calendar week (Mon-Sun) containing that day
            d = d0 - timedelta(days=7 * n)
            mon = d - timedelta(days=d.weekday())
            return _ts(mon), _ts(mon + timedelta(days=7))
        return _day(d0 - timedelta(days=n))
    m = re.search(rf"\blast ({_DOW})\b", t)
    if m:
        back = (d0.weekday() - _DAYS[m[1]]) % 7 or 7
        return _day(d0 - timedelta(days=back))
    # explicit dates: "8 May 2023", "May 8, 2023", "on May 8" (year defaults to now's)
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\b,?\s*(\d{{4}})?", t)
    if m:
        return _day(datetime(int(m[3] or today.year), _MONTHS[m[2]], int(m[1])))
    m = re.search(rf"\b({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(\d{{4}})?", t)
    if m and 1 <= int(m[2]) <= 31:
        return _day(datetime(int(m[3] or today.year), _MONTHS[m[1]], int(m[2])))
    m = re.search(rf"\b(?:in\s+|during\s+|of\s+)?({_MON})\s+(\d{{4}})\b", t)
    if m:
        return _month(int(m[2]), _MONTHS[m[1]])
    m = re.search(rf"\bin ({_MON})\b", t)
    if m:
        y = today.year if _MONTHS[m[1]] <= today.month else today.year - 1  # most recent such month
        return _month(y, _MONTHS[m[1]])
    m = re.search(r"\b(?:in|during|of)\s+((?:19|20)\d{2})\b", t)
    if m:
        y = int(m[1])
        return _ts(datetime(y, 1, 1)), _ts(datetime(y + 1, 1, 1))
    return None
