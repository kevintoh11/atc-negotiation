"""Time helpers.

The dataset stores times as hhmm integers (e.g. 1435). Inside the system a time
is minutes since midnight of the scenario's day and keeps counting past midnight
(00:30 the next day is 1470, written "24:30"), so times always compare correctly.
"""


def hhmm_to_min(v) -> int:
    v = int(float(v))
    return (v // 100) * 60 + (v % 100)


def fmt(m) -> str:
    """Continuous clock: 1470 -> "24:30" (never wraps, so sorting and comparing stay correct)."""
    if m is None:
        return "--:--"
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def parse(s) -> int:
    """Accept 'HH:MM' (HH may be >= 24), 'HHMM' or an int of minutes."""
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    if ":" in s:
        h, m = s.split(":")[:2]
        return int(h) * 60 + int(m)
    return hhmm_to_min(s)


def parse_near(s, ref: int) -> int:
    """Parse, then pick the occurrence of that clock time nearest `ref`.
    Lets "00:30" from an LLM mean 24:30 when the flight is due around midnight."""
    v = parse(s) % 1440
    return min((v + k * 1440 for k in range(-1, 3)), key=lambda t: abs(t - ref))
