"""Operational limits from the organisers' workbook (data/challenge2_operational_constraints.xlsx).

The workbook's values are used as given. Where the workbook does not say HOW a value
applies, we make an assumption; every one is listed in ASSUMPTIONS (the rules require
any changed or interpreted limit to be stated in the demo), shown in the app and README.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

WORKBOOK = Path(__file__).resolve().parent.parent / "data" / "challenge2_operational_constraints.xlsx"

APPROACH_KT = 140          # converts wake distances (nm) to time on final approach
ARRIVAL_SHARE = 0.5        # share of max_runway_ops_per_hour used by arrivals
HELD_SHARE = 0.5           # share of WEATHER_DELAY assumed already flown in holding
REPORT_BEFORE_DEP = 60     # flight-crew report time before scheduled departure (min)
DECISION_LEAD = 30         # the negotiation happens this long before the first ETA (min)

ASSUMPTIONS = [
    f"Wake separation (nm) is converted to minutes at {APPROACH_KT} kt on final approach, rounded up.",
    f"{ARRIVAL_SHARE:.0%} of max_runway_ops_per_hour are arrivals; the cap scales with runways open.",
    "Weather: WEATHER_DELAY > 15 closes one runway (at least one stays open); > 45 closes half of them. "
    "Both add 1 min to every landing separation.",
    f"Fuel on arrival = 30 min final reserve + 45 min spare (workbook), minus {HELD_SHARE:.0%} of "
    "WEATHER_DELAY assumed already spent holding (spare never below 5 min).",
    f"Flight-crew duty starts {REPORT_BEFORE_DEP} min before this leg's scheduled departure "
    "(earlier legs are not in the data), so the 13 h limit rarely binds.",
    "Ground has each flight's planned gate plus spare_gates_at_any_time spare gates; spare gate sizes "
    "are simulated. Other traffic uses gates we do not model.",
    "Ground crews work 8 h shifts with simulated, staggered start times; overtime counts 1.5x.",
    f"The negotiation runs {DECISION_LEAD} min before the first arrival; a gate change needs >= 20 min notice "
    "before landing.",
    "Fuel emergency (<= 8 min spare at ETA): other traffic is sequenced around its ETA slot and the hourly "
    "capacity cap is waived for it; wake separation still applies.",
    "Super (A380) wake rows are kept but unused: no A380s in the 2015 US domestic data.",
]


@dataclass
class AirportLimits:
    code: str
    size_class: str
    active_runways: int
    gates: int
    max_runway_ops_per_hour: int
    min_turnaround_min: int
    spare_gates: int


@dataclass
class Limits:
    airports: dict[str, AirportLimits]
    default_airport: AirportLimits
    wake_min: dict[tuple[str, str], int]          # (leader, follower) -> minutes; S/M/H codes
    wake_nm: dict[tuple[str, str], float]
    reserve_min: int = 30
    spare_fuel_min: int = 45
    burn_kg_h_narrow: int = 2400
    burn_kg_h_wide: int = 6500
    crew_duty_h: int = 13
    ground_shift_h: int = 8
    overtime_mult: float = 1.5
    gate_change_notice_min: int = 20
    source: str = "workbook"
    notes: list[str] = field(default_factory=list)

    def airport(self, code: str) -> AirportLimits:
        return self.airports.get(code, self.default_airport)


_CAT = {"Super (A380)": "J", "Heavy": "H", "Large": "M", "Small": "S"}


def _num(text, default):
    m = re.search(r"[\d,]+(?:\.\d+)?", str(text))
    return float(m.group().replace(",", "")) if m else default


def nm_to_min(nm: float) -> int:
    return math.ceil(nm / (APPROACH_KT / 60))


@lru_cache(maxsize=1)
def load_limits(path: str = str(WORKBOOK)) -> Limits:
    cap = pd.read_excel(path, sheet_name="airport_capacity", header=2)
    airports = {r.IATA_CODE: AirportLimits(r.IATA_CODE, r.size_class, int(r.active_runways), int(r.gates),
                                           int(r.max_runway_ops_per_hour), int(r.min_turnaround_min),
                                           int(r.spare_gates_at_any_time))
                for r in cap.itertuples() if isinstance(r.IATA_CODE, str)}
    nonhub = next((a for a in airports.values() if a.size_class == "Non-hub"), None)

    wake = pd.read_excel(path, sheet_name="wake_separation", header=2)
    explicit, other_nm = {}, 3.0
    for r in wake.itertuples():
        lead = str(r.leader).strip()
        if lead.lower().startswith("any other"):
            other_nm = float(r.min_separation_nm)
        elif lead in _CAT and str(r.follower).strip() in _CAT:
            explicit[(_CAT[lead], _CAT[str(r.follower).strip()])] = float(r.min_separation_nm)
    wake_nm = {(a, b): explicit.get((a, b), other_nm) for a in "SMH" for b in "SMH"}

    fc = pd.read_excel(path, sheet_name="fuel_and_crew")
    rule = {str(r.rule).lower(): r.value for r in fc.itertuples()}

    def val(key, default):
        return next((_num(v, default) for k, v in rule.items() if key in k), default)

    return Limits(
        airports=airports,
        default_airport=nonhub or AirportLimits("?", "Non-hub", 1, 8, 20, 35, 2),
        wake_min={k: nm_to_min(v) for k, v in wake_nm.items()}, wake_nm=wake_nm,
        reserve_min=int(val("final fuel reserve", 30)), spare_fuel_min=int(val("spare fuel", 45)),
        burn_kg_h_narrow=int(val("narrowbody", 2400)), burn_kg_h_wide=int(val("widebody", 6500)),
        crew_duty_h=int(val("flight crew max duty", 13)), ground_shift_h=int(val("ground crew shift", 8)),
        overtime_mult=1.5, gate_change_notice_min=int(val("gate change notice", 20)),
        source=Path(path).name)


def limits_table(lim: Limits, airport_code: str) -> list[tuple[str, str]]:
    """(limit, value used) rows for the demo: what the workbook says, as applied."""
    ap = lim.airport(airport_code)
    return [
        ("Airport size class", ap.size_class),
        ("Active runways", str(ap.active_runways)),
        ("Max runway ops / hour", f"{ap.max_runway_ops_per_hour} (arrivals: {int(ap.max_runway_ops_per_hour * ARRIVAL_SHARE)})"),
        ("Spare gates at any time", str(ap.spare_gates)),
        ("Min turnaround", f"{ap.min_turnaround_min} min"),
        ("Final fuel reserve", f"{lim.reserve_min} min"),
        ("Spare fuel on arrival", f"{lim.spare_fuel_min} min"),
        ("Holding burn", f"{lim.burn_kg_h_narrow} kg/h narrowbody, {lim.burn_kg_h_wide} kg/h widebody"),
        ("Flight crew max duty", f"{lim.crew_duty_h} h"),
        ("Ground crew shift", f"{lim.ground_shift_h} h, overtime x{lim.overtime_mult}"),
        ("Gate change notice", f">= {lim.gate_change_notice_min} min before arrival"),
        ("Wake separation (min)", ", ".join(f"{a}->{b} {lim.wake_nm[(a, b)]:g}nm={m}m"
                                            for (a, b), m in lim.wake_min.items() if a in "HM")),
    ]