"""Scenario builder: real DOT rows in, full world state out.

REAL (straight from flights.csv): date, airline, flight number, tail number,
origin/destination, scheduled times, departure/arrival delay, WEATHER_DELAY,
LATE_AIRCRAFT_DELAY, distance. Airport names/coordinates and airline names
come from airports.csv / airlines.csv.

SIMULATED (seeded, so every run of the same scenario is identical): runway
and taxiway layout, gates, crew shifts, aircraft type per tail, fuel on board,
passenger and connection counts, route options and turbulence.
The disruption itself is derived from the row: high WEATHER_DELAY closes runways,
adds spacing between landings and raises turbulence; LATE_AIRCRAFT_DELAY blocks a
taxiway and leaves a large gate occupied.

CLOCK: everything is built around each flight's DELAYED arrival (ETA = scheduled
arrival + departure delay, i.e. what is known while it is airborne), not its
schedule, so outcomes are not decided before the negotiation starts. Connecting
flights and next departures are simulated with a spread of slack around the ETA:
some connections are already lost, some are safe, and some can only be saved by
landing early - that last group is where negotiation matters.

TRAFFIC: the other real arrivals at the same airport around the same time (from
flights.csv, disrupted or not) take fixed landing slots on the modelled runways,
so landing time is a scarce resource and holding/fuel become real trade-offs.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from .constraints import sep_minutes
from .limits import ARRIVAL_SHARE, DECISION_LEAD, HELD_SHARE, REPORT_BEFORE_DEP, load_limits
from .schemas import Airport, Crew, Flight, Gate, OccupiedSlot, RouteOption, Runway, Taxiway, WorldState
from .timeutil import fmt, hhmm_to_min

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Synthetic fleet per airline: (type, wake, size, seats)
FLEET = {
    "UA": [("B737-800", "M", "M", 166), ("A320", "M", "M", 150), ("B777-200", "H", "L", 276)],
    "AA": [("A321", "M", "M", 187), ("B737-800", "M", "M", 160), ("B767-300", "H", "L", 209)],
    "DL": [("B757-200", "M", "M", 199), ("A320", "M", "M", 157), ("B767-400", "H", "L", 238)],
    "WN": [("B737-700", "M", "M", 143)],
    "US": [("A320", "M", "M", 150), ("A321", "M", "M", 187)],
    "B6": [("A320", "M", "M", 150), ("E190", "S", "S", 100)],
    "AS": [("B737-900", "M", "M", 178)],
    "NK": [("A320", "M", "M", 178)],
    "F9": [("A320", "M", "M", 180)],
    "VX": [("A320", "M", "M", 146)],
    "HA": [("A330-200", "H", "L", 278)],
    "OO": [("CRJ-700", "S", "S", 70), ("E175", "S", "S", 76)],
    "EV": [("CRJ-200", "S", "S", 50), ("ERJ-145", "S", "S", 50)],
    "MQ": [("E175", "S", "S", 76), ("ERJ-145", "S", "S", 50)],
}

USE_COLS = ["YEAR", "MONTH", "DAY", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "SCHEDULED_ARRIVAL",
            "ARRIVAL_DELAY", "DISTANCE", "WEATHER_DELAY", "LATE_AIRCRAFT_DELAY", "CANCELLED", "DIVERTED"]

BACKGROUND_WINDOW = (-15, 60)    # other arrivals with ETA in [first ETA - 15, first ETA + 60] take slots


# --------------------------------------------------------------------------
# Loading and selecting rows
# --------------------------------------------------------------------------
def load_reference():
    airports = pd.read_csv(DATA_DIR / "airports.csv")
    airlines = pd.read_csv(DATA_DIR / "airlines.csv")
    return airports, airlines


def eta_minutes(sched_arr_hhmm, sched_dep_hhmm, dep_delay) -> tuple[int, int]:
    """(scheduled arrival, ETA) in continuous minutes of the departure day.
    Overnight flights (arrival hhmm < departure hhmm) arrive the next day: +1440."""
    sched = hhmm_to_min(sched_arr_hhmm)
    if sched_dep_hhmm is not None and int(float(sched_arr_hhmm)) < int(float(sched_dep_hhmm)):
        sched += 1440
    return sched, sched + max(0, int(float(dep_delay) * 0.9))


def load_flights(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in USE_COLS, low_memory=False)
    airports, _ = load_reference()
    codes = set(airports.IATA_CODE)
    # October 2015 uses numeric airport IDs; keep rows we can resolve.
    df = df[df.ORIGIN_AIRPORT.astype(str).isin(codes) & df.DESTINATION_AIRPORT.astype(str).isin(codes)].copy()
    for c in ("WEATHER_DELAY", "LATE_AIRCRAFT_DELAY", "DEPARTURE_DELAY", "ARRIVAL_DELAY", "CANCELLED", "DIVERTED"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "CANCELLED" in df:   # cancelled flights never arrive, so they take no landing slot
        df = df[df.CANCELLED == 0]
    df = df.dropna(subset=["SCHEDULED_ARRIVAL"])
    times = [eta_minutes(a, d, dd) for a, d, dd in
             zip(df.SCHEDULED_ARRIVAL, df.SCHEDULED_DEPARTURE, df.DEPARTURE_DELAY)]
    df["SCHED_ARR_MIN"] = [t[0] for t in times]
    df["ETA_MIN"] = [t[1] for t in times]
    df["ETA_HOUR"] = df.ETA_MIN // 60          # cluster on when flights actually arrive, not their schedule
    return df.reset_index(drop=True)


def disrupted(df: pd.DataFrame) -> pd.DataFrame:
    """The brief: WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20."""
    return df[(df.WEATHER_DELAY > 15) | (df.LATE_AIRCRAFT_DELAY > 20)]


KEY = ["YEAR", "MONTH", "DAY", "DESTINATION_AIRPORT", "ETA_HOUR"]


def find_clusters(df: pd.DataFrame, min_flights: int = 2) -> pd.DataFrame:
    """Disrupted arrivals at the same airport arriving in the same hour - real contention."""
    d = disrupted(df)
    g = (d.groupby(KEY).size().reset_index(name="n").sort_values(["n"] + KEY, ascending=[False] + [True] * 5))
    return g[g.n >= min_flights].reset_index(drop=True)


def cluster_rows(df: pd.DataFrame, key: dict, max_flights: int = 4) -> pd.DataFrame:
    """The disrupted flights that negotiate. Any beyond max_flights are not dropped:
    they become other traffic (see background_rows) and the count is reported."""
    d = disrupted(df)
    m = pd.Series(True, index=d.index)
    for k in KEY:
        m &= d[k] == key[k]
    rows = d[m].sort_values("ETA_MIN")
    out = rows.head(max_flights).copy()
    out.attrs["overflow"] = len(rows) - len(out)
    return out


def background_rows(df: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Other real arrivals at the same airport around the same time (disrupted or not)."""
    first = rows.iloc[0]
    a0 = int(rows.ETA_MIN.min())
    m = ((df.YEAR == first.YEAR) & (df.MONTH == first.MONTH) & (df.DAY == first.DAY)
         & (df.DESTINATION_AIRPORT == first.DESTINATION_AIRPORT)
         & (df.ETA_MIN >= a0 + BACKGROUND_WINDOW[0]) & (df.ETA_MIN <= a0 + BACKGROUND_WINDOW[1]))
    bg = df[m]
    return bg[~bg.index.isin(rows.index)].sort_values("ETA_MIN")


def scenario(df: pd.DataFrame, rows: pd.DataFrame, scenario_id: Optional[str] = None) -> WorldState:
    """Build a world from negotiating rows plus the real traffic around them."""
    return build_world(rows, background_rows(df, rows), scenario_id=scenario_id)


# --------------------------------------------------------------------------
# Building the world
# --------------------------------------------------------------------------
def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _haversine_nm(lat1, lon1, lat2, lon2) -> float:
    r = 3440.065
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _alternates(airports: pd.DataFrame, code: str, k: int = 3) -> dict[str, int]:
    me = airports[airports.IATA_CODE == code]
    if me.empty or pd.isna(me.iloc[0].LATITUDE):
        return {}
    lat, lon = me.iloc[0].LATITUDE, me.iloc[0].LONGITUDE
    cands = []
    for r in airports.dropna(subset=["LATITUDE"]).itertuples():
        if r.IATA_CODE == code:
            continue
        nm = _haversine_nm(lat, lon, r.LATITUDE, r.LONGITUDE)
        if nm > 40:   # skip airports so close they share the weather
            cands.append((nm, r.IATA_CODE))
    cands.sort()
    # ~250 kt at low altitude => ~4.2 nm/min, plus 8 min for approach
    return {c: int(nm / 4.2 + 8) for nm, c in cands[:k]}


def _times(r) -> tuple[int, int]:
    if hasattr(r, "ETA_MIN") and not pd.isna(r.ETA_MIN):
        return int(r.SCHED_ARR_MIN), int(r.ETA_MIN)
    return eta_minutes(r.SCHEDULED_ARRIVAL, getattr(r, "SCHEDULED_DEPARTURE", None), r.DEPARTURE_DELAY)


def build_world(rows: pd.DataFrame, background: Optional[pd.DataFrame] = None,
                scenario_id: Optional[str] = None) -> WorldState:
    if rows.empty:
        raise ValueError("no rows to build a scenario from")
    airports, airlines = load_reference()
    first = rows.iloc[0]
    dest = str(first.DESTINATION_AIRPORT)
    date = f"{int(first.YEAR)}-{int(first.MONTH):02d}-{int(first.DAY):02d}"
    seed = _seed(date, dest, *rows.FLIGHT_NUMBER.astype(str))
    rng = random.Random(seed)
    background = background if background is not None else rows.iloc[0:0]

    # weather is the airport's, not the flight's: worst WEATHER_DELAY among all arrivals in the window
    wx = float(max(rows.WEATHER_DELAY.max(), background.WEATHER_DELAY.max() if len(background) else 0))
    late = float(rows.LATE_AIRCRAFT_DELAY.max())
    weather_hit = wx > 15
    severe_wx = wx > 45
    late_hit = late > 20
    n = len(rows)
    times = {f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)}": _times(r) for r in rows.itertuples()}
    a0 = min(eta for _, eta in times.values())          # the clock: when the first disrupted flight arrives

    # --- runways and capacity (workbook) ----------------------------------
    lim = load_limits()
    apl = lim.airport(dest)
    rwy_ids = ["28R", "28L", "22L", "22R", "10L", "10R", "4L", "4R"][:apl.active_runways]
    runways = [Runway(id=r, length_ft=rng.choice([9000, 10000, 11500, 13000])) for r in rwy_ids]
    if len(runways) > 1:                                   # weather closures (stated assumption)
        n_closed = len(runways) // 2 if severe_wx else (1 if weather_hit else 0)
        for r in runways[len(runways) - n_closed:]:
            r.open, r.note = False, ("closed: thunderstorm cell / crosswind over limits" if severe_wx
                                     else "closed: crosswind over limits")
    extra_spacing = 1 if weather_hit else 0
    n_open = sum(r.open for r in runways)
    arr_cap = max(1, int(apl.max_runway_ops_per_hour * ARRIVAL_SHARE * n_open / apl.active_runways))
    tw_ids = ["A", "B", "C", "D", "E"]
    taxiways = [Taxiway(id=t) for t in tw_ids]
    if late_hit:
        taxiways[1].blocked = True
        taxiways[1].note = "disabled aircraft awaiting tow"

    # aircraft type per flight (seeded per flight id so it is stable)
    picks = {}
    for r in rows.itertuples():
        fid = f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)}"
        picks[fid] = random.Random(_seed(seed, fid)).choice(FLEET.get(r.AIRLINE, FLEET["AA"]))
    # gates: each flight's planned gate (fits it) + the workbook's spare gates (sizes simulated)
    gates, planned = [], {}

    def add_gate(size, free_from, note=None):
        gid = f"B{2 + 3 * len(gates)}"
        taxi_min, taxi_route = {}, {}
        for rw in rwy_ids:
            taxi_min[rw] = rng.randint(5, 14)
            taxi_route[rw] = rng.sample(tw_ids, k=rng.choice([1, 2]))
        gates.append(Gate(id=gid, size=size, free_from=free_from, taxi_min=taxi_min,
                          taxi_route=taxi_route, note=note))
        return gid

    for k, (fid, p) in enumerate(picks.items()):
        if late_hit and k == 0:   # the late inbound is still on this flight's planned gate
            planned[fid] = add_gate(p[2], a0 + 45, "occupied: previous aircraft delayed (late inbound)")
        else:
            planned[fid] = add_gate(p[2], a0 - 60)
    for _ in range(apl.spare_gates):
        busy = rng.random() < 0.35
        add_gate(rng.choice(["S", "M", "M", "L"]), a0 + rng.randint(0, 40) if busy else a0 - 60, "spare")
    # A blocked taxiway closes some routes, never all: each planned gate keeps an open route from every
    # runway (real airports have redundant paths). Spare gates may still sit behind the blockage.
    blocked = {t.id for t in taxiways if t.blocked}
    clear = [t for t in tw_ids if t not in blocked]
    for g in gates:
        if g.id in planned.values():
            for rw, route in g.taxi_route.items():
                if blocked & set(route):
                    g.taxi_route[rw] = [t for t in route if t not in blocked] or [rng.choice(clear)]
    # ground crews: 8 h shifts (workbook), staggered simulated start times
    shift = lim.ground_shift_h * 60
    crews = [Crew(id=f"C{i + 1}", shift_end=a0 - shift + rng.randint(20, 240) + shift,
                  busy_until=a0 + rng.choice([-30, -10, 10])) for i in range(n + 1)]

    # --- other traffic: real arrivals sequenced first-come-first-served onto the open runways,
    # within hourly capacity. Flow control keeps one slot per negotiating flight free in each hour;
    # other traffic beyond that is metered into the next hour.
    open_ids = [r.id for r in runways if r.open]
    bg_cap = max(0, arr_cap - n)
    # A fuel emergency gets priority: its ETA slot on the first open runway is protected, and other
    # traffic is sequenced around it (real ATC moves other flights back for a fuel-critical aircraft).
    protected = []
    for r in rows.itertuples():
        fid = f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)}"
        if max(5, int(lim.spare_fuel_min - HELD_SHARE * float(r.WEATHER_DELAY))) <= 8:
            protected.append(OccupiedSlot(flight_id=fid, runway=open_ids[0], time=times[fid][1], wake=picks[fid][1]))
    slots: list[OccupiedSlot] = list(protected)
    for r in background.itertuples():
        wake = random.Random(_seed(seed, "bg", r.AIRLINE, r.FLIGHT_NUMBER)).choice(
            FLEET.get(r.AIRLINE, FLEET["AA"]))[1]
        _, eta = _times(r)
        best = None
        for rw in open_ids:
            t = eta
            while (any(s.runway == rw and abs(s.time - t) < sep_minutes(s.wake, wake, extra_spacing) for s in slots)
                   or sum(1 for s in slots if s.time // 60 == t // 60) >= bg_cap):
                t += 1
                if t > eta + 180:
                    break
            if best is None or t < best[1]:
                best = (rw, t)
        if best[1] <= eta + 180:
            slots.append(OccupiedSlot(flight_id=f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)}", runway=best[0],
                                      time=best[1], wake=wake))
    slots = [s for s in slots if s not in protected]     # the protected slot belongs to the negotiating flight

    ap_row = airports[airports.IATA_CODE == dest]
    ap_name = ap_row.iloc[0].AIRPORT if not ap_row.empty else dest
    lat = None if ap_row.empty or pd.isna(ap_row.iloc[0].LATITUDE) else float(ap_row.iloc[0].LATITUDE)
    lon = None if ap_row.empty or pd.isna(ap_row.iloc[0].LONGITUDE) else float(ap_row.iloc[0].LONGITUDE)
    airport = Airport(code=dest, name=ap_name, lat=lat, lon=lon, runways=runways, taxiways=taxiways,
                      gates=gates, crews=crews, occupied_slots=slots, extra_spacing=extra_spacing,
                      size_class=apl.size_class, max_arrivals_per_hour=arr_cap,
                      decision_time=a0 - DECISION_LEAD, gate_change_notice=lim.gate_change_notice_min)
    alts = _alternates(airports, dest)
    name_of = dict(zip(airlines.IATA_CODE, airlines.AIRLINE))

    # --- flights --------------------------------------------------------
    flights = []
    for i, r in enumerate(rows.itertuples()):
        fid = f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)}"
        frng = random.Random(_seed(seed, fid))
        frng.random()   # keep the stream aligned with the fleet pick above
        actype, wake, size, seats = picks[fid]
        sched_arr, eta = times[fid]
        # Workbook: 30 min reserve + 45 min spare on arrival; part of a weather delay was already spent holding.
        spare = max(5, int(lim.spare_fuel_min - HELD_SHARE * float(r.WEATHER_DELAY)))
        fuel = lim.reserve_min + spare
        # Duty counts from the ACTUAL departure: after a long delay the crew reports late or is swapped.
        dep_min = hhmm_to_min(r.SCHEDULED_DEPARTURE) if hasattr(r, "SCHEDULED_DEPARTURE") else sched_arr - 120
        duty_end = dep_min + max(0, int(r.DEPARTURE_DELAY)) - REPORT_BEFORE_DEP + lim.crew_duty_h * 60
        pax = int(seats * frng.uniform(0.78, 0.95))
        conn = int(pax * frng.uniform(0.15, 0.35))
        wx_level = 2 if r.WEATHER_DELAY > 45 else (1 if r.WEATHER_DELAY > 15 else 0)
        shift_d, fuel_d = frng.randint(3, 6), frng.randint(2, 5)
        routes = [
            RouteOption(id="STD", description="published standard arrival", eta_shift=0, fuel_delta=0,
                        turbulence=wx_level),
            RouteOption(id="DIRECT", description="direct routing through the weather area",
                        eta_shift=-shift_d, fuel_delta=fuel_d, turbulence=min(3, wx_level + 1)),
            RouteOption(id="WX_AVOID", description="longer arrival around the weather",
                        eta_shift=frng.randint(4, 8), fuel_delta=-frng.randint(3, 6), turbulence=0),
        ]
        # Slack is spread around the DELAYED arrival (gate-in ~ ETA + 10): from already lost to comfortable.
        conn_deadline = eta + 10 + frng.randint(-10, 40)
        turn = apl.min_turnaround_min
        next_dep = None if frng.random() < 0.2 else eta + 10 + turn + frng.randint(-10, 40)
        flights.append(Flight(
            flight_id=fid, airline=r.AIRLINE, airline_name=name_of.get(r.AIRLINE, r.AIRLINE),
            origin=str(r.ORIGIN_AIRPORT), dest=dest, tail=str(r.TAIL_NUMBER), aircraft_type=actype,
            wake=wake, size=size, sched_arr=sched_arr, eta=eta, fuel_at_eta=fuel, reserve=lim.reserve_min,
            pax=pax, connecting_pax=conn, connection_deadline=conn_deadline,
            min_turnaround=turn, next_departure=next_dep, routes=routes, alternates=alts,
            planned_gate=planned[fid], crew_duty_end=duty_end,
            burn_kg_h=lim.burn_kg_h_wide if wake == "H" else lim.burn_kg_h_narrow,
            weather_delay=float(r.WEATHER_DELAY), late_aircraft_delay=float(r.LATE_AIRCRAFT_DELAY),
            actual_arrival_delay=float(r.ARRIVAL_DELAY),
            source_row={k: (v.item() if hasattr(v, "item") else v) for k, v in r._asdict().items() if k != "Index"},
        ))

    parts = []
    closed = [r.id for r in runways if not r.open]
    parts.append(f"{apl.size_class}: {n_open}/{apl.active_runways} runways open, arrival capacity {arr_cap}/h")
    if weather_hit:
        parts.append(f"weather (max WEATHER_DELAY {wx:.0f} min): runway {', '.join(closed)} closed, "
                     f"+{extra_spacing} min spacing between landings")
    if late_hit:
        parts.append(f"late inbound aircraft (max LATE_AIRCRAFT_DELAY {late:.0f} min): taxiway "
                     f"{taxiways[1].id} blocked, planned gate {gates[0].id} occupied until {fmt(gates[0].free_from)}")
    if slots:
        parts.append(f"{len(slots)} other arrivals already sequenced on {', '.join(open_ids)}")
    overflow = rows.attrs.get("overflow", 0)
    if overflow:
        parts.append(f"{overflow} more disrupted flight(s) handled as other traffic")
    weather = ("Thunderstorms near the field, gusting crosswind; moderate turbulence on approach"
               if severe_wx else "Showers in the area, light turbulence" if weather_hit else "VMC, no significant weather")
    return WorldState(scenario_id=scenario_id or f"{dest}-{date}-{fmt(a0).replace(':', '')}",
                      date=date, airport=airport, flights=flights,
                      disruption="; ".join(parts) or "none", weather=weather, seed=seed)