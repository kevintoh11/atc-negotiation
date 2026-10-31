"""Deterministic constraint checker. Pure Python, no AI.

Every candidate resolution passes through `check()` before any agent can
accept it, and again before the final decision is committed. An LLM can
argue for anything; it cannot get a violation past this file.

Hard constraints (never negotiable):
  FUEL_RESERVE        fuel at landing >= reserve (30 min)
  SEVERE_TURBULENCE   no route with severe turbulence
  RUNWAY_CLOSED       landing runway must exist and be open
  EARLY_ARRIVAL       cannot land before the aircraft can physically arrive
  WAKE_SEPARATION     spacing behind the previous lander on the same runway (incl. other traffic,
                      plus extra spacing in bad weather)
  GATE_UNKNOWN / GATE_SIZE / GATE_OVERLAP  (a gate still occupied means waiting on the apron, not a violation)
  TAXIWAY_BLOCKED     runway->gate taxi route may not use a blocked taxiway
  CREW_UNKNOWN / CREW_OVERLAP  (a busy crew delays the turnaround, not a violation)
  RUNWAY_CAPACITY     landings in a clock hour <= the airport's arrival capacity (workbook);
                      waived for a fuel emergency (<= 8 min spare), separation never is
  GATE_CHANGE_NOTICE  a gate change needs >= 20 min notice before landing (workbook)
  CREW_DUTY           flight crew on blocks within the 13 h duty limit (workbook)
  DIVERT_RANGE        a diversion must be reachable with reserve fuel
  MISSING             every flight must be resolved
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from .schemas import Assignment, Candidate, Flight, WorldState
from .timeutil import fmt

# Minimum minutes between consecutive landings, (leader wake, follower wake), from the organisers'
# workbook (FAA JO 7110.65 simplified, nm converted to minutes - see limits.py). Fallback if missing.
_DEFAULT_SEP = {("H", "H"): 2, ("H", "M"): 3, ("H", "S"): 3, ("M", "H"): 2, ("M", "M"): 2, ("M", "S"): 2,
                ("S", "H"): 2, ("S", "M"): 2, ("S", "S"): 2}
try:
    from .limits import load_limits
    WAKE_SEP_MIN: dict[tuple[str, str], int] = dict(load_limits().wake_min)
except Exception:  # noqa: BLE001 - workbook missing or unreadable
    WAKE_SEP_MIN = dict(_DEFAULT_SEP)
SIZE_RANK = {"S": 0, "M": 1, "L": 2}


def fuel_emergency(f) -> bool:
    """<= 8 min spare fuel at ETA: priority landing. The hourly capacity cap is waived (separation is not)."""
    return f.fuel_at_eta - f.reserve <= 8


def sep_minutes(leader_wake: str, follower_wake: str, extra: int = 0) -> int:
    return WAKE_SEP_MIN[(leader_wake, follower_wake)] + extra


class Violation(BaseModel):
    code: str
    flight_id: Optional[str] = None
    message: str


@dataclass
class FlightMetrics:
    """Soft outcomes for one flight under one assignment (used for scoring)."""
    diverted: bool
    earliest: int
    holding: int
    fuel_at_landing: int
    fuel_margin: int
    delay: int                 # gate arrival vs scheduled arrival (as DOT ARRIVAL_DELAY)
    turbulence: int
    taxi: int
    gate_in: Optional[int]
    ready: Optional[int]
    conn_missed: int
    turnaround_delay: int
    overtime: int
    gate_wait: int = 0          # minutes on the apron waiting for an occupied gate
    crew_wait: int = 0          # minutes at the gate waiting for a busy ground crew


def metrics(world: WorldState, f: Flight, a: Assignment) -> Optional[FlightMetrics]:
    route = f.route(a.route)
    if route is None:
        return None
    earliest = f.eta + route.eta_shift
    if a.divert_to:
        mins = f.alternates.get(a.divert_to, 999)
        fuel = f.fuel_at_eta + route.fuel_delta - mins
        return FlightMetrics(True, earliest, 0, fuel, fuel - f.reserve, 0, route.turbulence,
                             0, None, None, f.connecting_pax, 0, 0)
    if a.landing is None:
        return None
    holding = a.landing - earliest
    fuel = f.fuel_at_eta + route.fuel_delta - max(0, holding)
    gate = world.airport.gate(a.gate) if a.gate else None
    taxi = gate.taxi_min.get(a.runway, 10) if gate else 10
    gate_wait = max(0, gate.free_from - (a.landing + taxi)) if gate else 0
    gate_in = a.landing + taxi + gate_wait          # wait on the apron if the gate is still occupied
    crew = world.airport.crew(a.crew) if a.crew else None
    crew_wait = max(0, crew.busy_until - gate_in) if crew else 0   # turnaround starts when the crew is free
    ready = gate_in + crew_wait + f.min_turnaround
    conn_missed = f.connecting_pax if (f.connection_deadline and gate_in > f.connection_deadline) else 0
    tad = max(0, ready - f.next_departure) if f.next_departure else 0
    overtime = max(0, ready - crew.shift_end) if crew else 0
    return FlightMetrics(False, earliest, holding, fuel, fuel - f.reserve,
                         gate_in - f.sched_arr, route.turbulence, taxi,   # delay at the gate, like DOT ARRIVAL_DELAY
                         gate_in, ready, conn_missed, tad, overtime, gate_wait, crew_wait)


def check_flight(world: WorldState, f: Flight, a: Optional[Assignment]) -> list[Violation]:
    v: list[Violation] = []
    fid = f.flight_id
    if a is None:
        return [Violation(code="MISSING", flight_id=fid, message=f"{fid} has no assignment")]
    route = f.route(a.route)
    if route is None:
        return [Violation(code="ROUTE_UNKNOWN", flight_id=fid, message=f"{fid}: route '{a.route}' does not exist")]
    if route.turbulence >= 3:
        v.append(Violation(code="SEVERE_TURBULENCE", flight_id=fid,
                           message=f"{fid}: route {route.id} crosses severe turbulence"))
    m = metrics(world, f, a)
    if a.divert_to:
        if a.divert_to not in f.alternates:
            v.append(Violation(code="DIVERT_RANGE", flight_id=fid, message=f"{fid}: {a.divert_to} is not a known alternate"))
        elif m and m.fuel_at_landing < f.reserve:
            v.append(Violation(code="DIVERT_RANGE", flight_id=fid,
                               message=f"{fid}: diversion to {a.divert_to} leaves {m.fuel_at_landing} min fuel < {f.reserve} reserve"))
        return v
    if a.landing is None or a.runway is None:
        return v + [Violation(code="MISSING", flight_id=fid, message=f"{fid}: no runway/landing time")]

    rwy = world.airport.runway(a.runway)
    if rwy is None or not rwy.open:
        why = f" ({rwy.note})" if rwy and rwy.note else ""
        v.append(Violation(code="RUNWAY_CLOSED", flight_id=fid, message=f"{fid}: runway {a.runway} unavailable{why}"))
    if m.holding < 0:
        v.append(Violation(code="EARLY_ARRIVAL", flight_id=fid,
                           message=f"{fid}: landing {fmt(a.landing)} is before earliest possible {fmt(m.earliest)} via {a.route}"))
    if m.fuel_at_landing < f.reserve:
        v.append(Violation(code="FUEL_RESERVE", flight_id=fid,
                           message=f"{fid}: fuel at landing {m.fuel_at_landing} min < {f.reserve} min reserve "
                                   f"({m.holding} min holding)"))

    gate = world.airport.gate(a.gate) if a.gate else None
    if gate is None:
        v.append(Violation(code="GATE_UNKNOWN", flight_id=fid, message=f"{fid}: gate '{a.gate}' does not exist"))
    else:
        if SIZE_RANK[gate.size] < SIZE_RANK[f.size]:
            v.append(Violation(code="GATE_SIZE", flight_id=fid,
                               message=f"{fid}: {f.aircraft_type} (size {f.size}) does not fit gate {gate.id} (size {gate.size})"))
        for tid in gate.taxi_route.get(a.runway, []):
            tw = world.airport.taxiway(tid)
            if tw and tw.blocked:
                v.append(Violation(code="TAXIWAY_BLOCKED", flight_id=fid,
                                   message=f"{fid}: route {a.runway}->{gate.id} uses taxiway {tid}, blocked ({tw.note})"))

    if gate is not None and f.planned_gate and a.gate != f.planned_gate:
        notice = a.landing - world.airport.decision_time
        if notice < world.airport.gate_change_notice:
            v.append(Violation(code="GATE_CHANGE_NOTICE", flight_id=fid,
                               message=f"{fid}: gate change {f.planned_gate}->{a.gate} with {notice} min notice before "
                                       f"landing; needs {world.airport.gate_change_notice}"))
    if f.crew_duty_end is not None and m.gate_in > f.crew_duty_end:
        v.append(Violation(code="CREW_DUTY", flight_id=fid,
                           message=f"{fid}: on blocks {fmt(m.gate_in)}, past flight-crew duty limit "
                                   f"{fmt(f.crew_duty_end)}"))

    if gate is not None and f.planned_gate and a.gate != f.planned_gate:
        notice = a.landing - world.airport.decision_time
        if notice < world.airport.gate_change_notice:
            v.append(Violation(code="GATE_CHANGE_NOTICE", flight_id=fid,
                               message=f"{fid}: gate change {f.planned_gate}->{a.gate} with {notice} min notice "
                                       f"before landing; needs {world.airport.gate_change_notice}"))
    if f.crew_duty_end is not None and m.gate_in > f.crew_duty_end:
        v.append(Violation(code="CREW_DUTY", flight_id=fid,
                           message=f"{fid}: on blocks {fmt(m.gate_in)}, past flight-crew duty limit "
                                   f"{fmt(f.crew_duty_end)}"))

    crew = world.airport.crew(a.crew) if a.crew else None
    if crew is None:
        v.append(Violation(code="CREW_UNKNOWN", flight_id=fid, message=f"{fid}: crew '{a.crew}' does not exist"))
    return v


def _overlap(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def check(world: WorldState, cand: Candidate) -> list[Violation]:
    """All hard-constraint violations for a full candidate resolution."""
    v: list[Violation] = []
    for f in world.flights:
        v += check_flight(world, f, cand.assignments.get(f.flight_id))

    # cross-flight checks -------------------------------------------------
    landed = []
    for fid, a in cand.assignments.items():
        f = world.flight(fid)
        if f is None:
            v.append(Violation(code="UNKNOWN_FLIGHT", flight_id=fid, message=f"unknown flight {fid}"))
            continue
        if a.divert_to or a.landing is None:
            continue
        m = metrics(world, f, a)
        if m is not None:
            landed.append((f, a, m))

    # wake separation, per runway, consecutive landings - including other traffic already sequenced
    extra = world.airport.extra_spacing
    by_rwy: dict[str, list] = {}
    for f, a, m in landed:
        by_rwy.setdefault(a.runway, []).append((a.landing, f.flight_id, f.wake, True))
    for s in world.airport.occupied_slots:
        if s.runway in by_rwy:
            by_rwy[s.runway].append((s.time, s.flight_id, s.wake, False))
    for rwy, seq in by_rwy.items():
        seq.sort(key=lambda x: (x[0], x[3]))
        for (t0, lid, lw, lneg), (t1, fid, fw, fneg) in zip(seq, seq[1:]):
            if not (lneg or fneg):
                continue                    # two background flights: not ours to check
            need = sep_minutes(lw, fw, extra)
            if t1 - t0 < need:
                culprit = fid if fneg else lid
                other = " (other traffic)" if not (lneg and fneg) else ""
                v.append(Violation(code="WAKE_SEPARATION", flight_id=culprit,
                                   message=f"{fid} ({fw}) lands {t1 - t0} min behind {lid} ({lw}){other} on {rwy}; "
                                           f"needs {need} min"))

    # runway capacity: landings per clock hour, other traffic included
    cap = world.airport.max_arrivals_per_hour
    buckets: dict[int, list] = {}
    for f, a, m in landed:
        buckets.setdefault(a.landing // 60, []).append((a.landing, f.flight_id))
    for s in world.airport.occupied_slots:
        if s.time // 60 in buckets:
            buckets[s.time // 60].append((s.time, None))
    for hour, items in buckets.items():
        if len(items) > cap:
            ours = sorted([x for x in items if x[1] and not fuel_emergency(world.flight(x[1]))],
                          reverse=True)[:len(items) - cap]
            for t, fid in ours:
                v.append(Violation(code="RUNWAY_CAPACITY", flight_id=fid,
                                   message=f"{fid}: {len(items)} landings booked in {fmt(hour * 60)}-"
                                           f"{fmt(hour * 60 + 59)}; arrival capacity is {cap}/h"))

    # gate and crew double-booking
    for key, code in (("gate", "GATE_OVERLAP"), ("crew", "CREW_OVERLAP")):
        groups: dict[str, list] = {}
        for f, a, m in landed:
            res = getattr(a, key)
            if res:
                start = m.gate_in + (m.crew_wait if key == "crew" else 0)
                groups.setdefault(res, []).append((start, m.ready, f.flight_id))
        for res, items in groups.items():
            items.sort()
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a0, a1, fa = items[i]
                    b0, b1, fb = items[j]
                    if _overlap(a0, a1, b0, b1):
                        v.append(Violation(code=code, flight_id=fb,
                                           message=f"{key} {res} double-booked: {fa} {fmt(a0)}-{fmt(a1)} "
                                                   f"vs {fb} {fmt(b0)}-{fmt(b1)}"))
    return v