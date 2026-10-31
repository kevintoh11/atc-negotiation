"""Deterministic non-AI fallback solver (and repair helper).

Greedy allocation in the arbiter's priority order: each flight, most urgent
first, takes the cheapest assignment that is still feasible given the
flights already placed. Same constraint checker rules, same cost function.

Used when:
  * no candidate is accepted by every agent within 3 turns
  * an agent times out or returns invalid output
It is also used by the mediator to repair the agents' own combined plan
(keep their choices, re-place only the flights that violate constraints).
"""
from __future__ import annotations

from typing import Optional

from .arbiter import Urgency, flight_cost, priority_order
from .constraints import SIZE_RANK, fuel_emergency, metrics, sep_minutes
from .schemas import Assignment, Candidate, Flight, WorldState

MAX_HOLD_SEARCH = 120   # minutes after earliest arrival to search for a slot


def _runway_ok(world, f: Flight, rwy: str, t: int, placed: list[tuple[Flight, Assignment]]) -> bool:
    extra = world.airport.extra_spacing
    others = [(b.landing, g.wake) for g, b in placed
              if b.runway == rwy and b.landing is not None and not b.divert_to]
    others += [(s.time, s.wake) for s in world.airport.occupied_slots if s.runway == rwy]
    hour = t // 60                                     # hourly arrival capacity, all runways
    booked = sum(1 for g, b in placed if b.landing is not None and not b.divert_to and b.landing // 60 == hour)
    booked += sum(1 for s in world.airport.occupied_slots if s.time // 60 == hour)
    if booked >= world.airport.max_arrivals_per_hour and not fuel_emergency(f):
        return False
    for tt, wake in others:
        if tt <= t:
            if t - tt < sep_minutes(wake, f.wake, extra):
                return False
        elif tt - t < sep_minutes(f.wake, wake, extra):
            return False
    return True


def _busy(resource_key: str, rid: str, start: int, end: int, world, placed) -> bool:
    for g, b in placed:
        if getattr(b, resource_key) != rid or b.landing is None or b.divert_to:
            continue
        m = metrics(world, g, b)
        if m and start < m.ready and m.gate_in < end:
            return True
    return False


def _best_gate_crew(world, f: Flight, rwy: str, t: int, route: str, placed, u: Urgency):
    best, best_cost = None, float("inf")
    blocked = {tw.id for tw in world.airport.taxiways if tw.blocked}
    for gate in world.airport.gates:
        if SIZE_RANK[gate.size] < SIZE_RANK[f.size]:
            continue
        if any(tid in blocked for tid in gate.taxi_route.get(rwy, [])):
            continue
        gate_in = max(t + gate.taxi_min.get(rwy, 10), gate.free_from)    # apron wait if still occupied
        ready = gate_in + f.min_turnaround
        if _busy("gate", gate.id, gate_in, ready, world, placed):
            continue
        for crew in world.airport.crews:
            start = max(gate_in, crew.busy_until)          # a busy crew delays the turnaround
            if _busy("crew", crew.id, start, start + f.min_turnaround, world, placed):
                continue
            a = Assignment(flight_id=f.flight_id, route=route, runway=rwy, landing=t, gate=gate.id, crew=crew.id)
            # small tie-breaker: prefer the smallest gate that fits (gate utilisation)
            c = flight_cost(world, f, a, u) + 0.1 * (SIZE_RANK[gate.size] - SIZE_RANK[f.size])
            if c < best_cost:
                best, best_cost = a, c
    return best, best_cost


def place_flight(world: WorldState, f: Flight, placed, u: Urgency, route_lock: Optional[str] = None) -> Assignment:
    best, best_cost = None, float("inf")
    for route in f.routes:
        if route.turbulence >= 3 or (route_lock and route.id != route_lock):
            continue
        earliest = f.eta + route.eta_shift
        latest = earliest + (f.fuel_at_eta + route.fuel_delta - f.reserve)   # last time with reserve intact
        for rwy in world.airport.runways:
            if not rwy.open:
                continue
            for t in range(earliest, min(latest, earliest + MAX_HOLD_SEARCH) + 1):
                if not _runway_ok(world, f, rwy.id, t, placed):
                    continue
                a, c = _best_gate_crew(world, f, rwy.id, t, route.id, placed, u)
                if a is not None:
                    if c < best_cost:
                        best, best_cost = a, c
                    break          # earliest feasible time on this runway/route found
    if best is not None:
        return best
    # No safe landing slot within fuel limits -> divert to best reachable alternate.
    for route in sorted(f.routes, key=lambda r: -r.fuel_delta):
        if route.turbulence >= 3:
            continue
        for alt, mins in sorted(f.alternates.items(), key=lambda kv: kv[1]):
            if f.fuel_at_eta + route.fuel_delta - mins >= f.reserve:
                return Assignment(flight_id=f.flight_id, route=route.id, divert_to=alt)
    # Nothing safe exists: land at the earliest separated slot rather than divert out of range.
    # The checker will flag what is broken and the protocol escalates to a human controller.
    for t in range(f.eta, f.eta + MAX_HOLD_SEARCH):
        for rwy in (r for r in world.airport.runways if r.open):
            if _runway_ok(world, f, rwy.id, t, placed):
                gate = f.planned_gate or world.airport.gates[0].id
                return Assignment(flight_id=f.flight_id, route="STD", runway=rwy.id, landing=t, gate=gate,
                                  crew=world.airport.crews[0].id)
    alt = min(f.alternates, key=f.alternates.get) if f.alternates else None
    return Assignment(flight_id=f.flight_id, divert_to=alt)


def solve(world: WorldState, urg: dict[str, Urgency], cid="FALLBACK", source="fallback", rnd=0,
          fixed: Optional[dict[str, Assignment]] = None, route_locks: Optional[dict[str, str]] = None) -> Candidate:
    fixed = dict(fixed or {})
    placed = [(world.flight(fid), a) for fid, a in fixed.items()]
    out = dict(fixed)
    for fid in priority_order(urg):
        if fid in out:
            continue
        f = world.flight(fid)
        lock = (route_locks or {}).get(fid)
        a = place_flight(world, f, placed, urg[fid], lock)
        if lock and a.divert_to:            # locked route impossible -> free choice
            a = place_flight(world, f, placed, urg[fid])
        out[fid] = a
        placed.append((f, a))
    return Candidate(id=cid, source=source, round=rnd, assignments=out)


def repair(world: WorldState, cand: Candidate, bad_flights: set[str], urg, cid: str, rnd: int,
           keep_routes: bool = False) -> Candidate:
    """Keep every assignment that is not involved in a violation; re-place the rest.
    keep_routes=True re-places the flights but honours the route each one asked for."""
    keep = {fid: a for fid, a in cand.assignments.items()
            if fid not in bad_flights and world.flight(fid) is not None}
    locks = {fid: cand.assignments[fid].route for fid in bad_flights if fid in cand.assignments} if keep_routes else None
    return solve(world, urg, cid=cid, source=f"mediator_repair({cand.id})", rnd=rnd, fixed=keep, route_locks=locks)