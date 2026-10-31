"""Structured data for the whole system.

Two groups:
  * World state  - what the scenario looks like (airport, flights).
  * Messages     - what agents are allowed to say to each other.

Every message an agent sends must validate against one of these models,
so nothing free-form ever reaches the negotiation logic.
Times inside messages are "HH:MM" strings (easier for LLMs);
times inside the world state are minutes since midnight (easier for code).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Wake = Literal["S", "M", "H"]          # small / medium (FAA "large") / heavy
Size = Literal["S", "M", "L"]          # gate & aircraft size class
Level = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LEVELS: list[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# --------------------------------------------------------------------------
# World state
# --------------------------------------------------------------------------
class Runway(BaseModel):
    id: str
    length_ft: int
    open: bool = True
    note: Optional[str] = None


class Taxiway(BaseModel):
    id: str
    blocked: bool = False
    note: Optional[str] = None


class Gate(BaseModel):
    id: str
    size: Size
    free_from: int                       # minute of day the gate becomes free
    taxi_min: dict[str, int]             # runway id -> taxi-in minutes
    taxi_route: dict[str, list[str]]     # runway id -> taxiways used
    note: Optional[str] = None


class Crew(BaseModel):
    id: str
    shift_end: int
    busy_until: int


class RouteOption(BaseModel):
    id: str
    description: str
    eta_shift: int        # minutes vs standard arrival (negative = earlier)
    fuel_delta: int       # minutes of fuel vs standard (negative = burns more)
    turbulence: int       # 0 none, 1 light, 2 moderate, 3 severe (forbidden)


class Flight(BaseModel):
    flight_id: str
    airline: str
    airline_name: str
    origin: str
    dest: str
    tail: str
    aircraft_type: str
    wake: Wake
    size: Size
    sched_arr: int
    eta: int                      # earliest arrival on the standard route
    fuel_at_eta: int              # minutes of fuel left at that ETA
    reserve: int = 30             # minutes of fuel that must remain at landing
    pax: int
    connecting_pax: int
    connection_deadline: Optional[int] = None   # latest gate-in that keeps connections
    min_turnaround: int
    next_departure: Optional[int] = None        # same aircraft's next leg
    routes: list[RouteOption]
    alternates: dict[str, int] = Field(default_factory=dict)  # diversion airport -> flight minutes
    planned_gate: Optional[str] = None       # the gate the flight was scheduled into
    crew_duty_end: Optional[int] = None      # flight crew must be on blocks by this time (13 h duty)
    burn_kg_h: int = 2400                    # holding fuel burn (workbook: narrowbody / widebody)
    weather_delay: float = 0
    late_aircraft_delay: float = 0
    actual_arrival_delay: Optional[float] = None
    source_row: dict = Field(default_factory=dict)

    def route(self, rid: str) -> Optional[RouteOption]:
        return next((r for r in self.routes if r.id == rid), None)


class OccupiedSlot(BaseModel):
    """A landing slot already taken by other traffic (real arrivals not in this negotiation)."""
    flight_id: str
    runway: str
    time: int
    wake: Wake


class Airport(BaseModel):
    code: str
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    runways: list[Runway]
    taxiways: list[Taxiway]
    gates: list[Gate]
    crews: list[Crew]
    occupied_slots: list[OccupiedSlot] = Field(default_factory=list)
    extra_spacing: int = 0      # minutes added to every wake minimum (low visibility / weather)
    size_class: str = "Non-hub"
    max_arrivals_per_hour: int = 99   # from workbook max_runway_ops_per_hour (arrival share, runways open)
    decision_time: int = 0            # when the negotiation happens (minute of day)
    gate_change_notice: int = 20      # a gate change needs this many minutes before landing

    def runway(self, rid):
        return next((r for r in self.runways if r.id == rid), None)

    def gate(self, gid):
        return next((g for g in self.gates if g.id == gid), None)

    def crew(self, cid):
        return next((c for c in self.crews if c.id == cid), None)

    def taxiway(self, tid):
        return next((t for t in self.taxiways if t.id == tid), None)


class WorldState(BaseModel):
    scenario_id: str
    date: str
    airport: Airport
    flights: list[Flight]
    disruption: str
    weather: str
    seed: int

    def flight(self, fid: str) -> Optional[Flight]:
        return next((f for f in self.flights if f.flight_id == fid), None)


# --------------------------------------------------------------------------
# Agent messages (round 1: positions)
# --------------------------------------------------------------------------
class UrgencyClaim(BaseModel):
    level: Level
    reason: str


class PilotPosition(BaseModel):
    flight_id: str
    preferred_route: str
    earliest_landing: str          # HH:MM
    latest_safe_landing: str       # HH:MM, last landing time that keeps fuel reserve
    urgency: UrgencyClaim
    message: str


class SlotProposal(BaseModel):
    flight_id: str
    runway: str
    landing: str                   # HH:MM
    route: str = "STD"


class ATCPosition(BaseModel):
    slots: list[SlotProposal]
    message: str


class GateProposal(BaseModel):
    flight_id: str
    gate: str
    crew: str


class GroundPosition(BaseModel):
    gates: list[GateProposal]
    urgency: Optional[UrgencyClaim] = None
    message: str


# --------------------------------------------------------------------------
# Agent messages (rounds 2-3: votes and counter-offers)
# --------------------------------------------------------------------------
PatchOp = Literal["set_route", "set_slot", "swap_slots", "set_gate", "set_crew"]

# Which role may use which lever. Enforced in code, not by prompt.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "PILOT": {"set_route"},
    "ATC": {"set_slot", "swap_slots"},
    "GROUND": {"set_gate", "set_crew"},
}


class Patch(BaseModel):
    op: PatchOp
    flight_id: str
    route: Optional[str] = None
    runway: Optional[str] = None
    landing: Optional[str] = None      # HH:MM
    gate: Optional[str] = None
    crew: Optional[str] = None
    other_flight_id: Optional[str] = None


class CounterProposal(BaseModel):
    base_candidate: str
    patches: list[Patch]
    concession: str        # what this costs the proposing agent
    reason: str


class Rejection(BaseModel):
    candidate: str
    reason: str


class Vote(BaseModel):
    accept: list[str] = Field(default_factory=list)
    reject: list[Rejection] = Field(default_factory=list)   # a list, not a dict: structured outputs need fixed keys
    counter: Optional[CounterProposal] = None
    message: str = ""

    def add_reject(self, candidate: str, reason: str) -> None:
        self.reject.append(Rejection(candidate=candidate, reason=reason))

    def reject_map(self) -> dict[str, str]:
        return {r.candidate: r.reason for r in self.reject}


# --------------------------------------------------------------------------
# Resolutions
# --------------------------------------------------------------------------
class Assignment(BaseModel):
    flight_id: str
    route: str = "STD"
    runway: Optional[str] = None
    landing: Optional[int] = None
    gate: Optional[str] = None
    crew: Optional[str] = None
    divert_to: Optional[str] = None


class Candidate(BaseModel):
    id: str
    source: str            # who produced it: agents / mediator_repair / arbiter_priority / counter:<agent> / fallback
    round: int
    assignments: dict[str, Assignment]
