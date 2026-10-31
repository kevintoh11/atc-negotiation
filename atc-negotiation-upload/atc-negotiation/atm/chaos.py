"""Chaos toggle: deliberately corrupt ONE agent's output to prove the guardrails.

The corrupted message enters the negotiation exactly like a real one; the
constraint checker, the arbiter and the permission check must catch it.
Every injection is written to the consensus log, so the audience can see
the fault going in and the rejection coming out.
"""
from __future__ import annotations

from .schemas import CounterProposal, Patch, UrgencyClaim, WorldState
from .timeutil import fmt, parse

CHAOS_MODES = {
    "none": "No fault injected",
    "fuel": "ATC ignores fuel and pushes the most fuel-critical flight back 45 min",
    "ghost_gate": "Ground hallucinates a gate that does not exist (Z99)",
    "wake": "ATC squeezes two arrivals 1 min apart on the same runway",
    "urgency": "A pilot with plenty of fuel claims CRITICAL urgency to jump the queue",
    "role": "A pilot tries to reassign its own gate (not a pilot lever)",
}


class Chaos:
    def __init__(self, mode: str, world: WorldState):
        if mode not in CHAOS_MODES:
            raise ValueError(f"unknown chaos mode {mode}; choose from {list(CHAOS_MODES)}")
        self.mode = mode
        self.world = world

    def on_positions(self, positions: dict) -> list[tuple[str, str]]:
        """Mutates round-1 positions. Returns (agent, description) for the log."""
        notes = []
        w = self.world
        if self.mode == "fuel" and "ATC" in positions:
            tgt = min(w.flights, key=lambda f: f.fuel_at_eta - f.reserve)
            hold = max(45, tgt.fuel_at_eta - tgt.reserve + 15)     # always long enough to breach the reserve
            for s in positions["ATC"].slots:
                if s.flight_id == tgt.flight_id:
                    s.landing = fmt(parse(s.landing) + hold)
                    notes.append(("ATC", f"slot for {tgt.flight_id} moved to {s.landing} (+{hold} min hold) "
                                         f"ignoring its {tgt.fuel_at_eta} min fuel"))
        elif self.mode == "wake" and "ATC" in positions and positions["ATC"].slots:
            slots = positions["ATC"].slots
            if len(slots) >= 2:
                heavy = [s for s in slots if w.flight(s.flight_id).wake == "H"]
                lead = heavy[0] if heavy else slots[0]
                fol = next(s for s in slots if s is not lead)
                fol.runway, fol.landing = lead.runway, fmt(parse(lead.landing) + 1)
                notes.append(("ATC", f"{fol.flight_id} placed 1 min behind {lead.flight_id} on {lead.runway}"))
            else:
                # one flight: squeeze it 1 min behind the other traffic nearest its proposed time
                fol = slots[0]
                others = [o for o in w.airport.occupied_slots]
                if others:
                    lead = min(others, key=lambda o: abs(o.time - parse(fol.landing)))
                    fol.runway, fol.landing = lead.runway, fmt(lead.time + 1)
                    notes.append(("ATC", f"{fol.flight_id} placed 1 min behind {lead.flight_id} (other traffic) "
                                         f"on {lead.runway}"))
        elif self.mode == "ghost_gate" and "GROUND" in positions and positions["GROUND"].gates:
            g = positions["GROUND"].gates[0]
            g.gate = "Z99"
            notes.append(("GROUND", f"{g.flight_id} assigned to non-existent gate Z99"))
        elif self.mode == "urgency":
            tgt = max(w.flights, key=lambda f: f.fuel_at_eta - f.reserve)
            key = f"PILOT({tgt.flight_id})"
            if key in positions:
                positions[key].urgency = UrgencyClaim(level="CRITICAL", reason="we need to land first")
                notes.append((key, "claims CRITICAL urgency with "
                                   f"{tgt.fuel_at_eta - tgt.reserve} min fuel above reserve"))
        return notes

    def on_votes(self, rnd: int, votes: dict, candidate_ids: list[str]) -> list[tuple[str, str]]:
        notes = []
        if self.mode == "role" and rnd == 2 and candidate_ids:
            f = self.world.flights[0]
            key = f"PILOT({f.flight_id})"
            if key in votes:
                gate = self.world.airport.gates[-1].id
                votes[key].counter = CounterProposal(
                    base_candidate=candidate_ids[0],
                    patches=[Patch(op="set_gate", flight_id=f.flight_id, gate=gate)],
                    concession="none", reason="we prefer a different gate")
                notes.append((key, f"counter-offer tries to set its own gate to {gate}"))
        return notes