"""The three agent roles: ATC, Pilot (one per flight) and Ground.

Each agent:
  * sees only its own domain's data (`view`) - private state
  * speaks only in structured messages (schemas.py)
  * may only pull its own levers (ROLE_PERMISSIONS, enforced by the protocol)
  * runs as its own LLM call with its own system prompt (never one combined prompt)

Two backends per role:
  * LLM     - Claude, via llm.py (set ANTHROPIC_API_KEY, run with --llm)
  * offline - a rule-based policy with the SAME inputs and outputs, so the whole
              pipeline is testable and demo-able without an API key or network.
"""
from __future__ import annotations

import json
from typing import Optional

from .constraints import SIZE_RANK, WAKE_SEP_MIN
from .llm import LLMClient
from .schemas import (ATCPosition, CounterProposal, GateProposal, GroundPosition, Patch, PilotPosition,
                      SlotProposal, UrgencyClaim, Vote, WorldState)
from .timeutil import fmt, parse

SHARED_RULES = """\
You are one agent in a multi-agent air traffic negotiation. Other agents represent
other domains and have information you do not.
Rules:
- Communicate ONLY in the JSON format requested.
- A deterministic constraint checker validates every proposal. Safety rules (fuel reserve,
  wake separation, closed runways, blocked taxiways, gate size, double-booking, severe
  turbulence) can never be traded away. Candidates that fail the checker cannot be accepted.
- The negotiation has at most 3 turns (one message from each agent per turn). Be decisive: prefer accepting a reasonable option
  over holding out for your ideal.
- If you reject a candidate, give a concrete reason. If you counter-offer, use only your own
  levers and state what the concession costs you.
- Times are HH:MM local on a continuous clock: after midnight they continue as 24:MM, 25:MM.
- Participants: one ATC agent (runway sequence and landing times), one GROUND agent (gates and
  crews), and one PILOT agent per arriving flight. There are no other agents.
- Keep 'message' and every reason to at most 2 short sentences, and include the numbers that
  drive your view (minutes, fuel, passengers). No markdown."""


class BaseAgent:
    role = "BASE"
    goals = ""
    levers = ""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    @property
    def name(self) -> str:
        return self.role

    def system_prompt(self) -> str:
        return f"You are the {self.role} agent.\nYour goals: {self.goals}\nYour levers: {self.levers}\n\n{SHARED_RULES}"

    # ---- round 1
    def view(self, world: WorldState) -> dict:
        raise NotImplementedError

    def declare(self, world: WorldState):
        v = self.view(world)
        if self.llm:
            prompt = (f"Turn 1 - declare your position.\nYour private view of the situation:\n"
                      f"{_compact(v)}")
            return self.llm.complete_json(self.system_prompt(), prompt, self.position_model, role=self.name)
        return self.offline_declare(v)

    # ---- rounds 2-3
    def vote(self, ctx: dict) -> Vote:
        if self.llm:
            task = ("Vote on the candidate resolutions below. Only candidates with \"feasible\": true can be "
                    "accepted; infeasible ones are shown with the checker's reasons for information. "
                    "Put the ids you can live with in 'accept' and every other feasible id in 'reject' with a "
                    "concrete reason.")
            if ctx["allow_counter"]:
                task += (" You MAY add one 'counter' that changes one candidate using only your levers "
                         f"(allowed ops: {sorted(ctx['allowed_ops'])}). {PATCH_HELP} Say what the concession "
                         "costs you. If you reject a feasible candidate, use your counter to propose the change "
                         "that would fix your objection; leave 'counter' null only if no lever of yours helps.")
            else:
                task += (" This is the FINAL round: no counter-offers (counter must be null). "
                         "'previous_round_votes' shows what every agent accepted and rejected last round. "
                         "Consensus needs EVERY agent to accept the same candidate: accept every candidate you "
                         "can safely live with, especially ones the others already accepted. Reject only what is "
                         "genuinely unworkable for your domain. If nothing is accepted by all, a deterministic "
                         "fallback decides for everyone.")
            prompt = f"Turn {ctx['round']}. {task}\n\n{_compact(ctx)}"
            return self.llm.complete_json(self.system_prompt(), prompt, Vote, role=self.name)
        return self.offline_vote(ctx)


PATCH_HELP = ("Patch fields by op: set_route -> flight_id, route; set_slot -> flight_id, runway, landing (HH:MM); "
              "swap_slots -> flight_id, other_flight_id; set_gate -> flight_id, gate; set_crew -> flight_id, crew. "
              "Leave unused fields null.")


def _compact(obj) -> str:
    """Compact JSON keeps prompts small; the vote context grows with candidates x flights."""
    return json.dumps(obj, separators=(",", ":"), default=lambda o: sorted(o) if isinstance(o, set) else str(o))


# ==========================================================================
# PILOT (one instance per flight)
# ==========================================================================
class PilotAgent(BaseAgent):
    role = "PILOT"
    position_model = PilotPosition
    goals = ("preserve fuel reserves, ensure flight safety, minimise turbulence, "
             "and protect passengers' connections. You speak for ONE flight. "
             "Declare urgency from fuel margin above reserve at your earliest landing: CRITICAL <= 8 min, "
             "HIGH <= 15 min, MEDIUM <= 25 min or connections at risk, otherwise LOW. Never trade a "
             "margin of 15 min or less for comfort.")
    levers = ("choose the arrival route (set_route): STD, DIRECT (shorter, saves fuel, may be bumpier), "
              "or WX_AVOID (smooth, longer, burns fuel); accept a later slot; declare urgency honestly.")

    def __init__(self, flight_id: str, llm=None):
        super().__init__(llm)
        self.flight_id = flight_id

    @property
    def name(self) -> str:
        return f"PILOT({self.flight_id})"

    def view(self, world):
        f = world.flight(self.flight_id)
        return {
            "flight": {
                "flight_id": f.flight_id, "airline": f.airline_name, "origin": f.origin, "dest": f.dest,
                "aircraft_type": f.aircraft_type, "wake": f.wake, "scheduled_arrival": fmt(f.sched_arr),
                "eta_standard_route": fmt(f.eta), "fuel_at_eta_min": f.fuel_at_eta, "reserve_min": f.reserve,
                "pax": f.pax, "connecting_pax": f.connecting_pax,
                "connections_need_gate_by": fmt(f.connection_deadline),
                "routes": [{**r.model_dump(),
                            "earliest_landing": fmt(f.eta + r.eta_shift),
                            "fuel_at_landing_if_no_holding_min": f.fuel_at_eta + r.fuel_delta,
                            "margin_above_reserve_if_no_holding_min": f.fuel_at_eta + r.fuel_delta - f.reserve,
                            "allowed": r.turbulence < 3}
                           for r in f.routes],
                "route_notes": "fuel_delta < 0 means the route BURNS more fuel than STD. Every minute of "
                               "holding costs one more minute of fuel. Turbulence 3 is severe: forbidden.",
            },
            "atis": {"weather": world.weather,
                     "runways": [{"id": r.id, "open": r.open, "note": r.note} for r in world.airport.runways]},
        }

    def offline_declare(self, v) -> PilotPosition:
        f = v["flight"]
        eta = parse(f["eta_standard_route"])
        usable = [r for r in f["routes"] if r["turbulence"] < 3]
        # earliest arrival, but avoid moderate turbulence unless it buys > 6 min
        pref = min(usable, key=lambda r: r["eta_shift"] + 6 * max(0, r["turbulence"] - 1))
        earliest = eta + pref["eta_shift"]
        margin = f["fuel_at_eta_min"] + pref["fuel_delta"] - f["reserve_min"]
        latest = earliest + margin
        if margin <= 8:
            level = "CRITICAL"
        elif margin <= 15:
            level = "HIGH"
        elif f["connecting_pax"] > 30 and parse(f["connections_need_gate_by"]) - earliest < 20:
            level = "MEDIUM"
        else:
            level = "LOW"
        return PilotPosition(
            flight_id=f["flight_id"], preferred_route=pref["id"], earliest_landing=fmt(earliest),
            latest_safe_landing=fmt(latest),
            urgency=UrgencyClaim(level=level, reason=f"{margin} min fuel above reserve via {pref['id']}; "
                                                     f"{f['connecting_pax']} connecting pax"),
            message=f"{f['flight_id']} requests {pref['id']}, can land from {fmt(earliest)}, "
                    f"must land by {fmt(latest)} to keep reserve.")

    def candidate_view(self, world, cand, mets, feasible, violations) -> dict:
        m = mets.get(self.flight_id)
        a = cand.assignments.get(self.flight_id)
        mine = None
        if m and a:
            mine = {"diverted_to": a.divert_to, "runway": a.runway, "landing": fmt(a.landing),
                    "route": a.route, "holding_min": m.holding, "fuel_at_landing_min": m.fuel_at_landing,
                    "fuel_margin_min": m.fuel_margin, "delay_min": m.delay, "turbulence": m.turbulence,
                    "gate": a.gate, "gate_in": fmt(m.gate_in), "connections_missed": m.conn_missed}
        return {"id": cand.id, "source": cand.source, "feasible": feasible,
                "violations": [x.message for x in violations], "my_flight": mine}

    def offline_vote(self, ctx) -> Vote:
        fid = self.flight_id
        feas = [c for c in ctx["candidates"] if c["feasible"] and c["my_flight"]]
        vote = Vote(message=f"{fid} reviewed {len(feas)} feasible options")
        if not feas:
            return vote
        best_delay = min(c["my_flight"]["delay_min"] for c in feas if not c["my_flight"]["diverted_to"]) \
            if any(not c["my_flight"]["diverted_to"] for c in feas) else 0
        any_keeps_conn = any(c["my_flight"]["connections_missed"] == 0 for c in feas)
        final = ctx["round"] >= 3          # deadline pressure: concede on comfort, never on safety
        for c in feas:
            m = c["my_flight"]
            if m["diverted_to"]:
                vote.add_reject(c["id"], f"diverts us to {m['diverted_to']}")
            elif m["fuel_margin_min"] < 3:
                vote.add_reject(c["id"], f"only {m['fuel_margin_min']} min above reserve - too thin")
            elif m["delay_min"] > best_delay + (25 if final else 12):
                vote.add_reject(c["id"], f"{m['delay_min']} min late vs {best_delay} available elsewhere")
            elif m["connections_missed"] and any_keeps_conn:
                vote.add_reject(c["id"], f"misses {m['connections_missed']} connections when another option keeps them")
            elif not final and m["turbulence"] >= 2 and any(o["my_flight"]["turbulence"] < m["turbulence"]
                                              for o in feas if not o["my_flight"]["diverted_to"]):
                vote.add_reject(c["id"], "moderate turbulence when a smoother option exists")
            else:
                vote.accept.append(c["id"])
        if ctx["allow_counter"] and feas:
            vote.counter = self._counter(ctx, feas, vote)
        return vote

    def _counter(self, ctx, feas, vote) -> Optional[CounterProposal]:
        """Offer a route change on the best option, saying what it costs us."""
        fid = self.flight_id
        landed = [c for c in feas if not c["my_flight"]["diverted_to"]]
        if not landed:
            return None
        base = min(landed, key=lambda c: c["my_flight"]["delay_min"])
        m = base["my_flight"]
        routes = {r["id"]: r for r in ctx["my_routes"]}
        cur = routes.get(m["route"])
        if cur is None:
            return None
        # (a) moderate turbulence: offer the smooth route if fuel allows (we pay time + fuel)
        if m["turbulence"] >= 2:
            smooth = [r for r in routes.values() if r["turbulence"] < m["turbulence"]
                      and m["fuel_margin_min"] + (r["fuel_delta"] - cur["fuel_delta"]) > 15]   # never trade a margin of 15 or less for comfort
            if smooth:
                r = min(smooth, key=lambda r: (r["turbulence"], r["eta_shift"]))
                return CounterProposal(
                    base_candidate=base["id"], patches=[Patch(op="set_route", flight_id=fid, route=r["id"])],
                    concession=f"arrive {r['eta_shift'] - cur['eta_shift']} min later and burn "
                               f"{cur['fuel_delta'] - r['fuel_delta']} min more fuel",
                    reason=f"{r['id']} avoids turbulence level {m['turbulence']} for {fid}'s passengers")
        # (b) holding: offer a faster route (we accept a bumpier ride)
        if m["holding_min"] > 4 or not vote.accept:
            fast = [r for r in routes.values() if r["turbulence"] <= 2 and r["eta_shift"] < cur["eta_shift"]]
            if fast:
                r = min(fast, key=lambda r: r["eta_shift"])
                return CounterProposal(
                    base_candidate=base["id"], patches=[Patch(op="set_route", flight_id=fid, route=r["id"])],
                    concession=f"accept turbulence level {r['turbulence']} on {r['id']}",
                    reason=f"{r['id']} arrives {cur['eta_shift'] - r['eta_shift']} min sooner and saves "
                           f"{r['fuel_delta'] - cur['fuel_delta']} min fuel")
        return None


# ==========================================================================
# ATC (one per airport)
# ==========================================================================
class ATCAgent(BaseAgent):
    role = "ATC"
    position_model = ATCPosition
    goals = ("keep runway wake separation, keep traffic off blocked taxiways, maximise runway/sector throughput. "
             "Other traffic already holds the landing slots listed in other_traffic_slots; fit these arrivals "
             "around them.")
    levers = ("landing sequence, runway and time for every arrival (set_slot, swap_slots); "
              "speed control / vectoring is expressed as a later landing time.")

    def view(self, world):
        return {
            "airport": world.airport.code,
            "runways": [r.model_dump() for r in world.airport.runways],
            "taxiways": [t.model_dump() for t in world.airport.taxiways],
            "wake_separation_min": {f"{a}->{b}": v + world.airport.extra_spacing
                                    for (a, b), v in WAKE_SEP_MIN.items()},
            "weather_extra_spacing_min": world.airport.extra_spacing,
            "other_traffic_slots": [{"flight_id": s.flight_id, "runway": s.runway, "landing": fmt(s.time),
                                     "wake": s.wake} for s in sorted(world.airport.occupied_slots,
                                                                     key=lambda s: s.time)],
            "arrivals": [{"flight_id": f.flight_id, "aircraft_type": f.aircraft_type, "wake": f.wake,
                          "eta_standard_route": fmt(f.eta),
                          "routes": [{"id": r.id, "eta_shift": r.eta_shift, "turbulence": r.turbulence,
                                      "allowed": r.turbulence < 3}
                                     for r in f.routes]}
                         for f in world.flights],
            "weather": world.weather,
        }

    def offline_declare(self, v) -> ATCPosition:
        """First-come-first-served on open runways with wake spacing, around other traffic.
        ATC cannot see fuel."""
        open_rwys = [r["id"] for r in v["runways"] if r["open"]]
        sep = {tuple(k.split("->")): s for k, s in v["wake_separation_min"].items()}
        placed: list[tuple[str, str, int]] = [(s["runway"], s["wake"], parse(s["landing"]))
                                              for s in v.get("other_traffic_slots", [])]
        slots = []
        for f in sorted(v["arrivals"], key=lambda f: parse(f["eta_standard_route"])):
            best = None
            for rw in open_rwys:
                t = parse(f["eta_standard_route"])
                while any(r == rw and ((pt <= t and t - pt < sep[(pw, f["wake"])]) or
                                       (pt > t and pt - t < sep[(f["wake"], pw)]))
                          for r, pw, pt in placed):
                    t += 1
                if best is None or t < best[1]:
                    best = (rw, t)
            placed.append((best[0], f["wake"], best[1]))
            slots.append(SlotProposal(flight_id=f["flight_id"], runway=best[0], landing=fmt(best[1]), route="STD"))
        closed = [r["id"] for r in v["runways"] if not r["open"]]
        return ATCPosition(slots=slots,
                           message=f"Sequencing {len(slots)} arrivals first-come-first-served on "
                                   f"{', '.join(open_rwys)}" + (f"; {', '.join(closed)} closed" if closed else ""))

    def candidate_view(self, world, cand, mets, feasible, violations) -> dict:
        rows, landings, holding = [], [], 0
        for fid, a in cand.assignments.items():
            m = mets.get(fid)
            rows.append({"flight_id": fid, "runway": a.runway, "landing": fmt(a.landing), "route": a.route,
                         "holding_min": m.holding if m else None, "diverted_to": a.divert_to})
            if a.landing is not None and not a.divert_to:
                landings.append(a.landing)
            holding += max(0, m.holding) if m else 0
        first_eta = min(f.eta for f in world.flights)
        return {"id": cand.id, "source": cand.source, "feasible": feasible,
                "violations": [x.message for x in violations], "sequence": rows,
                "total_holding_min": holding,
                "makespan_min": (max(landings) - first_eta) if landings else None}

    def offline_vote(self, ctx) -> Vote:
        feas = [c for c in ctx["candidates"] if c["feasible"]]
        vote = Vote(message=f"ATC reviewed {len(feas)} feasible options")
        if not feas:
            return vote
        best_mk = min(c["makespan_min"] or 0 for c in feas)
        best_hold = min(c["total_holding_min"] for c in feas)
        k = 2 if ctx["round"] >= 3 else 1   # final round: wider tolerance
        for c in feas:
            if (c["makespan_min"] or 0) > best_mk + 6 * k:
                vote.add_reject(c["id"], f"throughput: last landing {c['makespan_min'] - best_mk} min later than best")
            elif c["total_holding_min"] > best_hold + 20 * k:
                vote.add_reject(c["id"], f"{c['total_holding_min']} min total holding vs {best_hold} possible")
            else:
                vote.accept.append(c["id"])
        # counter: resequence so a higher-priority flight is not stuck behind a lower one,
        # but only if the higher-priority flight can actually be there by the earlier slot
        if ctx["allow_counter"]:
            rank = {fid: i for i, fid in enumerate(ctx["priority_order"])}
            earliest = {f["flight_id"]: {r["id"]: parse(f["eta_standard_route"]) + r["eta_shift"] for r in f["routes"]}
                        for f in ctx.get("arrivals", [])}
            base = min(feas, key=lambda c: c["total_holding_min"])
            seq = [s for s in base["sequence"] if s["landing"] != "--:--" and not s["diverted_to"]]
            for rw in sorted({s["runway"] for s in seq}):
                on = sorted([s for s in seq if s["runway"] == rw], key=lambda s: parse(s["landing"]))
                for a, b in zip(on, on[1:]):
                    can_move_up = earliest.get(b["flight_id"], {}).get(b["route"], 0) <= parse(a["landing"])
                    if rank.get(b["flight_id"], 99) < rank.get(a["flight_id"], 99) and can_move_up:
                        vote.counter = CounterProposal(
                            base_candidate=base["id"],
                            patches=[Patch(op="swap_slots", flight_id=a["flight_id"], other_flight_id=b["flight_id"])],
                            concession="re-sequence; no throughput loss",
                            reason=f"{b['flight_id']} has higher priority than {a['flight_id']}")
                        return vote
        return vote


# ==========================================================================
# GROUND OPS (one per airport)
# ==========================================================================
class GroundAgent(BaseAgent):
    role = "GROUND"
    position_model = GroundPosition
    goals = ("optimise gate utilisation, avoid ground-crew overtime (costs 1.5x), minimise turnaround delays. "
             "Keep flights on their planned gate where possible; a gate change needs >= 20 min notice before landing.")
    levers = ("gate and crew assignment for every arrival (set_gate, set_crew), including gate swaps. "
              "Only the gates and crews listed in your view exist; never invent new ones.")

    def view(self, world):
        # Ground does not own the taxiway picture - ATC does. Cross-domain conflicts are caught by the checker.
        return {
            "gates": [{"id": g.id, "size": g.size, "free_from": fmt(g.free_from), "note": g.note,
                       "taxi_min_by_runway": g.taxi_min} for g in world.airport.gates],
            "crews": [{"id": c.id, "shift_end": fmt(c.shift_end), "busy_until": fmt(c.busy_until)}
                      for c in world.airport.crews],
            "gate_change_notice_min": world.airport.gate_change_notice,
            "note": "An occupied gate means the aircraft waits on the apron until it frees up.",
            "arrivals": [{"flight_id": f.flight_id, "planned_gate": f.planned_gate,
                          "aircraft_type": f.aircraft_type, "size": f.size,
                          "eta_standard_route": fmt(f.eta), "min_turnaround_min": f.min_turnaround,
                          "next_departure": fmt(f.next_departure) if f.next_departure else None,
                          "connecting_pax": f.connecting_pax}
                         for f in world.flights],
        }

    def offline_declare(self, v) -> GroundPosition:
        gates = v["gates"]
        crews = v["crews"]
        used: dict[str, list] = {}
        crew_used: dict[str, list] = {}
        out = []
        for f in sorted(v["arrivals"], key=lambda f: parse(f["eta_standard_route"])):
            est_in = parse(f["eta_standard_route"]) + 8
            end = est_in + f["min_turnaround_min"]
            fits = [g for g in gates if SIZE_RANK[g["size"]] >= SIZE_RANK[f["size"]]]
            plan = next((g for g in fits if g["id"] == f.get("planned_gate")), None)
            if plan and parse(plan["free_from"]) <= est_in + 15 and not any(
                    s < end and est_in < e for s, e in used.get(plan["id"], [])):
                fits = [plan]                      # keep the planned gate unless it means a long apron wait
            free = [g for g in fits if parse(g["free_from"]) <= est_in
                    and not any(s < end and est_in < e for s, e in used.get(g["id"], []))]
            # gate utilisation: smallest gate that fits, then shortest taxi
            pool = free or sorted(fits, key=lambda g: parse(g["free_from"]))
            g = min(pool, key=lambda g: (SIZE_RANK[g["size"]], min(g["taxi_min_by_runway"].values()))) if free else pool[0]
            used.setdefault(g["id"], []).append((est_in, end))
            ok_crews = [c for c in crews if parse(c["busy_until"]) <= est_in
                        and not any(s < end and est_in < e for s, e in crew_used.get(c["id"], []))]
            c = (min([c for c in ok_crews if parse(c["shift_end"]) >= end], key=lambda c: parse(c["shift_end"]),
                     default=None) or (max(ok_crews, key=lambda c: parse(c["shift_end"])) if ok_crews else crews[0]))
            crew_used.setdefault(c["id"], []).append((est_in, end))
            out.append(GateProposal(flight_id=f["flight_id"], gate=g["id"], crew=c["id"]))
        return GroundPosition(gates=out, message=f"Gates assigned for {len(out)} arrivals, smallest-fit first")

    def candidate_view(self, world, cand, mets, feasible, violations) -> dict:
        rows, ot, tad = [], 0, 0
        for fid, a in cand.assignments.items():
            m = mets.get(fid)
            rows.append({"flight_id": fid, "gate": a.gate, "crew": a.crew,
                         "gate_in": fmt(m.gate_in) if m else None, "ready": fmt(m.ready) if m else None,
                         "overtime_min": m.overtime if m else 0,
                         "turnaround_delay_min": m.turnaround_delay if m else 0})
            ot += m.overtime if m else 0
            tad += m.turnaround_delay if m else 0
        return {"id": cand.id, "source": cand.source, "feasible": feasible,
                "violations": [x.message for x in violations], "gates": rows,
                "total_overtime_min": ot, "total_turnaround_delay_min": tad,
                "crews": [{"id": c.id, "shift_end": fmt(c.shift_end), "busy_until": fmt(c.busy_until)}
                          for c in world.airport.crews]}

    def offline_vote(self, ctx) -> Vote:
        feas = [c for c in ctx["candidates"] if c["feasible"]]
        vote = Vote(message=f"GROUND reviewed {len(feas)} feasible options")
        if not feas:
            return vote
        best_ot = min(c["total_overtime_min"] for c in feas)
        best_tad = min(c["total_turnaround_delay_min"] for c in feas)
        k = 3 if ctx["round"] >= 3 else 1   # final round: pay some overtime rather than deadlock
        for c in feas:
            if c["total_overtime_min"] > best_ot + 10 * k:
                vote.add_reject(c["id"], f"{c['total_overtime_min']} crew overtime min vs {best_ot} possible")
            elif c["total_turnaround_delay_min"] > best_tad + 15 * k:
                vote.add_reject(c["id"], f"pushes next departures {c['total_turnaround_delay_min']} min")
            else:
                vote.accept.append(c["id"])
        # counter: move an overtime turnaround onto a crew with shift time left
        if ctx["allow_counter"]:
            base = min(feas, key=lambda c: (c["total_overtime_min"], c["total_turnaround_delay_min"]))
            busy = {g["crew"] for g in base["gates"]}
            for g in base["gates"]:
                if g["overtime_min"] > 0 and g["ready"] not in (None, "--:--"):
                    spare = [c for c in base["crews"] if c["id"] not in busy
                             and parse(c["busy_until"]) <= parse(g["gate_in"])
                             and parse(c["shift_end"]) >= parse(g["ready"])]
                    if spare:
                        vote.counter = CounterProposal(
                            base_candidate=base["id"],
                            patches=[Patch(op="set_crew", flight_id=g["flight_id"], crew=spare[0]["id"])],
                            concession=f"bring crew {spare[0]['id']} onto the apron early",
                            reason=f"removes {g['overtime_min']} min overtime on {g['flight_id']}")
                        break
        return vote