"""Controller briefing. Explains the decision; has NO decision power.

Offline: a template. With --llm: Claude rewrites the same facts as a short
briefing (it only sees the already-decided, already-checked outcome).
"""
from __future__ import annotations

import json

from .constraints import metrics
from .timeutil import fmt


def facts(result) -> dict:
    w, cand = result.world, result.decision
    flights = []
    for f in w.flights:
        a = cand.assignments[f.flight_id]
        m = metrics(w, f, a)
        u = result.urgency[f.flight_id]
        if a.divert_to:
            flights.append({"flight": f.flight_id, "outcome": f"DIVERT to {a.divert_to}", "priority": u.level})
            continue
        flights.append({
            "flight": f.flight_id, "priority": u.level, "priority_reasons": u.reasons,
            "earliest_possible_landing": fmt(m.earliest),
            "landing": fmt(a.landing), "runway": a.runway, "route": a.route, "holding_min": m.holding,
            "fuel_at_landing_min": m.fuel_at_landing, "reserve_min": f.reserve, "gate": a.gate, "crew": a.crew,
            "gate_in": fmt(m.gate_in), "connections_missed": m.conn_missed, "of_connecting": f.connecting_pax,
            "crew_overtime_min": m.overtime, "next_leg_delay_min": m.turnaround_delay,
        })
    return {"airport": w.airport.code, "disruption": w.disruption, "decided_by": result.decided_by,
            "turns": result.rounds_used, "fallback_reason": result.fallback_reason,
            "binding": result.priority[0] if result.priority else None, "flights": flights}


def template_briefing(result) -> str:
    fx = facts(result)
    head = (f"{fx['airport']}: {fx['disruption']}. "
            + (f"Agreed by all agents in {fx['turns']} turns."
               if fx["decided_by"] == "consensus"
               else f"No consensus ({fx['fallback_reason']}); deterministic fallback applied."))
    lines = [head]
    for r in fx["flights"]:
        if "outcome" in r:
            lines.append(f"- {r['flight']} [{r['priority']}]: {r['outcome']}.")
            continue
        lines.append(
            f"- {r['flight']} [{r['priority']}]: land {r['landing']} {r['runway']} via {r['route']}"
            f" ({r['holding_min']} min hold), gate {r['gate']} / crew {r['crew']}. Fuel at landing "
            f"{r['fuel_at_landing_min']} min (reserve {r['reserve_min']}); "
            f"{r['connections_missed']}/{r['of_connecting']} connections missed.")
    top = next((r for r in fx["flights"] if r["flight"] == fx["binding"]), None)
    if top and "priority_reasons" in top:
        lines.append(f"Binding priority: {top['flight']} - {top['priority_reasons'][0]}.")
    return "\n".join(lines)


def llm_briefing(result, llm) -> str:
    system = ("You brief an air traffic controller on a decision already made. Use only the facts given; "
              "you cannot change the decision. Plain text, no markdown. Write these lines in order:\n"
              "1. The situation, in one line.\n"
              "2. One line per flight: landing time, runway, route, gate, fuel at landing vs reserve.\n"
              "3. The binding constraint. Priority decides who is protected, not who lands first: a flight "
              "cannot land before its earliest_possible_landing, so use that to explain the sequence.\n"
              "4. The main trade-off made.\n"
              "5. How it was decided (consensus in N turns, or fallback and why).")
    text = llm.complete_text(system, json.dumps(facts(result), indent=1), role="SUMMARISER")
    if len([ln for ln in text.splitlines() if ln.strip()]) < 3:
        raise ValueError(f"briefing too short: {text!r}")   # caller keeps the template briefing
    return text