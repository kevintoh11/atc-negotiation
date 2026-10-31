"""Compare today's process against the negotiated plan across many disruptions.

    python evaluate.py                                   # sample data, offline agents, one row per scenario
    python evaluate.py --data data/disruptions.csv --limit 300
    python evaluate.py --data data/disruptions.csv --limit 20 --llm   # Claude agents vs offline, same rows
    python evaluate.py --data data/disruptions.csv --limit 100 --chaos all  # inject every fault, count catches
    python evaluate.py --mode cluster                    # optional extension: several flights at once

BASELINE ("today's process"): each domain plans alone, nobody coordinates.
  ATC sequences first-come-first-served around the other traffic, standard routes, and
  cannot see fuel. Ground picks gates and crews from ETAs, without the actual landing
  times or the taxiway picture. Nothing is cross-checked, so conflicts are only found
  later (today: phone calls).
NEGOTIATED: the full system (agents + checker + arbiter + fallback).

Reported as raw minutes and passenger counts, never the arbiter's cost, because the
negotiation optimises that cost and scoring with it would flatter the system.
Baseline metrics are measured on the plan as made, conflicts included; they are counted
separately rather than resolved, so the baseline is if anything shown optimistically.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from statistics import mean

from atm.agents import ATCAgent, GroundAgent
from atm.constraints import check, metrics
from atm.chaos import CHAOS_MODES
from atm.protocol import negotiate
from atm.scenario import cluster_rows, disrupted, find_clusters, load_flights, scenario
from atm.schemas import Assignment, Candidate
from atm.timeutil import fmt, parse_near

METRICS = ["holding_min", "holding_fuel_kg", "connections_missed", "crew_overtime_min", "next_leg_delay_min",
           "apron_wait_min", "gate_changes", "flights_below_reserve", "diversions", "hard_conflicts"]


def baseline_plan(world) -> Candidate:
    atc, gnd = ATCAgent(), GroundAgent()
    slots = {s.flight_id: s for s in atc.offline_declare(atc.view(world)).slots}
    gates = {g.flight_id: g for g in gnd.offline_declare(gnd.view(world)).gates}
    asg = {}
    for f in world.flights:
        s, g = slots[f.flight_id], gates[f.flight_id]
        asg[f.flight_id] = Assignment(flight_id=f.flight_id, route="STD", runway=s.runway,
                                      landing=parse_near(s.landing, f.eta), gate=g.gate, crew=g.crew)
    return Candidate(id="BASELINE", source="baseline", round=0, assignments=asg)


def outcome(world, cand) -> dict:
    rows = [(f, cand.assignments[f.flight_id]) for f in world.flights]
    ms = [(f, a, metrics(world, f, a)) for f, a in rows]
    return {
        "holding_min": sum(max(0, m.holding) for f, a, m in ms if not m.diverted),
        "connections_missed": sum(m.conn_missed for f, a, m in ms),
        "crew_overtime_min": sum(m.overtime for f, a, m in ms),
        "next_leg_delay_min": sum(m.turnaround_delay for f, a, m in ms),
        "apron_wait_min": sum(m.gate_wait + m.crew_wait for f, a, m in ms),
        "holding_fuel_kg": int(sum(max(0, m.holding) / 60 * f.burn_kg_h for f, a, m in ms if not m.diverted)),
        "gate_changes": sum(1 for f, a, m in ms if f.planned_gate and a.gate != f.planned_gate),
        "flights_below_reserve": sum(m.fuel_at_landing < f.reserve for f, a, m in ms),
        "diversions": sum(m.diverted for f, a, m in ms),
        "hard_conflicts": len(check(world, cand)),
        "min_fuel_margin": min(m.fuel_margin for f, a, m in ms),
    }


def run_all(df, items, llm, label, a=None):
    """Negotiate every scenario. With Claude agents each scenario gets fresh clients so usage is per scenario."""
    results, flight_rows = [], []
    for i, rows in enumerate(items):
        world = scenario(df, rows)
        bplan = baseline_plan(world)
        base = outcome(world, bplan)
        clients = build_clients(a) if llm is not None else None
        t0 = time.perf_counter()
        res = negotiate(world, llm=clients)
        secs = time.perf_counter() - t0
        neg = outcome(world, res.decision)
        u = res.usage or {}
        for f in world.flights:     # who individually gave something up?
            bm, nm = (metrics(world, f, c.assignments[f.flight_id]) for c in (bplan, res.decision))
            later = None if (nm.diverted or bm.diverted) else nm.gate_in - bm.gate_in
            flight_rows.append({"scenario": world.scenario_id, "flight": f.flight_id,
                                "priority": res.urgency[f.flight_id].level, "gate_in_later_min": later,
                                "diverted": nm.diverted, "today_legal": base["hard_conflicts"] == 0,
                                "conn_missed_base": bm.conn_missed, "conn_missed_neg": nm.conn_missed,
                                "fuel_margin_base": bm.fuel_margin, "fuel_margin_neg": nm.fuel_margin})
        results.append({"scenario": world.scenario_id, "flights": len(world.flights),
                        "other_traffic": len(world.airport.occupied_slots), "decided_by": res.decided_by,
                        "rounds": res.rounds_used, "escalated": res.escalated,
                        "agent_errors": sum(e["event"] == "AGENT_ERROR" for e in res.log.entries),
                        "checker_rejects": sum(e["actor"] == "CHECKER" and e["event"] == "REJECT"
                                               for e in res.log.entries),
                        "llm_calls": u.get("calls", 0), "llm_failed": u.get("failed", 0),
                        "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                        "seconds": round(secs, 2), "decision": decision_key(res.decision),
                        **{f"base_{k}": v for k, v in base.items()}, **{f"neg_{k}": v for k, v in neg.items()}})
        print(f"\r[{label}] {i + 1}/{len(items)} {world.scenario_id:<26}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return results, flight_rows


def build_clients(a):
    from atm.llm import build_llms
    return build_llms(model=getattr(a, "model", None), pilot_model=getattr(a, "pilot_model", None))


def decision_key(cand) -> str:
    """Compact fingerprint of a plan, to tell whether two runs ended in the same decision."""
    return ";".join(f"{fid}:{x.route}/{x.runway}/{x.landing}/{x.gate}/{x.crew}"
                    for fid, x in sorted(cand.assignments.items()))


def llm_section(llm_res: list[dict], off_res: list[dict]) -> list[str]:
    """Claude agents vs offline (rule-based) agents on exactly the same scenarios."""
    n = len(llm_res)

    def col(rs, f):
        return f(rs)

    rows = [
        ("Consensus (no fallback)", lambda rs: f"{sum(r['decided_by'] == 'consensus' for r in rs)}/{n}"),
        ("Fallback decided", lambda rs: f"{sum(r['decided_by'] == 'fallback' for r in rs)}/{n}"),
        ("Escalated to a controller", lambda rs: str(sum(r["escalated"] for r in rs))),
        ("Mean turns (limit 3)", lambda rs: f"{mean(r['rounds'] for r in rs):.2f}"),
        ("Final plans with a hard conflict", lambda rs: str(sum(r["neg_hard_conflicts"] > 0 for r in rs))),
        ("Unsafe proposals caught by the checker", lambda rs: str(sum(r["checker_rejects"] for r in rs))),
        ("Agent errors (bad/failed output -> handled)", lambda rs: str(sum(r["agent_errors"] for r in rs))),
        ("Holding (min, total)", lambda rs: str(sum(r["neg_holding_min"] for r in rs))),
        ("Connections missed (total)", lambda rs: str(sum(r["neg_connections_missed"] for r in rs))),
        ("Crew overtime (min, total)", lambda rs: str(sum(r["neg_crew_overtime_min"] for r in rs))),
        ("Apron wait (min, total)", lambda rs: str(sum(r["neg_apron_wait_min"] for r in rs))),
        ("Gate changes (total)", lambda rs: str(sum(r["neg_gate_changes"] for r in rs))),
        ("Mean wall time per scenario (s)", lambda rs: f"{mean(r['seconds'] for r in rs):.1f}"),
    ]
    out = ["", f"## Claude agents vs offline agents (same {n} scenarios)", "",
           "| | Claude agents | Offline agents |", "|---|---|---|"]
    out += [f"| {name} | {col(llm_res, f)} | {col(off_res, f)} |" for name, f in rows]
    same = sum(l["decision"] == o["decision"] for l, o in zip(llm_res, off_res))
    calls = sum(r["llm_calls"] for r in llm_res)
    tin, tout = sum(r["input_tokens"] for r in llm_res), sum(r["output_tokens"] for r in llm_res)
    out += ["", f"- Same final plan as the offline agents: {same}/{n}",
            f"- LLM calls: {calls} ({sum(r['llm_failed'] for r in llm_res)} failed), "
            f"{calls / n:.1f} per scenario",
            f"- Tokens: {tin:,} in / {tout:,} out ({(tin + tout) / n:,.0f} per scenario); "
            "check the Anthropic Console usage page for the exact cost",
            f"- Slowest scenario: {max(r['seconds'] for r in llm_res):.1f} s"]
    worse = [(l, o) for l, o in zip(llm_res, off_res)
             if (l["decided_by"] == "fallback") > (o["decided_by"] == "fallback") or l["rounds"] > o["rounds"]]
    if worse:
        out += ["", "Scenarios where Claude agents needed more turns or the fallback (read these logs):", ""]
        out += [f"- {l['scenario']}: Claude {l['decided_by']} in {l['rounds']} turns, "
                f"offline {o['decided_by']} in {o['rounds']}" for l, o in worse[:10]]
    return out


# what a caught fault looks like in the consensus log, per chaos mode
EXPECTED_CATCH = {
    "fuel": ("CHECKER", "REJECT", "FUEL_RESERVE"),
    "ghost_gate": ("CHECKER", "REJECT", "GATE_UNKNOWN"),
    "wake": ("CHECKER", "REJECT", "WAKE_SEPARATION"),
    "urgency": ("ARBITER", "DOWNWEIGHT", None),
    "role": ("PROTOCOL", "PERMISSION_DENIED", None),
}


def caught(entries: list[dict], mode: str) -> bool:
    actor, event, code = EXPECTED_CATCH[mode]
    return any(e["actor"] == actor and e["event"] == event
               and (code is None or (e.get("data") or {}).get("code") == code) for e in entries)


def chaos_sweep(df, items, a, llm=None):
    """Inject each fault into each scenario. A fault counts as caught when the guardrail that owns it
    fires in the consensus log, and as contained when the final plan still has no hard conflict."""
    modes = [m for m in EXPECTED_CATCH] if a.chaos == "all" else [a.chaos]
    rows = []
    for i, scen_rows in enumerate(items):
        world = scenario(df, scen_rows)
        clean = negotiate(world, llm=build_clients(a) if llm else None)
        clean_key = decision_key(clean.decision)
        for mode in modes:
            res = negotiate(world, llm=build_clients(a) if llm else None, chaos=mode)
            ents = res.log.entries
            injected = any(e["actor"] == "CHAOS" and e["event"] == "INJECT" for e in ents)
            # a CRITICAL claim is only a fault if the world state says LOW or MEDIUM: the arbiter
            # tolerates a claim one level above (HIGH -> CRITICAL), and priority comes from world state anyway
            applicable = injected
            if mode == "urgency" and injected:
                tgt = max(world.flights, key=lambda f: f.fuel_at_eta - f.reserve)
                applicable = res.urgency[tgt.flight_id].level in ("LOW", "MEDIUM")
            final_viol = check(world, res.decision)
            rows.append({"scenario": world.scenario_id, "mode": mode, "injected": injected,
                         "applicable": applicable, "caught": applicable and caught(ents, mode),
                         "final_hard_conflicts": len(final_viol), "escalated": res.escalated,
                         "decided_by": res.decided_by, "rounds": res.rounds_used,
                         "same_as_clean_run": decision_key(res.decision) == clean_key,
                         "clean_escalated": clean.escalated})
        print(f"\r[chaos] {i + 1}/{len(items)} {world.scenario_id:<26}", end="", file=sys.stderr)
    print(file=sys.stderr)
    if not rows:
        sys.exit("no disrupted scenarios in the data")

    lines = [f"# Chaos sweep: {len(items)} scenarios x {len(modes)} fault type(s) "
             f"({'Claude' if llm else 'offline'} agents)", "",
             f"Data: `{a.data}`. Each fault is injected into one agent's real message; the guardrail that owns "
             "it must fire in the consensus log (caught) and the final plan must still pass every hard "
             "constraint (contained).", "",
             "| Fault | What is injected | Guardrail expected | Injected | Caught | Final plan safe | "
             "Same plan as clean run | Fallback |", "|---|---|---|---|---|---|---|---|"]
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        app = [r for r in rs if r["applicable"]]
        actor, event, code = EXPECTED_CATCH[mode]
        safe = [r for r in rs if r["final_hard_conflicts"] == 0 or r["clean_escalated"]]
        lines.append(f"| {mode} | {CHAOS_MODES[mode]} | {actor} {event}{' ' + code if code else ''} | "
                     f"{len(app)}/{len(rs)} | {sum(r['caught'] for r in app)}/{len(app)} | "
                     f"{len(safe)}/{len(rs)} | {sum(r['same_as_clean_run'] for r in rs)}/{len(rs)} | "
                     f"{sum(r['decided_by'] == 'fallback' for r in rs)} |")
    app = [r for r in rows if r["applicable"]]
    missed = [r for r in app if not r["caught"]]
    unsafe = [r for r in rows if r["final_hard_conflicts"] > 0 and not r["clean_escalated"]]
    lines += ["", f"**{len(app)} faults injected, {len(app) - len(missed)} caught, "
                  f"{len(unsafe)} reached a final plan.**", ""]
    na = [r for r in rows if not r["applicable"]]
    if na:
        why = {"wake": "no other traffic in the window to squeeze behind",
               "urgency": "the flight really was HIGH or CRITICAL, so a CRITICAL claim is true or within the "
                          "one-level tolerance (priority is set from world state either way)",
               "role": "no counter-offer turn"}
        lines += [f"Not counted: {mode} {sum(r['mode'] == mode for r in na)} ({why.get(mode, 'not injectable')})"
                  for mode in modes if any(r["mode"] == mode for r in na)]
    if missed:
        lines += ["", "Missed (read these logs):", ""] + [f"- {r['scenario']} {r['mode']}" for r in missed[:10]]
    if unsafe:
        lines += ["", "Unsafe final plans:", ""] + [f"- {r['scenario']} {r['mode']}" for r in unsafe[:10]]
    lines += ["", "\"Same plan as clean run\" below 100% is not a failure: when the rejected proposal was one an "
                  "agent preferred, the agents settle on the next-best legal plan (read chaos.csv for which).",
              "", "Scenarios that were already escalated without any fault (no legal plan exists) are counted "
                  "as safe: the system hands them to a controller rather than deciding."]

    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "chaos.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = "\n".join(lines)
    (Path(a.out) / "chaos.md").write_text(report)
    print(report)
    print(f"\nWritten: {a.out}/chaos.csv, {a.out}/chaos.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample_flights.csv")
    ap.add_argument("--limit", type=int, default=300, help="max scenarios (rows or clusters) to evaluate")
    ap.add_argument("--mode", choices=["row", "cluster"], default="row",
                    help="row: each disrupted row is one scenario (the challenge rules); cluster: extension")
    ap.add_argument("--max-flights", type=int, default=4)
    ap.add_argument("--llm", action="store_true",
                    help="Claude agents, compared with offline agents on the same rows (costly: keep --limit ~20)")
    ap.add_argument("--model", default=None, help="ATC/Ground/summariser model (default: atm.llm)")
    ap.add_argument("--pilot-model", default=None, help="pilot model (default: atm.llm)")
    ap.add_argument("--chaos", choices=[m for m in CHAOS_MODES if m != "none"] + ["all"], default=None,
                    help="fault-injection sweep: inject this fault (or every fault) into each scenario")
    ap.add_argument("--out", default="out/eval")
    a = ap.parse_args()

    llm = None
    if a.llm:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("--llm needs ANTHROPIC_API_KEY set")
        llm = True          # fresh clients are built per scenario in run_all

    df = load_flights(a.data)
    if a.mode == "row":
        d = disrupted(df)
        if len(d) > a.limit:
            d = d.sample(a.limit, random_state=42)
        items = [d.loc[[idx]] for idx in d.index]
    else:
        items = [cluster_rows(df, key, a.max_flights)
                 for key in find_clusters(df, min_flights=1).head(a.limit).to_dict("records")]
    if a.chaos:
        return chaos_sweep(df, items, a, llm)
    if llm is not None:
        off_results, _ = run_all(df, items, None, "offline")
    results, flight_rows = run_all(df, items, llm, "Claude" if llm else "offline", a)
    if not results:
        sys.exit("no disrupted scenarios in the data")

    n = len(results)
    unit = "disrupted rows (one flight, three agents each)" if a.mode == "row" else "disrupted airport-hours"
    lines = [f"# Evaluation: {n} {unit} ({'Claude' if llm else 'offline'} agents)", "",
             f"Data: `{a.data}`. Flights negotiated: {sum(r['flights'] for r in results)}; "
             f"other traffic taking slots: {sum(r['other_traffic'] for r in results)}.", "",
             "## Process", "",
             f"- Agreement (consensus) rate: **{sum(r['decided_by'] == 'consensus' for r in results) / n:.0%}**",
             f"- Fallback rate: **{sum(r['decided_by'] == 'fallback' for r in results) / n:.0%}**",
             f"- Mean turns: {mean(r['rounds'] for r in results):.2f} (limit 3)",
             f"- Escalated to a controller (no plan satisfies every hard constraint): "
             f"**{sum(r['escalated'] for r in results)}**",
             f"- Baseline plans with at least one hard conflict: "
             f"**{sum(r['base_hard_conflicts'] > 0 for r in results) / n:.0%}**", "",
             "## Outcomes (totals across all scenarios)", "",
             "| Metric | Today's process | Negotiated | Better in | Worse in |",
             "|---|---|---|---|---|"]
    for k in METRICS:
        b, g = sum(r[f"base_{k}"] for r in results), sum(r[f"neg_{k}"] for r in results)
        better = sum(r[f"neg_{k}"] < r[f"base_{k}"] for r in results)
        worse = sum(r[f"neg_{k}"] > r[f"base_{k}"] for r in results)
        lines.append(f"| {k.replace('_', ' ')} | {b} | {g} | {better} | {worse} |")

    # like for like: only scenarios where today's plan was itself legal
    legal = [r for r in results if r["base_hard_conflicts"] == 0]
    lines += ["", f"## Like for like: the {len(legal)} scenarios where today's plan had no hard conflict", "",
              "In the other scenarios today's plan only works by breaking a rule (e.g. taxiing through a blocked "
              "taxiway), so its numbers flatter it.", "",
              "| Metric | Today's process | Negotiated | Better in | Worse in |", "|---|---|---|---|---|"]
    for k in METRICS[:-1]:
        b, g = sum(r[f"base_{k}"] for r in legal), sum(r[f"neg_{k}"] for r in legal)
        lines.append(f"| {k.replace('_', ' ')} | {b} | {g} | {sum(r[f'neg_{k}'] < r[f'base_{k}'] for r in legal)} | "
                     f"{sum(r[f'neg_{k}'] > r[f'base_{k}'] for r in legal)} |")

    # the clearest case where negotiation does worse, and why
    def regret(r):
        return max(r[f"neg_{k}"] - r[f"base_{k}"] for k in ("holding_min", "connections_missed", "crew_overtime_min",
                                                            "next_leg_delay_min", "apron_wait_min"))
    worst = max(results, key=regret)
    lines += ["", "## A case where negotiation does worse", ""]
    if regret(worst) <= 0:
        lines.append("None: in every scenario the negotiated plan was at least as good on every metric.")
    else:
        lines += [f"**{worst['scenario']}** ({worst['decided_by']}, {worst['rounds']} turns)", "",
                  "| Metric | Today's process | Negotiated |", "|---|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {worst[f'base_{k}']} | {worst[f'neg_{k}']} |"
                  for k in METRICS + ["min_fuel_margin"]]
        lines += ["", "Read the consensus log for this scenario to see which trade-off the agents made "
                      "(typically: fuel safety, wake spacing or another flight's connections were protected "
                      "at this metric's expense, or the baseline only looks better because its plan "
                      "contains a hard conflict)."]
    # flight level: totals can improve while individual flights pay for it
    losers = sorted(flight_rows, key=lambda r: (r["diverted"], r["conn_missed_neg"] - r["conn_missed_base"],
                                                r["gate_in_later_min"] or 0), reverse=True)
    paid = [r for r in losers if r["diverted"] or (r["gate_in_later_min"] or 0) > 0
            or r["conn_missed_neg"] > r["conn_missed_base"]]
    lines += ["", "## Who paid for it (flight level)", "",
              f"{len(paid)} of {len(flight_rows)} flights reached the gate later or lost more connections "
              "than under today's process. Totals can improve while individual flights give something up, usually "
              "time traded for fuel margin, or a slot ceded to a more urgent flight:", "",
              "| Scenario | Flight | Priority | Gate-in later (min) | Connections missed (today → negotiated) "
              "| Fuel margin (today → negotiated) | Today's plan legal? |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['scenario']} | {r['flight']} | {r['priority']} | "
              f"{'diverted' if r['diverted'] else r['gate_in_later_min']} | "
              f"{r['conn_missed_base']} → {r['conn_missed_neg']} | {r['fuel_margin_base']} → {r['fuel_margin_neg']} | "
              f"{'yes' if r['today_legal'] else 'no'} |"
              for r in paid[:8]] or ["| none | | | | | | |"]
    if llm is not None:
        lines += llm_section(results, off_results)
    lines += ["", "Caveats: offline agents unless stated; simulated fuel, gates, crews and connections "
                  "(see README); baseline conflicts are counted, not resolved."]

    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name, rows in (("results.csv", results), ("flights.csv", flight_rows)):
        with open(Path(a.out) / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    report = "\n".join(lines)
    (Path(a.out) / "summary.md").write_text(report)
    print(report)
    print(f"\nWritten: {a.out}/results.csv, {a.out}/flights.csv, {a.out}/summary.md")


if __name__ == "__main__":
    main()