"""Command-line runner. Default use: one disrupted row = the disruption, three agents.

    python run.py --list                          # disrupted rows (and multi-flight clusters)
    python run.py --row 0                         # one disrupted row = the disruption, 3 agents
    python run.py --cluster 0                     # optional extension: several flights at once
    python run.py --row 0 --chaos fuel            # inject a fault, watch it get caught
    python run.py --row 0 --llm                   # Claude agents (needs ANTHROPIC_API_KEY)
    python run.py --data data/disruptions.csv ... # use the real filtered DOT data
"""
import argparse
import sys
from pathlib import Path

from atm.chaos import CHAOS_MODES
from atm.constraints import metrics
from atm.protocol import negotiate
from atm.scenario import cluster_rows, disrupted, find_clusters, load_flights, scenario
from atm.timeutil import fmt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample_flights.csv")
    ap.add_argument("--list", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cluster", type=int, help="index from --list (multi-flight contention)")
    g.add_argument("--row", "--single", dest="single", type=int,
                   help="index of a disrupted row from --list: that row is the disruption (3 agents)")
    ap.add_argument("--max-flights", type=int, default=4)
    ap.add_argument("--chaos", default="none", choices=list(CHAOS_MODES))
    ap.add_argument("--llm", action="store_true", help="use Claude agents instead of offline policies")
    ap.add_argument("--model", default=None, help="ATC, Ground and summariser model (default: ATM_MODEL or claude-sonnet-5-5)")
    ap.add_argument("--pilot-model", default=None, help="pilot model (default: ATM_PILOT_MODEL or claude-haiku-5-5)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per agent call before it counts as failed")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    df = load_flights(a.data)
    clusters = find_clusters(df, min_flights=1)
    if a.list or (a.cluster is None and a.single is None):
        d = disrupted(df).reset_index(drop=True)
        print(f"{len(d)} disrupted rows (WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20). Use --row N:")
        show = d[["YEAR", "MONTH", "DAY", "AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
                  "SCHEDULED_ARRIVAL", "DEPARTURE_DELAY", "WEATHER_DELAY", "LATE_AIRCRAFT_DELAY"]]
        print(show.head(30).to_string())
        print("\nOptional extension, several disrupted flights at once (--cluster N):")
        print(clusters.head(10).to_string())
        return

    if a.cluster is not None:
        rows = cluster_rows(df, clusters.iloc[a.cluster].to_dict(), a.max_flights)
    else:
        rows = disrupted(df).iloc[[a.single]]
    world = scenario(df, rows)

    llm = None
    if a.llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("--llm needs ANTHROPIC_API_KEY set (export ANTHROPIC_API_KEY=...). Run without --llm for offline agents.")
        from atm.llm import build_llms
        llm = build_llms(a.model, a.pilot_model)

    res = negotiate(world, llm=llm, chaos=a.chaos, timeout=a.timeout)

    print(f"\n=== {world.scenario_id} | {world.airport.name} ===")
    print(f"Disruption: {world.disruption}\nWeather: {world.weather}\n")
    print("CONSENSUS LOG")
    for line in res.log.lines():
        print(line)
    print("\nDECISION", f"({res.decided_by}, {res.rounds_used} turns)")
    for f in world.flights:
        x = res.decision.assignments[f.flight_id]
        m = metrics(world, f, x)
        if x.divert_to:
            print(f"  {f.flight_id:<8} DIVERT {x.divert_to}")
        else:
            print(f"  {f.flight_id:<8} {fmt(x.landing)} {x.runway:<4} {x.route:<8} hold {m.holding:>3}m  "
                  f"fuel {m.fuel_at_landing:>3}m  gate {x.gate}/{x.crew}  conn missed {m.conn_missed}")
    print("\nBRIEFING\n" + res.briefing)
    if res.usage:
        u = res.usage
        print(f"\nLLM USAGE: {u['calls']} calls ({u['failed']} failed), {u['input_tokens']} in / "
              f"{u['output_tokens']} out tokens, slowest call {u['slowest_call_s']}s")
        for role, d in u["by_role"].items():
            print(f"  {role:<11} {d['model']:<22} {d['calls']:>3} calls  {d['seconds']:>6.1f}s total")
    ok, bad = res.log.verify()
    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"{world.scenario_id}_{a.chaos}.jsonl"
    res.log.write_jsonl(str(path))
    print(f"\nLog: {path}  (hash chain {'verified' if ok else f'BROKEN at {bad}'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
