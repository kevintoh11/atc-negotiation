"""Streamlit demo.   streamlit run app.py"""
import json
import os

import pandas as pd
import streamlit as st

from atm.chaos import CHAOS_MODES
from atm.constraints import metrics
from atm.protocol import negotiate
from atm.scenario import cluster_rows, disrupted, find_clusters, load_flights, scenario
from atm.timeutil import fmt

st.set_page_config(page_title="ATC Negotiation", layout="wide")

EVENT_STYLE = {
    "REJECT": "background-color: rgba(220, 53, 69, .15)",
    "PERMISSION_DENIED": "background-color: rgba(220, 53, 69, .25)",
    "BLOCKED": "background-color: rgba(220, 53, 69, .25)",
    "DOWNWEIGHT": "background-color: rgba(255, 193, 7, .25)",
    "INJECT": "background-color: rgba(255, 140, 0, .30)",
    "FALLBACK": "background-color: rgba(255, 193, 7, .30)",
    "COUNTER": "background-color: rgba(13, 110, 253, .12)",
    "MERGE": "background-color: rgba(13, 110, 253, .12)",
    "OK": "background-color: rgba(25, 135, 84, .12)",
    "ACCEPT": "background-color: rgba(25, 135, 84, .08)",
    "DECIDE": "background-color: rgba(25, 135, 84, .30); font-weight: 600",
}


@st.cache_data
def _load(path):
    return load_flights(path)


def _style_log(df):
    return df.style.apply(lambda r: [EVENT_STYLE.get(r.Event, "")] * len(r), axis=1)


def render_replay():
    """Step through a saved consensus log turn by turn. No API calls."""
    import copy
    import glob

    from atm.log import ConsensusLog
    from atm.replay import load as load_replay

    with st.sidebar:
        st.header("Saved log")
        files = sorted(glob.glob("demo/*.jsonl")) + sorted(glob.glob("out/*.jsonl"))
        choice = st.selectbox("Log file", files) if files else None
        up = st.file_uploader("...or upload a .jsonl log", type=["jsonl"])
        tamper = st.checkbox("Simulate tampering",
                             help="Edits one old entry in memory to show that the hash chain detects it")
    if up is not None:
        rep, key = load_replay(up.getvalue().decode().splitlines()), up.name
    elif choice:
        rep, key = load_replay(choice), choice
    else:
        st.info("No saved logs in demo/ or out/ yet. Run a scenario first, then copy its log into demo/.")
        return

    if st.session_state.get("replay_key") != key:
        st.session_state.replay_key, st.session_state.replay_idx = key, 0
    n = len(rep.stages)

    def step(d):
        st.session_state.replay_idx = max(0, min(n - 1, st.session_state.replay_idx + d))

    st.title("Replay: multi-agent disruption negotiation")
    st.caption(rep.scenario)
    meta = rep.meta or {}
    models = meta.get("models") or {}
    st.caption(f"Recorded in {meta.get('mode', '?')} mode · chaos: {meta.get('chaos', 'none')}"
               + (" · " + ", ".join(f"{k}: {v}" for k, v in models.items() if v) if any(models.values()) else ""))

    check_log = rep.log
    if tamper:
        check_log = ConsensusLog()
        check_log.entries = copy.deepcopy(rep.log.entries)
        i = min(5, len(check_log.entries) - 1)
        check_log.entries[i]["reason"] += " [edited after the fact]"
    ok, bad = check_log.verify()
    (st.success if ok else st.error)(
        "Hash chain verified: no entry has been altered" if ok
        else f"Hash chain BROKEN at entry {bad}: the log was modified after it was written")

    c1, c2, c3 = st.columns([1, 6, 1])
    c1.button("Previous", on_click=step, args=(-1,), disabled=st.session_state.replay_idx == 0, width="stretch")
    c3.button("Next", on_click=step, args=(1,), disabled=st.session_state.replay_idx == n - 1, width="stretch")
    idx = st.session_state.replay_idx
    stage = rep.stages[idx]
    c2.progress((idx + 1) / n, text=f"{stage.name}  ({idx + 1} of {n}: " + " → ".join(s.name for s in rep.stages) + ")")

    if stage.name == "Decision":
        m1, m2 = st.columns(2)
        m1.metric("Decided by", (rep.decided_by or "?").capitalize())
        m2.metric("Turns", sum(1 for s in rep.stages if s.name.startswith("Turn")))
        if rep.outcomes:
            st.dataframe(pd.DataFrame(rep.outcomes), hide_index=True, width="stretch")
        if rep.briefing:
            st.subheader("Controller briefing")
            st.text(rep.briefing)
        if rep.usage:
            u = rep.usage
            st.caption(f"LLM: {u['calls']} calls ({u['failed']} failed) · {u['input_tokens']:,} in / "
                       f"{u['output_tokens']:,} out tokens · slowest call {u['slowest_call_s']}s")

    entries = check_log.entries[:stage.end]
    ldf = pd.DataFrame([{"New": "●" if i >= stage.start else "", "#": e["seq"], "Turn": e["round"],
                         "Actor": e["actor"], "Event": e["event"], "Detail": e["reason"]}
                        for i, e in enumerate(entries)])
    st.subheader(f"Consensus log up to {stage.name.lower()}")
    st.caption("● = new in this stage")
    st.dataframe(_style_log(ldf), hide_index=True, width="stretch", height=560)


source = st.sidebar.radio("Source", ["Live run", "Replay saved log"], horizontal=True)
if source == "Replay saved log":
    render_replay()
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Scenario")
    data_path = st.text_input("Flights CSV", "data/sample_flights.csv",
                              help="data/sample_flights.csv (made-up rows) or the real filtered file from "
                                   "scripts/prepare_data.py")
    try:
        df = _load(data_path)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load {data_path}: {e}")
        st.stop()
    dis = disrupted(df)
    mode = st.radio("Mode", ["Single flight (3 agents)", "Multi-flight contention"],
                    help="Single flight follows the challenge rules: one disrupted row, three agents. "
                         "Multi-flight is an optional extension.")
    if mode.startswith("Multi"):
        cl = find_clusters(df, min_flights=1)
        labels = [f"{r.DESTINATION_AIRPORT} {int(r.YEAR)}-{int(r.MONTH):02d}-{int(r.DAY):02d} "
                  f"{int(r.ETA_HOUR):02d}:00  ({r.n} flights)" for r in cl.itertuples()]
        idx = st.selectbox("Airport-hour cluster (by arrival time)", range(len(labels)), format_func=lambda i: labels[i])
        max_f = st.slider("Max flights", 1, 6, 4)
        rows = cluster_rows(df, cl.iloc[idx].to_dict(), max_f)
    else:
        labels = [f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)} {r.ORIGIN_AIRPORT}->{r.DESTINATION_AIRPORT} "
                  f"wx {r.WEATHER_DELAY:.0f} / late {r.LATE_AIRCRAFT_DELAY:.0f}" for r in dis.itertuples()]
        idx = st.selectbox("Disrupted row", range(len(labels)), format_func=lambda i: labels[i])
        rows = dis.iloc[[idx]]

    st.header("Agents")
    backend = st.radio("Backend", ["Offline policies", "Claude (LLM)"],
                       help="Offline runs rule-based agents with the same inputs/outputs; no API key needed.")
    model = pilot_model = None
    if backend.startswith("Claude"):
        model = st.text_input("ATC / Ground model", os.environ.get("ATM_MODEL", "claude-sonnet-5-5"))
        pilot_model = st.text_input("Pilot model", os.environ.get("ATM_PILOT_MODEL", "claude-haiku-5-5"))
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Set ANTHROPIC_API_KEY in the environment before starting Streamlit.")

    st.header("Chaos toggle")
    chaos = st.selectbox("Inject a fault", list(CHAOS_MODES), format_func=lambda k: f"{k}: {CHAOS_MODES[k]}")
    run = st.button("Run negotiation", type="primary", width="stretch")

world = scenario(df, rows)

if run:
    llm = None
    if backend.startswith("Claude"):
        from atm.llm import build_llms
        llm = build_llms(model, pilot_model)
    with st.spinner("Agents negotiating..."):
        st.session_state["res"] = negotiate(world, llm=llm, chaos=chaos)
        st.session_state["res_id"] = (world.scenario_id, chaos, backend)

res = st.session_state.get("res")
if res is not None and res.world.scenario_id != world.scenario_id:
    res = None

# ---------------------------------------------------------------- header
st.title("Multi-agent disruption negotiation")
st.caption(f"{world.airport.name} ({world.airport.code}) · {world.date} · {len(world.flights)} flight(s)")
c1, c2 = st.columns([3, 2])
c1.markdown(f"**Disruption:** {world.disruption}")
c2.markdown(f"**Weather:** {world.weather}")

if res is None:
    st.info("Pick a scenario in the sidebar and press **Run negotiation**.")

tabs = st.tabs(["Decision", "Negotiation", "Consensus log", "World state"])

# ---------------------------------------------------------------- decision
with tabs[0]:
    if res:
        verdict = ("Consensus" if res.decided_by == "consensus" else "Deterministic fallback")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Decided by", verdict)
        m2.metric("Turns", f"{res.rounds_used} / 3")
        m3.metric("Candidates checked", len(res.candidates))
        m4.metric("Hard-constraint rejections", sum(len(v) for v in res.checks.values()))
        if res.fallback_reason:
            st.warning(f"Fallback reason: {res.fallback_reason}")
        if res.usage:
            u = res.usage
            st.caption(f"LLM: {u['calls']} calls ({u['failed']} failed) · {u['input_tokens']:,} in / "
                       f"{u['output_tokens']:,} out tokens · slowest call {u['slowest_call_s']}s · "
                       + " · ".join(f"{r}: {d['model']}" for r, d in u["by_role"].items()))
        out = []
        for f in world.flights:
            a = res.decision.assignments[f.flight_id]
            m = metrics(world, f, a)
            u = res.urgency[f.flight_id]
            out.append({"Flight": f.flight_id, "Priority": u.level, "Type": f.aircraft_type,
                        "Landing": "DIVERT " + a.divert_to if a.divert_to else fmt(a.landing),
                        "Runway": a.runway, "Route": a.route, "Hold (min)": m.holding,
                        "Fuel at landing": m.fuel_at_landing, "Reserve": f.reserve,
                        "Gate": a.gate, "Crew": a.crew, "Gate in": fmt(m.gate_in),
                        "Conn. missed": f"{m.conn_missed}/{f.connecting_pax}",
                        "Crew OT (min)": m.overtime, "Actual 2015 arr. delay": f.actual_arrival_delay})
        st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
        st.subheader("Controller briefing")
        st.text(res.briefing)

# ---------------------------------------------------------------- negotiation
with tabs[1]:
    if res:
        st.subheader("Turn 1: positions")
        cols = st.columns(3)
        for i, (name, pos) in enumerate(res.positions.items()):
            with cols[i % 3].container(border=True):
                st.markdown(f"**{name}**")
                st.caption(pos.message)
                if hasattr(pos, "urgency") and pos.urgency:
                    st.markdown(f"Urgency claim: `{pos.urgency.level}`")
                with st.expander("message JSON"):
                    st.json(pos.model_dump())

        st.subheader("Arbiter priority (from world state)")
        st.dataframe(pd.DataFrame([{"Rank": i + 1, "Flight": fid, "Level": res.urgency[fid].level,
                                    "Score": round(res.urgency[fid].score, 1),
                                    "Why": "; ".join(res.urgency[fid].reasons)}
                                   for i, fid in enumerate(res.priority)]), hide_index=True, width="stretch")

        st.subheader("Candidates and constraint checks")
        cand_rows = []
        for cid, cand in res.candidates.items():
            v = res.checks[cid]
            cand_rows.append({"Id": cid, "Turn": cand.round, "Source": cand.source,
                              "Checker": "PASS" if not v else f"FAIL ({len(v)})",
                              "Cost": res.costs[cid], "Chosen": "★" if res.decision and cid == res.decision.id else "",
                              "Violations": "; ".join(x.message for x in v)})
        st.dataframe(pd.DataFrame(cand_rows), hide_index=True, width="stretch")

        for rnd in sorted(res.votes):
            st.subheader(f"Turn {rnd}: votes")
            feas = [cid for cid in res.candidates if not res.checks[cid] and res.candidates[cid].round < rnd]
            matrix, reasons = [], []
            for name, v in res.votes[rnd].items():
                row = {"Agent": name}
                for cid in feas:
                    row[cid] = "✅" if cid in v.accept else ("❌" if cid in v.reject_map() else "·")
                matrix.append(row)
                for cid, why in v.reject_map().items():
                    reasons.append(f"**{name}** rejects {cid}: {why}")
                if v.counter:
                    reasons.append(f"**{name}** counter on {v.counter.base_candidate}: "
                                   f"{[p.op + ':' + p.flight_id for p in v.counter.patches]}; "
                                   f"concession: {v.counter.concession}")
            st.dataframe(pd.DataFrame(matrix), hide_index=True, width="stretch")
            with st.expander("Reasons and counter-offers"):
                st.markdown("\n\n".join(reasons) or "none")

# ---------------------------------------------------------------- log
with tabs[2]:
    if res:
        ok, bad = res.log.verify()
        (st.success if ok else st.error)(
            f"Hash chain {'verified' if ok else f'broken at entry {bad}'} · {len(res.log.entries)} entries")
        ldf = pd.DataFrame([{"#": e["seq"], "Turn": e["round"], "Actor": e["actor"], "Event": e["event"],
                             "Flight": e["flight_id"], "Detail": e["reason"], "Hash": e["hash"][:12]}
                            for e in res.log.entries])
        st.dataframe(ldf.style.apply(lambda r: [EVENT_STYLE.get(r.Event, "")] * len(r), axis=1),
                     hide_index=True, width="stretch", height=600)
        st.download_button("Download log (JSONL)",
                           "\n".join(json.dumps(e, default=str) for e in res.log.entries),
                           file_name=f"{world.scenario_id}_{st.session_state['res_id'][1]}.jsonl")

# ---------------------------------------------------------------- world
with tabs[3]:
    from atm.limits import ASSUMPTIONS, load_limits, limits_table
    with st.expander("Operational limits used (organisers' workbook) and our stated assumptions", expanded=True):
        st.table(pd.DataFrame(limits_table(load_limits(), world.airport.code), columns=["Limit", "Value used"]))
        st.markdown("**Assumptions** (where the workbook does not say how a value applies):\n\n"
                    + "\n".join(f"- {x}" for x in ASSUMPTIONS))
    st.markdown("**Flights** (top rows: real DOT data; fuel, pax, aircraft type: simulated, seeded)")
    st.dataframe(pd.DataFrame([{
        "Flight": f.flight_id, "Tail": f.tail, "From": f.origin, "Sched arr": fmt(f.sched_arr),
        "WEATHER_DELAY": f.weather_delay, "LATE_AIRCRAFT_DELAY": f.late_aircraft_delay,
        "Type": f.aircraft_type, "Wake": f.wake, "ETA": fmt(f.eta), "Fuel at ETA": f.fuel_at_eta,
        "Pax": f.pax, "Connecting": f.connecting_pax, "Conn. gate-in by": fmt(f.connection_deadline),
        "Next leg": fmt(f.next_departure) if f.next_departure else "-",
        "Routes": ", ".join(f"{r.id}({r.eta_shift:+d}m, fuel {r.fuel_delta:+d}, turb {r.turbulence})"
                            for r in f.routes)} for f in world.flights]), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.markdown("**Runways**")
    a.dataframe(pd.DataFrame([r.model_dump() for r in world.airport.runways]), hide_index=True)
    a.markdown("**Taxiways**")
    a.dataframe(pd.DataFrame([t.model_dump() for t in world.airport.taxiways]), hide_index=True)
    b.markdown("**Gates**")
    b.dataframe(pd.DataFrame([{"Gate": g.id, "Size": g.size, "Free from": fmt(g.free_from),
                               "Taxi routes": g.taxi_route, "Note": g.note} for g in world.airport.gates]),
                hide_index=True)
    b.markdown("**Crews**")
    b.dataframe(pd.DataFrame([{"Crew": c.id, "Busy until": fmt(c.busy_until), "Shift ends": fmt(c.shift_end)}
                              for c in world.airport.crews]), hide_index=True)
