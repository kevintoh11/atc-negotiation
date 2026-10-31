"""Core guarantees. Run: python -m pytest -q"""
import json
from types import SimpleNamespace

import pytest

from atm.chaos import CHAOS_MODES
from atm.constraints import check
from atm.llm import AgentOutputError, LLMClient
from atm.log import ConsensusLog
from atm.protocol import negotiate
from atm.scenario import cluster_rows, disrupted, find_clusters, load_flights, scenario

DATA = "data/sample_flights.csv"


def worlds():
    df = load_flights(DATA)
    cl = find_clusters(df, min_flights=1)
    out = [scenario(df, cluster_rows(df, cl.iloc[i].to_dict())) for i in range(len(cl))]
    out.append(scenario(df, disrupted(df).iloc[[0]]))          # single flight, exactly 3 agents
    return out


def test_only_disrupted_rows_selected():
    d = disrupted(load_flights(DATA))
    assert ((d.WEATHER_DELAY > 15) | (d.LATE_AIRCRAFT_DELAY > 20)).all()
    assert len(d) == 10


@pytest.mark.parametrize("world", worlds(), ids=lambda w: w.scenario_id)
def test_final_decision_always_safe_and_bounded(world):
    res = negotiate(world)
    assert check(world, res.decision) == []                   # zero hard-constraint violations
    assert res.rounds_used <= 3                               # turn limit
    assert set(res.decision.assignments) == {f.flight_id for f in world.flights}
    assert res.log.verify() == (True, None)


def test_scenario_is_deterministic():
    w1, w2 = worlds()[0], worlds()[0]
    assert w1.model_dump() == w2.model_dump()


@pytest.mark.parametrize("mode", [m for m in CHAOS_MODES if m != "none"])
def test_chaos_is_caught(mode):
    world = worlds()[0]
    res = negotiate(world, chaos=mode)
    events = {(e["actor"], e["event"]) for e in res.log.entries}
    assert ("CHAOS", "INJECT") in events
    caught = {"fuel": ("CHECKER", "REJECT"), "ghost_gate": ("CHECKER", "REJECT"), "wake": ("CHECKER", "REJECT"),
              "urgency": ("ARBITER", "DOWNWEIGHT"), "role": ("PROTOCOL", "PERMISSION_DENIED")}[mode]
    assert caught in events
    assert check(world, res.decision) == []


# ---- fake Anthropic SDK client: same shape as client.messages.parse / .create -------------
class _Resp(SimpleNamespace):
    pass


class FakeMessages:
    """Answers like the offline policies would, through the REAL LLMClient code path."""
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.calls = []

    def parse(self, model, max_tokens, system, messages, output_format):
        from atm import agents
        self.calls.append(model)
        user = messages[-1]["content"]
        role = system.split("You are the ")[1].split(" agent")[0]
        if self.stop_reason != "end_turn":
            return _Resp(parsed_output=None, stop_reason=self.stop_reason, usage=None)
        if user.startswith("Turn 1"):
            view = json.loads(user.split("situation:\n", 1)[1])
            agent = {"PILOT": lambda: agents.PilotAgent(view["flight"]["flight_id"]),
                     "ATC": agents.ATCAgent, "GROUND": agents.GroundAgent}[role]()
            out = agent.offline_declare(view)
        else:
            ctx = json.loads(user.split("\n\n", 1)[1])
            agent = {"PILOT": lambda: agents.PilotAgent(ctx["my_flight_id"]),
                     "ATC": agents.ATCAgent, "GROUND": agents.GroundAgent}[role]()
            out = agent.offline_vote(ctx)
        out = output_format.model_validate_json(out.model_dump_json())   # what the SDK would hand back
        return _Resp(parsed_output=out, stop_reason="end_turn",
                     usage=SimpleNamespace(input_tokens=len(user) // 4, output_tokens=50))

    briefing = "Situation line.\nFlight line.\nBinding constraint line.\nTrade-off line.\nDecided by consensus."

    def create(self, model, max_tokens, system, messages):
        return _Resp(content=[SimpleNamespace(type="text", text=self.briefing)],
                     stop_reason="end_turn", usage=SimpleNamespace(input_tokens=100, output_tokens=40))


def fake_llms(stop_reason="end_turn"):
    client = SimpleNamespace(messages=FakeMessages(stop_reason))
    main = LLMClient(model="main-model", client=client)
    pilot = LLMClient(model="pilot-model", client=client)
    return {"ATC": main, "GROUND": main, "SUMMARISER": main, "PILOT": pilot}, client


def test_llm_code_path_end_to_end():
    world = worlds()[0]
    llms, client = fake_llms()
    res = negotiate(world, llm=llms)
    assert check(world, res.decision) == []
    assert not any(e["event"] == "AGENT_ERROR" for e in res.log.entries)
    assert res.briefing == FakeMessages.briefing
    # agents hear each other: round-3 context carries round-2 votes
    assert 3 not in res.votes or res.votes[2]
    assert res.usage["calls"] > 0 and res.usage["failed"] == 0
    # per-role models: pilots used the pilot model, ATC/Ground the main model
    assert "pilot-model" in client.messages.calls and "main-model" in client.messages.calls


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_or_truncation_triggers_fallback(stop):
    world = worlds()[0]
    llms, _ = fake_llms(stop_reason=stop)
    res = negotiate(world, llm=llms)
    assert res.decided_by == "fallback"
    assert "failed in round 1" in res.fallback_reason
    assert check(world, res.decision) == []
    assert any(e["event"] == "AGENT_ERROR" and stop in e["reason"] for e in res.log.entries)


def test_api_exception_triggers_fallback():
    class Boom:
        def parse(self, **kw):
            raise ConnectionError("network down")
    llm = LLMClient(model="m", client=SimpleNamespace(messages=Boom()))
    res = negotiate(worlds()[0], llm=llm)
    assert res.decided_by == "fallback"
    assert check(res.world, res.decision) == []


def test_agent_output_error_raised_on_refusal():
    llms, _ = fake_llms("refusal")
    from atm.schemas import Vote
    with pytest.raises(AgentOutputError):
        llms["ATC"].complete_json("You are the ATC agent.", "Turn 2. x\n\n{}", Vote)


def test_log_tamper_detected(tmp_path):
    res = negotiate(worlds()[0])
    p = tmp_path / "log.jsonl"
    res.log.write_jsonl(str(p))
    log = ConsensusLog.read_jsonl(str(p))
    assert log.verify() == (True, None)
    log.entries[5]["reason"] = "edited after the fact"
    ok, where = log.verify()
    assert not ok and where == 5


def test_short_llm_briefing_falls_back_to_template():
    llms, client = fake_llms()
    client.messages.briefing = "Runway 28L is closed"
    res = negotiate(worlds()[0], llm=llms)
    assert res.briefing.count("\n") >= 2            # template, not the one-liner
    assert any(e["actor"] == "SUMMARISER" and e["event"] == "ERROR" for e in res.log.entries)


def test_vote_context_shares_other_agents_votes():
    from atm.protocol import Negotiator
    n = Negotiator(worlds()[0])
    res = n.run()
    if 2 in res.votes:
        ctx = n._vote_ctx(res, n.atc, 3, list(res.candidates), False)
        assert "previous_round_votes" in ctx and "GROUND" in ctx["previous_round_votes"]
        assert "ATC" not in ctx["what_others_declared"]


def test_cut_off_briefing_falls_back_to_template():
    llms, client = fake_llms()
    orig = client.messages.create
    def truncated(**kw):
        r = orig(**kw)
        r.stop_reason = "max_tokens"
        return r
    client.messages.create = truncated
    res = negotiate(worlds()[0], llm=llms)
    assert "Binding priority" in res.briefing          # template briefing


def test_replay_stages_from_saved_log(tmp_path):
    from atm.replay import load
    res = negotiate(worlds()[0], chaos="fuel")
    p = tmp_path / "run.jsonl"
    res.log.write_jsonl(str(p))
    rep = load(str(p))
    names = [s.name for s in rep.stages]
    assert names[0] == "Setup" and names[-1] == "Decision" and "Turn 1" in names
    # stages tile the whole log with no gaps or overlaps
    assert rep.stages[0].start == 0 and rep.stages[-1].end == len(rep.log.entries)
    assert all(a.end == b.start for a, b in zip(rep.stages, rep.stages[1:]))
    assert len(rep.outcomes) == len(res.world.flights)
    assert rep.decided_by == res.decided_by and rep.briefing



# ---- clock, traffic, midnight -------------------------------------------------------
def test_world_is_built_around_delayed_arrival():
    for w in worlds():
        a0 = min(f.eta for f in w.flights)
        for f in w.flights:
            # connections and next legs are placed around the ETA, not the (earlier) schedule
            assert f.eta + 10 - 10 <= f.connection_deadline <= f.eta + 10 + 40
            if f.next_departure:
                assert f.next_departure >= f.eta + 10 + f.min_turnaround - 10
        for c in w.airport.crews:
            assert c.shift_end >= a0 + 20


def test_background_traffic_is_real_and_respected():
    df = load_flights(DATA)
    w = scenario(df, cluster_rows(df, find_clusters(df, 1).iloc[0].to_dict()))   # ORD
    assert len(w.airport.occupied_slots) >= 6
    neg = {f.flight_id for f in w.flights}
    assert not neg & {s.flight_id for s in w.airport.occupied_slots}            # nobody counted twice
    res = negotiate(w)
    assert check(w, res.decision) == []
    # put a negotiated flight on top of a background slot: the checker must object
    s0 = w.airport.occupied_slots[0]
    bad = res.decision.model_copy(deep=True)
    fid = next(iter(neg))
    bad.assignments[fid].runway, bad.assignments[fid].landing = s0.runway, s0.time
    assert any(v.code == "WAKE_SEPARATION" for v in check(w, bad))


def test_weather_cuts_runway_capacity():
    df = load_flights(DATA)
    w = scenario(df, cluster_rows(df, find_clusters(df, 1).iloc[0].to_dict()))   # ORD, WEATHER_DELAY 52
    # workbook: ORD is a Large hub, 4 runways, 90 ops/h. Severe weather closes half.
    assert len(w.airport.runways) == 4 and sum(r.open for r in w.airport.runways) == 2
    assert w.airport.max_arrivals_per_hour == 22 and w.airport.extra_spacing == 1


def test_workbook_limits_are_used():
    from atm.constraints import WAKE_SEP_MIN
    from atm.limits import load_limits
    lim = load_limits()
    assert lim.airport("ORD").active_runways == 4 and lim.airport("JFK").size_class == "Medium hub"
    assert WAKE_SEP_MIN == lim.wake_min and WAKE_SEP_MIN[("H", "S")] == 3     # 6 nm at 140 kt
    df = load_flights(DATA)
    w = scenario(df, disrupted(df).iloc[[0]])
    f = w.flights[0]
    assert f.reserve == 30 and f.fuel_at_eta == 30 + max(5, int(45 - 0.5 * f.weather_delay))
    assert f.min_turnaround == lim.airport(w.airport.code).min_turnaround_min
    assert len(w.airport.gates) == len(w.flights) + lim.airport(w.airport.code).spare_gates
    assert f.planned_gate and f.crew_duty_end


def test_gate_change_notice_and_apron_wait():
    df = load_flights(DATA)
    w = scenario(df, disrupted(df).iloc[[0]])
    res = negotiate(w)
    f = w.flights[0]
    bad = res.decision.model_copy(deep=True)
    a = bad.assignments[f.flight_id]
    other = next(g for g in w.airport.gates if g.id != f.planned_gate and g.size == "L") \
        if any(g.size == "L" and g.id != f.planned_gate for g in w.airport.gates) else None
    if other:
        a.gate = other.id
        w2 = w.model_copy(deep=True)
        w2.airport.decision_time = a.landing - 5               # only 5 min notice
        assert any(v.code == "GATE_CHANGE_NOTICE" for v in check(w2, bad))
    # an occupied gate means waiting on the apron, never a hard violation
    w3 = w.model_copy(deep=True)
    w3.airport.gate(res.decision.assignments[f.flight_id].gate).free_from += 120
    from atm.constraints import metrics as M
    assert M(w3, f, res.decision.assignments[f.flight_id]).gate_wait > 0
    assert not any(v.code.startswith("GATE_BUSY") for v in check(w3, res.decision))


def test_midnight_clock():
    from atm.timeutil import fmt, parse, parse_near
    assert fmt(1470) == "24:30" and parse("24:30") == 1470
    assert parse_near("00:30", 1460) == 1470 and parse_near("23:50", 1460) == 1430
    df = load_flights(DATA)
    cl = find_clusters(df, 1)
    jfk = cl[cl.DESTINATION_AIRPORT == "JFK"].iloc[0].to_dict()
    w = scenario(df, cluster_rows(df, jfk))
    assert all(f.eta >= 1440 for f in w.flights)          # arrivals after midnight stay on the same clock
    res = negotiate(w)
    assert check(w, res.decision) == [] and res.decided_by == "consensus"


def test_cluster_overflow_becomes_traffic():
    df = load_flights(DATA)
    key = find_clusters(df, 1).iloc[0].to_dict()               # ORD has 4 disrupted flights
    rows = cluster_rows(df, key, max_flights=2)
    assert rows.attrs["overflow"] == 2
    w = scenario(df, rows)
    assert len(w.flights) == 2 and "handled as other traffic" in w.disruption



def test_evaluation_runs_and_reports(tmp_path):
    import subprocess
    import sys
    out = tmp_path / "eval"
    r = subprocess.run([sys.executable, "evaluate.py", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summary = (out / "summary.md").read_text()
    for needle in ("Agreement (consensus) rate", "Fallback rate", "holding min", "connections missed",
                   "crew overtime min", "Who paid for it"):
        assert needle in summary
    assert (out / "results.csv").exists() and (out / "flights.csv").exists()
    import csv
    rows = list(csv.DictReader(open(out / "results.csv")))
    assert all(int(r["neg_hard_conflicts"]) == 0 for r in rows)       # negotiated plans are always safe



def test_fuel_outranks_passengers():
    from atm import arbiter
    w = worlds()[0]
    f_fuel, f_pax = w.flights[0].model_copy(), w.flights[1].model_copy()
    f_fuel.fuel_at_eta, f_fuel.connecting_pax, f_fuel.next_departure = f_fuel.reserve + 10, 0, None
    f_pax.fuel_at_eta, f_pax.connecting_pax = f_pax.reserve + 24, 200
    f_pax.connection_deadline, f_pax.next_departure = f_pax.eta + 10, f_pax.eta + 10 + f_pax.min_turnaround
    w2 = w.model_copy(update={"flights": [f_pax, f_fuel]})
    urg = arbiter.assess(w2)
    assert arbiter.priority_order(urg)[0] == f_fuel.flight_id      # 10 min fuel beats 200 passengers
    assert urg[f_pax.flight_id].tier == 2 and urg[f_pax.flight_id].reasons[0].startswith("200 connecting")


# ---- evaluate.py: chaos sweep and Claude-vs-offline comparison ---------------------------
def _eval_items():
    df = load_flights(DATA)
    d = disrupted(df)
    return df, [d.loc[[i]] for i in d.index[:4]]


def test_chaos_sweep_catches_every_applicable_fault(tmp_path):
    import csv
    import evaluate
    df, items = _eval_items()
    evaluate.chaos_sweep(df, items, SimpleNamespace(data=DATA, out=str(tmp_path), chaos="all"))
    rows = list(csv.DictReader(open(tmp_path / "chaos.csv")))
    assert {r["mode"] for r in rows} == set(evaluate.EXPECTED_CATCH)
    applicable = [r for r in rows if r["applicable"] == "True"]
    assert applicable and all(r["caught"] == "True" for r in applicable)
    assert all(r["final_hard_conflicts"] == "0" for r in rows)


def test_llm_comparison_uses_fresh_clients_per_scenario(tmp_path, monkeypatch):
    import evaluate
    monkeypatch.setattr(evaluate, "build_clients", lambda a: fake_llms()[0])
    df, items = _eval_items()
    llm_res, _ = evaluate.run_all(df, items, True, "Claude")
    off_res, _ = evaluate.run_all(df, items, None, "offline")
    # the fake client answers like the offline policies, so plans must match
    assert [r["decision"] for r in llm_res] == [r["decision"] for r in off_res]
    assert all(r["llm_calls"] > 0 for r in llm_res) and all(r["llm_calls"] == 0 for r in off_res)
    # usage is per scenario, not cumulative
    assert len({r["llm_calls"] for r in llm_res}) <= 2
    text = "\n".join(evaluate.llm_section(llm_res, off_res))
    assert "Same final plan as the offline agents: 4/4" in text