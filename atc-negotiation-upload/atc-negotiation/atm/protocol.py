"""Negotiation protocol: at most 3 turns, then a guaranteed answer.

One TURN = each of the three agent roles sends one structured message (organisers' rules).
Checker, arbiter and mediator entries are deterministic code, not agent messages.

Turn 1   DECLARE  every agent states its position in parallel (independent LLM calls).
                   Arbiter ranks flights from world state and checks urgency claims.
                   Candidates built: agents' combined plan, a mediator repair of it,
                   and the arbiter-priority plan. Each goes through the checker.
Turn 2   VOTE     every agent accepts / rejects each FEASIBLE candidate and may make
                   one counter-offer using only its own levers. Counters are checked.
Turn 3   FINAL    vote again over all feasible candidates; no new counters.
Decide            a feasible candidate accepted by EVERY agent -> lowest arbiter cost.
                   Otherwise (or on any agent error/timeout) -> deterministic fallback.
Every step is written to the hash-chained consensus log.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Optional

from . import arbiter, fallback
from .agents import ATCAgent, GroundAgent, PilotAgent
from .chaos import Chaos
from .constraints import Violation, check, metrics
from .log import ConsensusLog
from .schemas import ROLE_PERMISSIONS, Assignment, Candidate, Patch, Vote, WorldState
from .llm import usage_summary
from .summariser import llm_briefing, template_briefing
from .timeutil import fmt, parse_near

MAX_ROUNDS = 3


@dataclass
class NegotiationResult:
    world: WorldState
    urgency: dict
    priority: list[str]
    positions: dict = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    checks: dict[str, list[Violation]] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)
    votes: dict[int, dict[str, Vote]] = field(default_factory=dict)
    decision: Optional[Candidate] = None
    decided_by: str = ""
    rounds_used: int = 0
    fallback_reason: Optional[str] = None
    briefing: str = ""
    usage: Optional[dict] = None
    escalated: bool = False            # no plan satisfies every hard constraint: a controller must decide
    log: ConsensusLog = field(default_factory=ConsensusLog)


class AgentFailure(RuntimeError):
    pass


class Negotiator:
    def __init__(self, world: WorldState, llm=None, chaos: str = "none", timeout: float = 120.0):
        self.world = world
        self.llm = llm
        self.timeout = timeout
        self.chaos = Chaos(chaos, world)
        # llm: None (offline), one LLMClient for every role, or {"ATC"|"GROUND"|"PILOT"|"SUMMARISER": LLMClient}
        self.atc = ATCAgent(self._llm_for("ATC"))
        self.ground = GroundAgent(self._llm_for("GROUND"))
        self.pilots = {f.flight_id: PilotAgent(f.flight_id, self._llm_for("PILOT")) for f in world.flights}
        self.log = ConsensusLog()
        self._n = 0

    # ------------------------------------------------------------------ helpers
    def _llm_for(self, role: str):
        return self.llm.get(role) if isinstance(self.llm, dict) else self.llm

    @property
    def agents(self) -> dict:
        d = {"ATC": self.atc, "GROUND": self.ground}
        d.update({p.name: p for p in self.pilots.values()})
        return d

    def _parallel(self, jobs: dict) -> dict:
        """Run each agent's call concurrently - separate calls, separate prompts."""
        out = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {name: ex.submit(fn) for name, fn in jobs.items()}
            for name, fut in futs.items():
                try:
                    out[name] = fut.result(timeout=self.timeout)
                except FutTimeout:
                    out[name] = AgentFailure(f"timeout after {self.timeout}s")
                except Exception as e:  # noqa: BLE001 - any agent failure triggers fallback
                    out[name] = AgentFailure(f"{type(e).__name__}: {e}")
        return out

    def _next_id(self) -> str:
        self._n += 1
        return f"C{self._n}"

    def _evaluate(self, res: NegotiationResult, cand: Candidate, rnd: int) -> bool:
        viol = check(self.world, cand)
        res.candidates[cand.id] = cand
        res.checks[cand.id] = viol
        res.costs[cand.id] = arbiter.candidate_cost(self.world, cand, res.urgency)
        if viol:
            for v in viol:
                self.log.add("CHECKER", "REJECT", f"{cand.id}: {v.message}", rnd, v.flight_id,
                             {"candidate": cand.id, "code": v.code})
        else:
            self.log.add("CHECKER", "OK", f"{cand.id} ({cand.source}) passes all hard constraints; "
                                          f"cost {res.costs[cand.id]}", rnd, None,
                         {"candidate": cand.id, "cost": res.costs[cand.id]})
        return not viol

    def _evaluate_or_repair(self, res, cand: Candidate, protected: set[str], rnd: int, already_checked=False):
        """Check a candidate; if it fails, keep the proposed changes and re-place the flights they displaced."""
        if not already_checked and self._evaluate(res, cand, rnd):
            return
        violators = {x.flight_id for x in res.checks[cand.id] if x.flight_id}
        bad = violators - protected
        keep_routes = False
        if not bad:          # only the proposing flights clash: re-time them but keep their requested routes
            bad, keep_routes = violators, True
        if bad:
            fixed = fallback.repair(self.world, cand, bad, res.urgency, self._next_id(), rnd, keep_routes)
            self.log.add("MEDIATOR", "REPAIR", f"{fixed.id} = {cand.id} with {sorted(bad)} re-placed"
                         + (" (requested routes kept)" if keep_routes else ""), rnd)
            self._evaluate(res, fixed, rnd)

    def _feasible(self, res) -> list[str]:
        return [cid for cid, v in res.checks.items() if not v]

    def _from_positions(self, positions: dict) -> Candidate:
        asg = {f.flight_id: Assignment(flight_id=f.flight_id) for f in self.world.flights}
        atc = positions.get("ATC")
        for s in (atc.slots if atc else []):
            if s.flight_id in asg:
                a = asg[s.flight_id]
                a.runway, a.route = s.runway, s.route
                try:
                    a.landing = parse_near(s.landing, self.world.flight(s.flight_id).eta)
                except (ValueError, TypeError):
                    a.landing = None
        gnd = positions.get("GROUND")
        for g in (gnd.gates if gnd else []):
            if g.flight_id in asg:
                asg[g.flight_id].gate, asg[g.flight_id].crew = g.gate, g.crew
        return Candidate(id=self._next_id(), source="agents_combined", round=1, assignments=asg)

    def _apply(self, base: Candidate, patches: list[Patch], source: str, rnd: int) -> Candidate:
        asg = {k: v.model_copy() for k, v in base.assignments.items()}
        for p in patches:
            a = asg.get(p.flight_id)
            f = self.world.flight(p.flight_id)
            if a is None or f is None:
                raise ValueError(f"unknown flight {p.flight_id}")
            if p.op == "set_route":
                old, new = f.route(a.route), f.route(p.route or "")
                if new is None:
                    raise ValueError(f"unknown route {p.route}")
                if a.landing is not None and old is not None:
                    # same holding, new arrival time: the pilot "arrives earlier/later" with the route
                    a.landing = max(f.eta + new.eta_shift, a.landing + (new.eta_shift - old.eta_shift))
                a.route = new.id
            elif p.op == "set_slot":
                a.runway = p.runway or a.runway
                a.landing = parse_near(p.landing, f.eta)
            elif p.op == "swap_slots":
                b = asg.get(p.other_flight_id or "")
                if b is None:
                    raise ValueError(f"unknown flight {p.other_flight_id}")
                a.runway, b.runway = b.runway, a.runway
                a.landing, b.landing = b.landing, a.landing
            elif p.op == "set_gate":
                a.gate = p.gate
            elif p.op == "set_crew":
                a.crew = p.crew
        return Candidate(id=self._next_id(), source=source, round=rnd, assignments=asg)

    def _vote_ctx(self, res, agent, rnd: int, cand_ids: list[str], allow_counter: bool) -> dict:
        mets = {}
        views = []
        for cid in cand_ids:
            cand = res.candidates[cid]
            mets = {fid: metrics(self.world, self.world.flight(fid), a)
                    for fid, a in cand.assignments.items() if self.world.flight(fid)}
            views.append(agent.candidate_view(self.world, cand, mets, not res.checks[cid], res.checks[cid]))
        ctx = {"round": rnd, "allow_counter": allow_counter,
               "allowed_ops": ROLE_PERMISSIONS[agent.role],
               "priority_order": res.priority,
               "priority": [{"flight_id": fid, "level": res.urgency[fid].level,
                             "reasons": res.urgency[fid].reasons} for fid in res.priority],
               "candidates": views}
        if agent.role == "ATC":
            ctx["arrivals"] = agent.view(self.world)["arrivals"]      # earliest arrival per route, for swaps
        if agent.role == "PILOT":
            ctx["my_flight_id"] = agent.flight_id
            ctx["my_routes"] = [r.model_dump() for r in self.world.flight(agent.flight_id).routes]
        # agents hear each other: round-1 statements, and every vote from the previous round
        ctx["what_others_declared"] = {n: p.message for n, p in res.positions.items() if n != agent.name}
        prev = res.votes.get(rnd - 1)
        if prev:
            ctx["previous_round_votes"] = {
                n: {"accept": v.accept, "reject": v.reject_map(),
                    "counter": (f"{v.counter.base_candidate} + "
                                f"{[p.op + ':' + p.flight_id for p in v.counter.patches]}; "
                                f"concession: {v.counter.concession}") if v.counter else None}
                for n, v in prev.items()}
        return ctx

    def _log_position(self, name, pos):
        if name == "ATC":
            seq = ", ".join(f"{s.flight_id}@{s.landing}/{s.runway}" for s in pos.slots)
            self.log.add(name, "PROPOSE", f"{pos.message} | {seq}", 1, None, pos.model_dump())
        elif name == "GROUND":
            gates = ", ".join(f"{g.flight_id}->{g.gate}/{g.crew}" for g in pos.gates)
            self.log.add(name, "PROPOSE", f"{pos.message} | {gates}", 1, None, pos.model_dump())
        else:
            self.log.add(name, "PROPOSE", f"{pos.message} Urgency {pos.urgency.level}: {pos.urgency.reason}",
                         1, pos.flight_id, pos.model_dump())

    # ------------------------------------------------------------------ voting round
    def _vote_round(self, res, rnd: int, allow_counter: bool) -> Optional[list[str]]:
        cand_ids = list(res.candidates)
        feasible = set(self._feasible(res))
        jobs = {name: (lambda a=a: a.vote(self._vote_ctx(res, a, rnd, cand_ids, allow_counter)))
                for name, a in self.agents.items()}
        out = self._parallel(jobs)
        failed = {n: r for n, r in out.items() if isinstance(r, Exception)}
        for n, err in failed.items():
            self.log.add(n, "AGENT_ERROR", str(err), rnd)
        if failed:
            raise AgentFailure(f"{', '.join(failed)} failed in round {rnd}")
        votes: dict[str, Vote] = out
        for agent_name, note in self.chaos.on_votes(rnd, votes, sorted(feasible)):
            self.log.add("CHAOS", "INJECT", f"{agent_name}: {note}", rnd)
        res.votes[rnd] = votes

        for name, v in votes.items():
            fid = name[6:-1] if name.startswith("PILOT(") else None
            bad = [c for c in v.accept if c not in feasible]
            for c in bad:
                self.log.add("CHECKER", "BLOCKED", f"{name} accepted {c}, which fails hard constraints - ignored",
                             rnd, fid)
            v.accept = [c for c in v.accept if c in feasible]
            if v.accept:
                self.log.add(name, "ACCEPT", ", ".join(v.accept) + (f" - {v.message}" if v.message else ""),
                             rnd, fid, {"accept": v.accept})
            for c, why in v.reject_map().items():
                self.log.add(name, "REJECT", f"{c}: {why}", rnd, fid, {"candidate": c})

        # counter-offers -----------------------------------------------------
        valid_patches: list[Patch] = []
        for name, v in votes.items():
            if not v.counter:
                continue
            agent = self.agents[name]
            fid = getattr(agent, "flight_id", None)
            if not allow_counter:
                self.log.add(name, "IGNORED", "counter-offers are closed in the final round", rnd, fid)
                continue
            co = v.counter
            denied = [p for p in co.patches if p.op not in ROLE_PERMISSIONS[agent.role]
                      or (agent.role == "PILOT" and p.flight_id != fid)]
            if denied:
                self.log.add("PROTOCOL", "PERMISSION_DENIED",
                             f"{name} tried {[p.op for p in denied]} - not a {agent.role} lever; counter dropped",
                             rnd, fid, co.model_dump())
                continue
            if co.base_candidate not in res.candidates:
                self.log.add("PROTOCOL", "INVALID", f"{name} countered unknown candidate {co.base_candidate}", rnd, fid)
                continue
            try:
                new = self._apply(res.candidates[co.base_candidate], co.patches, f"counter:{name}", rnd)
            except ValueError as e:
                self.log.add("PROTOCOL", "INVALID", f"{name} counter rejected: {e}", rnd, fid)
                continue
            self.log.add(name, "COUNTER", f"{new.id} = {co.base_candidate} + "
                                          f"{', '.join(p.op + ':' + p.flight_id for p in co.patches)}. "
                                          f"Concession: {co.concession}. Why: {co.reason}",
                         rnd, fid, co.model_dump())
            if self._evaluate(res, new, rnd):
                valid_patches += co.patches      # only concessions that are safe on their own get merged
            else:
                self._evaluate_or_repair(res, new, {p.flight_id for p in co.patches}, rnd, already_checked=True)

        # mediator: combine every agent's safe concessions into one package deal
        if len(valid_patches) >= 2 and feasible:
            base_id = min(feasible, key=lambda c: res.costs[c])
            try:
                merged = self._apply(res.candidates[base_id], valid_patches, "mediator_merge", rnd)
            except ValueError as e:
                self.log.add("MEDIATOR", "INVALID", f"could not merge counters: {e}", rnd)
            else:
                self.log.add("MEDIATOR", "MERGE", f"{merged.id} = {base_id} + all {len(valid_patches)} counter "
                                                  f"patches combined", rnd)
                self._evaluate_or_repair(res, merged, {p.flight_id for p in valid_patches}, rnd)

        unanimous = [cid for cid in feasible if all(cid in v.accept for v in votes.values())]
        return unanimous

    # ------------------------------------------------------------------ main
    def run(self) -> NegotiationResult:
        w = self.world
        urg = arbiter.assess(w)
        prio = arbiter.priority_order(urg)
        res = NegotiationResult(world=w, urgency=urg, priority=prio, log=self.log)
        self.log.add("SYSTEM", "START", f"{w.scenario_id}: {len(w.flights)} flight(s) at {w.airport.code}; "
                                        f"{w.disruption}", None, None,
                     {"flights": [f.flight_id for f in w.flights], "chaos": self.chaos.mode,
                      "mode": "llm" if self.llm else "offline",
                      "models": {r: getattr(self._llm_for(r), "model", None)
                                 for r in ("ATC", "GROUND", "PILOT", "SUMMARISER")}})
        try:
            unanimous = self._round1_and_2(res)
            res.rounds_used = 2
            if unanimous:
                agreed = min(res.costs[c] for c in unanimous)
                pending = [c for c in self._feasible(res)
                           if res.candidates[c].round == 2 and res.costs[c] < agreed]
                if pending:   # cheaper counter-offers on the table: one final vote including them
                    res.rounds_used = 3
                    final = self._vote_round(res, 3, allow_counter=False)
                    unanimous = final or unanimous
            else:
                res.rounds_used = 3
                unanimous = self._vote_round(res, 3, allow_counter=False)
            if unanimous:
                best = min(unanimous, key=lambda c: res.costs[c])
                self._commit(res, res.candidates[best], "consensus")
            else:
                self._fallback(res, "no candidate accepted by every agent within 3 turns")
        except AgentFailure as e:
            self._fallback(res, str(e))

        res.briefing = template_briefing(res)
        if self._llm_for("SUMMARISER"):
            try:
                res.briefing = llm_briefing(res, self._llm_for("SUMMARISER"))
            except Exception as e:  # noqa: BLE001 - briefing is optional, template stays
                self.log.add("SUMMARISER", "ERROR", f"LLM briefing failed, template used: {e}")
        self.log.add("SUMMARISER", "BRIEFING", res.briefing.splitlines()[0], None, None, {"text": res.briefing})
        if self.llm:
            res.usage = usage_summary(self.llm)
            u = res.usage
            self.log.add("SYSTEM", "USAGE", f"{u['calls']} LLM calls ({u['failed']} failed), "
                                            f"{u['input_tokens']} in / {u['output_tokens']} out tokens, "
                                            f"slowest call {u['slowest_call_s']}s", None, None, u)
        ok, _ = self.log.verify()
        self.log.add("SYSTEM", "END", f"decided by {res.decided_by} after {res.rounds_used} turn(s); "
                                      f"log chain {'intact' if ok else 'BROKEN'}")
        return res

    def _round1_and_2(self, res) -> list[str]:
        w = self.world
        # ---- Turn 1: declare (all agents in parallel)
        out = self._parallel({name: (lambda a=a: a.declare(w)) for name, a in self.agents.items()})
        failed = {n: r for n, r in out.items() if isinstance(r, Exception)}
        for n, err in failed.items():
            self.log.add(n, "AGENT_ERROR", str(err), 1)
        if failed:
            raise AgentFailure(f"{', '.join(failed)} failed in round 1")
        positions = out
        for agent_name, note in self.chaos.on_positions(positions):
            self.log.add("CHAOS", "INJECT", f"{agent_name}: {note}", 1)
        res.positions = positions
        for name, pos in positions.items():
            self._log_position(name, pos)

        # arbiter: verify urgency claims against world state, publish priority
        for fid, p in self.pilots.items():
            pos = positions[p.name]
            ok, note = arbiter.verify_claim(res.urgency[fid].level, pos.urgency)
            self.log.add("ARBITER", "VERIFY" if ok else "DOWNWEIGHT", f"{p.name}: {note}", 1, fid)
        if positions["GROUND"].urgency:
            self.log.add("ARBITER", "NOTED", f"GROUND urgency claim '{positions['GROUND'].urgency.level}' "
                                            "recorded; flight priority is set from world state only", 1)
        self.log.add("ARBITER", "RANK", " > ".join(f"{fid}[{res.urgency[fid].level}]" for fid in res.priority),
                     1, None, {fid: {"level": res.urgency[fid].level, "score": round(res.urgency[fid].score, 1),
                                     "reasons": res.urgency[fid].reasons} for fid in res.priority})

        # candidates
        c1 = self._from_positions(positions)
        if not self._evaluate(res, c1, 1):
            bad = {v.flight_id for v in res.checks[c1.id] if v.flight_id}
            c2 = fallback.repair(w, c1, bad, res.urgency, self._next_id(), 1)
            self.log.add("MEDIATOR", "REPAIR", f"{c2.id} = {c1.id} with {sorted(bad)} re-placed", 1)
            self._evaluate(res, c2, 1)
        c3 = fallback.solve(w, res.urgency, cid=self._next_id(), source="arbiter_priority", rnd=1)
        self.log.add("MEDIATOR", "PROPOSE", f"{c3.id}: allocation in arbiter priority order", 1)
        self._evaluate(res, c3, 1)

        # ---- Round 2: vote + counter-offers
        return self._vote_round(res, 2, allow_counter=True)

    def _commit(self, res, cand: Candidate, how: str):
        final = check(self.world, cand)    # final safety gate, always re-run
        if final:
            self._fallback(res, f"chosen candidate {cand.id} failed final check")
            return
        res.decision, res.decided_by = cand, how
        self.log.add("SYSTEM", "DECIDE", f"{cand.id} ({cand.source}) by {how}; cost {res.costs.get(cand.id)}",
                     res.rounds_used, None, {"candidate": cand.id})
        self._log_outcomes(res)

    def _fallback(self, res, reason: str):
        res.fallback_reason = reason
        self.log.add("SYSTEM", "FALLBACK", reason, res.rounds_used or None)
        cand = fallback.solve(self.world, res.urgency, cid="FALLBACK", source="fallback", rnd=res.rounds_used)
        if not self._evaluate(res, cand, res.rounds_used):
            res.escalated = True
            codes = sorted({v.code for v in res.checks[cand.id]})
            self.log.add("SYSTEM", "ESCALATE", f"no plan satisfies every hard constraint ({', '.join(codes)}); "
                                               "fallback plan shown, controller decision required",
                         res.rounds_used or None, None, {"codes": codes})
        res.decision, res.decided_by = cand, "fallback"
        self.log.add("FALLBACK", "DECIDE", f"deterministic allocation in priority order; cost {res.costs['FALLBACK']}",
                     res.rounds_used or None, None, {"candidate": "FALLBACK"})
        self._log_outcomes(res)

    def _log_outcomes(self, res):
        for f in self.world.flights:
            a = res.decision.assignments[f.flight_id]
            m = metrics(self.world, f, a)
            if a.divert_to:
                txt = f"DIVERT to {a.divert_to}, fuel on arrival {m.fuel_at_landing} min"
            else:
                txt = (f"land {fmt(a.landing)} {a.runway} via {a.route}, hold {m.holding} min, fuel {m.fuel_at_landing} "
                       f"min, gate {a.gate}/{a.crew} at {fmt(m.gate_in)}, connections missed {m.conn_missed}")
            self.log.add("SYSTEM", "OUTCOME", f"{f.flight_id}: {txt}", None, f.flight_id, a.model_dump())


def negotiate(world: WorldState, llm=None, chaos: str = "none", timeout: float = 120.0) -> NegotiationResult:
    return Negotiator(world, llm=llm, chaos=chaos, timeout=timeout).run()