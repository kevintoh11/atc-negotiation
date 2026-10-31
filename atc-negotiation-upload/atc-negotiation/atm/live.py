"""Live streaming of a negotiation, for the dashboard.

The engine is unchanged: this module only *observes* atm.protocol.Negotiator.
  - every consensus-log entry is emitted the moment it is written (StreamingLog)
  - every agent call is emitted when it starts ("typing") and when it answers
so a UI can show each message as it is sent instead of a spinner.

It also applies optional what-if constraints (close a runway, block a taxiway,
cut fuel, add spacing) to a world before the agents see it. The constraint
checker then enforces them like any other rule.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .log import ConsensusLog
from .protocol import NegotiationResult, Negotiator
from .schemas import WorldState


# --------------------------------------------------------------------------
# What-if constraints
# --------------------------------------------------------------------------
@dataclass
class WhatIf:
    close_runways: list[str] = field(default_factory=list)
    block_taxiways: list[str] = field(default_factory=list)
    fuel_cut_min: int = 0          # minutes of fuel removed from every negotiating flight
    extra_spacing_min: int = 0     # minutes added to every wake minimum (low visibility)

    def any(self) -> bool:
        return bool(self.close_runways or self.block_taxiways or self.fuel_cut_min or self.extra_spacing_min)

    def describe(self) -> list[str]:
        out = []
        if self.close_runways:
            out.append("runway " + ", ".join(self.close_runways) + " closed")
        if self.block_taxiways:
            out.append("taxiway " + ", ".join(self.block_taxiways) + " blocked")
        if self.fuel_cut_min:
            out.append(f"{self.fuel_cut_min} min less fuel on board")
        if self.extra_spacing_min:
            out.append(f"+{self.extra_spacing_min} min spacing between landings (low visibility)")
        return out


def apply_whatif(world: WorldState, w: Optional[WhatIf]) -> WorldState:
    """Return a copy of the world with the user's extra constraints applied."""
    if not w or not w.any():
        return world
    new = world.model_copy(deep=True)
    ap = new.airport
    open_before = sum(r.open for r in ap.runways) or 1
    for r in ap.runways:
        if r.id in w.close_runways and r.open:
            r.open, r.note = False, "closed by what-if"
    open_after = sum(r.open for r in ap.runways)
    if open_after < open_before:      # the hourly cap scales with the runways open (workbook assumption)
        ap.max_arrivals_per_hour = max(1, round(ap.max_arrivals_per_hour * open_after / open_before))
    for t in ap.taxiways:
        if t.id in w.block_taxiways:
            t.blocked, t.note = True, "blocked by what-if"
    for f in new.flights:
        if w.fuel_cut_min:
            f.fuel_at_eta = max(1, f.fuel_at_eta - w.fuel_cut_min)
    ap.extra_spacing += w.extra_spacing_min
    new.disruption = new.disruption + "; what-if: " + "; ".join(w.describe())
    return new


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------
class StreamingLog(ConsensusLog):
    """A ConsensusLog that also emits each entry as it is written (hashing is unchanged)."""

    def __init__(self, emit):
        super().__init__()
        self._emit = emit

    def add(self, *args, **kwargs) -> dict:
        entry = super().add(*args, **kwargs)
        self._emit(("log", dict(entry)))
        return entry


def _message_of(obj) -> str:
    return (getattr(obj, "message", None) or "").strip()


def _instrument(agent, emit, t0: float):
    """Wrap one agent's declare/vote so the UI sees when it starts thinking and when it answers."""
    for meth in ("declare", "vote"):
        orig = getattr(agent, meth)

        def wrapped(*args, _orig=orig, _meth=meth, **kwargs):
            rnd = 1 if _meth == "declare" else (args[0].get("round") if args and isinstance(args[0], dict) else None)
            start = time.monotonic()
            emit(("start", agent.name, rnd, start - t0))
            try:
                out = _orig(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - re-raised; the protocol turns it into a fallback
                emit(("fail", agent.name, rnd, f"{type(e).__name__}: {e}", time.monotonic() - start))
                raise
            emit(("said", agent.name, rnd, _message_of(out), time.monotonic() - start))
            return out

        setattr(agent, meth, wrapped)


class LiveRun:
    """Run one negotiation in a background thread and expose its events as a queue.

    Events (tuples):
      ("start", agent, turn, t)            the agent's call began (t = seconds since run start)
      ("said",  agent, turn, text, secs)   the agent answered; text is its own message
      ("fail",  agent, turn, error, secs)  the agent's call failed (the protocol falls back)
      ("log",   entry)                     a consensus-log entry was written (hash-chained)
      ("done",  result) / ("crash", error) the run finished
    """

    def __init__(self, world: WorldState, llm=None, chaos: str = "none", timeout: float = 120.0):
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.result: Optional[NegotiationResult] = None
        self.error: Optional[BaseException] = None
        self.t0 = time.monotonic()
        self.neg = Negotiator(world, llm=llm, chaos=chaos, timeout=timeout)
        self.neg.log = StreamingLog(self.events.put)
        for agent in self.neg.agents.values():
            _instrument(agent, self.events.put, self.t0)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self.result = self.neg.run()
            self.events.put(("done", self.result))
        except BaseException as e:  # noqa: BLE001 - surfaced to the UI
            self.error = e
            self.events.put(("crash", f"{type(e).__name__}: {e}"))

    def start(self) -> "LiveRun":
        self.t0 = time.monotonic()
        self.thread.start()
        return self

    def iter_events(self, poll: float = 0.1):
        """Yield events until the run is over; yields None every `poll` s while waiting (for timers)."""
        while True:
            try:
                ev = self.events.get(timeout=poll)
            except queue.Empty:
                if not self.thread.is_alive() and self.events.empty():
                    return
                yield None
                continue
            yield ev
            if ev[0] in ("done", "crash"):
                return
