"""Replay a saved consensus log stage by stage, with no API calls.

A log is split into stages the audience can step through:
  Setup      START, chaos injections        (before turn 1)
  Turn 1     positions, arbiter, checker
  Turn 2     votes, counter-offers, checks
  Turn 3     final votes (if reached)
  Decision   DECIDE / FALLBACK onwards: outcomes, briefing, usage, END
One turn = each of the three agent roles sends one structured message.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .log import ConsensusLog


@dataclass
class Stage:
    name: str
    start: int          # index of first entry in this stage
    end: int            # index after the last entry in this stage


@dataclass
class Replay:
    log: ConsensusLog
    stages: list[Stage]
    scenario: str
    meta: dict = field(default_factory=dict)
    outcomes: list[dict] = field(default_factory=list)
    briefing: Optional[str] = None
    usage: Optional[dict] = None
    decided_by: Optional[str] = None


def _decision_index(entries: list[dict]) -> int:
    for i, e in enumerate(entries):
        if (e["actor"] == "SYSTEM" and e["event"] in ("DECIDE", "FALLBACK")) or e["actor"] == "FALLBACK":
            return i
    return len(entries)


def build(log: ConsensusLog) -> Replay:
    entries = log.entries
    if not entries:
        raise ValueError("empty log")
    d = _decision_index(entries)
    stages: list[Stage] = []
    first_r1 = next((i for i, e in enumerate(entries[:d]) if e["round"] == 1), d)
    stages.append(Stage("Setup", 0, first_r1))
    cursor = first_r1
    for rnd in (1, 2, 3):
        idx = [i for i, e in enumerate(entries[:d]) if e["round"] == rnd]
        if not idx:
            continue
        end = max(idx) + 1
        stages.append(Stage(f"Turn {rnd}", cursor, end))
        cursor = end
    stages.append(Stage("Decision", cursor, len(entries)))

    start = entries[0]
    rep = Replay(log=log, stages=stages, scenario=start.get("reason", ""), meta=start.get("data") or {})
    for e in entries:
        if e["event"] == "OUTCOME":
            fid, _, text = e["reason"].partition(": ")
            rep.outcomes.append({"Flight": fid, "Outcome": text})
        elif e["event"] == "BRIEFING":
            rep.briefing = (e.get("data") or {}).get("text") or e["reason"]
        elif e["event"] == "USAGE":
            rep.usage = e.get("data")
        elif e["event"] == "END":
            rep.decided_by = "consensus" if "by consensus" in e["reason"] else "fallback"
    return rep


def load(path_or_lines) -> Replay:
    if isinstance(path_or_lines, str):
        return build(ConsensusLog.read_jsonl(path_or_lines))
    log = ConsensusLog()
    log.entries = [json.loads(line) for line in path_or_lines if line.strip()]
    return build(log)
