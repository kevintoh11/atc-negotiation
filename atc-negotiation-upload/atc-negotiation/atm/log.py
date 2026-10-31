"""Traceable consensus log: append-only and hash-chained.

Each entry stores the SHA-256 hash of the previous entry. Editing any old
entry changes its hash, which no longer matches what the next entry
recorded, so `verify()` detects tampering.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class ConsensusLog:
    def __init__(self):
        self.entries: list[dict] = []

    def add(self, actor: str, event: str, reason: str = "", round: Optional[int] = None,
            flight_id: Optional[str] = None, data: Any = None) -> dict:
        entry = {
            "seq": len(self.entries),
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "round": round,
            "actor": actor,
            "event": event,
            "flight_id": flight_id,
            "reason": reason,
            "data": data,
            "prev_hash": self.entries[-1]["hash"] if self.entries else GENESIS,
        }
        entry["hash"] = _digest(entry)
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, Optional[int]]:
        prev = GENESIS
        for i, e in enumerate(self.entries):
            if e["prev_hash"] != prev or _digest(e) != e["hash"]:
                return False, i
            prev = e["hash"]
        return True, None

    def write_jsonl(self, path: str) -> None:
        with open(path, "w") as fh:
            for e in self.entries:
                fh.write(json.dumps(e, default=str) + "\n")

    @staticmethod
    def read_jsonl(path: str) -> "ConsensusLog":
        log = ConsensusLog()
        with open(path) as fh:
            log.entries = [json.loads(line) for line in fh if line.strip()]
        return log

    def lines(self) -> list[str]:
        out = []
        for e in self.entries:
            rnd = f"T{e['round']}" if e["round"] is not None else "--"     # T = turn
            out.append(f"{e['seq']:>3}  {rnd:<3} {e['actor']:<16} {e['event']:<15} {e['reason']}")
        return out
