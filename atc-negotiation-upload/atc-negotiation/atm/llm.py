"""Anthropic SDK wrapper: one call in, one validated Pydantic object out.

Uses structured outputs (`client.messages.parse(output_format=Model)`): the API
constrains Claude's reply to the message schema, and the SDK validates it and
returns the object. No JSON is ever pulled out of free text.

Any of these makes the agent FAIL (and the protocol fall back deterministically):
  * API error / timeout (after the SDK's own retries)
  * stop_reason "refusal" or "max_tokens" (output may not match the schema)
  * no parsed object
Every call is recorded (tokens, latency) so a run can report its cost and speed.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Stronger model for the agents that reason over every flight; a smaller, faster
# one for the (more numerous) per-flight pilot agents. Override with env vars or CLI flags.
DEFAULT_MODEL = os.environ.get("ATM_MODEL", "claude-sonnet-5-5")
DEFAULT_PILOT_MODEL = os.environ.get("ATM_PILOT_MODEL", "claude-haiku-5-5")


class AgentOutputError(RuntimeError):
    pass


@dataclass
class CallStat:
    role: str
    model: str
    ok: bool
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    error: Optional[str] = None


class LLMClient:
    def __init__(self, model: Optional[str] = None, max_tokens: int = 4000, timeout: float = 60.0,
                 max_retries: int = 2, client=None):
        if client is None:
            import anthropic  # lazy: offline mode needs neither SDK nor key
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.stats: list[CallStat] = []
        self._lock = threading.Lock()

    # -- the only methods that touch the network (tests override these) --------
    def _parse(self, system: str, user: str, schema: Type[T]):
        resp = self.client.messages.parse(model=self.model, max_tokens=self.max_tokens, system=system,
                                          messages=[{"role": "user", "content": user}], output_format=schema)
        return resp.parsed_output, resp.stop_reason, getattr(resp, "usage", None)

    def _text(self, system: str, user: str):
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                           messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.stop_reason, getattr(resp, "usage", None)

    # ---------------------------------------------------------------------------
    def _record(self, role, t0, ok, stop=None, usage=None, error=None):
        st = CallStat(role=role, model=self.model, ok=ok, seconds=round(time.monotonic() - t0, 2),
                      input_tokens=getattr(usage, "input_tokens", 0) or 0,
                      output_tokens=getattr(usage, "output_tokens", 0) or 0,
                      stop_reason=stop, error=error)
        with self._lock:
            self.stats.append(st)

    def complete_json(self, system: str, user: str, schema: Type[T], role: str = "agent") -> T:
        t0 = time.monotonic()
        try:
            obj, stop, usage = self._parse(system, user, schema)
        except Exception as e:  # noqa: BLE001 - every API/validation failure is an agent failure
            self._record(role, t0, False, error=f"{type(e).__name__}: {e}")
            raise AgentOutputError(f"{type(e).__name__}: {e}") from e
        if stop in ("refusal", "max_tokens") or obj is None:
            self._record(role, t0, False, stop, usage, error=f"stop_reason={stop}")
            raise AgentOutputError(f"unusable output (stop_reason={stop})")
        self._record(role, t0, True, stop, usage)
        return obj

    def complete_text(self, system: str, user: str, role: str = "summariser") -> str:
        t0 = time.monotonic()
        try:
            text, stop, usage = self._text(system, user)
        except Exception as e:  # noqa: BLE001
            self._record(role, t0, False, error=f"{type(e).__name__}: {e}")
            raise
        if stop == "max_tokens" or not text.strip():
            self._record(role, t0, False, stop, usage, error=f"stop_reason={stop}")
            raise AgentOutputError(f"text cut off (stop_reason={stop})")
        self._record(role, t0, True, stop, usage)
        return text.strip()


def build_llms(model: Optional[str] = None, pilot_model: Optional[str] = None, **kw) -> dict[str, LLMClient]:
    """One client per role group. ATC, Ground and the summariser share the main model."""
    main = LLMClient(model=model or DEFAULT_MODEL, **kw)
    pilot = LLMClient(model=pilot_model or DEFAULT_PILOT_MODEL, client=main.client, **kw) \
        if (pilot_model or DEFAULT_PILOT_MODEL) != main.model else main
    return {"ATC": main, "GROUND": main, "SUMMARISER": main, "PILOT": pilot}


def usage_summary(llms) -> dict:
    clients = {id(c): c for c in (llms.values() if isinstance(llms, dict) else [llms]) if c is not None}
    stats = [s for c in clients.values() for s in c.stats]
    by_role: dict[str, dict] = {}
    for s in stats:
        key = s.role.split("(")[0]
        d = by_role.setdefault(key, {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
                                     "seconds": 0.0, "model": s.model})
        d["calls"] += 1
        d["failed"] += 0 if s.ok else 1
        d["input_tokens"] += s.input_tokens
        d["output_tokens"] += s.output_tokens
        d["seconds"] = round(d["seconds"] + s.seconds, 2)
    return {"calls": len(stats), "failed": sum(not s.ok for s in stats),
            "input_tokens": sum(s.input_tokens for s in stats), "output_tokens": sum(s.output_tokens for s in stats),
            "slowest_call_s": max((s.seconds for s in stats), default=0), "by_role": by_role}
