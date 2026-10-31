"""Apple-style dashboard for the ATC / Pilot / Ground disruption negotiation (Airbus track, Challenge 2).

    streamlit run dashboard.py

Live        pick a disrupted flights.csv row, add constraints, and watch the three agents negotiate
            message by message as each one is sent (rule-based agents offline, or Claude agents
            when ANTHROPIC_API_KEY is set).
Replays     step through saved consensus logs (three recorded Claude runs plus fault and fallback cases).
Evaluation  read from evaluate.py's output files (out/eval, out/eval_llm, out/chaos), so the numbers
            match the README.
The engine (atm/) is unchanged; atm/live.py only observes it.
"""
from __future__ import annotations

import copy
import html
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from atm.chaos import CHAOS_MODES
from atm.constraints import metrics
from atm.limits import ASSUMPTIONS, load_limits, limits_table
from atm.live import LiveRun, WhatIf, apply_whatif
from atm.log import ConsensusLog
from atm.scenario import disrupted, load_flights, scenario
from atm.timeutil import fmt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "disruptions.csv"
MAX_CLAUDE_RUNS = int(os.environ.get("CLAUDE_MAX_RUNS", "5"))      # per browser session, protects the API credit
WORKBOOK = ROOT / "data" / "challenge2_operational_constraints.xlsx"

st.set_page_config(page_title="ATC · Pilot · Ground negotiation", page_icon="✈️", layout="wide",
                   initial_sidebar_state="collapsed")

# =====================================================================================================
# Style: Apple-like. System font, generous white space, white cards on a light grey canvas,
# pill buttons, one accent blue. Agents are colour-coded the same way everywhere.
# =====================================================================================================
CSS = """
<style>
:root{--ink:#1d1d1f;--ink2:#6e6e73;--ink3:#86868b;--line:#e5e5ea;--fill:#f5f5f7;--blue:#0071e3;
--green:#1d8a3a;--red:#d70015;--orange:#b25000;--sf:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Helvetica Neue",Helvetica,Arial,sans-serif;}
.stApp{background:var(--fill);}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp button p, .stApp [data-baseweb="select"] div, .stApp [data-testid="stMarkdownContainer"]{font-family:var(--sf)!important;}
.stApp{-webkit-font-smoothing:antialiased;color:var(--ink);}
header[data-testid="stHeader"]{background:rgba(245,245,247,.72);backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px);}
[data-testid="stSidebarCollapsedControl"],[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1200px;padding-top:3.4rem;padding-bottom:5rem;}
.stButton>button,.stDownloadButton>button{border-radius:980px;font-weight:500;border:1px solid #d2d2d7;background:#fff;color:var(--ink);transition:all .2s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--blue);color:var(--blue);}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--blue)!important;border-color:var(--blue)!important;color:#fff!important;}
.stButton>button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{background:#0077ed!important;color:#fff!important;}
.stButton>button[kind="primary"] p{color:#fff!important;}
[data-testid="stExpander"] details{border-radius:16px;border:1px solid var(--line);background:#fff;}
[data-baseweb="select"]>div,.stTextInput input{border-radius:12px!important;}
div[class*="st-key-card"]{background:#fff;border-radius:22px;padding:22px 24px 20px;box-shadow:0 1px 2px rgba(0,0,0,.04),0 6px 24px rgba(0,0,0,.05);}
div[class*="st-key-chat"]{gap:.35rem!important;}
div[class*="st-key-chat"] [data-testid="stMarkdownContainer"],div[class*="st-key-card"] [data-testid="stMarkdownContainer"]{margin-bottom:0!important;}
div[class*="st-key-chat"] [data-testid="stElementContainer"]{margin:0!important;}
div[class*="st-key-nav"] [data-testid="stButtonGroup"]{justify-content:center;}
/* ------------- brand + type ------------- */
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px;color:var(--ink);padding-top:6px;}
.brand .logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#0071e3,#8e44ad);display:grid;place-items:center;color:#fff;font-size:15px;}
.eyebrow{color:#b64400;font-weight:600;font-size:17px;margin:0 0 6px;}
.hero{text-align:center;padding:26px 0 8px;}
.hero h1{font-size:clamp(40px,6vw,68px);line-height:1.05;font-weight:700;letter-spacing:-.035em;margin:0 0 14px;}
.hero p.lede{font-size:clamp(18px,2vw,23px);line-height:1.4;color:var(--ink2);max-width:780px;margin:0 auto 22px;}
.sec{font-size:clamp(28px,3.4vw,40px);font-weight:700;letter-spacing:-.025em;line-height:1.1;margin:46px 0 8px;}
.sub{font-size:17px;color:var(--ink2);max-width:760px;margin:0 0 20px;line-height:1.45;}
.h3{font-size:19px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;}
.muted{color:var(--ink2);} .small{font-size:13px;color:var(--ink2);line-height:1.45;}
/* ------------- tiles ------------- */
.kpis{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin:8px 0 6px;}
.kpi{background:#fff;border-radius:20px;padding:18px 20px;box-shadow:0 1px 2px rgba(0,0,0,.04),0 6px 24px rgba(0,0,0,.05);}
.kpi .v{font-size:clamp(28px,3vw,40px);white-space:nowrap;font-weight:600;letter-spacing:-.03em;line-height:1.05;margin:6px 0 4px;font-variant-numeric:tabular-nums;}
.kpi .l{font-size:13px;color:var(--ink2);font-weight:500;}
.kpi .s{font-size:12.5px;color:var(--ink2);}
.kpi.good .v{color:var(--green);} .kpi.blue .v{color:var(--blue);}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));}
.box{background:#fff;border-radius:22px;padding:22px 22px 20px;box-shadow:0 1px 2px rgba(0,0,0,.04),0 6px 24px rgba(0,0,0,.05);}
.box .n{width:30px;height:30px;border-radius:50%;background:var(--fill);display:grid;place-items:center;font-weight:600;font-size:14px;margin-bottom:12px;color:var(--blue);}
.box h4{font-size:17px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em;}
.box p{font-size:14.5px;color:var(--ink2);line-height:1.45;margin:0;}
.chip{display:inline-block;padding:2px 10px;border-radius:980px;font-size:12px;font-weight:500;background:var(--fill);color:var(--ink);margin:2px 4px 2px 0;white-space:nowrap;}
.chip.ok{background:#e8f6ec;color:var(--green);} .chip.bad{background:#fdecee;color:var(--red);} .chip.warn{background:#fff4e5;color:var(--orange);} .chip.blue{background:#e8f1fd;color:var(--blue);}
/* ------------- spec list ------------- */
.spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px 18px;margin-top:10px;}
.spec .k{font-size:12px;color:var(--ink2);} .spec .val{font-size:17px;font-weight:600;letter-spacing:-.01em;}
/* ------------- tables ------------- */
table.tbl{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;}
table.tbl th{text-align:left;font-weight:600;font-size:12px;color:var(--ink2);padding:8px 10px;border-bottom:1px solid var(--line);}
table.tbl td{padding:9px 10px;border-bottom:1px solid #f0f0f2;vertical-align:top;}
table.tbl td.num,table.tbl th.num{text-align:right;white-space:nowrap;}
table.tbl tr:last-child td{border-bottom:none;}
.stApp table.tbl th,.stApp table.tbl td{border-left:none!important;border-right:none!important;border-top:none!important;background:transparent;}
.stApp table.tbl{border:none!important;}
/* ------------- metric tiles ------------- */
.mt{background:#fff;border-radius:20px;padding:16px 18px;box-shadow:0 1px 2px rgba(0,0,0,.04),0 6px 24px rgba(0,0,0,.05);}
.mt .l{font-size:13px;color:var(--ink2);font-weight:500;min-height:2.4em;}
.mt .row{display:flex;align-items:baseline;gap:8px;margin:6px 0;flex-wrap:wrap;}
.mt .a{font-size:17px;color:var(--ink3);text-decoration:line-through;text-decoration-color:rgba(0,0,0,.25);}
.mt .b{font-size:30px;font-weight:600;letter-spacing:-.02em;}
.mt .bar{height:6px;border-radius:3px;background:var(--fill);overflow:hidden;margin:2px 0 4px;}
.mt .bar span{display:block;height:100%;border-radius:3px;}
/* ------------- requirement rows ------------- */
.req{display:grid;grid-template-columns:auto 1fr;gap:14px;padding:16px 0;border-bottom:1px solid #f0f0f2;}
.req:last-child{border-bottom:none;}
.req .tick{width:26px;height:26px;border-radius:50%;background:#e8f6ec;color:var(--green);display:grid;place-items:center;font-weight:700;font-size:14px;}
.req .t{font-weight:600;font-size:15.5px;margin:0 0 4px;color:var(--ink)!important;}
.req .h{font-size:14px;color:var(--ink2);margin:0 0 6px;line-height:1.45;}
/* ------------- chat (iMessage-like) ------------- */
.msg{display:flex;gap:10px;align-items:flex-end;}
.msg.new{animation:pop .38s cubic-bezier(.2,.8,.2,1);}
.av{width:34px;height:34px;border-radius:50%;color:#fff;font-weight:700;font-size:10.5px;display:grid;place-items:center;flex:none;letter-spacing:.02em;}
.bw{max-width:min(680px,86%);}
.who{font-size:12px;color:var(--ink2);margin:0 0 3px 12px;}
.bub{padding:10px 14px;border-radius:20px 20px 20px 6px;font-size:15px;line-height:1.42;color:var(--ink);}
.rec{margin-top:8px;padding-top:7px;border-top:1px solid rgba(0,0,0,.08);font-size:13px;line-height:1.4;}
.rec div{margin:3px 0;}
.rec .acc{color:var(--green);font-weight:600;} .rec .rej{color:var(--red);font-weight:600;} .rec .ctr{color:var(--blue);font-weight:600;} .rec .pro{color:var(--ink2);font-weight:600;}
.rid{color:var(--ink3);font-size:11px;margin-left:4px;font-variant-numeric:tabular-nums;}
.note{align-self:center;margin:0 auto;max-width:92%;text-align:center;font-size:13px;line-height:1.4;padding:6px 14px;border-radius:14px;width:fit-content;}
.note.info{background:#ececf0;color:#3a3a3c;} .note.ok{background:#e8f6ec;color:#17702f;} .note.bad{background:#fdecee;color:#b0001a;}
.note.warn{background:#fff4e5;color:#924000;} .note.fault{background:#b0001a;color:#fff;}
.note.new{animation:pop .38s cubic-bezier(.2,.8,.2,1);}
.div{display:flex;align-items:center;gap:12px;color:var(--ink3);font-size:12px;font-weight:600;margin:12px 0 2px;}
.div:before,.div:after{content:"";flex:1;height:1px;background:var(--line);}
.big{border-radius:18px;padding:14px 18px;margin:4px 0;}
.big.ok{background:#e8f6ec;} .big.warn{background:#fff4e5;} .big.bad{background:#fdecee;}
.big .t{font-weight:600;font-size:16px;margin:0 0 2px;} .big .d{font-size:13.5px;color:var(--ink2);}
.brief{background:#fff;border:1px solid var(--line);border-radius:18px;padding:14px 18px;font-size:14px;line-height:1.5;white-space:pre-wrap;}
.typing{display:flex;flex-direction:column;gap:8px;margin-top:6px;}
.dots{display:inline-flex;gap:4px;padding:12px 14px;border-radius:20px 20px 20px 6px;background:#e9e9eb;}
.dots i{width:7px;height:7px;border-radius:50%;background:#8e8e93;display:block;animation:blink 1.2s infinite ease-in-out;}
.dots i:nth-child(2){animation-delay:.18s;} .dots i:nth-child(3){animation-delay:.36s;}
@keyframes blink{0%,80%,100%{opacity:.3;transform:translateY(0)}40%{opacity:1;transform:translateY(-2px)}}
@keyframes pop{from{opacity:0;transform:translateY(8px) scale(.985)}to{opacity:1;transform:none}}
/* ------------- turn tracker ------------- */
.trk{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:2px 0 10px;}
.st{display:flex;align-items:center;gap:6px;font-size:13px;font-weight:500;color:var(--ink3);padding:5px 12px;border-radius:980px;background:var(--fill);}
.st.done{color:var(--green);background:#e8f6ec;} .st.now{color:#fff;background:var(--blue);} .st.skip{text-decoration:line-through;opacity:.7;}
.trk .sep{width:14px;height:1px;background:#d2d2d7;}
.live{margin-left:auto;font-size:13px;color:var(--ink2);display:flex;align-items:center;gap:6px;font-variant-numeric:tabular-nums;}
.live .rd{width:8px;height:8px;border-radius:50%;background:#ff3b30;animation:blink 1.4s infinite;}
.empty{text-align:center;padding:38px 10px;color:var(--ink2);}
.empty .avs{display:flex;justify-content:center;gap:10px;margin-bottom:14px;}
.foot{font-size:12px;color:var(--ink3);text-align:center;margin-top:48px;line-height:1.6;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s: str, where=st):
    where.markdown(s, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x)).replace("$", "&#36;")


def num(x, d=0) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:,.{d}f}"


def table_html(rows: list[dict], cols: list[tuple[str, str]], numeric=()) -> str:
    head = "".join(f'<th class="{"num" if k in numeric else ""}">{esc(lbl)}</th>' for k, lbl in cols)
    body = "".join("<tr>" + "".join(f'<td class="{"num" if k in numeric else ""}">{r.get(k, "")}</td>'
                                    for k, _ in cols) + "</tr>" for r in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# =====================================================================================================
# Data (cached)
# =====================================================================================================
@st.cache_data(show_spinner=False)
def flights_df():
    return load_flights(DATA)


@st.cache_data(show_spinner=False)
def disrupted_df():
    d = disrupted(flights_df()).reset_index(drop=True)
    d["SID"] = [f"{r.DESTINATION_AIRPORT}-{int(r.YEAR)}-{int(r.MONTH):02d}-{int(r.DAY):02d}-"
                f"{fmt(int(r.ETA_MIN)).replace(':', '')}" for r in d.itertuples()]
    d["LABEL"] = [f"{r.AIRLINE}{int(r.FLIGHT_NUMBER)}  {r.ORIGIN_AIRPORT} → {r.DESTINATION_AIRPORT}  ·  "
                  f"{int(r.YEAR)}-{int(r.MONTH):02d}-{int(r.DAY):02d}  ·  weather {r.WEATHER_DELAY:.0f} min, "
                  f"late aircraft {r.LATE_AIRCRAFT_DELAY:.0f} min" for r in d.itertuples()]
    return d


def world_for(i: int):
    return scenario(flights_df(), disrupted(flights_df()).reset_index(drop=True).iloc[[i]])


@st.cache_data(show_spinner=False)
def eval_data():
    def rd(p):
        p = ROOT / p
        return pd.read_csv(p) if p.exists() else None
    return rd("out/eval/results.csv"), rd("out/eval_llm/results.csv"), rd("out/eval/flights.csv"), rd("out/chaos/chaos.csv")


def process_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    s = {"n": n, "agree": (df.decided_by == "consensus").mean(), "fallback": (df.decided_by == "fallback").mean(),
         "turns": df.rounds.mean(), "max_turns": int(df.rounds.max()),
         "escalated": int(df.escalated.astype(str).eq("True").sum()),
         "final_conf": int((df.neg_hard_conflicts > 0).sum()), "base_conf": int((df.base_hard_conflicts > 0).sum())}
    for c in ("llm_calls", "llm_failed", "checker_rejects", "input_tokens", "output_tokens", "seconds"):
        s[c] = df[c].sum() if c in df else None
    return s


FEATURED = [
    ("ATL-2015-06-09-2151", "Thunderstorm at Atlanta",
     "2 of 4 runways closed, 206 min weather delay. A fuel-critical arrival needs priority."),
    ("ATL-2015-09-03-2017", "Blocked taxiway, occupied gate",
     "Late inbound aircraft: taxiway B blocked and the planned gate still occupied."),
    ("PHL-2015-01-06-1659", "Three turns to agree",
     "The first vote fails; a counter-offer is needed before everyone accepts."),
    ("PDX-2015-03-02-1458", "Where it does worse",
     "Today's plan looks better here, but it breaks a rule. Watch the trade-off."),
]

REPLAYS = [
    ("demo/1_storm_fuel_priority.jsonl", "Claude · Thunderstorm, fuel priority",
     "Recorded live run with Claude agents. The checker rejects a plan that would land below fuel reserve."),
    ("demo/2_taxiway_catch.jsonl", "Claude · Blocked taxiway caught",
     "Recorded Claude run. Ground's plan would taxi through a blocked taxiway; the checker stops it."),
    ("demo/3_chaos_fuel.jsonl", "Claude · Injected fuel fault",
     "Recorded Claude run. A fault makes ATC ignore fuel; the checker blocks it and the agents settle on a safe plan."),
    ("showcase/chaos_wake.jsonl", "Fault · Wake spacing squeezed",
     "ATC is made to put two arrivals 1 min apart on one runway. WAKE_SEPARATION catches it."),
    ("showcase/chaos_ghost.jsonl", "Fault · Ghost gate Z99",
     "Ground invents a gate that does not exist. GATE_UNKNOWN catches it."),
    ("showcase/chaos_urgency.jsonl", "Fault · False CRITICAL claim",
     "A pilot with plenty of fuel claims CRITICAL. The arbiter downweights it from world state."),
    ("showcase/chaos_role.jsonl", "Fault · Pilot changes its own gate",
     "A pilot uses a lever that is not a pilot lever. The protocol denies the permission."),
    ("showcase/three_turns.jsonl", "Three turns to agree",
     "No candidate is accepted by all in turn 2; a final vote in turn 3 reaches consensus."),
    ("showcase/worse_case.jsonl", "Where negotiation does worse",
     "The negotiated plan costs this flight connections; today's plan only worked by breaking a rule."),
    ("showcase/fallback_outage.jsonl", "AI outage → non-AI fallback",
     "Ground's AI call fails (simulated outage). The deterministic fallback decides and the log says why."),
]

# =====================================================================================================
# Chat rendering: one renderer for live runs, stored runs and saved logs
# =====================================================================================================
AGENTS = {  # key: (initials, colour, bubble tint)
    "ATC": ("ATC", "#0071e3", "#e8f1fd"),
    "PILOT": ("PLT", "#1d8a3a", "#e9f7ee"),
    "GROUND": ("GND", "#8e44ad", "#f4ebfa"),
}
CODE = {"CHECKER": "Safety checker", "ARBITER": "Arbiter", "MEDIATOR": "Mediator", "PROTOCOL": "Protocol",
        "FALLBACK": "Fallback solver", "SYSTEM": "System", "SUMMARISER": "Summariser", "CHAOS": "Fault injected"}
AGENT_EVENTS = {"PROPOSE", "ACCEPT", "REJECT", "COUNTER", "IGNORED", "AGENT_ERROR"}
TURN_TITLE = {1: "Turn 1 · each agent declares its position", 2: "Turn 2 · vote and counter-offer",
              3: "Turn 3 · final vote"}
VERB = {"REJECT": "blocked", "OK": "passed", "BLOCKED": "ignored an unsafe acceptance", "PERMISSION_DENIED": "denied",
        "DOWNWEIGHT": "downweighted a claim", "VERIFY": "verified", "RANK": "set the priority order",
        "REPAIR": "repaired a plan", "PROPOSE": "added a plan", "MERGE": "merged counter-offers",
        "NOTED": "noted", "INVALID": "rejected an invalid message", "FALLBACK": "triggered the fallback",
        "ESCALATE": "escalated to a controller", "ERROR": "error"}


def akey(actor: str) -> str:
    return "PILOT" if actor.startswith("PILOT") else actor


def is_agent(actor: str) -> bool:
    return akey(actor) in AGENTS


def alabel(actor: str) -> str:
    if actor.startswith("PILOT("):
        return "Pilot · " + actor[6:-1]
    return {"ATC": "ATC · air traffic control", "GROUND": "Ground · gates and crews"}.get(actor, actor)


def avatar(actor: str) -> str:
    ini, col, _ = AGENTS[akey(actor)]
    return f'<div class="av" style="background:{col}">{ini}</div>'


def _record_line(e: dict) -> str:
    ev, r, rid = e["event"], e["reason"], f'<span class="rid">#{e["seq"]}</span>'
    if ev == "PROPOSE":
        body = r.split(" | ", 1)[1] if " | " in r else ""
        return f'<div><span class="pro">Proposal on record</span> {esc(body)}{rid}</div>' if body else \
               f'<div><span class="pro">Position on record</span>{rid}</div>'
    if ev == "ACCEPT":
        return f'<div><span class="acc">✓ Accepts</span> {esc(r.split(" - ", 1)[0])}{rid}</div>'
    if ev == "REJECT":
        return f'<div><span class="rej">✕ Rejects</span> {esc(r)}{rid}</div>'
    if ev == "COUNTER":
        return f'<div><span class="ctr">↺ Counter-offer</span> {esc(r)}{rid}</div>'
    if ev == "IGNORED":
        return f'<div><span class="pro">Ignored</span> {esc(r)}{rid}</div>'
    return f'<div><span class="rej">Error</span> {esc(r)}{rid}</div>'


def bubble_html(item: dict, new: bool) -> str:
    actor, recs = item["actor"], item["records"]
    text = item.get("said")
    if not text and recs:      # saved log: the message is inside the record
        r0 = recs[0]
        if r0["event"] == "PROPOSE":
            text = r0["reason"].split(" | ", 1)[0]
        elif r0["event"] == "ACCEPT" and " - " in r0["reason"]:
            text = r0["reason"].split(" - ", 1)[1]
    tint = AGENTS[akey(actor)][2]
    meta = TURN_TITLE.get(item["rnd"], "").split(" · ")[0]
    if item.get("secs") is not None:
        meta += f" · answered in {item['secs']:.1f} s" if item["secs"] >= 0.05 else " · rule-based, instant"
    rec = "".join(_record_line(e) for e in recs)
    body = (f'<div>{esc(text)}</div>' if text else "") + (f'<div class="rec">{rec}</div>' if rec else "")
    return (f'<div class="msg{" new" if new else ""}">{avatar(actor)}<div class="bw">'
            f'<div class="who"><b>{esc(alabel(actor))}</b> · {esc(meta)}</div>'
            f'<div class="bub" style="background:{tint}">{body or "…"}</div></div></div>')


def note_html(e: dict) -> str:
    a, ev, r, rid = e["actor"], e["event"], e["reason"], f'<span class="rid">#{e["seq"]}</span>'
    if a == "SYSTEM" and ev == "START":
        d = e.get("data") or {}
        mode = "Claude agents" if d.get("mode") == "llm" else "rule-based agents"
        fault = d.get("chaos", "none")
        return (f'<div class="note info new">Negotiation opened with {mode}'
                f'{" · fault: " + esc(fault) if fault and fault != "none" else ""}{rid}</div>')
    if ev == "DECIDE":
        if a == "FALLBACK":
            return (f'<div class="big warn new"><div class="t">The non-AI fallback decided</div>'
                    f'<div class="d">{esc(r)}{rid}</div></div>')
        return (f'<div class="big ok new"><div class="t">Consensus: every agent accepted the same plan</div>'
                f'<div class="d">{esc(r)}{rid}</div></div>')
    if ev == "OUTCOME":
        fid, _, txt = r.partition(": ")
        return f'<div class="note info new"><b>{esc(fid)}</b> {esc(txt)}{rid}</div>'
    if ev == "BRIEFING":
        text = (e.get("data") or {}).get("text") or r
        return (f'<div class="new" style="animation:pop .38s"><div class="who" style="margin-left:4px">'
                f'<b>Briefing for controller sign-off</b> · written by the summariser{rid}</div>'
                f'<div class="brief">{esc(text)}</div></div>')
    if ev == "USAGE":
        return f'<div class="note info new">{esc(r)}{rid}</div>'
    if ev == "END":
        ok = "intact" in r
        return f'<div class="note {"ok" if ok else "bad"} new">Log closed · {esc(r)}{rid}</div>'
    tone = "fault" if a == "CHAOS" else \
        "bad" if ev in ("REJECT", "BLOCKED", "PERMISSION_DENIED", "INVALID", "ESCALATE", "AGENT_ERROR", "ERROR") else \
        "warn" if ev in ("DOWNWEIGHT", "FALLBACK") else "ok" if ev in ("OK", "VERIFY") else "info"
    who = CODE.get(a, alabel(a))
    verb = "" if a == "CHAOS" else VERB.get(ev, ev.lower())
    icon = {"fault": "⚠ ", "bad": "⛔ ", "ok": "✓ ", "warn": "! "}.get(tone, "")
    return f'<div class="note {tone} new">{icon}<b>{esc(who)}</b> {esc(verb)}{" · " if verb else ": "}{esc(r)}{rid}</div>'


def tracker_html(stage: str, turns_seen: set, finished: bool, status: str) -> str:
    steps = ["Setup", "Turn 1", "Turn 2", "Turn 3", "Decision"]
    cur = steps.index(stage) if stage in steps else 0
    out = []
    for i, s in enumerate(steps):
        if finished and s == "Turn 3" and 3 not in turns_seen:
            cls, lab = "st skip", "Turn 3 not needed"
        elif i < cur or (finished and i == cur):
            cls, lab = "st done", "✓ " + s
        elif i == cur:
            cls, lab = "st now", s
        else:
            cls, lab = "st", s
        out.append(f'<span class="{cls}">{lab}</span>')
    return f'<div class="trk">{"<span class=sep></span>".join(out)}<span class="live">{status}</span></div>'


class Chat:
    """Feeds events into a Streamlit container as chat bubbles and notices."""

    def __init__(self, box, tracker_ph, typing_ph, live: bool):
        self.box, self.tracker_ph, self.typing_ph, self.live = box, tracker_ph, typing_ph, live
        self.bubbles: dict = {}
        self.turn, self.stage, self.turns_seen = 0, "Setup", set()
        self.typing: dict = {}
        self.events: list = []
        self.t0 = time.monotonic()
        self.finished = False
        self.unsafe_caught = 0

    def _add(self, s: str):
        ph = self.box.empty()
        H(s, ph)
        return ph

    def _turn(self, rnd):
        if rnd and rnd > self.turn:
            self.turn, self.stage = rnd, f"Turn {rnd}"
            self.turns_seen.add(rnd)
            self._add(f'<div class="div">{esc(TURN_TITLE.get(rnd, f"Turn {rnd}"))}</div>')
            self.draw_tracker()

    def draw_tracker(self, status: str | None = None):
        if self.tracker_ph is None:
            return
        if status is None:
            el = time.monotonic() - self.t0
            status = (f'<span class="rd"></span>Live · {el:.1f} s' if self.live and not self.finished
                      else (f"Finished in {el:.1f} s" if self.live else "Replay"))
        H(tracker_html(self.stage, self.turns_seen, self.finished, status), self.tracker_ph)

    def draw_typing(self):
        if self.typing_ph is None:
            return
        if not self.typing:
            self.typing_ph.empty()
            return
        now = time.monotonic()
        rows = "".join(
            f'<div class="msg">{avatar(a)}<div class="bw"><div class="who"><b>{esc(alabel(a))}</b> is thinking · '
            f'{now - t:.1f} s</div><div class="dots"><i></i><i></i><i></i></div></div></div>'
            for a, (_, t) in self.typing.items())
        H(f'<div class="typing">{rows}</div>', self.typing_ph)

    def tick(self):
        self.draw_typing()
        self.draw_tracker()

    def feed(self, ev: tuple):
        kind = ev[0]
        if kind not in ("done", "crash"):
            self.events.append(ev)
        if kind == "start":
            _, actor, rnd, _t = ev
            self._turn(rnd)
            self.typing[actor] = (rnd, time.monotonic())
            self.draw_typing()
        elif kind == "said":
            _, actor, rnd, text, secs = ev
            self.typing.pop(actor, None)
            self._turn(rnd)
            item = {"actor": actor, "rnd": rnd, "said": text, "secs": secs, "records": []}
            item["ph"] = self._add(bubble_html(item, new=True))
            self.bubbles[(actor, rnd)] = item
            self.draw_typing()
        elif kind == "fail":
            _, actor, rnd, err, secs = ev
            self.typing.pop(actor, None)
            self._add(f'<div class="note bad new">⛔ <b>{esc(alabel(actor))}</b> failed after {secs:.1f} s · '
                      f'{esc(err)}</div>')
            self.draw_typing()
        elif kind == "log":
            e = ev[1]
            a, evn = e["actor"], e["event"]
            if a in ("CHECKER", "PROTOCOL") and evn in ("REJECT", "BLOCKED", "PERMISSION_DENIED", "INVALID"):
                self.unsafe_caught += 1
            if a == "ARBITER" and evn == "DOWNWEIGHT":
                self.unsafe_caught += 1
            if (a == "SYSTEM" and evn in ("DECIDE", "FALLBACK")) or a == "FALLBACK":
                if self.stage != "Decision":
                    self.stage = "Decision"
                    self._add('<div class="div">Decision</div>')
            else:
                self._turn(e.get("round"))
            if is_agent(a) and evn in AGENT_EVENTS:
                key = (a, e.get("round"))
                item = self.bubbles.get(key)
                if item is None:
                    item = {"actor": a, "rnd": e.get("round"), "said": None, "secs": None, "records": [e]}
                    item["ph"] = self._add(bubble_html(item, new=True))
                    self.bubbles[key] = item
                else:
                    item["records"].append(e)
                    H(bubble_html(item, new=False), item["ph"])
            else:
                self._add(note_html(e))
            self.draw_tracker()
        elif kind in ("done", "crash"):
            self.finish()
            if kind == "crash":
                self._add(f'<div class="note bad new">The run stopped: {esc(ev[1])}</div>')

    def finish(self):
        self.finished = True
        self.typing.clear()
        self.draw_typing()
        self.draw_tracker()


def chat_frame(key: str, live: bool):
    """A card holding the tracker, the chat and the typing indicators."""
    card = st.container(key=f"card_{key}")
    with card:
        tracker = st.empty()
        box = st.container(key=f"chat_{key}")
        typing = st.empty()
    return Chat(box, tracker, typing, live)


def feed_all(chat: Chat, events, pace: float = 0.0):
    for ev in events:
        if pace and ev[0] == "said":
            time.sleep(pace)
        chat.feed(ev)
        if pace and ev[0] == "log":
            time.sleep(pace * 0.45)


# =====================================================================================================
# Navigation
# =====================================================================================================
PAGES = ["Overview", "Live", "Replays", "Evaluation", "Requirements", "Rules & limits"]
st.session_state.setdefault("page", "Overview")
st.session_state.setdefault("_last_page", "Overview")


def _keep_page():
    if st.session_state.page is None:          # clicking the selected segment deselects it: undo that
        st.session_state.page = st.session_state._last_page
    st.session_state._last_page = st.session_state.page


def goto(page: str, **state):
    st.session_state.page = page
    st.session_state._last_page = page
    for k, v in state.items():
        st.session_state[k] = v


c1, c2 = st.columns([1, 3], vertical_alignment="center")
H('<div class="brand"><div class="logo">✈</div>ATC · Pilot · Ground</div>', c1)
with c2.container(key="nav"):
    st.segmented_control("Section", PAGES, key="page", on_change=_keep_page, label_visibility="collapsed")
page = st.session_state.page or "Overview"

results, llm_results, flights_eval, chaos = eval_data()
OFF = process_stats(results) if results is not None else None
LLM = process_stats(llm_results) if llm_results is not None else None


def chaos_totals():
    if chaos is None:
        return None
    return int(chaos.applicable.sum()), int((chaos.caught & chaos.applicable).sum()), \
        int((chaos.final_hard_conflicts > 0).sum())


# =====================================================================================================
# OVERVIEW
# =====================================================================================================
def page_overview():
    H('<div class="hero"><p class="eyebrow">Airbus track · Challenge 2</p>'
      '<h1>Three agents.<br>One safe plan.</h1>'
      '<p class="lede">When weather or a late aircraft disrupts an arrival, an ATC agent, a Pilot agent and a '
      'Ground agent negotiate a new landing time, route, gate and crew in at most three turns. Code checks '
      'every safety rule, and a non-AI fallback decides if they cannot agree.</p></div>')
    _, b1, b2, _ = st.columns([2, 1.3, 1.3, 2])
    b1.button("Watch a live negotiation", type="primary", on_click=goto, args=("Live",), width="stretch")
    b2.button("See the evidence", on_click=goto, args=("Evaluation",), width="stretch")

    if OFF:
        inj, caught, leaked = chaos_totals() or (None, None, None)
        H('<div class="sec" style="text-align:center">The results, measured.</div>'
          f'<p class="sub" style="margin:0 auto 20px;text-align:center">Every number below is computed from '
          f'evaluate.py on {OFF["n"]} real disrupted flights from the 2015 DOT data.</p>')
        tiles = [
            ("good", "Agreement within 3 turns", f'{OFF["agree"]:.0%}', f'rule-based agents, {OFF["n"]} scenarios'),
            ("", "Turns used on average", f'{OFF["turns"]:.2f}', "the limit is three"),
            ("good", "Final plans breaking a safety rule", f'{OFF["final_conf"]}',
             f'today\'s process: {OFF["base_conf"]} of {OFF["n"]}'),
            ("good", "Injected faults caught", f"{caught}/{inj}" if inj else "–",
             f"{leaked} reached a final plan" if inj else "run evaluate.py --chaos all"),
        ]
        if LLM:
            tiles.append(("blue", "Claude agents: agreement", f'{LLM["agree"]:.0%}',
                          f'{LLM["n"]} live runs, {int(LLM["llm_failed"] or 0)} failed AI calls'))
        H('<div class="kpis">' + "".join(f'<div class="kpi {c}"><div class="l">{esc(l)}</div><div class="v">{v}</div>'
                                         f'<div class="s">{esc(s)}</div></div>' for c, l, v, s in tiles) + "</div>")

    H('<div class="sec">How it works.</div><p class="sub">Language models argue and explain. Safety rules, '
      'scheduling and the fallback are plain code that cannot be talked round.</p>')
    steps = [
        ("A real disruption comes in", "One row of flights.csv with WEATHER_DELAY over 15 or LATE_AIRCRAFT_DELAY "
         "over 20 minutes, plus the real traffic around it and the organisers' workbook limits."),
        ("Three agents, three turns", "ATC, Pilot and Ground each get their own prompt and a private view. Turn 1 "
         "they declare positions, turn 2 they vote and counter-offer, turn 3 is a final vote."),
        ("Code checks every plan", "A deterministic checker tests fuel reserve, wake separation, runway capacity, "
         "blocked taxiways, gates, crew duty and turbulence. An unsafe plan can never be accepted."),
        ("Decide, record, brief", "Consensus or the non-AI fallback. Every step goes into a hash-chained log, and a "
         "briefing goes to the human controller for sign-off."),
    ]
    H('<div class="grid">' + "".join(f'<div class="box"><div class="n">{i + 1}</div><h4>{esc(t)}</h4><p>{esc(d)}</p>'
                                     f'</div>' for i, (t, d) in enumerate(steps)) + "</div>")

    H('<div class="sec">Built for the people who fix disruptions today.</div><p class="sub">Today each team plans '
      'alone and they coordinate by phone, so conflicts surface late: a gate behind a closed taxiway, a slot the '
      'fuel cannot reach, a crew already off shift.</p>')
    people = [
        ("ATC", "Approach and tower supervisor", "Sequences arrivals with wake spacing while weather cuts runway "
         "capacity and taxiways close."),
        ("PILOT", "Captain and airline dispatcher", "Protects the fuel reserve, avoids turbulence and keeps "
         "passengers' onward connections."),
        ("GROUND", "Station operations and ramp", "Reassigns gates and crews when an aircraft runs late, without "
         "overtime or turnaround delays."),
    ]
    H('<div class="grid">' + "".join(
        f'<div class="box"><div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">'
        f'{avatar(k)}<h4 style="margin:0">{esc(t)}</h4></div><p>{esc(d)}</p></div>' for k, t, d in people) + "</div>")

    H('<div class="sec">Every requirement, covered.</div>')
    req = pd.read_csv(ROOT / "dashboard_content" / "requirements.csv")
    short = req[req.id.isin(["B5", "B6", "B7", "B8", "B9", "A1", "A2", "A3"])]
    H('<div class="box">' + "".join(
        f'<div class="req"><div class="tick">✓</div><div><p class="t">{esc(r.requirement)}</p>'
        f'<p class="h">{esc(r.how_met)}</p></div></div>' for r in short.itertuples()) + "</div>")
    st.button("See all 17 requirements", on_click=goto, args=("Requirements",))


# =====================================================================================================
# LIVE
# =====================================================================================================
def situation_html(world) -> str:
    f = world.flights[0]
    ap = world.airport
    rw = "".join(f'<span class="chip {"" if r.open else "bad"}">{esc(r.id)}{"" if r.open else " closed"}</span>'
                 for r in ap.runways)
    tw = "".join(f'<span class="chip bad">taxiway {esc(t.id)} blocked</span>' for t in ap.taxiways if t.blocked)
    spare = f.fuel_at_eta - f.reserve
    spec = [
        ("Flight", f"{f.flight_id} · {f.origin} → {f.dest}"),
        ("Aircraft", f"{f.aircraft_type} ({f.wake})"),
        ("Weather delay", f"{f.weather_delay:.0f} min"),
        ("Late-aircraft delay", f"{f.late_aircraft_delay:.0f} min"),
        ("Earliest arrival", fmt(f.eta)),
        ("Fuel above reserve", f"{spare} min"),
        ("Connecting passengers", f"{f.connecting_pax} of {f.pax}"),
        ("Other arrivals", f"{len(ap.occupied_slots)} already sequenced"),
    ]
    return (f'<div class="h3">{esc(ap.name)} ({esc(ap.code)}) · {esc(world.date)}</div>'
            f'<div class="small">{esc(world.disruption)}</div>'
            f'<div class="spec">' + "".join(f'<div><div class="k">{esc(k)}</div><div class="val">{esc(v)}</div></div>'
                                           for k, v in spec) + "</div>"
            f'<div style="margin-top:12px"><span class="small">Runways </span>{rw} {tw}</div>'
            f'<div class="small" style="margin-top:8px">Real DOT data: flight, times, delays, airports. Simulated '
            f'and seeded: fuel, passengers, gates, crews, routes.</div>')


def decision_panel(res, world, chat: Chat, cfg_key: str):
    ok, bad = res.log.verify()
    rejected = sum(len(v) for v in res.checks.values())
    tiles = [
        ("good" if res.decided_by == "consensus" else "", "Decided by",
         "Consensus" if res.decided_by == "consensus" else "Fallback", res.fallback_reason or "all three agents accepted"),
        ("", "Turns used", f"{res.rounds_used}/3", "one message per agent per turn"),
        ("", "Plans checked", f"{len(res.candidates)}", f"{rejected} rule violations found"),
        ("good" if ok else "", "Hash chain", "Intact" if ok else f"Broken at {bad}", f"{len(res.log.entries)} log entries"),
    ]
    if res.usage:
        u = res.usage
        tiles.append(("blue", "Claude calls", f"{u['calls']}", f"{u['failed']} failed · "
                      f"{u['input_tokens'] + u['output_tokens']:,} tokens"))
    H('<div class="kpis">' + "".join(f'<div class="kpi {c}"><div class="l">{esc(l)}</div><div class="v" '
                                     f'style="font-size:30px">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
                                     for c, l, v, s in tiles) + "</div>")
    if res.escalated:
        H('<div class="big bad"><div class="t">Escalated to a controller</div><div class="d">No plan satisfies every '
          'hard constraint. The fallback plan is shown, but a human must decide.</div></div>')
    rows = []
    for f in world.flights:
        a = res.decision.assignments[f.flight_id]
        m = metrics(world, f, a)
        u = res.urgency[f.flight_id]
        rows.append({
            "f": f"<b>{esc(f.flight_id)}</b>", "p": f'<span class="chip {"bad" if u.level == "CRITICAL" else "warn" if u.level == "HIGH" else ""}">{esc(u.level)}</span>',
            "land": esc("Divert to " + a.divert_to) if a.divert_to else esc(f"{fmt(a.landing)} on {a.runway}"),
            "route": esc(a.route), "hold": num(m.holding) if m else "–",
            "fuel": f"{num(m.fuel_at_landing)} min (reserve {f.reserve})" if m else "–",
            "gate": esc(f"{a.gate} / {a.crew}"), "in": esc(fmt(m.gate_in)) if m and m.gate_in else "–",
            "conn": f"{num(m.conn_missed)} of {f.connecting_pax}" if m else "–",
        })
    with st.container(key="card_outcome"):
        H('<div class="h3">The revised plan</div><p class="small">What the controller is asked to sign off.</p>')
        H(table_html(rows, [("f", "Flight"), ("p", "Priority"), ("land", "Landing"), ("route", "Route"),
                            ("hold", "Hold (min)"), ("fuel", "Fuel at landing"), ("gate", "Gate / crew"),
                            ("in", "At gate"), ("conn", "Connections missed")], numeric=("hold",)))
        c1, c2 = st.columns(2)
        c1.download_button("Download the consensus log (JSONL)",
                           "\n".join(json.dumps(e, default=str) for e in res.log.entries),
                           file_name=f"{cfg_key}.jsonl", width="stretch")
        with c2.expander("Structured messages (JSON)"):
            st.caption("Exactly what each agent sent: Pydantic schemas, validated before use.")
            st.json({"turn 1 positions": {k: v.model_dump() for k, v in res.positions.items()},
                     **{f"turn {r} votes": {k: v.model_dump() for k, v in vs.items()} for r, vs in res.votes.items()}},
                    expanded=False)


def page_live():
    d = disrupted_df()
    H('<div class="sec" style="margin-top:18px">Watch the agents negotiate.</div>'
      '<p class="sub">Pick a disrupted flight, add your own constraints if you like, and press Run. Each message '
      'appears the moment its agent sends it; the safety checker and arbiter answer in between.</p>')
    left, right = st.columns([1, 2.05], gap="large")
    with left:
        with st.container(key="card_controls"):
            H('<div class="h3">1 · Choose a disruption</div>')
            src = st.segmented_control("Scenario source", ["Featured", "Any of 1,290"], default="Featured",
                                       key="live_src", label_visibility="collapsed") or "Featured"
            sid_to_i = {s: i for i, s in enumerate(d.SID)}
            if src == "Featured":
                feats = [f for f in FEATURED if f[0] in sid_to_i]
                pick = st.radio("Featured scenario", range(len(feats)), key="live_feat",
                                format_func=lambda i: feats[i][1], captions=[f[2] for f in feats],
                                label_visibility="collapsed")
                row_i = sid_to_i[feats[pick][0]]
            else:
                row_i = st.selectbox("Disrupted row", range(len(d)), key="live_row", format_func=lambda i: d.LABEL[i],
                                     help="Every row of flights.csv with WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20.")
            base_world = world_for(int(row_i))

            H('<div class="h3" style="margin-top:18px">2 · Choose the agents</div>')
            backend = st.segmented_control("Agents", ["Rule-based", "Claude"], default="Rule-based", key="live_backend",
                                           label_visibility="collapsed") or "Rule-based"
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            model = pilot_model = None
            claude_ok = has_key
            if backend == "Claude":
                passcode = os.environ.get("CLAUDE_PASSCODE")
                used = st.session_state.get("claude_runs", 0)
                if not has_key:
                    st.warning("Claude agents are not switched on for this copy of the dashboard. Use Rule-based, "
                               "or open Replays to see recorded Claude runs.")
                elif passcode and st.text_input("Passcode for Claude runs", type="password", key="live_pass",
                                                help="Given with the submission. It protects the API credit.") != passcode:
                    claude_ok = False
                    st.caption("Enter the passcode to run Claude agents. Rule-based runs and Replays need none.")
                elif used >= MAX_CLAUDE_RUNS:
                    claude_ok = False
                    st.caption(f"This session has used its {MAX_CLAUDE_RUNS} Claude runs. Reload the page for more.")
                else:
                    st.caption(f"About 7 Claude calls and 15–20 s per run · {MAX_CLAUDE_RUNS - used} runs left "
                               f"in this session.")
                with st.expander("Models"):
                    model = st.text_input("ATC, Ground, summariser", os.environ.get("ATM_MODEL", "claude-sonnet-5-5"))
                    pilot_model = st.text_input("Pilot", os.environ.get("ATM_PILOT_MODEL", "claude-haiku-5-5"))
                pace = 0.0
            else:
                speed = st.select_slider("Reading speed", ["Fast", "Normal", "Slow"], value="Normal", key="live_speed",
                                         help="Rule-based agents answer in milliseconds, so their messages are "
                                              "spaced out for reading. Claude runs stream in real time.")
                pace = {"Fast": 0.2, "Normal": 0.65, "Slow": 1.2}[speed]

            H('<div class="h3" style="margin-top:18px">3 · Add constraints (optional)</div>')
            with st.expander("What-if: change the situation"):
                open_rw = [r.id for r in base_world.airport.runways if r.open]
                if len(open_rw) > 1:
                    close = st.multiselect("Close runways", open_rw, key="wi_rw", max_selections=len(open_rw) - 1,
                                           help="At least one runway must stay open.")
                else:
                    close = []
                    st.caption(f"Only runway {open_rw[0] if open_rw else '–'} is open, so none can be closed.")
                free_tw = [t.id for t in base_world.airport.taxiways if not t.blocked]
                block = st.multiselect("Block taxiways", free_tw, key="wi_tw")
                cut = st.slider("Less fuel on board (min)", 0, 45, 0, 5, key="wi_fuel")
                spacing = st.slider("Low visibility: extra spacing per landing (min)", 0, 3, 0, key="wi_sp")
            chaos = st.selectbox("Inject a fault", list(CHAOS_MODES), key="live_chaos",
                                 format_func=lambda k: "None" if k == "none" else f"{k}: {CHAOS_MODES[k]}",
                                 help="Corrupts one agent's real message to prove the guardrails catch it.")
            wi = WhatIf(close_runways=close, block_taxiways=block, fuel_cut_min=cut, extra_spacing_min=spacing)
            world = apply_whatif(base_world, wi)
            run = st.button("Run negotiation", type="primary", width="stretch",
                            disabled=backend == "Claude" and not claude_ok)

    cfg_key = f"{world.scenario_id}_{backend}_{chaos}_{'-'.join(wi.describe()) or 'base'}".replace(" ", "")
    with right:
        with st.container(key="card_situation"):
            H(situation_html(world))
        stored = st.session_state.get("live_run")
        if run:
            llm = None
            if backend == "Claude":
                from atm.llm import build_llms
                llm = build_llms(model, pilot_model)
                st.session_state.claude_runs = st.session_state.get("claude_runs", 0) + 1
            chat = chat_frame("live", live=True)
            lr = LiveRun(world, llm=llm, chaos=chaos).start()
            chat.t0 = lr.t0
            for ev in lr.iter_events():
                if ev is None:
                    chat.tick()
                    continue
                if pace and ev[0] == "said":
                    end = time.monotonic() + pace
                    while time.monotonic() < end:
                        chat.tick()
                        time.sleep(0.1)
                chat.feed(ev)
                if pace and ev[0] == "log":
                    time.sleep(pace * 0.45)
            secs = time.monotonic() - chat.t0
            chat.draw_tracker(f"Finished in {secs:.1f} s")
            if lr.result is not None:
                st.session_state.live_run = {"key": cfg_key, "events": chat.events, "res": lr.result,
                                             "world": world, "secs": secs}
                decision_panel(lr.result, world, chat, cfg_key)
        elif stored and stored["key"] == cfg_key:
            chat = chat_frame("live", live=True)
            feed_all(chat, stored["events"])
            chat.finish()
            chat.draw_tracker(f"Finished in {stored['secs']:.1f} s")
            decision_panel(stored["res"], stored["world"], chat, cfg_key)
        else:
            with st.container(key="card_empty"):
                H('<div class="empty"><div class="avs">' + avatar("ATC") + avatar("PILOT(x)") + avatar("GROUND") +
                  '</div><div class="h3">Ready when you are</div><p class="small">Press <b>Run negotiation</b>. '
                  'You will see each agent think, speak, vote and counter-offer, and every safety check as it '
                  'happens.</p></div>')


# =====================================================================================================
# REPLAYS
# =====================================================================================================
def page_replays():
    H('<div class="sec" style="margin-top:18px">Replay a recorded negotiation.</div>'
      '<p class="sub">Saved consensus logs, played back message by message. The three Claude runs were recorded '
      'live; the others use the rule-based agents to show each guardrail. No API key needed.</p>')
    items = [r for r in REPLAYS if (ROOT / r[0]).exists()]
    left, right = st.columns([1, 2.05], gap="large")
    with left:
        with st.container(key="card_rcontrols"):
            H('<div class="h3">Choose a recording</div>')
            i = st.radio("Recording", range(len(items)), key="rep_pick", format_func=lambda k: items[k][1],
                         captions=[r[2] for r in items], label_visibility="collapsed")
            speed = st.select_slider("Playback speed", ["Fast", "Normal", "Slow"], value="Normal", key="rep_speed")
            tamper = st.toggle("Tamper with entry #5", key="rep_tamper",
                               help="Edits one old entry in memory. The hash chain shows exactly where.")
            play = st.button("Play", type="primary", width="stretch")
    path = ROOT / items[i][0]
    log = ConsensusLog.read_jsonl(str(path))
    if tamper and len(log.entries) > 5:
        log.entries = copy.deepcopy(log.entries)
        log.entries[5]["reason"] += " [edited after the fact]"
    ok, bad = log.verify()
    start = log.entries[0]
    d = start.get("data") or {}
    models = d.get("models") or {}
    with right:
        with st.container(key="card_rhead"):
            H(f'<div class="h3">{esc(items[i][1])}</div><div class="small">{esc(start["reason"])}</div>'
              f'<div style="margin-top:10px">'
              f'<span class="chip {"blue" if d.get("mode") == "llm" else ""}">'
              f'{"Claude agents" if d.get("mode") == "llm" else "Rule-based agents"}</span>'
              + ("".join(f'<span class="chip">{esc(k)}: {esc(v)}</span>' for k, v in models.items() if v)
                 if d.get("mode") == "llm" else "")
              + (f'<span class="chip bad">fault: {esc(d.get("chaos"))}</span>' if d.get("chaos") not in (None, "none") else "")
              + (f'<span class="chip ok">✓ Hash chain verified · {len(log.entries)} entries</span>' if ok else
                 f'<span class="chip bad">⛔ Hash chain broken at entry #{bad}: the log was edited</span>')
              + "</div>")
        chat = chat_frame("replay", live=False)
        pace = {"Fast": 0.15, "Normal": 0.45, "Slow": 0.9}[speed] if play else 0.0
        feed_all(chat, [("log", e) for e in log.entries], pace)
        chat.finish()


# =====================================================================================================
# EVALUATION
# =====================================================================================================
METRICS = [("connections_missed", "Connecting passengers who miss their flight"),
           ("crew_overtime_min", "Ground-crew overtime (min)"),
           ("holding_min", "Airborne holding (min)"), ("holding_fuel_kg", "Fuel burned holding (kg)"),
           ("next_leg_delay_min", "Delay to the aircraft's next leg (min)"),
           ("apron_wait_min", "Waiting on the apron for a gate (min)"), ("gate_changes", "Gate changes"),
           ("flights_below_reserve", "Flights below fuel reserve"), ("diversions", "Diversions"),
           ("hard_conflicts", "Safety-rule conflicts in the plan")]


def metric_tiles(df: pd.DataFrame):
    out = []
    for m, label in METRICS:
        b, n = df["base_" + m], df["neg_" + m]
        tb, tn = b.sum(), n.sum()
        better, worse = int((n < b).sum()), int((n > b).sum())
        if tn < tb:
            pct = (tb - tn) / tb
            chip, col = f'<span class="chip ok">↓ {"<1%" if pct < .01 else f"{pct:.0%}"}</span>', "#1d8a3a"
        elif tn > tb:
            chip, col = f'<span class="chip warn">↑ {(tn - tb) / max(tb, 1):.0%} worse</span>', "#b25000"
        else:
            chip, col = '<span class="chip">no change</span>', "#86868b"
        mx = max(tb, tn, 1)
        out.append(
            f'<div class="mt"><div class="l">{esc(label)}</div><div class="row"><span class="a">{num(tb)}</span>'
            f'<span class="b">{num(tn)}</span>{chip}</div>'
            f'<div class="bar"><span style="width:{tb / mx * 100:.1f}%;background:#c7c7cc"></span></div>'
            f'<div class="bar"><span style="width:{tn / mx * 100:.1f}%;background:{col}"></span></div>'
            f'<div class="small">Better in {better} · worse in {worse} scenarios</div></div>')
    H('<div class="grid">' + "".join(out) + "</div>")


def page_evaluation():
    if results is None:
        st.info("No evaluation yet. Run `python evaluate.py --data data/disruptions.csv` first.")
        return
    H('<div class="sec" style="margin-top:18px">Does it converge, safely?</div>'
      f'<p class="sub">Each of the {OFF["n"]} scenarios is one disrupted row of flights.csv. Rule-based agents ran '
      f'all of them; Claude agents ran {LLM["n"] if LLM else 0} of the same rows.</p>')
    rows = []
    for name, s in (("Rule-based agents", OFF), ("Claude agents", LLM)):
        if s:
            rows.append({"a": f"<b>{name}</b>", "n": num(s["n"]), "ag": f'{s["agree"]:.0%}', "fb": f'{s["fallback"]:.0%}',
                         "t": f'{s["turns"]:.2f}', "mx": num(s["max_turns"]), "es": num(s["escalated"]),
                         "fc": num(s["final_conf"]), "bc": num(s["base_conf"])})
    H('<div class="box">' + table_html(rows, [("a", "Agents"), ("n", "Scenarios"), ("ag", "Agreement"),
                                               ("fb", "Fallback"), ("t", "Mean turns"), ("mx", "Max turns"),
                                               ("es", "Escalated"), ("fc", "Final plans breaking a rule"),
                                               ("bc", "Today's plans breaking a rule")],
                                       numeric=("n", "ag", "fb", "t", "mx", "es", "fc", "bc")) + "</div>")

    H('<div class="sec">Today\'s process vs negotiated.</div><p class="sub">Today\'s process is modelled as each team '
      'planning alone: first-come-first-served landing, standard route, Ground\'s first-choice gate and crew, '
      'nothing cross-checked. Grey is today, colour is negotiated. Lower is better everywhere.</p>')
    like = results[results.base_hard_conflicts == 0]
    scope = st.segmented_control("Scope", [f"All {len(results)} scenarios",
                                           f"Like-for-like: {len(like)} where today's plan is legal"],
                                 default=f"All {len(results)} scenarios", key="ev_scope",
                                 label_visibility="collapsed") or "All"
    use_like = scope.startswith("Like")
    metric_tiles(like if use_like else results)
    H('<p class="small" style="margin-top:10px">Like-for-like drops the scenarios where today\'s plan only works by '
      'breaking a rule (for example taxiing through a blocked taxiway), because those plans flatter their own numbers. '
      'Apron waiting is the one measure that gets worse: negotiated plans more often keep an aircraft waiting on the '
      'apron for a free gate, in both scopes.</p>')

    if chaos is not None:
        inj, caught, leaked = chaos_totals()
        H(f'<div class="sec">{caught} of {inj} injected faults caught.</div><p class="sub">Each fault corrupts one agent\'s '
          f'real message in {chaos.scenario.nunique()} scenarios. The guardrail that owns it must fire, and the final '
          f'plan must still pass every hard constraint. {leaked} faults reached a final plan.</p>')
        g = chaos.groupby("mode")
        expected = {"fuel": "Checker · FUEL_RESERVE", "ghost_gate": "Checker · GATE_UNKNOWN",
                    "wake": "Checker · WAKE_SEPARATION", "urgency": "Arbiter · DOWNWEIGHT",
                    "role": "Protocol · PERMISSION_DENIED"}
        rows = []
        for mode, x in g:
            a = int(x.applicable.sum())
            rows.append({"m": f"<b>{esc(mode)}</b>", "w": esc(CHAOS_MODES.get(mode, "")), "g": esc(expected.get(mode, "")),
                         "i": f"{a}/{len(x)}", "c": f'<span class="chip ok">{int((x.caught & x.applicable).sum())}/{a}</span>',
                         "s": f"{int((x.final_hard_conflicts == 0).sum())}/{len(x)}"})
        H('<div class="box">' + table_html(rows, [("m", "Fault"), ("w", "What is injected"), ("g", "Guardrail"),
                                                   ("i", "Applicable"), ("c", "Caught"), ("s", "Final plan safe")],
                                           numeric=("i", "c", "s")) +
          '<p class="small" style="margin-top:10px">Not applicable: a wake squeeze needs other traffic in the window; '
          'a CRITICAL claim is not false when the flight really is HIGH or CRITICAL.</p></div>')

    if LLM:
        ids = set(llm_results.scenario)
        off = results[results.scenario.isin(ids)]
        H('<div class="sec">Claude agents vs rule-based, same flights.</div><p class="sub">The same '
          f'{len(ids)} disrupted rows, negotiated once by Claude agents and once by rule-based agents.</p>')

        def row(label, f, fmt_=lambda v: num(v)):
            return {"m": esc(label), "c": fmt_(f(llm_results)), "o": fmt_(f(off))}
        rows = [row("Agreement", lambda x: (x.decided_by == "consensus").mean(), lambda v: f"{v:.0%}"),
                row("Mean turns", lambda x: x.rounds.mean(), lambda v: f"{v:.2f}"),
                row("Final plans breaking a rule", lambda x: (x.neg_hard_conflicts > 0).sum()),
                row("Connections missed", lambda x: x.neg_connections_missed.sum()),
                row("Crew overtime (min)", lambda x: x.neg_crew_overtime_min.sum()),
                row("Next-leg delay (min)", lambda x: x.neg_next_leg_delay_min.sum()),
                row("Apron wait (min)", lambda x: x.neg_apron_wait_min.sum())]
        c1, c2 = st.columns([1.4, 1], gap="medium")
        H('<div class="box">' + table_html(rows, [("m", "Measure"), ("c", "Claude"), ("o", "Rule-based")],
                                           numeric=("c", "o")) + "</div>", c1)
        tok = (LLM["input_tokens"] or 0) + (LLM["output_tokens"] or 0)
        H(f'<div class="box"><div class="h3">Cost of the Claude runs</div><div class="spec">'
          f'<div><div class="k">AI calls</div><div class="val">{num(LLM["llm_calls"])}</div></div>'
          f'<div><div class="k">Failed calls</div><div class="val">{num(LLM["llm_failed"])}</div></div>'
          f'<div><div class="k">Unsafe proposals caught</div><div class="val">{num(LLM["checker_rejects"])}</div></div>'
          f'<div><div class="k">Tokens</div><div class="val">{num(tok)}</div></div>'
          f'<div><div class="k">Seconds per scenario</div><div class="val">{LLM["seconds"] / LLM["n"]:.1f}</div></div>'
          f'</div><p class="small" style="margin-top:12px">ATC, Ground and the summariser run on Claude Sonnet, the '
          f'pilot on Claude Haiku. Each agent is a separate call with its own prompt.</p></div>', c2)

    if flights_eval is not None:
        paid = flights_eval[(flights_eval.gate_in_later_min > 0) | (flights_eval.conn_missed_neg > flights_eval.conn_missed_base)]
        paid = paid.sort_values("gate_in_later_min", ascending=False)
        H(f'<div class="sec">Where it does worse.</div><p class="sub">Totals improve while individual flights can give '
          f'something up. {len(paid)} of {len(flights_eval)} flights reached the gate later or lost more connections '
          f'than under today\'s process. In most, today\'s plan was only possible by breaking a rule.</p>')
        rows = [{"s": esc(r.scenario), "f": f"<b>{esc(r.flight)}</b>", "p": esc(r.priority),
                 "g": num(r.gate_in_later_min), "c": f"{num(r.conn_missed_base)} → {num(r.conn_missed_neg)}",
                 "fu": f"{num(r.fuel_margin_base)} → {num(r.fuel_margin_neg)}",
                 "l": '<span class="chip">yes</span>' if str(r.today_legal) == "True" else '<span class="chip warn">no</span>'}
                for r in paid.itertuples()]
        H('<div class="box">' + table_html(rows, [("s", "Scenario"), ("f", "Flight"), ("p", "Priority"),
                                                   ("g", "At gate later (min)"), ("c", "Connections missed"),
                                                   ("fu", "Fuel margin (min)"), ("l", "Today's plan legal?")],
                                           numeric=("g",)) + "</div>")
        st.button("Replay the worst case", on_click=goto, args=("Replays",),
                  kwargs={"rep_pick": next((k for k, r in enumerate(REPLAYS) if "worse_case" in r[0]), 0)})
    H('<p class="small" style="margin-top:16px">Caveats: fuel, gates, crews and connections are simulated (seeded per '
      'scenario); other traffic\'s landing slots are fixed, so results are conservative; today\'s process is modelled, '
      'not observed.</p>')


# =====================================================================================================
# REQUIREMENTS
# =====================================================================================================
def page_requirements():
    req = pd.read_csv(ROOT / "dashboard_content" / "requirements.csv")
    H('<div class="sec" style="margin-top:18px">The brief, line by line.</div><p class="sub">Every requirement from '
      'the Challenge 2 brief, its "what to avoid" list and the rules of play in the organisers\' workbook, with how '
      'it is met and where to see it in this dashboard.</p>')
    for grp, title in (("Challenge brief", "From the challenge brief"), ("Avoid", "What the brief says to avoid"),
                       ("Workbook rules", "Rules of play (organisers' workbook)")):
        sub = req[req.group == grp]
        H(f'<div class="h3" style="margin:26px 0 10px">{esc(title)}</div><div class="box">' + "".join(
            f'<div class="req"><div class="tick">✓</div><div><p class="t"><span class="chip">{esc(r.id)}</span> '
            f'{esc(r.requirement)}</p><p class="h">{esc(r.how_met)}</p>'
            f'<span class="chip blue">See it: {esc(r.where_to_see)}</span>'
            f'{"" if str(r.status) == "Met" else " <span class=" + chr(34) + "chip warn" + chr(34) + ">" + esc(r.status) + "</span>"}'
            f'</div></div>' for r in sub.itertuples()) + "</div>")
    rub = pd.read_csv(ROOT / "dashboard_content" / "rubric.csv")
    H('<div class="sec">How it maps to the judging criteria.</div>')
    rows = [{"c": f"<b>{esc(r.criterion)}</b>", "w": f"{int(r.weight_pct)}%", "j": esc(r.judges_look_for),
             "d": esc(r.where_in_dashboard)} for r in rub.itertuples()]
    H('<div class="box">' + table_html(rows, [("c", "Criterion"), ("w", "Weight"), ("j", "Judges look for"),
                                               ("d", "Where in the dashboard")], numeric=("w",)) + "</div>")


# =====================================================================================================
# RULES & LIMITS
# =====================================================================================================
@st.cache_data(show_spinner=False)
def workbook():
    x = pd.ExcelFile(WORKBOOK)
    cap = pd.read_excel(x, "airport_capacity", header=2)
    wake = pd.read_excel(x, "wake_separation", header=2)
    fuel = pd.read_excel(x, "fuel_and_crew", header=0)
    rules = pd.read_excel(x, "rules_of_play", header=0)
    return cap, wake, fuel, rules


def page_limits():
    cap, wake, fuel, rules = workbook()
    H('<div class="sec" style="margin-top:18px">The rules the agents cannot bend.</div><p class="sub">Read from '
      'the organisers\' workbook (challenge2_operational_constraints.xlsx) at run time by atm/limits.py. Nothing '
      'here is retyped.</p>')
    H('<div class="box">' + table_html(
        [{"t": f"<b>{esc(r.topic)}</b>", "d": esc(r.detail)} for r in rules.itertuples()],
        [("t", "Rule of play"), ("d", "Detail")]) + "</div>")

    H('<div class="sec">Look up an airport.</div>')
    codes = sorted(cap.IATA_CODE.dropna().astype(str))
    code = st.selectbox("Airport", codes, index=codes.index("ATL") if "ATL" in codes else 0, key="lim_ap",
                        format_func=lambda c: f"{c} · {cap.set_index('IATA_CODE').AIRPORT.get(c, '')}")
    lt = limits_table(load_limits(), code)
    H('<div class="box">' + table_html([{"k": esc(k), "v": f"<b>{esc(v)}</b>"} for k, v in lt],
                                       [("k", "Limit"), ("v", "Value used")]) + "</div>")

    c1, c2 = st.columns(2, gap="medium")
    cls = cap.groupby("size_class", sort=False).agg(airports=("IATA_CODE", "count"), runways=("active_runways", "first"),
                                                    ops=("max_runway_ops_per_hour", "first"), gates=("gates", "first"),
                                                    spare=("spare_gates_at_any_time", "first"),
                                                    turn=("min_turnaround_min", "first")).reset_index()
    H('<div class="box"><div class="h3">Capacity by airport size</div>' + table_html(
        [{"s": f"<b>{esc(r.size_class)}</b>", "a": num(r.airports), "r": num(r.runways), "o": num(r.ops),
          "g": num(r.gates), "sp": num(r.spare), "t": num(r.turn)} for r in cls.itertuples()],
        [("s", "Size"), ("a", "Airports"), ("r", "Runways"), ("o", "Ops/h"), ("g", "Gates"), ("sp", "Spare"),
         ("t", "Turn (min)")], numeric=("a", "r", "o", "g", "sp", "t")) + "</div>", c1)
    H('<div class="box"><div class="h3">Wake separation</div>' + table_html(
        [{"l": esc(r[0]), "f": esc(r[1]) if pd.notna(r[1]) else "(any)", "n": num(r[2])} for r in wake.itertuples(index=False)],
        [("l", "Leader"), ("f", "Follower"), ("n", "Min. (nm)")], numeric=("n",)) + "</div>", c2)
    H('<div class="box" style="margin-top:14px"><div class="h3">Fuel and crew</div>' + table_html(
        [{"r": f"<b>{esc(r.rule)}</b>", "v": esc(r.value), "a": esc(r.applies_to)} for r in fuel.itertuples()],
        [("r", "Rule"), ("v", "Value"), ("a", "Applies to")]) + "</div>")

    H('<div class="sec">Stated assumptions.</div><p class="sub">The workbook asks teams to state any limit they '
      'change or interpret. These are ours.</p>')
    H('<div class="box">' + "".join(f'<div class="req"><div class="tick" style="background:#e8f1fd;color:#0071e3">'
                                    f'{i + 1}</div><div><p class="h" style="color:#1d1d1f;margin:3px 0 0">{esc(a)}</p>'
                                    f'</div></div>' for i, a in enumerate(ASSUMPTIONS)) + "</div>")


# =====================================================================================================
{"Overview": page_overview, "Live": page_live, "Replays": page_replays, "Evaluation": page_evaluation,
 "Requirements": page_requirements, "Rules & limits": page_limits}[page]()

H('<div class="foot">Data: 2015 US DOT flights.csv (real flights, times and delays) · organisers\' operational '
  'constraints workbook · fuel, passengers, gates, crews and routes simulated and seeded per scenario.<br>'
  'Airbus track, Challenge 2 · Autonomous SG 2026</div>')
