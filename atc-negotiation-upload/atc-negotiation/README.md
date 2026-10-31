# Multi-Agent Disruption Negotiation (Challenge #2)

Pick one disrupted row of the 2015 DOT `flights.csv`. Three agents, **ATC**, **Pilot** and
**Ground**, negotiate a revised arrival time, gate and flight plan for that flight within
**3 turns** (one structured message from each agent per turn). Every proposal is checked by
deterministic safety code against the organisers' constraints workbook; with no agreement after
3 turns, a deterministic non-AI fallback decides and the log says why. Every step is written to a
hash-chained consensus log.

## Quick start

```bash
pip install -r requirements.txt

python run.py --list                      # disrupted rows (WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20)
python run.py --row 0                     # that row is the disruption: ATC + Pilot + Ground
python run.py --row 0 --chaos fuel        # inject a fault, watch it get caught
python evaluate.py                        # today's process vs negotiated, every disrupted row
streamlit run app.py                      # demo UI (live run or replay)
python -m pytest -q                       # 31 tests

# Claude agents (live API, structured outputs via client.messages.parse)
export ANTHROPIC_API_KEY=...
python run.py --row 0 --llm               # ATC/Ground/summariser: claude-sonnet-5-5, pilot: claude-haiku-5-5
python run.py --row 0 --llm --model <id> --pilot-model <id>   # or ATM_MODEL / ATM_PILOT_MODEL
```

**Data.** `data/sample_flights.csv` holds made-up rows in the exact `flights.csv` format so
everything runs immediately. With the real file (put it in `data/`):

```bash
python scripts/prepare_data.py data/flights.csv --sample 300   # -> data/disruptions.csv
python run.py --data data/disruptions.csv --list
python evaluate.py --data data/disruptions.csv --limit 300
```

`prepare_data.py` keeps every arrival, disrupted or not, at each sampled airport and hour plus
the hours either side. The non-disrupted ones are the other traffic that takes landing slots.

## Challenge rules → where they are met

| Rule (brief, clarifications, workbook) | Implementation |
|---|---|
| Input: one row with WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20 is the disruption | `run.py --row N`, `scenario.disrupted()` |
| Three agents: ATC, Pilot, Ground, separate domains | `agents.py`: one LLM call per agent, own system prompt and private view |
| One turn = each agent sends one structured message; consensus within 3 turns | `protocol.py`; messages are Pydantic schemas (`schemas.py`) |
| No consensus after 3 turns → deterministic fallback, log says why | `fallback.py`; `FALLBACK` log entry with the reason |
| Explicit constraint checking | `constraints.py`, on every candidate and again before commit |
| Use the workbook's limits; state any changed limit | `limits.py` loads `data/challenge2_operational_constraints.xlsx`; assumptions listed below and in the app |
| Output: revised arrival time, gate or flight plan + consensus log of each turn | decision per flight; `log.py` JSONL with turns `T1`–`T3`, hash-chained |
| ATC: wake separation, taxiway blockage, throughput | `WAKE_SEPARATION`, `TAXIWAY_BLOCKED`, `RUNWAY_CAPACITY` checks; ATC votes on holding and makespan |
| Pilot: fuel reserve, safety, turbulence, connections | `FUEL_RESERVE`, `SEVERE_TURBULENCE`, `CREW_DUTY` checks; route levers trade turbulence, time and fuel |
| Ground: gate utilisation, crew overtime, turnaround | planned gate + spares, `GATE_CHANGE_NOTICE`, 8 h shifts, overtime ×1.5, next-leg delay |
| **Avoid:** one prompt for three roles / endless loops / ignoring safety | separate calls; 3-turn limit; LLMs never judge hard constraints (chaos tests prove it) |

## Operational limits (organisers' workbook)

Loaded from `data/challenge2_operational_constraints.xlsx` by `atm/limits.py`:

| Workbook value | Used as |
|---|---|
| `active_runways`, `max_runway_ops_per_hour` (per airport) | runways modelled; hourly landing cap (`RUNWAY_CAPACITY`) |
| `spare_gates_at_any_time` | Ground has the flight's planned gate plus this many spares |
| `min_turnaround_min` | turnaround at that airport |
| Wake separation (FAA JO 7110.65, simplified, nm) | ATC spacing between consecutive landings |
| Final reserve 30 min; spare fuel on arrival 45 min | fuel on arrival; `FUEL_RESERVE` |
| Holding burn 2,400 / 6,500 kg/h | fuel cost of holding in the evaluation |
| Flight crew max duty 13 h | `CREW_DUTY` |
| Ground crew shift 8 h, overtime 1.5× | crew shifts; overtime cost |
| Gate change notice ≥ 20 min | `GATE_CHANGE_NOTICE` |

**Stated assumptions** (where the workbook does not say how a value applies; also shown in the app):

- Wake separation (nm) is converted to minutes at 140 kt on final approach, rounded up.
- 50% of `max_runway_ops_per_hour` are arrivals; the cap scales with runways open.
- Weather is the airport's: the worst WEATHER_DELAY among arrivals in the window. > 15 closes one
  runway (at least one stays open), > 45 closes half; both add 1 min to every landing separation.
- Fuel on arrival = 30 min reserve + 45 min spare, minus 50% of WEATHER_DELAY assumed already spent
  holding (spare never below 5 min).
- Flight-crew duty starts 60 min before this leg's scheduled departure (earlier legs aren't in the data).
- Ground has each flight's planned gate plus the spare gates; spare gate sizes are simulated.
- Ground crews work 8 h shifts with simulated, staggered starts.
- The negotiation runs 30 min before the first arrival.
- An occupied gate means waiting on the apron (not airborne holding); a busy crew delays the
  turnaround. Neither is a hard violation; both are costed.
- Super (A380) wake rows are unused: no A380s in the 2015 US domestic data.

## Architecture

```
flights.csv row ──► Scenario Builder ◄── constraints workbook
                          │
                          ▼
                    World State          airport (runways, capacity, gates, crews), the flight,
                          │              other real traffic holding landing slots
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Pilot agent     ATC agent      Ground agent      separate calls, prompts, private views
          └───────────────┼───────────────┘
                          ▼
                   Priority Arbiter      urgency from world state (fuel first), not from claims
                          │
   ┌──── Negotiation Protocol (≤ 3 turns) ───────────────────────────┐
   │  proposals ──► Constraint Checker ──► accept / reject / counter  │
   │                (pure Python)          mediator repairs & merges  │
   └──────────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
       consensus ✔              no consensus / agent error / invalid output
            │                            │
            ▼                            ▼
     Final decision            Deterministic Fallback Solver
            └─────────────┬──────────────┘
                          ▼
                     Summariser          briefing, no decision power
                          ▼
           Consensus Log (append-only JSON, hash-chained)
```

| File | Component |
|---|---|
| `atm/limits.py` | Loads the organisers' workbook; lists stated assumptions |
| `atm/scenario.py` | Scenario builder: the row, other traffic, airport from the workbook |
| `atm/schemas.py` | Every message type (Pydantic) and the role → lever permission table |
| `atm/agents.py` | ATC, Pilot, Ground: private views, prompts, LLM and offline backends |
| `atm/arbiter.py` | Urgency tiers (fuel first), claim verification, the single cost function |
| `atm/constraints.py` | Hard-constraint checker and per-flight metrics |
| `atm/protocol.py` | The 3-turn protocol, mediator, permission checks, decision |
| `atm/fallback.py` | Deterministic greedy allocator (also used for repairs) |
| `atm/chaos.py` | Fault injection |
| `atm/summariser.py` | Controller briefing |
| `atm/log.py` / `atm/replay.py` | Hash-chained consensus log; replay by turn |
| `atm/llm.py` | Anthropic SDK wrapper: structured outputs, per-role models, usage stats |

## The protocol

**Turn 1, declare.** All three agents run in parallel. The Pilot states its preferred route,
earliest and latest-safe landing, and an urgency claim. ATC proposes a landing time and runway
around the other traffic (it cannot see fuel). Ground proposes a gate and crew (it does not own
the taxiway picture). The arbiter ranks by world state and flags inflated claims. Candidates are
built and checked: the agents' combined plan, a mediator repair of it, and the arbiter-priority plan.

**Turn 2, vote.** Each agent accepts or rejects every feasible candidate with a reason, and may
make one counter-offer using only its own levers, stating what it costs. Counter-offers are
checked; the mediator repairs and merges safe ones into a package.

**Turn 3, final vote.** Only if needed. No new counter-offers; agents see every turn-2 vote.

**Decide.** A feasible candidate accepted by every agent wins (lowest arbiter cost). Otherwise,
or if any agent errors, times out or returns invalid output, the fallback decides. The chosen
plan is re-checked before commit. Checker, arbiter and mediator entries are deterministic code,
not agent messages, so they don't count as turns.

## Evaluation

`python evaluate.py` treats each disrupted row as one scenario and compares **today's process**
(each domain plans alone: first-come-first-served landing around other traffic, standard route,
Ground's first-choice gate and crew, nothing cross-checked) against the **negotiated** plan. It
writes `out/eval/summary.md`, `results.csv` and `flights.csv`: agreement and fallback rates, turns,
holding (minutes and kg of fuel), missed connections, crew overtime, next-leg delay, apron wait,
gate changes, flights below reserve, diversions and hard conflicts, in raw units (never the
arbiter's cost, which the negotiation optimises), plus a case where negotiation does worse and a
flight-level table of who gave something up. Arrival delay is measured at the gate, as in the DOT
data. Use offline agents for large sweeps; `--llm --limit 10` spot-checks Claude agents.
`--mode cluster` evaluates the optional multi-flight extension.

## Real vs simulated

**Real (DOT data):** the disrupted row (date, airline, flight, tail, origin, destination, schedule,
delays, WEATHER_DELAY, LATE_AIRCRAFT_DELAY) and the other arrivals around it, which take landing
slots. Airport and airline names, coordinates for diversion alternates.

**Workbook (synthetic, organisers'):** runways, capacity, gates, turnaround, wake separation,
fuel reserve and spare, burn rates, crew duty and shifts, gate-change notice.

**Simulated (seeded, same input → same world):** aircraft type per tail, passengers and
connections, taxiway layout, spare gate sizes, crew start times, route options and turbulence.
Connections and next departures are placed around the delayed arrival with a spread of slack, so
the negotiation decides outcomes. Times run continuously past midnight (`24:30`).

**Not modelled:** other traffic's landing slots are fixed (real ATC could resequence them, so our
results are conservative); flow control keeps one slot per hour for the negotiating flight.

## Chaos toggle

| Mode | Fault | Caught by |
|---|---|---|
| `fuel` | ATC pushes the flight back 45 min, ignoring fuel | Checker: `FUEL_RESERVE` |
| `ghost_gate` | Ground assigns non-existent gate Z99 | Checker: `GATE_UNKNOWN` |
| `wake` | ATC squeezes arrivals together | Checker: `WAKE_SEPARATION` |
| `urgency` | A healthy flight claims CRITICAL | Arbiter: `DOWNWEIGHT` |
| `role` | Pilot tries to set its own gate | Protocol: `PERMISSION_DENIED` |

## Replay mode (demo insurance)

`streamlit run app.py` → **Replay saved log**. Plays any saved `.jsonl` log turn by turn without
calling the API. **Simulate tampering** edits one old entry in memory and the hash-chain check
turns red.

```bash
mkdir -p demo && cp out/<scenario>_<chaos>.jsonl demo/<name>.jsonl   # keep good live runs
```

## Live LLM agents (Anthropic SDK)

Each agent call uses `client.messages.parse(output_format=<Pydantic model>)`, so the reply is
constrained to the message schema and arrives validated. An API error, timeout, `refusal` or
`max_tokens` stop counts as an agent failure and the fallback decides. Tokens and latency are
logged as a `USAGE` entry. Offline policies receive the same views and return the same schemas,
so everything is testable without a key.

## Optional extension: multi-flight contention

`run.py --cluster N` / `evaluate.py --mode cluster` negotiate several disrupted arrivals at once
(one Pilot agent per flight). Not required by the rules; useful to show competition between flights.

## Cost function weights (`arbiter.py`)

Per flight: `urgency_weight × gate delay + 1.5 × holding + 0.5 × missed connections + turbulence
(0/2/8) + 1.2 × crew overtime (0.8 × workbook's 1.5) + 1.0 × next-leg delay + 3 × minutes of fuel
margin below 15 + 0.5 × apron/crew wait + 5 per gate change`; diversion 400 × urgency weight;
plus `0.3 × makespan` for runway throughput. Priority tiers are set by fuel first, so passenger
counts never outrank a fuel-critical flight.
