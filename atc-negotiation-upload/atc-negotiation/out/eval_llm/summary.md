# Evaluation: 20 disrupted rows (one flight, three agents each) (Claude agents)

Data: `data/disruptions.csv`. Flights negotiated: 20; other traffic taking slots: 70.

## Process

- Agreement (consensus) rate: **100%**
- Fallback rate: **0%**
- Mean turns: 2.05 (limit 3)
- Escalated to a controller (no plan satisfies every hard constraint): **0**
- Baseline plans with at least one hard conflict: **35%**

## Outcomes (totals across all scenarios)

| Metric | Today's process | Negotiated | Better in | Worse in |
|---|---|---|---|---|
| holding min | 5 | 1 | 1 | 1 |
| holding fuel kg | 200 | 40 | 1 | 1 |
| connections missed | 104 | 104 | 0 | 0 |
| crew overtime min | 6 | 0 | 1 | 0 |
| next leg delay min | 23 | 3 | 5 | 0 |
| apron wait min | 0 | 1 | 0 | 1 |
| gate changes | 18 | 18 | 0 | 0 |
| flights below reserve | 0 | 0 | 0 | 0 |
| diversions | 0 | 0 | 0 | 0 |
| hard conflicts | 7 | 0 | 7 | 0 |

## Like for like: the 13 scenarios where today's plan had no hard conflict

In the other scenarios today's plan only works by breaking a rule (e.g. taxiing through a blocked taxiway), so its numbers flatter it.

| Metric | Today's process | Negotiated | Better in | Worse in |
|---|---|---|---|---|
| holding min | 5 | 1 | 1 | 1 |
| holding fuel kg | 200 | 40 | 1 | 1 |
| connections missed | 39 | 39 | 0 | 0 |
| crew overtime min | 6 | 0 | 1 | 0 |
| next leg delay min | 20 | 3 | 4 | 0 |
| apron wait min | 0 | 1 | 0 | 1 |
| gate changes | 11 | 11 | 0 | 0 |
| flights below reserve | 0 | 0 | 0 | 0 |
| diversions | 0 | 0 | 0 | 0 |

## A case where negotiation does worse

**JFK-2015-12-10-3104** (consensus, 2 turns)

| Metric | Today's process | Negotiated |
|---|---|---|
| holding min | 0 | 0 |
| holding fuel kg | 0 | 0 |
| connections missed | 0 | 0 |
| crew overtime min | 6 | 0 |
| next leg delay min | 0 | 0 |
| apron wait min | 0 | 1 |
| gate changes | 1 | 1 |
| flights below reserve | 0 | 0 |
| diversions | 0 | 0 |
| hard conflicts | 0 | 0 |
| min fuel margin | 45 | 45 |

Read the consensus log for this scenario to see which trade-off the agents made (typically: fuel safety, wake spacing or another flight's connections were protected at this metric's expense, or the baseline only looks better because its plan contains a hard conflict).

## Who paid for it (flight level)

0 of 20 flights reached the gate later or lost more connections than under today's process. Totals can improve while individual flights give something up, usually time traded for fuel margin, or a slot ceded to a more urgent flight:

| Scenario | Flight | Priority | Gate-in later (min) | Connections missed (today → negotiated) | Fuel margin (today → negotiated) | Today's plan legal? |
|---|---|---|---|---|---|---|
| none | | | | | | |

## Claude agents vs offline agents (same 20 scenarios)

| | Claude agents | Offline agents |
|---|---|---|
| Consensus (no fallback) | 20/20 | 20/20 |
| Fallback decided | 0/20 | 0/20 |
| Escalated to a controller | 0 | 0 |
| Mean turns (limit 3) | 2.05 | 2.00 |
| Final plans with a hard conflict | 0 | 0 |
| Unsafe proposals caught by the checker | 8 | 7 |
| Agent errors (bad/failed output -> handled) | 0 | 0 |
| Holding (min, total) | 1 | 1 |
| Connections missed (total) | 104 | 104 |
| Crew overtime (min, total) | 0 | 0 |
| Apron wait (min, total) | 1 | 0 |
| Gate changes (total) | 18 | 18 |
| Mean wall time per scenario (s) | 15.0 | 0.0 |

- Same final plan as the offline agents: 15/20
- LLM calls: 143 (0 failed), 7.2 per scenario
- Tokens: 315,455 in / 69,341 out (19,240 per scenario); check the Anthropic Console usage page for the exact cost
- Slowest scenario: 21.2 s

Scenarios where Claude agents needed more turns or the fallback (read these logs):

- DFW-2015-02-16-1209: Claude consensus in 3 turns, offline consensus in 2

Caveats: offline agents unless stated; simulated fuel, gates, crews and connections (see README); baseline conflicts are counted, not resolved.