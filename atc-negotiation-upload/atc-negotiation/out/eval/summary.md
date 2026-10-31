# Evaluation: 300 disrupted rows (one flight, three agents each) (offline agents)

Data: `data/disruptions.csv`. Flights negotiated: 300; other traffic taking slots: 1202.

## Process

- Agreement (consensus) rate: **100%**
- Fallback rate: **0%**
- Mean turns: 2.01 (limit 3)
- Escalated to a controller (no plan satisfies every hard constraint): **0**
- Baseline plans with at least one hard conflict: **25%**

## Outcomes (totals across all scenarios)

| Metric | Today's process | Negotiated | Better in | Worse in |
|---|---|---|---|---|
| holding min | 37 | 12 | 7 | 8 |
| holding fuel kg | 2915 | 889 | 7 | 8 |
| connections missed | 2604 | 1807 | 37 | 4 |
| crew overtime min | 523 | 60 | 32 | 1 |
| next leg delay min | 657 | 602 | 32 | 7 |
| apron wait min | 862 | 1121 | 24 | 46 |
| gate changes | 264 | 263 | 5 | 4 |
| flights below reserve | 0 | 0 | 0 | 0 |
| diversions | 0 | 0 | 0 | 0 |
| hard conflicts | 74 | 0 | 74 | 0 |

## Like for like: the 226 scenarios where today's plan had no hard conflict

In the other scenarios today's plan only works by breaking a rule (e.g. taxiing through a blocked taxiway), so its numbers flatter it.

| Metric | Today's process | Negotiated | Better in | Worse in |
|---|---|---|---|---|
| holding min | 37 | 9 | 7 | 5 |
| holding fuel kg | 2915 | 701 | 7 | 5 |
| connections missed | 2234 | 1411 | 33 | 0 |
| crew overtime min | 433 | 47 | 26 | 0 |
| next leg delay min | 546 | 451 | 24 | 1 |
| apron wait min | 781 | 848 | 19 | 30 |
| gate changes | 190 | 194 | 0 | 4 |
| flights below reserve | 0 | 0 | 0 | 0 |
| diversions | 0 | 0 | 0 | 0 |

## A case where negotiation does worse

**PDX-2015-03-02-1458** (consensus, 2 turns)

| Metric | Today's process | Negotiated |
|---|---|---|
| holding min | 0 | 0 |
| holding fuel kg | 0 | 0 |
| connections missed | 0 | 51 |
| crew overtime min | 0 | 13 |
| next leg delay min | 0 | 11 |
| apron wait min | 0 | 27 |
| gate changes | 1 | 0 |
| flights below reserve | 0 | 0 |
| diversions | 0 | 0 |
| hard conflicts | 1 | 0 |
| min fuel margin | 45 | 42 |

Read the consensus log for this scenario to see which trade-off the agents made (typically: fuel safety, wake spacing or another flight's connections were protected at this metric's expense, or the baseline only looks better because its plan contains a hard conflict).

## Who paid for it (flight level)

12 of 300 flights reached the gate later or lost more connections than under today's process. Totals can improve while individual flights give something up, usually time traded for fuel margin, or a slot ceded to a more urgent flight:

| Scenario | Flight | Priority | Gate-in later (min) | Connections missed (today → negotiated) | Fuel margin (today → negotiated) | Today's plan legal? |
|---|---|---|---|---|---|---|
| PDX-2015-03-02-1458 | UA1436 | LOW | 38 | 0 → 51 | 45 → 42 | no |
| IAD-2015-07-22-1725 | F9212 | LOW | 31 | 0 → 44 | 45 → 39 | no |
| JAX-2015-05-26-2521 | DL1090 | MEDIUM | 15 | 0 → 30 | 45 → 40 | no |
| PDX-2015-08-11-2201 | WN141 | MEDIUM | 40 | 0 → 25 | 45 → 40 | no |
| OAK-2015-04-02-1858 | WN750 | LOW | 36 | 0 → 0 | 45 → 39 | no |
| MHT-2015-07-30-1843 | WN864 | CRITICAL | 32 | 0 → 0 | 5 → 5 | no |
| BNA-2015-06-30-2211 | MQ3607 | MEDIUM | 16 | 7 → 7 | 45 → 40 | no |
| ATL-2015-09-03-1852 | DL1510 | LOW | 3 | 0 → 0 | 43 → 43 | no |

Caveats: offline agents unless stated; simulated fuel, gates, crews and connections (see README); baseline conflicts are counted, not resolved.