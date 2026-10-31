# Chaos sweep: 100 scenarios x 5 fault type(s) (offline agents)

Data: `data/disruptions.csv`. Each fault is injected into one agent's real message; the guardrail that owns it must fire in the consensus log (caught) and the final plan must still pass every hard constraint (contained).

| Fault | What is injected | Guardrail expected | Injected | Caught | Final plan safe | Same plan as clean run | Fallback |
|---|---|---|---|---|---|---|---|
| fuel | ATC ignores fuel and pushes the most fuel-critical flight back 45 min | CHECKER REJECT FUEL_RESERVE | 100/100 | 100/100 | 100/100 | 99/100 | 0 |
| ghost_gate | Ground hallucinates a gate that does not exist (Z99) | CHECKER REJECT GATE_UNKNOWN | 100/100 | 100/100 | 100/100 | 99/100 | 0 |
| wake | ATC squeezes two arrivals 1 min apart on the same runway | CHECKER REJECT WAKE_SEPARATION | 81/100 | 81/81 | 100/100 | 99/100 | 0 |
| urgency | A pilot with plenty of fuel claims CRITICAL urgency to jump the queue | ARBITER DOWNWEIGHT | 95/100 | 95/95 | 100/100 | 100/100 | 0 |
| role | A pilot tries to reassign its own gate (not a pilot lever) | PROTOCOL PERMISSION_DENIED | 100/100 | 100/100 | 100/100 | 100/100 | 0 |

**476 faults injected, 476 caught, 0 reached a final plan.**

Not counted: wake 19 (no other traffic in the window to squeeze behind)
Not counted: urgency 5 (the flight really was HIGH or CRITICAL, so a CRITICAL claim is true or within the one-level tolerance (priority is set from world state either way))

"Same plan as clean run" below 100% is not a failure: when the rejected proposal was one an agent preferred, the agents settle on the next-best legal plan (read chaos.csv for which).

Scenarios that were already escalated without any fault (no legal plan exists) are counted as safe: the system hands them to a controller rather than deciding.