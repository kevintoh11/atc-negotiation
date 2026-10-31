# Dashboard

```
streamlit run dashboard.py
```

Run it from the project folder, inside the same virtual environment as the rest of the project. Nothing new needs installing: it uses streamlit, pandas and openpyxl, which are already in `requirements.txt`. For Claude agents, set `ANTHROPIC_API_KEY` in that terminal before you start it.

## Pages

| Page | What it shows | Where the content comes from |
|---|---|---|
| Overview | The problem, who the system is for, how it works, headline results and the key requirements | `out/eval`, `out/eval_llm`, `out/chaos`, `dashboard_content/requirements.csv` |
| Live | Pick a disrupted row, add constraints, inject a fault, then watch each agent's message as it is sent | Computed live by the engine in `atm/` |
| Replays | Saved consensus logs played back message by message, plus a switch that shows the hash chain catching an edited entry | `demo/*.jsonl` (three Claude runs) and `showcase/*.jsonl` (rule-based) |
| Evaluation | Agreement, turns, today's process vs negotiated, like-for-like, the fault sweep, Claude vs rule-based, and where it does worse | `out/eval`, `out/eval_llm`, `out/chaos` |
| Requirements | Every line of the brief, the avoid list and the rules of play, with evidence | `dashboard_content/requirements.csv`, `dashboard_content/rubric.csv` |
| Rules & limits | The workbook tables, a lookup of each airport's limits, and the stated assumptions | `data/challenge2_operational_constraints.xlsx` via `atm/limits.py` |

## How the streaming works

`atm/live.py` runs the existing negotiator (`atm.protocol.Negotiator`) in a background thread. It changes nothing in the engine; it only listens to it:

- It swaps in a `StreamingLog`. This is the same hash-chained `ConsensusLog`, except that each entry is also sent to the screen the moment it is written.
- It wraps each agent's `declare` and `vote` calls. The dashboard shows "thinking" while a call runs and the agent's message as soon as it returns.

**Timing:**
- Claude agents stream in real time, at roughly 15 to 20 seconds per run.
- Rule-based agents answer in milliseconds, so their messages are spaced out for reading. Use the "Reading speed" control to change the spacing.

**What-if constraints:** You can close a runway, block a taxiway, remove fuel or add landing spacing. These changes are applied to the world before the agents see it, and the constraint checker enforces them like any other rule. At least one runway always stays open.

## Regenerating the data

```
python evaluate.py --data data/disruptions.csv                                    # out/eval
python evaluate.py --data data/disruptions.csv --limit 100 --chaos all --out out/chaos   # fault sweep
```

## Putting it online (Streamlit Community Cloud)

1. Upload the project to a GitHub repository. Leave out `.venv`, `.venv-win` and `data/flights.csv`: the dashboard does not use flights.csv, and the browser upload limit is 25 MB per file.
2. At share.streamlit.io choose **Create app**, pick the repository, and set the main file to `dashboard.py`.
3. Optional: to let visitors run Claude agents, open **Advanced settings → Secrets** and add:

```
ANTHROPIC_API_KEY = "sk-ant-..."
CLAUDE_PASSCODE = "pick-a-passcode"   # visitors must enter this before running Claude agents
CLAUDE_MAX_RUNS = "5"                 # Claude runs allowed per browser session
```

Without the key, the Live page still runs the rule-based agents, and Replays shows the recorded Claude runs.
