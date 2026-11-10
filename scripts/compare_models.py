"""Run the same summary prompt on two Gemini models and audit both with the same metric.

Writes data/model_comparison.csv (one row per model, day and run). Run from the project root:
    python scripts/compare_models.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "buffalo" can be imported

from buffalo import ai_summary, analytics, config, data_pipeline  # noqa: E402

MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]   # the app's default model and its larger sibling
RUNS = 2          # Gemini answers differ each time, so each model summarises each day twice
DAYS = 8          # the most recent 8 snapshots
THRESHOLD = 0.10  # same match threshold as the Newsroom Check
CACHE = config.CACHE_DIR / "model_runs"   # raw replies, so a rerun does not spend API calls again
CACHE.mkdir(parents=True, exist_ok=True)


def get_summary(model, day, run, titles):
    path = CACHE / f"{model}_{day}_{run}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), "ok"
    summary, status = ai_summary.ask_for_summary(model, titles)
    print(f"  {day}  {model}  run {run}: {status}")
    time.sleep(7)                     # stay under the free-tier requests-per-minute limit
    if summary:                       # only successes are cached; failures are retried next time
        path.write_text(json.dumps(summary), encoding="utf-8")
        return summary, "ok"
    return None, "invalid json" if "JSON" in status else "api error"


def main():
    df = data_pipeline.build_dataframe(data_pipeline.load_snapshots())
    days = sorted(df["snapshot"].unique())[-DAYS:]
    rows = []
    for day in days:
        snap = df[df["snapshot"] == day]
        titles = snap["title"].drop_duplicates().tolist()
        for model in MODELS:
            for run in range(1, RUNS + 1):
                summary, status = get_summary(model, day, run, titles)
                row = {"model": model, "snapshot": day, "run": run, "status": status,
                       "json_ok": bool(summary and summary.get("clusters")), "clusters": len((summary or {}).get("clusters", []))}
                if row["json_ok"]:
                    row.update(analytics.score_summary(snap, summary, THRESHOLD, config.CATEGORIES))
                rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(config.MODEL_RUNS_FILE, index=False)
    summary, _, test = analytics.compare_models(runs)
    print(summary.round(3).to_string(index=False))
    print("paired test:", test)


if __name__ == "__main__":
    main()
