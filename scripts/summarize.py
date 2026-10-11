#!/usr/bin/env python3
"""Aggregate completed runs, with explicit missing-review coverage for JR."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="suite plan.json")
    parser.add_argument("--annotations", type=Path, help="CSV: variant,prompt_id,seed,janus (0 or 1)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plans = json.loads(args.manifest.read_text())
    annotations = {}
    if args.annotations:
        with args.annotations.open() as f:
            for row in csv.DictReader(f):
                if row["janus"].strip() not in {"0", "1"}:
                    raise ValueError("janus must be 0 or 1; omit unreviewed rows")
                key = (row["variant"],row["prompt_id"],int(row["seed"]))
                if key in annotations:
                    raise ValueError("duplicate review row")
                annotations[key] = int(row["janus"])
    grouped = defaultdict(list)
    for plan in plans:
        metric_path = Path(plan["trial_dir"]) / "metrics.json"
        metrics = json.loads(metric_path.read_text()) if metric_path.exists() else None
        key = (plan["variant"],plan["prompt_id"],plan["seed"])
        grouped[plan["variant"]].append((plan, metrics, annotations.get(key)))
    table = []
    for variant, rows in grouped.items():
        scored = [m["mean_cosine_similarity_times_100"] for p,m,j in rows if m is not None]
        reviewed = [j for p,m,j in rows if j is not None]
        table.append({"variant":variant,"planned_models":len(rows),"clip_scored_models":len(scored),
                      "mean_clip_cosine_times_100":sum(scored)/len(scored) if scored else None,
                      "human_reviewed_models":len(reviewed),
                      "janus_rate_percent_on_reviewed":100*sum(reviewed)/len(reviewed) if reviewed else None,
                      "complete_review":len(reviewed)==len(rows)})
    result = {"summary":table,"note":"JR uses 3D models as denominator, never individual rendered views."}
    output = args.output or args.manifest.parent / "summary.json"
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
