#!/usr/bin/env python3
"""Durable sequential training, evidence audits and CPU CLIP evaluation.

Default: print the command only. Execution needs explicit --execute --allow-gpu.
An interrupted/failed run terminates its own process group and records the failure.
"""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcgen.run_io import MetricsSnapshotError, read_metrics_snapshot


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def print_log_tail(path, lines=80):
    if path is not None and path.exists():
        with path.open(errors="replace") as handle:
            tail = "".join(deque(handle, maxlen=lines))
        print(f"\nLast {lines} lines of {path}:\n{tail}", file=sys.stderr, flush=True)


def read_progress(plan, previous=None, allow_unstable=False):
    progress = []
    previous_by_trial = {item["trial_dir"]: item for item in (previous or [])}
    for run in plan:
        trial = Path(run["trial_dir"])
        files = sorted((trial / "csv_logs").glob("version_*/metrics.csv"))
        item = {"variant": run["variant"], "prompt_id": run["prompt_id"],
                "seed": run["seed"], "requested_steps": run["steps"],
                "logged_steps": 0, "trial_dir": str(trial),
                "checkpoint_exists": (trial / "ckpts/last.ckpt").exists(),
                "initial_step": run.get("initial_step", 0)}
        gradients, ratios = [], []
        if files:
            try:
                rows = read_metrics_snapshot(files[-1])
                for row in rows:
                    for key, raw in row.items():
                        if isinstance(key, str) and key.startswith("train/") and raw:
                            try:
                                number = float(raw)
                            except (ValueError, TypeError):
                                continue
                            if not math.isfinite(number):
                                raise RuntimeError(f"non-finite {key}: {trial}")
                    if row.get("train/loss_sds"):
                        try:
                            loss = float(row["train/loss_sds"])
                        except ValueError:
                            continue  # CSV may be in the middle of being flushed.
                        if not math.isfinite(loss):
                            raise RuntimeError(f"non-finite SDS loss: {trial}")
                        item["logged_steps"] += 1
                        item["latest_loss_sds"] = loss
                    for key, values in [("train/cert_grad_norm", gradients),
                                        ("train/cert_to_sds_grad_ratio", ratios)]:
                        if row.get(key):
                            try:
                                value = float(row[key])
                            except ValueError:
                                continue
                            if not math.isfinite(value):
                                raise RuntimeError(f"non-finite {key}: {trial}")
                            values.append(value)
                old = previous_by_trial.get(str(trial))
                if old and item["logged_steps"] < old["logged_steps"]:
                    raise MetricsSnapshotError(f"CSV snapshot is older or partially rewritten: {trial}")
            except MetricsSnapshotError as error:
                if not allow_unstable:
                    raise
                old = previous_by_trial.get(str(trial))
                checkpoint_exists = item["checkpoint_exists"]
                if old:
                    item = dict(old)
                item["checkpoint_exists"] = checkpoint_exists
                item["metrics_read_state"] = "retrying"
                item["metrics_read_warning"] = str(error)
                item["metrics_read_failures"] = (old or {}).get("metrics_read_failures", 0) + 1
                progress.append(item)
                continue
            if gradients:
                item["certainty_active_step_fraction"] = sum(v > 0 for v in gradients) / len(gradients)
                item["latest_cert_grad_norm"] = gradients[-1]
                if len(gradients) >= 200 and max(gradients[-200:]) == 0:
                    raise RuntimeError(f"certainty gradient was zero for 200 consecutive logged steps: {trial}")
            if ratios:
                item["latest_cert_to_sds_grad_ratio"] = ratios[-1]
        progress.append(item)
    return progress


def stop_process_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def build_runner_command(args, group):
    command = [sys.executable, str(ROOT / "scripts/run_experiments.py"),
               "--stage", args.stage, "--variants", *args.variants,
               "--prompts", str(args.prompts), "--seeds", *map(str, args.seeds),
               "--run-group", group, "--execute", "--allow-gpu"]
    if args.gpu is not None:
        command += ["--gpu", args.gpu]
    if args.prompt_ids:
        command += ["--prompt-ids", *args.prompt_ids]
    if args.steps is not None:
        command += ["--steps", str(args.steps)]
    if args.label:
        command += ["--label", args.label]
    if args.resume_from_suite:
        command += ["--resume-from-suite", args.resume_from_suite]
    for override in args.set:
        command += ["--set", override]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", help="Optional override; default preserves the scheduler allocation")
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "lcgen-paper", "lcgen-ddpm"],
                        default=["baseline", "lcgen-paper"])
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/prompts-mini.json")
    parser.add_argument("--prompt-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--label")
    parser.add_argument("--resume-from-suite", help="Resume matching unfinished runs into new output directories")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="CPU runtime and registry check; never starts training")
    args = parser.parse_args()
    if args.execute and not args.allow_gpu:
        parser.error("execution requires the user's GPU approval and --allow-gpu")
    cpu_environment = os.environ.copy()
    cpu_environment["CUDA_VISIBLE_DEVICES"] = "-1"
    preflight_command = [sys.executable, str(ROOT / "scripts/preflight_cpu.py"), "--registry"]
    if args.check_only:
        subprocess.run(preflight_command + ["--output", str(ROOT / "logs/runtime_preflight.json")],
                       cwd=ROOT, env=cpu_environment, check=True)
        return
    group = args.stage + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    command = build_runner_command(args, group)
    if not args.execute:
        print(json.dumps({"execution": False, "command_after_approval": command}, indent=2))
        return
    active_status = ROOT / "GPU_EXPERIMENT_STATUS.json"
    # Each submitted job owns its unique directory. A PID from a different
    # compute node must never block a scheduler-assigned, independent job.
    job = ROOT / "outputs/jobs" / group
    job.mkdir(parents=True, exist_ok=False)
    suite = ROOT / "outputs/manifests" / group
    status = {"group": group, "supervisor_pid": os.getpid(), "state": "starting",
              "started_at_utc": utc_now(), "gpu_approval_received": True,
              "job_dir": str(job), "suite_dir": str(suite), "runs": [],
              "gpu_override": args.gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "hostname": socket.gethostname(),
              "scheduler_job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID")),
              "log": str(job / "training.log"),
              "note": "LCGen is a documented self-implementation; JR requires human review."}

    def record():
        status["updated_at_utc"] = utc_now()
        write_json(job / "status.json", status)
        write_json(active_status, status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = None
    record()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    cpu_environment["PYTHONUNBUFFERED"] = "1"
    error_log = job / "preflight.log"
    try:
        status["state"] = "checking_runtime"
        status["preflight_log"] = str(error_log)
        record()
        print(f"Checking CPU runtime; job directory: {job}", flush=True)
        with error_log.open("w") as log:
            process = subprocess.Popen(preflight_command + ["--output", str(job / "preflight.json")],
                                       cwd=ROOT, env=cpu_environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if process.wait() != 0:
                raise RuntimeError(f"CPU runtime check failed before training; inspect {error_log}")
        print(f"CPU runtime check passed. Training log: {job / 'training.log'}", flush=True)
        error_log = job / "training.log"
        with (job / "training.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            status["runner_pid"] = process.pid
            status["state"] = "training"
            while True:
                if (suite / "plan.json").exists():
                    plan = json.loads((suite / "plan.json").read_text())
                    previous = status["runs"]
                    status["runs"] = read_progress(plan, previous=previous, allow_unstable=True)
                    old_warnings = {run["trial_dir"]: run.get("metrics_read_warning") for run in previous}
                    for run in status["runs"]:
                        warning = run.get("metrics_read_warning")
                        if warning and warning != old_warnings.get(run["trial_dir"]):
                            print(f"Metrics snapshot unavailable; keeping last valid progress and retrying: {warning}",
                                  file=sys.stderr, flush=True)
                record()
                code = process.poll()
                if code is not None:
                    break
                time.sleep(15)
        if code != 0:
            raise RuntimeError(f"training runner exited with code {code}; inspect {job / 'training.log'}")
        plan = json.loads((suite / "plan.json").read_text())
        for attempt in range(3):
            try:
                status["runs"] = read_progress(plan)
                break
            except MetricsSnapshotError:
                if attempt == 2:
                    raise
                print("Waiting for the completed CSV logs to become readable...", flush=True)
                time.sleep(2)
        status["state"] = "evaluating"
        record()
        error_log = job / "evaluation.log"
        with (job / "evaluation.log").open("w") as log:
            for run in plan:
                status["evaluating"] = {"variant": run["variant"], "prompt_id": run["prompt_id"]}
                record()
                commands = [
                    [sys.executable, str(ROOT / "scripts/audit_run.py"),
                     "--trial", run["trial_dir"], "--expected-steps", str(run["steps"])]
                    + (["--initial-step", str(run["initial_step"])] if run.get("initial_step", 0) else [])
                    + (["--require-cert"] if run["guidance"].get("cert_strength", 0) > 0 else []),
                    [sys.executable, str(ROOT / "scripts/evaluate.py"),
                     "--trial", run["trial_dir"], "--prompt", run["prompt"], "--device", "cpu"]]
                for evaluation in commands:
                    process = subprocess.Popen(evaluation, cwd=ROOT, env=cpu_environment,
                                               stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    if process.wait() != 0:
                        raise RuntimeError(f"evaluation/audit failed: {evaluation}")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/summarize.py"),
                 "--manifest", str(suite / "plan.json")], cwd=ROOT, env=cpu_environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if process.wait() != 0:
                raise RuntimeError("summary generation failed")
        with (job / "janus_review_template.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["variant", "prompt_id", "seed", "janus"])
            writer.writerows((r["variant"], r["prompt_id"], r["seed"], "") for r in plan)
        status.pop("evaluating", None)
        status["state"] = "training_and_clip_complete_human_janus_review_pending"
        status["finished_at_utc"] = utc_now()
        status["summary"] = str(suite / "summary.json")
        status["janus_review_template"] = str(job / "janus_review_template.csv")
        record()
    except BaseException as error:
        stop_process_group(process)
        status["state"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status["error"] = str(error)
        status["error_log"] = str(error_log)
        status["finished_at_utc"] = utc_now()
        record()
        print_log_tail(error_log)
        raise


if __name__ == "__main__":
    main()
