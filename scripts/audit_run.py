#!/usr/bin/env python3
"""Audit saved training evidence on CPU; never infer generation quality from loss."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcgen.run_io import read_metrics_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--require-cert", action="store_true")
    parser.add_argument("--initial-step", type=int, default=0, help="Saved checkpoint step for this resumed segment")
    args = parser.parse_args()
    expected_logged_steps = args.expected_steps - args.initial_step
    if not 0 <= args.initial_step < args.expected_steps:
        parser.error("initial-step must be nonnegative and less than expected-steps")
    metrics = sorted((args.trial / "csv_logs").glob("version_*/metrics.csv"))
    if not metrics:
        raise RuntimeError("training metrics are missing")
    series = {}
    rows = read_metrics_snapshot(metrics[-1])
    for row in rows:
        for key, value in row.items():
            if key.startswith("train/") and value:
                series.setdefault(key, []).append(float(value))
    logged_steps = {int(row["step"]) for row in rows if row.get("train/loss_sds")}
    if logged_steps != set(range(args.initial_step, args.expected_steps)):
        raise RuntimeError("training metrics do not cover every step of this run's recorded segment")
    required = ["train/loss_sds", "train/grad_norm"]
    if args.require_cert:
        required += ["train/cert_grad_norm", "train/cert_to_sds_grad_ratio",
                     "train/cert_nonzero_fraction"]
    for key in required:
        if key not in series or len(series[key]) < expected_logged_steps:
            raise RuntimeError(f"missing step metrics: {key}")
    for key, values in series.items():
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite logged metric: {key}")
    if max(series["train/grad_norm"]) <= 0:
        raise RuntimeError("SDS gradient is always zero")
    import torch
    # Lightning pickles the upstream configuration dataclass in the checkpoint.
    # Import that class with CUDA networks explicitly blocked, just as preflight
    # does. No geometry/network is constructed during this CPU-only audit.
    def forbidden(*args, **kwargs):
        raise RuntimeError("CUDA initialization is forbidden during checkpoint audit")
    torch.cuda._lazy_init = forbidden
    blocked = types.ModuleType("tinycudann")
    class BlockedNetwork:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CPU checkpoint audit cannot construct a CUDA network")
    for name in ["Encoding", "Network", "NetworkWithInputEncoding", "Module"]:
        setattr(blocked, name, BlockedNetwork)
    blocked.free_temporary_memory = lambda: None
    sys.modules["tinycudann"] = blocked
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "threestudio"))
    checkpoint_path = args.trial / "ckpts/last.ckpt"
    # This checkpoint is produced by our local, authorized training process.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert not torch.cuda.is_initialized()
    if checkpoint["global_step"] != args.expected_steps:
        raise RuntimeError("checkpoint step does not match the requested run")
    tensors = [v for k, v in checkpoint["state_dict"].items()
               if k.startswith("geometry.") and isinstance(v, torch.Tensor)]
    if not tensors or not all(bool(torch.isfinite(v).all()) for v in tensors):
        raise RuntimeError("invalid geometry parameters in checkpoint")
    moving_averages = [state["exp_avg"] for optimizer in checkpoint["optimizer_states"]
                      for state in optimizer["state"].values() if "exp_avg" in state]
    if not moving_averages or not all(bool(torch.isfinite(v).all()) for v in moving_averages):
        raise RuntimeError("invalid optimizer state")
    if not any(bool(v.ne(0).any()) for v in moving_averages):
        raise RuntimeError("no evidence of optimizer updates")
    frames = list((args.trial / f"save/it{args.expected_steps}-test").glob("*.png"))
    video = args.trial / f"save/it{args.expected_steps}-test.mp4"
    if not frames or not video.exists() or video.stat().st_size == 0:
        raise RuntimeError("test renders or turntable video missing")
    cert_active_steps = sum(v > 0 for v in series.get("train/cert_grad_norm", []))
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "trial": str(args.trial), "global_step": checkpoint["global_step"],
        "initial_step": args.initial_step,
        "metric_coverage_note": "Metrics cover this segment; earlier checkpoint segments may have incomplete logs.",
        "logged_training_steps": len(series["train/loss_sds"]),
        "finite_geometry_and_optimizer": True, "nonzero_optimizer_updates": True,
        "test_views": len(frames), "video": str(video),
        "certainty_active_steps": cert_active_steps,
        "certainty_active_step_fraction": cert_active_steps / expected_logged_steps,
        "metric_summary": {key: {"min": min(v), "median": statistics.median(v),
                                  "mean": statistics.mean(v), "max": max(v)}
                           for key, v in series.items()},
        "quality_note": "Passing execution checks does not establish Janus reduction or image quality."
    }
    (args.trial / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_cert and cert_active_steps == 0:
        raise RuntimeError("certainty gradient is zero for every training step")


if __name__ == "__main__":
    main()
