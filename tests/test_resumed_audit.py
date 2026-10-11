import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("missing_step", [False, True])
def test_resumed_audit_requires_all_steps_after_checkpoint(tmp_path, missing_step):
    trial = tmp_path / "resumed-trial"
    metrics = trial / "csv_logs/version_0/metrics.csv"
    metrics.parent.mkdir(parents=True)
    rows = [{"step": step, "train/loss_sds": 1, "train/grad_norm": 2,
             "train/cert_grad_norm": 1, "train/cert_to_sds_grad_ratio": 0.5,
             "train/cert_nonzero_fraction": 1} for step in ([2] if missing_step else [2, 3])]
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = trial / "ckpts/last.ckpt"
    checkpoint.parent.mkdir()
    torch.save({"global_step": 4, "state_dict": {"geometry.example": torch.ones(1)},
                "optimizer_states": [{"state": {0: {"exp_avg": torch.ones(1)}}}]}, checkpoint)
    frames = trial / "save/it4-test"
    frames.mkdir(parents=True)
    # Artifact presence is enough here: this regression concerns metric coverage.
    (frames / "0.png").write_bytes(b"render-fixture")
    (trial / "save/it4-test.mp4").write_bytes(b"video-fixture")
    script = Path(__file__).resolve().parents[1] / "scripts/audit_run.py"
    result = subprocess.run([sys.executable, str(script), "--trial", str(trial),
                             "--expected-steps", "4", "--initial-step", "2", "--require-cert"],
                            capture_output=True, text=True)
    if missing_step:
        assert result.returncode != 0
        assert "do not cover every step" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["global_step"] == 4
        assert report["initial_step"] == 2
        assert report["logged_training_steps"] == 2
        assert report["certainty_active_step_fraction"] == 1
