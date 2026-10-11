import csv
import importlib.util
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "comparison_suite_monitor", Path(__file__).resolve().parents[1] / "scripts/run_comparison_suite.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_run(tmp_path, rows):
    folder = tmp_path / "trial/csv_logs/version_0"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [{"variant": "lcgen-paper", "prompt_id": "corgi", "seed": 0,
             "steps": 10000, "trial_dir": str(tmp_path / "trial")}]


def test_monitor_reports_logged_progress(tmp_path):
    plan = fixture_run(tmp_path, [{"train/loss_sds": 1, "train/cert_grad_norm": 2},
                                  {"train/loss_sds": 3, "train/cert_grad_norm": 0}])
    progress = module.read_progress(plan)[0]
    assert progress["logged_steps"] == 2
    assert progress["certainty_active_step_fraction"] == 0.5
    assert not progress["checkpoint_exists"]


def test_monitor_rejects_nonfinite_regularizer(tmp_path):
    plan = fixture_run(tmp_path, [{"train/loss_sds": 1, "train/loss_orient": "nan"}])
    with pytest.raises(RuntimeError, match="non-finite train/loss_orient"):
        module.read_progress(plan)


def test_monitor_rejects_inactive_certainty_window(tmp_path):
    plan = fixture_run(tmp_path, [{"train/loss_sds": 1, "train/cert_grad_norm": 0}] * 200)
    with pytest.raises(RuntimeError, match="200 consecutive"):
        module.read_progress(plan)


@pytest.mark.parametrize("broken", [b"\x00" * 256, b"", b"train/loss_sds,train/cert_grad_norm\n1,",
                                   b"train/loss_sds,train/cert_grad_norm\nna,1\n"])
def test_live_monitor_keeps_previous_progress_until_csv_recovers(tmp_path, broken):
    plan = fixture_run(tmp_path, [{"train/loss_sds": 1, "train/cert_grad_norm": 2},
                                  {"train/loss_sds": 3, "train/cert_grad_norm": 4}])
    previous = module.read_progress(plan)
    csv_path = tmp_path / "trial/csv_logs/version_0/metrics.csv"
    good = csv_path.read_bytes()
    csv_path.write_bytes(broken)
    retrying = module.read_progress(plan, previous=previous, allow_unstable=True)
    assert retrying[0]["logged_steps"] == 2
    assert retrying[0]["latest_loss_sds"] == 3
    assert retrying[0]["metrics_read_state"] == "retrying"
    # The live monitor does not change or sanitize the original log.
    assert csv_path.read_bytes() == broken
    with pytest.raises(module.MetricsSnapshotError):
        module.read_progress(plan)
    csv_path.write_bytes(good)
    recovered = module.read_progress(plan, previous=retrying, allow_unstable=True)
    assert recovered[0]["logged_steps"] == 2
    assert "metrics_read_warning" not in recovered[0]


def test_live_monitor_preserves_progress_during_a_shorter_rewrite(tmp_path):
    rows = [{"train/loss_sds": 1}, {"train/loss_sds": 2}]
    plan = fixture_run(tmp_path, rows)
    previous = module.read_progress(plan)
    fixture_run(tmp_path, rows[:1])
    result = module.read_progress(plan, previous=previous, allow_unstable=True)[0]
    assert result["logged_steps"] == 2
    assert result["metrics_read_state"] == "retrying"


def test_live_monitor_still_rejects_real_nan_when_transient_reads_are_allowed(tmp_path):
    plan = fixture_run(tmp_path, [{"train/loss_sds": 1}])
    previous = module.read_progress(plan)
    fixture_run(tmp_path, [{"train/loss_sds": "nan"}])
    with pytest.raises(RuntimeError, match="non-finite train/loss_sds"):
        module.read_progress(plan, previous=previous, allow_unstable=True)


def test_submission_preserves_allocation_and_forwards_experiment_options():
    args = SimpleNamespace(stage="full", variants=["lcgen-paper"], prompts=Path("/tmp/prompts.json"),
                           seeds=[0, 1], gpu=None, prompt_ids=["corgi"], steps=500,
                           label="printed-sign", set=["system.guidance.cert_gradient_sign=paper"], resume_from_suite=None)
    command = module.build_runner_command(args, "submitted-test")
    assert "--gpu" not in command
    assert command[command.index("--seeds") + 1:command.index("--seeds") + 3] == ["0", "1"]
    assert command[command.index("--steps") + 1] == "500"
    assert command[command.index("--label") + 1] == "printed-sign"
    assert "system.guidance.cert_gradient_sign=paper" in command
    args.resume_from_suite = "full-interrupted"
    command = module.build_runner_command(args, "submitted-test")
    assert command[command.index("--resume-from-suite") + 1] == "full-interrupted"
    args.gpu = "MIG-explicit-device"
    command = module.build_runner_command(args, "submitted-test")
    assert command[command.index("--gpu") + 1] == "MIG-explicit-device"


def test_bash_entry_dry_run_does_not_execute():
    script = Path(__file__).resolve().parents[1] / "scripts/run_gpu_experiments.sh"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(["bash", str(script), "--dry-run"], env=environment,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["execution"] is False
    result = subprocess.run(["bash", str(script), "--dry-run", "--execute"], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2


def test_failed_cpu_preflight_stops_before_training_and_prints_root_cause(tmp_path, monkeypatch, capsys):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "preflight_cpu.py").write_text(
        "import os\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'\n"
        "raise ImportError('libX11.so.6: cannot open shared object file')\n")
    # This marker would be written if the training runner were incorrectly started.
    (scripts / "run_experiments.py").write_text(
        f"from pathlib import Path\nPath({str(tmp_path / 'training_started')!r}).touch()\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["suite", "--execute", "--allow-gpu"])
    # main installs handlers, so restore them when the test finishes.
    previous_term = module.signal.getsignal(module.signal.SIGTERM)
    previous_int = module.signal.getsignal(module.signal.SIGINT)
    try:
        with pytest.raises(RuntimeError, match="CPU runtime check failed before training"):
            module.main()
    finally:
        module.signal.signal(module.signal.SIGTERM, previous_term)
        module.signal.signal(module.signal.SIGINT, previous_int)
    assert "libX11.so.6: cannot open shared object file" in capsys.readouterr().err
    assert not (tmp_path / "training_started").exists()
    status = json.loads((tmp_path / "GPU_EXPERIMENT_STATUS.json").read_text())
    assert status["state"] == "failed"
    assert status["error_log"].endswith("preflight.log")


def test_check_only_hides_gpu_and_does_not_create_training_job(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    marker = tmp_path / "cpu_check_passed"
    (scripts / "preflight_cpu.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'\n"
        f"Path({str(marker)!r}).touch()\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "platform-assigned-device")
    monkeypatch.setattr(sys, "argv", ["suite", "--check-only"])
    module.main()
    assert marker.exists()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "platform-assigned-device"
    assert not (tmp_path / "outputs").exists()
