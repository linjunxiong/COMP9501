import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace
import zipfile

from omegaconf import OmegaConf
import pytest

from lcgen.run_io import checkpoint_global_step, MetricsSnapshotError, read_metrics_snapshot

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_runner", ROOT / "scripts/run_experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fake_checkpoint(path, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"epoch": 0, "global_step": step}, protocol=2))


def test_checkpoint_step_is_read_without_unpickling_or_torch(tmp_path):
    path = tmp_path / "last.ckpt"
    fake_checkpoint(path, 4200)
    assert checkpoint_global_step(path) == 4200


def test_snapshot_rejects_rewrite_that_changes_file_metadata(tmp_path, monkeypatch):
    path = tmp_path / "metrics.csv"
    path.write_text("train/loss_sds,step\n1,0\n")
    snapshots = iter([SimpleNamespace(st_ino=1, st_size=24, st_mtime_ns=1, st_ctime_ns=1),
                      SimpleNamespace(st_ino=1, st_size=24, st_mtime_ns=2, st_ctime_ns=2)])
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda target, *args, **kwargs:
                        next(snapshots) if target == path else original_stat(target, *args, **kwargs))
    with pytest.raises(MetricsSnapshotError, match="changed while"):
        read_metrics_snapshot(path)


def test_resume_uses_checkpoint_step_and_rejects_changed_training_config(tmp_path):
    source = tmp_path / "suite"
    source.mkdir()
    old_trial = tmp_path / "old-trial"
    checkpoint = old_trial / "ckpts/last.ckpt"
    fake_checkpoint(checkpoint, 4200)
    prompt = {"id": "corgi", "text": "a corgi"}
    original = runner.make_config("baseline", prompt, 0, 10000, "old-group")
    OmegaConf.save(original, source / "000-baseline-corgi-seed0.yaml")
    previous = [{"prompt_id": "corgi", "variant": "baseline", "seed": 0,
                 "trial_dir": str(old_trial)}]
    current = runner.make_config("baseline", prompt, 0, 10000, "new-group")
    result = runner.resume_info(current, prompt, "baseline", 0, source, previous)
    assert result["initial_step"] == 4200
    assert current.resume == str(checkpoint)
    assert current.tag != original.tag
    assert result["resume_checkpoint_sha256"]
    current.system.loss.lambda_sparsity = 0.5
    with pytest.raises(ValueError, match="configuration differs"):
        runner.resume_info(current, prompt, "baseline", 0, source, previous)


def test_unstarted_run_in_resume_suite_starts_from_zero(tmp_path):
    current = runner.make_config("baseline", {"id": "corgi", "text": "a corgi"}, 0, 10000, "new-group")
    previous = [{"prompt_id": "corgi", "variant": "baseline", "seed": 0, "trial_dir": str(tmp_path / "absent")}]
    result = runner.resume_info(current, {"id": "corgi"}, "baseline", 0, tmp_path, previous)
    assert result == {"initial_step": 0}
    assert not current.get("resume")
