#!/usr/bin/env python3
"""Plan by default. GPU execution requires BOTH --execute and --allow-gpu."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "threestudio"
sys.path.insert(0, str(ROOT))
from lcgen.run_io import checkpoint_global_step


def resume_info(cfg, prompt, variant, seed, source_dir, source_plan):
    """Only reuse an unfinished checkpoint with the same saved numerical configuration."""
    matches = [(index, run) for index, run in enumerate(source_plan)
               if (run["prompt_id"], run["variant"], run["seed"]) == (prompt["id"], variant, seed)]
    if len(matches) != 1:
        raise ValueError(f"resume suite must contain exactly one matching run: {prompt['id']}/{variant}/seed{seed}")
    index, old = matches[0]
    checkpoint = Path(old["trial_dir"]) / "ckpts/last.ckpt"
    if not checkpoint.exists():
        return {"initial_step": 0}
    original_path = source_dir / f"{index:03d}-{variant}-{prompt['id']}-seed{seed}.yaml"
    original = OmegaConf.to_container(OmegaConf.load(original_path), resolve=False)
    current = OmegaConf.to_container(cfg, resolve=False)
    for configuration in (original, current):
        for key in ("tag", "use_timestamp", "resume"):
            configuration.pop(key, None)
    if original != current:
        raise ValueError(f"resume configuration differs from saved run: {original_path}")
    step = checkpoint_global_step(checkpoint)
    if not 0 <= step < int(cfg.trainer.max_steps):
        raise ValueError(f"resume requires an unfinished checkpoint; {checkpoint} is already at step {step}")
    cfg.resume = str(checkpoint)
    return {"initial_step": step, "resume_from": str(checkpoint),
            "resume_source_suite": str(source_dir),
            "resume_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "resume_note": "Optimizer and global step resume; bitwise RNG continuity is not guaranteed."}


def make_config(variant, prompt, seed, steps, run_group):
    cfg = OmegaConf.load(ROOT / "configs/dreamfusion-sd-baseline.yaml")
    if variant != "baseline":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ROOT / f"configs/{variant}.yaml"))
    cfg.system.prompt_processor.prompt = prompt["text"]
    cfg.seed = seed
    cfg.tag = f"{run_group}-{prompt['id']}-seed{seed}"
    cfg.use_timestamp = False
    cfg.exp_root_dir = str(ROOT / "outputs")
    cfg.system.guidance.pretrained_model_name_or_path = str(ROOT / "models/sd21-base")
    cfg.system.prompt_processor.pretrained_model_name_or_path = str(ROOT / "models/sd21-base")
    cfg.trainer.max_steps = steps
    cfg.trainer.val_check_interval = min(200, steps)
    cfg.checkpoint.every_n_train_steps = steps
    cfg.data.n_test_views = 120 if steps >= 10000 else 24
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "lcgen-paper", "lcgen-ddpm"],
                        default=["baseline", "lcgen-paper"])
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/prompts-mini.json")
    parser.add_argument("--prompt-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--set", action="append", default=[], help="OmegaConf key=value override")
    parser.add_argument("--label", help="Explicit suffix for a separately named sensitivity variant")
    parser.add_argument("--run-group", help="Unique suite identifier for durable orchestration")
    parser.add_argument("--resume-from-suite", help="Previous suite group; unfinished checkpoints go into new output directories")
    parser.add_argument("--gpu", help="GPU visibility after approval; otherwise preserve scheduler assignment")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true", help="Use only after the user's GPU approval")
    args = parser.parse_args()
    if args.label and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label must contain only letters, numbers, hyphens or underscores")
    if args.run_group and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_group):
        parser.error("run-group must contain only letters, numbers, hyphens or underscores")
    if args.resume_from_suite and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.resume_from_suite):
        parser.error("resume-from-suite must be a suite group identifier")
    if args.execute and not args.allow_gpu:
        parser.error("GPU experiments are pending user approval; execution requires --allow-gpu")
    steps = args.steps if args.steps is not None else (100 if args.stage == "smoke" else 10000)
    if steps <= 0:
        parser.error("steps must be positive")
    prompts = json.loads(args.prompts.read_text())
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+",p["id"]) for p in prompts):
        parser.error("prompt IDs must contain only letters, numbers, hyphens or underscores")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must satisfy 0 <= seed < 2**32")
    if args.prompt_ids:
        known = {p["id"] for p in prompts}
        if set(args.prompt_ids) - known:
            parser.error("unknown prompt id")
        prompts = [p for p in prompts if p["id"] in args.prompt_ids]
    if not prompts:
        parser.error("no prompts selected")
    group = args.run_group or args.stage + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    source_dir = ROOT / "outputs/manifests" / args.resume_from_suite if args.resume_from_suite else None
    source_plan = json.loads((source_dir / "plan.json").read_text()) if source_dir else None
    plans = []
    resumptions = []
    for prompt in prompts:
        for seed in args.seeds:
            for variant in args.variants:
                cfg = make_config(variant, prompt, seed, steps, group)
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
                method_label = variant if not args.label else f"{variant}-{args.label}"
                if args.label:
                    cfg.name = f"{cfg.name}-{args.label}"
                trial = Path(cfg.exp_root_dir) / cfg.name / cfg.tag
                if source_dir:
                    resumptions.append(resume_info(cfg, prompt, method_label, seed, source_dir, source_plan))
                else:
                    if cfg.get("resume"):
                        parser.error("use --resume-from-suite, not a shared --set resume=... across methods")
                    resumptions.append({"initial_step": 0})
                plans.append((prompt, method_label, seed, cfg, trial))
    plan = [{"prompt": p["text"], "prompt_id": p["id"], "variant": v, "seed": s,
             "steps": int(c.trainer.max_steps), "trial_dir": str(d),
             "guidance": OmegaConf.to_container(c.system.guidance, resolve=True)}
            for p, v, s, c, d in plans]
    for run, resumption in zip(plan, resumptions):
        run.update(resumption)
    print(json.dumps({"execution": args.execute, "gpu_approval_required": not args.execute,
                      "runs": plan}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    model_inventory = ROOT / "models/sd21-base/verified_files.json"
    if not model_inventory.exists():
        parser.error("model download/verification is not complete")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = env["DIFFUSERS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["GIT_CONFIG_GLOBAL"] = str(ROOT / ".gitconfig.local")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if "," in env.get("CUDA_VISIBLE_DEVICES", ""):
        parser.error("this protocol uses one GPU; select one allocated device explicitly with --gpu")
    if env.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        parser.error("GPU is hidden; explicitly select the approved device with --gpu")
    suite_dir = ROOT / "outputs" / "manifests" / group
    suite_dir.mkdir(parents=True, exist_ok=False)
    (suite_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    for folder in ["lcgen", "configs", "scripts", "docs", "tests"]:
        shutil.copytree(ROOT / folder, suite_dir / "source" / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["requirements.txt", "requirements.lock.txt", "source_versions.json", "README.md"]:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, suite_dir / "source" / name)
    shutil.copytree(UPSTREAM / "custom/lcgen_extension",
                    suite_dir / "source/lcgen_extension",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["gpu_preflight.json", "gpu_smoke_summary.json"]:
        if (ROOT / "logs" / name).exists():
            shutil.copy2(ROOT / "logs" / name, suite_dir / "source" / name)
    for index, (prompt, variant, seed, cfg, trial) in enumerate(plans):
        config_path = suite_dir / f"{index:03d}-{variant}-{prompt['id']}-seed{seed}.yaml"
        OmegaConf.save(cfg, config_path)
        command = [sys.executable, "launch.py", "--config", str(config_path), "--train", "--gpu", "0"]
        log_path = suite_dir / f"{index:03d}-{variant}-{prompt['id']}-seed{seed}.log"
        print("Executing after explicit approval:", shlex.join(command), flush=True)
        metadata = {**plan[index], "config": str(config_path), "log": str(log_path),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "upstream_commit": "28d9d80d9d00f308244adfcf3be8b17ca0cb6465",
                    "model_inventory_sha256": hashlib.sha256(model_inventory.read_bytes()).hexdigest()}
        (suite_dir / f"{index:03d}-metadata.json").write_text(json.dumps(metadata, indent=2))
        started = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=UPSTREAM, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except KeyboardInterrupt:
                # Do not leave a GPU process running if the runner is interrupted.
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        metadata["exit_code"] = code
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["checkpoint_exists"] = (trial / "ckpts/last.ckpt").exists()
        (suite_dir / f"{index:03d}-metadata.json").write_text(json.dumps(metadata, indent=2))
        if code != 0 or not metadata["checkpoint_exists"]:
            raise RuntimeError(f"Run failed or checkpoint missing: inspect {log_path}")


if __name__ == "__main__":
    main()
