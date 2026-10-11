#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
environment["GIT_CONFIG_GLOBAL"] = str(root / ".gitconfig.local")
versions = {}
for relative in ["threestudio", "vendor/tiny-cuda-nn",
                 "vendor/tiny-cuda-nn/dependencies/cutlass",
                 "vendor/tiny-cuda-nn/dependencies/fmt",
                 "vendor/nvdiffrast", "vendor/envlight"]:
    versions[relative] = subprocess.check_output(
        ["git", "-C", str(root / relative), "rev-parse", "HEAD"],
        env=environment, text=True).strip()
(root / "source_versions.json").write_text(json.dumps(versions, indent=2))
print(json.dumps(versions, indent=2))
