#!/usr/bin/env python3
"""Compile CUDA C++ sources for sm_90 on CPU, without creating a CUDA context."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

import importlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_EXTENSIONS_DIR",str(ROOT/".cache/torch_extensions"))
os.environ.setdefault("MAX_JOBS","4")


def forbidden(*args,**kwargs):
    raise RuntimeError("GPU context creation is forbidden during offline compilation")


torch.cuda._lazy_init = forbidden
torch.cuda.init = forbidden
backend = importlib.import_module("nerfacc.cuda._backend")
if backend._C is None:
    raise RuntimeError("nerfacc CUDA extension did not compile")
print("Nerfacc offline compilation complete",flush=True)
from nvdiffrast.torch import ops
# nvdiffrast 0.3.3 clears TORCH_CUDA_ARCH_LIST immediately before its loader.
# Restore the explicit target at that boundary; otherwise PyTorch attempts to
# select an architecture from visible GPUs, which preparation must never use.
from torch.utils import cpp_extension
original_load = cpp_extension.load
def offline_load(*args, **kwargs):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
    return original_load(*args, **kwargs)
cpp_extension.load = offline_load
try:
    plugin = ops._get_plugin(gl=False)
finally:
    cpp_extension.load = original_load
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
print("nvdiffrast CUDA plugin offline compilation complete",flush=True)
assert not torch.cuda.is_initialized()
(ROOT/"logs/native_build.json").write_text(json.dumps({
    "gpu_used":False,"cuda_initialized":False,"architecture":"sm_90",
    "nerfacc_module":backend._C.__file__,"nvdiffrast_module":plugin.__file__,
},indent=2))
