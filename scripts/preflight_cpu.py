#!/usr/bin/env python3
"""CPU-only readiness checks. No training, device probing, or CUDA context."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("COMP9501_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-models", action="store_true", help="CPU weight loading and one diffusion forward")
    parser.add_argument("--registry", action="store_true", help="import threestudio with a blocked tinycudann stub")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/cpu_preflight.json")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    def forbidden_cuda_init(*args, **kwargs):
        raise RuntimeError("CUDA initialization is forbidden during CPU preparation")
    torch.cuda._lazy_init = forbidden_cuda_init
    torch.cuda.init = forbidden_cuda_init
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_used": False, "cuda_initialized": torch.cuda.is_initialized(), "checks": {}}
    for package in ["torch", "torchvision", "pytorch-lightning", "diffusers", "transformers",
                    "nerfacc", "tinycudann", "nvdiffrast", "envlight", "huggingface-hub"]:
        report["checks"][package] = importlib.metadata.version(package)
    for name in ["torchvision", "pytorch_lightning", "diffusers", "transformers", "nerfacc",
                 "nvdiffrast.torch", "envlight", "igl", "cv2"]:
        importlib.import_module(name)
    import cv2
    import numpy as np
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    encoded_ok, png = cv2.imencode(".png", image)
    assert encoded_ok and np.array_equal(cv2.imdecode(png, cv2.IMREAD_COLOR), image)
    binary = next(Path(cv2.__file__).parent.glob("cv2*.so"))
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    if "not found" in linked:
        raise RuntimeError(f"OpenCV has missing shared libraries:\n{linked}")
    libraries = {}
    for line in linked.splitlines():
        if " => " in line:
            name, target = line.strip().split(" => ", 1)
            libraries[name] = target.split(" (")[0]
    # The workstation has X11 libraries, while the platform container may not.
    # Require the project copies so workstation checks cannot mask that gap.
    required_runtime = ["libGL.so.1", "libGLX.so.0", "libGLdispatch.so.0", "libX11.so.6",
                        "libxcb.so.1", "libXau.so.6", "libXdmcp.so.6", "libbsd.so.0",
                        "libmd.so.0", "libglib-2.0.so.0", "libgthread-2.0.so.0", "libpcre.so.3"]
    for name in required_runtime:
        target = Path(libraries.get(name, "/missing"))
        if not target.is_relative_to(ROOT / "vendor/runtime"):
            raise RuntimeError(f"{name} must resolve from vendor/runtime, got {target}; "
                               "source scripts/env.sh and run scripts/install_runtime_libs.py")
    report["checks"]["opencv_png_roundtrip"] = "passed"
    report["checks"]["opencv_project_runtime_libraries"] = {name: libraries[name] for name in required_runtime}
    bindings = list((ROOT / ".venv/lib/python3.10/site-packages").glob("tinycudann_bindings/*90*.so"))
    if not bindings:
        raise RuntimeError("H100 sm_90 tiny-cuda-nn binary is missing")
    report["checks"]["tinycudann_sm90_binary"] = str(bindings[0])
    if args.registry:
        # tinycudann's real Python import initializes CUDA to select a binary.
        # A clearly blocked stub permits registration/config checks, never rendering.
        blocked = types.ModuleType("tinycudann")
        class BlockedNetwork:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("CPU registry check cannot instantiate a CUDA network")
        for name in ["Encoding", "Network", "NetworkWithInputEncoding", "Module"]:
            setattr(blocked, name, BlockedNetwork)
        blocked.free_temporary_memory = lambda: None
        sys.modules["tinycudann"] = blocked
        sys.path.insert(0, str(ROOT / "threestudio"))
        import threestudio
        from launch import load_custom_module
        assert load_custom_module(str(ROOT / "threestudio/custom/lcgen_extension"))
        from threestudio.utils.config import load_config
        cfg = load_config(str(ROOT / "configs/dreamfusion-sd-baseline.yaml"),
                          cli_args=["system.prompt_processor.prompt=a corgi",
                                    f"exp_root_dir={ROOT / 'logs/cpu-config-only'}",
                                    "tag=validation", "use_timestamp=false"], n_gpus=1)
        for key in ["dreamfusion-system", "stable-diffusion-guidance", "lcgen-stable-diffusion-guidance"]:
            assert threestudio.find(key) is not None
        assert cfg.system.loss.lambda_sparsity == 0.1
        assert cfg.system.background.random_aug
        report["checks"]["registry_with_blocked_cuda_networks"] = "passed; not a rendering test"
        report["checks"]["upstream_custom_extension_loader"] = "passed"
    if args.load_models:
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
        model = ROOT / "models/sd21-base"
        for path in [model, ROOT / "models/clip-vit-base-patch32"]:
            if not (path / "verified_files.json").exists():
                raise RuntimeError(f"weight verification incomplete: {path}")
        tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
        # Match the upstream guidance loader, but keep every component on CPU.
        pipe = StableDiffusionPipeline.from_pretrained(
            model, tokenizer=None, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False, torch_dtype=torch.float32, local_files_only=True)
        text_encoder, unet, vae = pipe.text_encoder, pipe.unet, pipe.vae
        scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler", local_files_only=True)
        assert scheduler.config.prediction_type == "epsilon"
        tokens = tokenizer(["a corgi", ""], padding="max_length",
                           max_length=tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            embeddings = text_encoder(tokens.input_ids)[0]
            sample = torch.zeros(2, 4, 8, 8)
            prediction = unet(sample, torch.tensor([500,500]), encoder_hidden_states=embeddings).sample
            encoded = vae.encode(torch.zeros(1,3,64,64)).latent_dist.mode()
            decoded = vae.decode(encoded).sample
        assert prediction.shape == sample.shape and torch.isfinite(prediction).all()
        assert decoded.shape == (1,3,64,64) and torch.isfinite(decoded).all()
        clip_path = ROOT / "models/clip-vit-base-patch32"
        clip = CLIPModel.from_pretrained(clip_path, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        from PIL import Image
        inputs = processor(text=["a corgi"], images=[Image.new("RGB",(224,224))],
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            result = clip(**inputs)
        assert torch.isfinite(result.logits_per_image).all()
        report["checks"]["sd21_cpu_text_unet_vae_forward"] = "passed"
        report["checks"]["upstream_diffusion_pipeline_cpu_loader"] = "passed"
        report["checks"]["clip_cpu_forward"] = "passed"
    assert not torch.cuda.is_initialized()
    report["cuda_initialized"] = False
    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
