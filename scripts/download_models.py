#!/usr/bin/env python3
"""Download pinned public weights, validating LFS SHA256 without importing torch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "sd21-base": ("sd2-community/stable-diffusion-2-1-base",
                  "4e63672c03103b6c636b8fb4119ba982469b2955"),
    "clip-vit-base-patch32": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["auto", "direct", "environment"], default="auto")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    args = parser.parse_args()
    network = args.network
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    inherited = urllib.request.build_opener()

    def open_url(url, headers=None):
        nonlocal network
        request = urllib.request.Request(url, headers={"User-Agent": "COMP9501-reproduction", **(headers or {})})
        if network in {"direct", "auto"}:
            try:
                response = direct.open(request, timeout=8 if network == "auto" else 30)
                if network == "auto":
                    network = "direct"
                return response
            except (OSError, urllib.error.URLError):
                if network == "direct":
                    raise
                print("Direct Hugging Face connection unavailable; using environment network settings.", flush=True)
                network = "environment"
        return inherited.open(request, timeout=60)

    for name, (repo, requested_revision) in MODELS.items():
        if args.model not in {"all", name}:
            continue
        destination = ROOT / "models" / name
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "download_manifest.json"
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text())
        else:
            url = f"https://huggingface.co/api/models/{repo}/revision/{requested_revision}?blobs=true"
            with open_url(url) as response:
                metadata = json.load(response)
            manifest_path.write_text(json.dumps(metadata, indent=2))
        revision = metadata["sha"]
        siblings = metadata["siblings"]
        safe_folders = {str(Path(x["rfilename"]).parent) for x in siblings
                        if x["rfilename"].endswith(".safetensors") and ".fp16." not in x["rfilename"]}
        selected = []
        for item in siblings:
            filename = item["rfilename"]
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe filename in remote manifest")
            if name == "sd21-base" and path.parts[0] in {"safety_checker", "feature_extractor"}:
                continue
            if name == "sd21-base" and len(path.parts) == 1 and path.suffix in {".bin", ".safetensors"}:
                # The monolithic checkpoint duplicates the Diffusers component weights.
                continue
            if ".fp16." in filename or filename.endswith((".onnx", ".ckpt", ".msgpack", ".h5")):
                continue
            if filename.endswith(".bin") and str(path.parent) in safe_folders:
                continue
            if path.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"} or filename == "README.md":
                selected.append(item)

        def download(item):
            filename = item["rfilename"]
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = item.get("lfs", {}).get("sha256")
            expected_size = item.get("size")
            if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
                if expected is None or sha256(target) == expected:
                    print(f"Verified {name}/{filename}", flush=True)
                    return
            partial = target.with_name(target.name + ".partial")
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}?download=true"
            for attempt in range(4):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with open_url(url, headers) as response:
                        resume = offset > 0 and response.status == 206
                        with partial.open("ab" if resume else "wb") as handle:
                            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                                handle.write(chunk)
                    if expected_size is not None and partial.stat().st_size != expected_size:
                        raise ValueError(f"size mismatch: {filename}")
                    if expected and sha256(partial) != expected:
                        partial.unlink()
                        raise ValueError(f"SHA256 mismatch: {filename}")
                    partial.replace(target)
                    print(f"Downloaded {name}/{filename} ({target.stat().st_size:,} bytes)", flush=True)
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))

        print(f"Downloading {repo}@{revision}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(download, selected))
        inventory = [{"file": item["rfilename"], "sha256": sha256(destination / item["rfilename"]),
                      "bytes": (destination / item["rfilename"]).stat().st_size} for item in selected]
        (destination / "verified_files.json").write_text(json.dumps(inventory, indent=2))
        print(f"Complete: {name}, {len(selected)} files", flush=True)


if __name__ == "__main__":
    main()
