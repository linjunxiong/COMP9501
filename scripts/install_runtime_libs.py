#!/usr/bin/env python3
"""Extract pinned OpenCV GL/X11/GLib libraries into this project, without apt install."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "vendor/runtime"
    downloads = ROOT / ".cache/debs"
    downloads.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    packages = json.loads((ROOT / "configs/runtime-packages.json").read_text())
    for package in packages:
        filename = downloads / Path(package["url"]).name
        if not filename.exists():
            temporary = filename.with_suffix(filename.suffix + ".tmp")
            with urllib.request.urlopen(package["url"], timeout=30) as response:
                temporary.write_bytes(response.read())
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != package["sha256"]:
                temporary.unlink()
                raise RuntimeError(f"SHA256 mismatch: {package['package']}")
            temporary.replace(filename)
        if hashlib.sha256(filename.read_bytes()).hexdigest() != package["sha256"]:
            raise RuntimeError(f"SHA256 mismatch in cache: {filename}")
        stream = subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(filename)],
                                  stdout=subprocess.PIPE)
        try:
            subprocess.run(["tar", "--extract", "--file=-", "--directory", str(destination),
                            "--no-same-owner"], stdin=stream.stdout, check=True)
        finally:
            stream.stdout.close()
            code = stream.wait()
        if code != 0:
            raise RuntimeError(f"cannot unpack {package['package']}")
        print(f"Ready: {package['package']}={package['version']}", flush=True)
    (destination / "packages.json").write_text(json.dumps(packages, indent=2) + "\n")


if __name__ == "__main__":
    main()
