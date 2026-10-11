#!/usr/bin/env python3
"""CLIP cosine similarity on RGB views; JR requires per-model human annotations."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.device == "cuda" and not args.allow_gpu:
        parser.error("GPU evaluation requires user approval and --allow-gpu")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    folders = list((args.trial / "save").glob("it*-test"))
    if not folders:
        parser.error("no test renders found")
    views = max(folders, key=lambda p: int(re.search(r"it(\d+)",p.name)[1]))
    images = sorted(views.glob("*.png"), key=lambda p: int(p.stem))
    if not images:
        parser.error("empty test-render directory")
    path = ROOT / "models/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(path, local_files_only=True).eval().to(args.device)
    processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
    scores = []
    with torch.inference_mode():
        tokens = processor(text=[args.prompt], return_tensors="pt", padding=True).to(args.device)
        text = model.get_text_features(**tokens)
        text = torch.nn.functional.normalize(text, dim=-1)
        for start in range(0,len(images),args.batch_size):
            batch = []
            for filename in images[start:start+args.batch_size]:
                with Image.open(filename) as original:
                    # Upstream saves RGB, normals and opacity in horizontal square tiles.
                    image = original.convert("RGB")
                    height = image.height
                    if image.width < height or image.width % height != 0:
                        raise ValueError(f"unexpected image-grid dimensions: {filename}")
                    batch.append(image.crop((0,0,height,height)))
            pixels = processor(images=batch, return_tensors="pt").to(args.device)
            features = model.get_image_features(**pixels)
            features = torch.nn.functional.normalize(features, dim=-1)
            scores.extend((features @ text.T).flatten().cpu().tolist())
    output = {"prompt":args.prompt, "num_views":len(images), "views":str(views),
              "clip_model":"openai/clip-vit-base-patch32", "device":args.device,
              "mean_cosine_similarity":sum(scores)/len(scores),
              "mean_cosine_similarity_times_100":100*sum(scores)/len(scores),
              "per_view_cosine":scores,
              "janus":None, "janus_note":"Requires a human decision per 3D model, not per view."}
    filename = args.trial / "metrics.json"
    filename.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!="per_view_cosine"},indent=2))


if __name__ == "__main__":
    main()
