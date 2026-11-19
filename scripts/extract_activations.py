"""Step 2: run texts through the probed model and save per-text mean residual activations.

  python scripts/extract_activations.py --model Qwen/Qwen2.5-7B-Instruct \
      --data data/stories.jsonl --out acts/qwen7b_stories --layers every:2

Writes <out>.npy (float16 [N, L, D]) and <out>.meta.json (layers, labels, topic ids).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from emoprobe.activations import mean_activations  # noqa: E402
from emoprobe.model import load_model, num_layers, parse_layers  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL from generate_data.py")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--layers", default="all", help="all | every:K | frac:0.66 | 10,14,18")
    p.add_argument("--start-token", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--load-in-4bit", action="store_true")
    a = p.parse_args()

    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{len(rows)} texts from {a.data}")

    model, tok = load_model(a.model, load_in_4bit=a.load_in_4bit)
    layers = parse_layers(a.layers, num_layers(model))
    print(f"{num_layers(model)} layers; extracting {layers}")

    acts = mean_activations(model, tok, [r["text"] for r in rows], layers,
                            start_token=a.start_token, batch_size=a.batch_size, max_length=a.max_length)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out + ".npy", acts)
    meta = {
        "model": a.model, "n_layers": num_layers(model), "layers": layers, "start_token": a.start_token,
        "ids": [r["id"] for r in rows], "kind": [r["kind"] for r in rows],
        "emotion": [r["emotion"] for r in rows], "topic_id": [r["topic_id"] for r in rows],
    }
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(f"saved {a.out}.npy {acts.shape}")


if __name__ == "__main__":
    main()
