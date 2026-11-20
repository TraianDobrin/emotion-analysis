"""Step 2: run every text through the model and save its mean activation at every layer.

  python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/happy_stories.jsonl --out acts/q3b_happy

Writes <out>.npy (float16 [N, layers, D]) and <out>.meta.json (labels and topic of each text).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from emoprobe.activations import activations  # noqa: E402
from emoprobe.model import decoder_layers, load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL from generate_data.py")
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true")
    a = p.parse_args()

    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    model, tok = load_model(a.model, a.load_in_4bit)
    layers = list(range(len(decoder_layers(model))))
    acts = activations(model, tok, [r["text"] for r in rows], layers, batch_size=a.batch_size)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out + ".npy", acts.astype(np.float16))  # half the disk space
    meta = {"model": a.model, "layers": layers,
            "emotion": [r["emotion"] for r in rows], "topic_id": [r["topic_id"] for r in rows]}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(f"saved {a.out}.npy {acts.shape}")


if __name__ == "__main__":
    main()
