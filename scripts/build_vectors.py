"""Step 3: compute emotion vectors and save them as a probe file.

  python scripts/build_vectors.py --stories acts/q3b_happy --neutral acts/q3b_neutral --out probes/q3b_happy.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from emoprobe.vectors import compute_emotion_vectors  # noqa: E402


def load(prefix):
    with open(prefix + ".meta.json") as f:
        return np.load(prefix + ".npy"), json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stories", required=True, help="activations of the stories (path without extension)")
    p.add_argument("--neutral", required=True, help="activations of the neutral dialogues")
    p.add_argument("--var-frac", type=float, default=0.5, help="share of neutral variance to project out")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    acts, meta = load(a.stories)
    neutral, _ = load(a.neutral)
    vectors, emotions, n_removed = compute_emotion_vectors(acts, meta["emotion"], neutral, a.var_frac)
    layers = meta["layers"]
    default_layer = round(2 * len(layers) / 3)  # the paper reads out at about 2/3 depth

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({
        "model_name": meta["model"], "emotions": emotions, "layers": layers, "default_layer": default_layer,
        "vectors": torch.from_numpy(vectors),
        "neutral_mean": torch.from_numpy(neutral.astype(np.float32).mean(0)),
    }, a.out)
    print(f"{len(emotions)} emotions x {len(layers)} layers, default layer {default_layer}; "
          f"neutral PCs removed per layer: {n_removed}")


if __name__ == "__main__":
    main()
