"""Step 3: compute emotion vectors (difference of means, neutral-PC denoising) and save a probe.

  python scripts/build_vectors.py --stories acts/qwen7b_stories --neutral acts/qwen7b_neutral \
      --out probes/qwen7b.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from emoprobe.probe import save_probe  # noqa: E402
from emoprobe.vectors import compute_emotion_vectors  # noqa: E402


def load(prefix):
    with open(prefix + ".meta.json") as f:
        meta = json.load(f)
    return np.load(prefix + ".npy"), meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stories", required=True, help="activation prefix for stories")
    p.add_argument("--neutral", default=None, help="activation prefix for neutral dialogues (recommended)")
    p.add_argument("--var-frac", type=float, default=0.5, help="neutral variance to project out")
    p.add_argument("--default-layer", type=int, default=None, help="default: extracted layer nearest 2/3 depth")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    acts, meta = load(a.stories)
    neutral = None
    if a.neutral:
        neutral, nmeta = load(a.neutral)
        assert nmeta["layers"] == meta["layers"], "stories and neutral must use the same layers"

    vectors, emotions, ncomp = compute_emotion_vectors(acts, meta["emotion"], neutral, a.var_frac)
    layers = meta["layers"]
    default = a.default_layer
    if default is None:
        target = 2 * meta["n_layers"] / 3
        default = min(layers, key=lambda l: abs(l - target))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_probe(
        a.out, model_name=meta["model"], layers=layers, emotions=emotions, vectors=vectors,
        story_mean=acts.astype(np.float32).mean(0),
        neutral_mean=None if neutral is None else neutral.astype(np.float32).mean(0),
        default_layer=default, start_token=meta["start_token"], n_components=ncomp,
    )
    print(f"{len(emotions)} emotions x {len(layers)} layers; default layer {default}")
    if ncomp:
        print(f"neutral PCs projected out per layer: {ncomp}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
