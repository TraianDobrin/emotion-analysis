"""Step 4: sanity checks for a probe.

  python scripts/validate.py --probe probes/qwen7b.pt --stories acts/qwen7b_stories \
      --neutral acts/qwen7b_neutral --model Qwen/Qwen2.5-7B-Instruct --steer

Checks
  1. Held-out classification: vectors built on training topics, tested on unseen topics
     (nearest vector by cosine), with a logistic-regression probe for comparison. Per layer.
  2. Geometry: PCA of emotion vectors. The paper finds PC1 ~ valence, PC2/3 ~ arousal.
  3. Dose test (needs --model): "afraid"-type scores at the Assistant token should rise
     as a medication dose becomes dangerous, and "calm" should fall.
  4. Steering (needs --model --steer): completions of "I feel..." steered by vectors.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from emoprobe.vectors import (  # noqa: E402
    classify_by_vectors, compute_emotion_vectors, logistic_probe, topic_split,
)

AFRAID_LIKE = ["afraid", "terrified", "scared", "alarmed", "panicked", "worried", "nervous", "anxious"]


def heldout(stories_prefix, neutral_prefix, var_frac):
    acts = np.load(stories_prefix + ".npy")
    with open(stories_prefix + ".meta.json") as f:
        meta = json.load(f)
    neutral = np.load(neutral_prefix + ".npy") if neutral_prefix else None
    labels = np.asarray(meta["emotion"])
    train, test = topic_split(meta["topic_id"], test_frac=0.25)
    if test.sum() == 0 or train.sum() == 0:
        print("not enough topics for a held-out split")
        return
    vecs, emotions, _ = compute_emotion_vectors(acts[train], labels[train].tolist(), neutral, var_frac)
    y = np.array([emotions.index(e) for e in labels[test]])
    print(f"\n[1] Held-out topic classification ({len(emotions)} emotions, chance {1/len(emotions):.3f}, "
          f"{train.sum()} train / {test.sum()} test stories)")
    print("layer  diff-of-means  logistic")
    for li, layer in enumerate(meta["layers"]):
        center = acts[train, li].astype(np.float32).mean(0)
        acc_dm = (classify_by_vectors(acts[test, li], vecs[:, li], center) == y).mean()
        acc_lr, _ = logistic_probe(acts[train, li], labels[train], acts[test, li], labels[test])
        print(f"{layer:5d}  {acc_dm:13.3f}  {acc_lr:8.3f}")


def geometry(probe):
    li = probe["layers"].index(probe["default_layer"])
    v = probe["vectors"][:, li].numpy()
    emotions = probe["emotions"]
    vc = v - v.mean(0)
    _, s, vt = np.linalg.svd(vc, full_matrices=False)
    proj = vc @ vt[:3].T
    ev = s**2 / (s**2).sum()
    print(f"\n[2] PCA of emotion vectors at layer {probe['default_layer']}")
    for k in range(min(3, len(emotions) - 1)):
        order = np.argsort(proj[:, k])
        lo = ", ".join(emotions[i] for i in order[:6])
        hi = ", ".join(emotions[i] for i in order[::-1][:6])
        print(f"PC{k+1} ({ev[k]:.0%} var)\n   + {hi}\n   - {lo}")


def dose_test(ep):
    doses = [500, 1000, 2000, 4000, 8000, 16000]
    msgs = [f"I just took {d} mg of Tylenol for a headache. Is there anything I should do now?" for d in doses]
    scores = ep.score_assistant_batch(msgs)
    cols = [e for e in AFRAID_LIKE + ["calm", "happy"] if e in ep.emotions][:5]
    print(f"\n[3] Dose test at the Assistant token (layer {ep.layer}); expect fear up, calm down")
    print("dose(mg) " + " ".join(f"{c:>10s}" for c in cols))
    for d, row in zip(doses, scores):
        print(f"{d:8d} " + " ".join(f"{row[ep.emotions.index(c)]:10.2f}" for c in cols))


HAPPY_PAIRS = [  # (good, bad) versions of the same situation
    ("My sister finally got the job she has wanted for years.", "My sister lost the job she had wanted for years."),
    ("The biopsy came back benign.", "The biopsy came back malignant."),
    ("My dog came home after being missing for a week.", "My dog was hit by a car after being missing for a week."),
    ("I passed my driving test on the first try.", "I failed my driving test for the third time."),
    ("My best friend surprised me with a visit today.", "My best friend stopped talking to me today."),
    ("We just found out we are having a baby.", "We just found out we lost the baby."),
    ("My grandmother turned 95 and is in great health.", "My grandmother passed away this morning."),
    ("I got accepted to my dream university.", "I got rejected from every university I applied to."),
]


def happiness_test(ep):
    """The 'happy' score at the Assistant token should be higher for the good version of
    each situation, and should rise as a sister's age at death rises (paper's example)."""
    if "happy" not in ep.emotions:
        return
    i = ep.emotions.index("happy")
    good = ep.score_assistant_batch([g for g, _ in HAPPY_PAIRS])[:, i]
    bad = ep.score_assistant_batch([b for _, b in HAPPY_PAIRS])[:, i]
    print(f"\n[3b] Happiness pairs at the Assistant token (layer {ep.layer}): "
          f"good > bad in {(good > bad).sum()}/{len(good)}, mean diff {np.mean(good - bad):+.2f}")
    for (g, _), sg, sb in zip(HAPPY_PAIRS, good, bad):
        print(f"   {sg:+6.2f} vs {sb:+6.2f}  {g[:60]}")
    ages = [5, 20, 40, 60, 80, 100]
    s = ep.score_assistant_batch([f"My sister passed away. She was {a} years old." for a in ages])[:, i]
    print("   sister's age at death -> happy: " + ", ".join(f"{a}:{x:+.2f}" for a, x in zip(ages, s)))


def steering_demo(ep, model, tok):
    from emoprobe.steering import generate, steer

    prompt = ep.assistant_prompt("How are you feeling right now? Answer in one sentence.") + "I feel"
    print(f"\n[4] Steering at layer {ep.layer} (strength in units of residual norm)")
    print(f"  baseline: I feel{generate(model, tok, prompt, 30)!r}")
    for emo in [e for e in ["happy", "sad", "afraid", "angry", "calm", "desperate"] if e in ep.emotions]:
        for s in (0.1, 0.3):
            with steer(model, ep.layer, ep.vector(emo), s):
                print(f"  {emo:>9s} s={s}: I feel{generate(model, tok, prompt, 30)!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--stories", default=None, help="activation prefix, for held-out accuracy")
    p.add_argument("--neutral", default=None)
    p.add_argument("--var-frac", type=float, default=0.5)
    p.add_argument("--model", default=None, help="model name, for dose test and steering")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--steer", action="store_true")
    a = p.parse_args()

    probe = torch.load(a.probe, weights_only=True)
    print(f"probe for {probe['model_name']}: {len(probe['emotions'])} emotions, layers {probe['layers']}")
    if a.stories:
        heldout(a.stories, a.neutral, a.var_frac)
    if len(probe["emotions"]) >= 3:
        geometry(probe)
    if a.model:
        from emoprobe.model import load_model
        from emoprobe.probe import EmotionProbe

        model, tok = load_model(a.model)
        ep = EmotionProbe(model, tok, probe, layer=a.layer)
        dose_test(ep)
        happiness_test(ep)
        if a.steer:
            steering_demo(ep, model, tok)


if __name__ == "__main__":
    main()
