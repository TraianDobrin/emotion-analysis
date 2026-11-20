"""Step 4: check that the probe measures emotion and not just surface words.

  python scripts/validate.py --probe probes/q3b_happy.pt --stories acts/q3b_happy --neutral acts/q3b_neutral --steer

1. Held-out topics: build vectors on some topics, classify stories from unseen topics.
2. Tylenol dose: fear should rise and calm fall as the dose becomes dangerous.
3. Happiness: good vs bad versions of a situation; a sister's age at death.
4. Steering (--steer): add a vector while the model completes "I feel...".
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from emoprobe.model import load_model  # noqa: E402
from emoprobe.probe import EmotionProbe  # noqa: E402
from emoprobe.steering import generate, steer  # noqa: E402
from emoprobe.vectors import classify, compute_emotion_vectors, topic_split  # noqa: E402

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


def heldout(stories, neutral, var_frac):
    acts = np.load(stories + ".npy")
    with open(stories + ".meta.json") as f:
        meta = json.load(f)
    labels = np.asarray(meta["emotion"])
    train, test = topic_split(meta["topic_id"])
    vecs, emotions, _ = compute_emotion_vectors(acts[train], labels[train], np.load(neutral + ".npy"), var_frac)
    y = np.array([emotions.index(e) for e in labels[test]])
    print(f"\n[1] Held-out topics: {len(emotions)} emotions (chance {1 / len(emotions):.2f}), "
          f"{train.sum()} train / {test.sum()} test stories")
    print("layer  emotion-vectors  logistic-probe")
    for layer in meta["layers"]:
        x_train, x_test = acts[train, layer].astype(np.float32), acts[test, layer].astype(np.float32)
        acc_vec = (classify(x_test, vecs[:, layer], x_train.mean(0)) == y).mean()
        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        acc_log = logistic.fit(x_train, labels[train]).score(x_test, labels[test])
        print(f"{layer:5d}  {acc_vec:15.2f}  {acc_log:14.2f}")


def dose_test(ep):
    doses = [500, 1000, 2000, 4000, 8000, 16000]
    scores = ep.score([f"I just took {d} mg of Tylenol for a headache. Is there anything I should do now?"
                       for d in doses])
    cols = [e for e in ["afraid", "calm", "happy"] if e in ep.emotions]
    print(f"\n[2] Tylenol dose at the Assistant token (layer {ep.layer}): expect afraid up, calm down")
    print("dose mg " + "".join(f"{c:>9s}" for c in cols))
    for d, row in zip(doses, scores):
        print(f"{d:7d} " + "".join(f"{row[ep.emotions.index(c)]:9.2f}" for c in cols))


def happiness_test(ep):
    happy = ep.emotions.index("happy")
    good = ep.score([g for g, _ in HAPPY_PAIRS])[:, happy]
    bad = ep.score([b for _, b in HAPPY_PAIRS])[:, happy]
    print(f"\n[3] 'happy' for good vs bad situations: good higher in {(good > bad).sum()}/{len(good)}, "
          f"mean difference {np.mean(good - bad):+.2f}")
    for (g, _), sg, sb in zip(HAPPY_PAIRS, good, bad):
        print(f"   {sg:+6.2f} vs {sb:+6.2f}  {g}")
    ages = [5, 20, 40, 60, 80, 100]
    s = ep.score([f"My sister passed away. She was {age} years old." for age in ages])[:, happy]
    print("   sister's age at death -> happy: " + ", ".join(f"{age}: {x:+.2f}" for age, x in zip(ages, s)))


def steering_demo(ep, model, tok):
    prompt = ep.assistant_prompt("How are you feeling right now? Answer in one sentence.") + "I feel"
    print(f"\n[4] Steering at layer {ep.layer} (strength = fraction of the residual norm)")
    print(f"   baseline:  I feel{generate(model, tok, prompt)!r}")
    for emotion in ep.emotions:
        for strength in (0.1, 0.3):
            with steer(model, ep.layer, ep.vector(emotion), strength):
                print(f"   {emotion} {strength}:  I feel{generate(model, tok, prompt)!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--stories", required=True, help="story activations (path without extension)")
    p.add_argument("--neutral", required=True, help="neutral activations (path without extension)")
    p.add_argument("--var-frac", type=float, default=0.5)
    p.add_argument("--steer", action="store_true")
    a = p.parse_args()

    heldout(a.stories, a.neutral, a.var_frac)
    probe = torch.load(a.probe, weights_only=True)
    model, tok = load_model(probe["model_name"])
    ep = EmotionProbe(model, tok, probe)
    dose_test(ep)
    happiness_test(ep)
    if a.steer:
        steering_demo(ep, model, tok)


if __name__ == "__main__":
    main()
