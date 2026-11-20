"""Tests that need no model: vector recipe, held-out classification, parsing, generation loop."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from emoprobe.generate import build_jobs, names_emotion, run  # noqa: E402
from emoprobe.prompts import parse_items  # noqa: E402
from emoprobe.vectors import classify, compute_emotion_vectors, neutral_components, topic_split  # noqa: E402


def synthetic(n_emotions=5, n_per=40, d=64, seed=0):
    """Story = its emotion's direction + a big random amount of a 'style' direction + noise.
    Neutral text contains the style direction too, so the recipe should remove it."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_emotions, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    style = rng.normal(size=d)
    style /= np.linalg.norm(style)
    acts = np.array([2 * dirs[e] + 8 * rng.normal() * style + 0.3 * rng.normal(size=d)
                     for e in range(n_emotions) for _ in range(n_per)])[:, None]
    labels = [f"e{e}" for e in range(n_emotions) for _ in range(n_per)]
    topics = [i % 10 for _ in range(n_emotions) for i in range(n_per)]
    neutral = (8 * rng.normal(size=(200, 1)) * style + 0.3 * rng.normal(size=(200, d)))[:, None]
    return acts, labels, topics, neutral, dirs, style


def test_vectors_find_emotions_and_drop_style():
    acts, labels, _, neutral, dirs, style = synthetic()
    vecs, emotions, n_removed = compute_emotion_vectors(acts, labels, neutral)
    raw, _, _ = compute_emotion_vectors(acts, labels)
    v = vecs[:, 0]
    assert emotions == [f"e{i}" for i in range(5)] and n_removed == [1]
    assert np.abs(v @ style).max() < 0.1 * np.abs(raw[:, 0] @ style).max() + 1e-2
    centered = dirs - dirs.mean(0)
    cos = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ (centered / np.linalg.norm(centered, axis=1, keepdims=True)).T
    assert np.all(np.diag(cos) > 0.9)


def test_heldout_classification():
    acts, labels, topics, neutral, _, _ = synthetic()
    labels = np.array(labels)
    train, test = topic_split(topics, test_frac=0.3)
    vecs, emotions, _ = compute_emotion_vectors(acts[train], labels[train], neutral)
    pred = classify(acts[test, 0], vecs[:, 0], acts[train, 0].mean(0))
    assert (pred == np.array([emotions.index(e) for e in labels[test]])).mean() > 0.9


def test_neutral_components_orthonormal():
    rng = np.random.default_rng(1)
    b = neutral_components(rng.normal(size=(100, 16)) @ rng.normal(size=(16, 16)))
    assert np.allclose(b @ b.T, np.eye(len(b)), atol=1e-5)


def test_parse_items():
    x, y, z = "x" * 50, "y" * 50, "z" * 50
    assert parse_items(f"[story 1]\n{x}\n\n{z}\n[story 2]\n{y}") == [f"{x}\n\n{z}", y]  # paper format
    assert parse_items(f"{x}\n###\n{y}\n###\nshort") == [x, y]
    assert parse_items(f"### Story 1:\n{x}\n### Story 2:\n{y}") == [x, y]
    assert parse_items(f"{x}\n\n{y}") == [x, y]  # no separators: split on blank lines


def test_names_emotion():
    assert names_emotion("She smiled happily.", "happy")
    assert names_emotion("Pure happiness.", "happy")
    assert not names_emotion("He sat in the saddle.", "sad")
    assert not names_emotion("In this context.", "content")


def test_run_filters_resumes_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    story = "She walked into the kitchen and put the kettle on for the second time."
    calls = []

    def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise TimeoutError  # first attempt fails, the retry succeeds
        text = f"[story 1]\n{story}\n[story 2]\nI was so happy that the kettle finally whistled for me."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    jobs = build_jobs("story", ["A kettle"], 2, ["happy"], "{n_stories} {topic} {emotion}")
    out = str(tmp_path / "s.jsonl")
    run(jobs, out, client, "m", workers=1)
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert [r["text"] for r in rows] == [story]  # the story naming "happy" was dropped
    run(jobs, out, client, "m", workers=1)  # already done: no new calls
    assert len(calls) == 2
