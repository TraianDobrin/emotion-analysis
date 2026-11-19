"""Model-free tests: vector construction, denoising, parsing, layer specs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from emoprobe.model import parse_layers  # noqa: E402
from emoprobe.prompts import parse_items, story_prompt  # noqa: E402
from emoprobe.vectors import (  # noqa: E402
    classify_by_vectors, compute_emotion_vectors, neutral_components, topic_split,
)


def _synthetic(seed=0, n_per=40, d=64, n_emotions=5, confound=8.0):
    """Stories = emotion direction + a large shared confound direction (also present
    in neutral text with varying strength) + noise."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_emotions, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    conf = rng.normal(size=d)
    conf /= np.linalg.norm(conf)
    acts, labels, topics = [], [], []
    for e in range(n_emotions):
        for i in range(n_per):
            c = rng.normal() * confound
            acts.append(2.0 * dirs[e] + c * conf + 0.3 * rng.normal(size=d))
            labels.append(f"e{e}")
            topics.append(i % 10)
    acts = np.array(acts)[:, None, :]  # one layer
    neutral = (rng.normal(size=(200, 1)) * confound) * conf + 0.3 * rng.normal(size=(200, d))
    return acts, labels, topics, neutral[:, None, :], dirs, conf


def test_vectors_recover_directions_and_remove_confound():
    acts, labels, _, neutral, dirs, conf = _synthetic()
    vecs, emotions, ncomp = compute_emotion_vectors(acts, labels, neutral, var_frac=0.5)
    assert emotions == [f"e{i}" for i in range(5)]
    assert ncomp == [1]
    v = vecs[:, 0]
    # confound removed (up to estimation noise in the neutral PC)
    raw, _, _ = compute_emotion_vectors(acts, labels, None)
    assert np.all(np.abs(v @ conf) < 0.1 * np.abs(raw[:, 0] @ conf).max() + 1e-2)
    # each vector most similar to its own true (mean-centered) direction
    centered = dirs - dirs.mean(0)
    cos = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ (
        centered / np.linalg.norm(centered, axis=1, keepdims=True)).T
    assert np.all(cos.argmax(1) == np.arange(5))
    assert np.all(np.diag(cos) > 0.9)


def test_heldout_classification_above_chance():
    acts, labels, topics, neutral, _, _ = _synthetic()
    train, test = topic_split(topics, test_frac=0.3)
    labels = np.array(labels)
    vecs, emotions, _ = compute_emotion_vectors(acts[train], labels[train].tolist(), neutral)
    pred = classify_by_vectors(acts[test, 0], vecs[:, 0], acts[train, 0].mean(0))
    y = np.array([emotions.index(e) for e in labels[test]])
    assert (pred == y).mean() > 0.9


def test_neutral_components_orthonormal():
    rng = np.random.default_rng(1)
    b = neutral_components(rng.normal(size=(100, 16)) @ rng.normal(size=(16, 16)), 0.5)
    assert np.allclose(b @ b.T, np.eye(len(b)), atol=1e-5)


def test_parse_items():
    text = "[story 1]\n" + "A" * 50 + "\n###\nStory 2: " + "B" * 50 + "\n###\n\nshort"
    items = parse_items(text)
    assert items == ["A" * 50, "B" * 50]
    assert parse_items("x" * 50 + "\n\n" + "y" * 50) == ["x" * 50, "y" * 50]
    assert parse_items("### Story 1:\n" + "x" * 50 + "\n\n### Story 2:\n" + "y" * 50) == ["x" * 50, "y" * 50]


def test_names_emotion():
    from emoprobe.generate import names_emotion

    assert names_emotion("She smiled happily.", "happy")
    assert names_emotion("Pure happiness.", "happy")
    assert names_emotion("He was AFRAID.", "afraid")
    assert names_emotion("They felt on edge all day.", "on edge")
    assert not names_emotion("He sat on the saddle.", "sad")
    assert not names_emotion("Out of context.", "content")
    assert not names_emotion("Her hands shook.", "afraid")


class _FakeClient:
    """Mimics openai.OpenAI().chat.completions.create; fails once per job to test retries."""

    def __init__(self):
        self.seen = set()
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kw):
        prompt = messages[0]["content"]
        if prompt not in self.seen:
            self.seen.add(prompt)
            raise RuntimeError("rate limited")
        good = "She gripped the railing until her knuckles went white, " * 2
        bad = "He was so afraid that he could not move from the spot at all."
        msg = type("M", (), {"content": f"{good}\n###\n{bad}\n###\n{good}"})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_openai_compatible_backend_resumes_and_filters(tmp_path, monkeypatch):
    from emoprobe import generate

    monkeypatch.setattr(generate.time, "sleep", lambda s: None)
    jobs = generate.build_jobs("story", ["Topic A", "Topic B"], ["afraid"], n_per_call=3)
    out = str(tmp_path / "s.jsonl")
    generate.run_openai_compatible(jobs[:1], out, "m", "u", "K", workers=2, client=_FakeClient())
    generate.run_openai_compatible(jobs, out, "m", "u", "K", workers=2, client=_FakeClient())
    import json

    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert len(rows) == 4  # 2 jobs x 2 stories; the story naming 'afraid' is dropped
    assert {r["topic"] for r in rows} == {"Topic A", "Topic B"}  # job A not redone


def test_prompt_and_layers():
    assert '"proud"' in story_prompt("A topic", "proud", 3)
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("every:2", 5) == [0, 2, 4]
    assert parse_layers("every:2", 4) == [0, 2, 3]
    assert parse_layers("frac:0.66", 24) == [16]
    assert parse_layers("3,5", 24) == [3, 5]
