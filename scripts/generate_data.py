"""Step 1: generate labelled emotion stories and neutral dialogues.

Examples
  # Real dataset with Claude (Message Batches API, needs ANTHROPIC_API_KEY or `ant auth login`)
  python scripts/generate_data.py --kind story   --backend anthropic --n-topics 20 --out data/stories.jsonl
  python scripts/generate_data.py --kind neutral --backend anthropic --n-topics 60 --out data/neutral.jsonl

  # Tiny local smoke test
  python scripts/generate_data.py --kind story --backend hf --hf-model Qwen/Qwen2.5-0.5B-Instruct \
      --emotions happy,afraid,calm,angry --n-topics 3 --n-per-call 2 --out data/smoke_stories.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emoprobe.emotions import CORE_EMOTIONS, EMOTIONS, TOPICS  # noqa: E402
from emoprobe.generate import build_jobs, run_anthropic_batches, run_hf  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["story", "neutral"], required=True)
    p.add_argument("--backend", choices=["anthropic", "hf"], default="anthropic")
    p.add_argument("--emotions", default="all", help="'all' (171), 'core' (16), or comma-separated list")
    p.add_argument("--topics-file", default=None, help="one topic per line (default: built-in topics)")
    p.add_argument("--n-topics", type=int, default=20)
    p.add_argument("--n-per-call", type=int, default=6, help="stories/dialogues requested per call")
    p.add_argument("--calls-per-topic", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--claude-model", default="claude-opus-5-5")
    p.add_argument("--effort", default="low")
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dry-run", action="store_true", help="print job count and one prompt, then exit")
    a = p.parse_args()

    if a.emotions == "all":
        emotions = EMOTIONS
    elif a.emotions == "core":
        emotions = CORE_EMOTIONS
    else:
        emotions = [e.strip() for e in a.emotions.split(",")]

    topics = TOPICS
    if a.topics_file:
        with open(a.topics_file, encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    topics = topics[: a.n_topics]

    jobs = build_jobs(a.kind, topics, emotions if a.kind == "story" else None, a.n_per_call, a.calls_per_topic)
    n_items = len(jobs) * a.n_per_call
    print(f"{len(jobs)} calls, ~{n_items} {a.kind} items")
    if a.dry_run:
        print("\n--- example prompt ---\n" + jobs[0].prompt)
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.backend == "anthropic":
        run_anthropic_batches(jobs, a.out, model=a.claude_model, effort=a.effort)
    else:
        if os.path.exists(a.out):
            sys.exit(f"{a.out} exists; delete it or choose another --out (hf backend appends).")
        run_hf(jobs, a.out, a.hf_model)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
