"""Step 1: generate labelled emotion stories and neutral dialogues.

Examples
  # Real dataset with Claude (Message Batches API, needs ANTHROPIC_API_KEY or `ant auth login`)
  python scripts/generate_data.py --kind story   --backend anthropic --n-topics 20 --out data/stories.jsonl
  python scripts/generate_data.py --kind neutral --backend anthropic --n-topics 60 --out data/neutral.jsonl

  # Cheap: DeepSeek (OpenAI-compatible API, needs DEEPSEEK_API_KEY); resumable
  python scripts/generate_data.py --kind story --backend deepseek --emotions core --n-topics 20 --out data/stories.jsonl

  # Any other OpenAI-compatible endpoint (OpenRouter, Groq, Gemini presets, or your own)
  python scripts/generate_data.py --kind story --backend openrouter --api-model qwen/qwen-2.5-72b-instruct ...

  # Tiny local smoke test
  python scripts/generate_data.py --kind story --backend hf --hf-model Qwen/Qwen2.5-0.5B-Instruct \
      --emotions happy,afraid,calm,angry --n-topics 3 --n-per-call 2 --out data/smoke_stories.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emoprobe.emotions import CORE_EMOTIONS, EMOTIONS, HAPPY_MIN, TOPICS  # noqa: E402
from emoprobe.generate import (  # noqa: E402
    PRESETS, REASONING_DEFAULTS, build_jobs, run_anthropic_batches, run_hf, run_openai_compatible,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["story", "neutral"], required=True)
    p.add_argument("--backend", default="anthropic",
                   choices=["anthropic", "hf", "deepseek", "openrouter", "groq", "gemini", "openai-compatible"])
    p.add_argument("--api-model", default=None, help="model for OpenAI-compatible backends (default: preset's)")
    p.add_argument("--base-url", default=None, help="for --backend openai-compatible")
    p.add_argument("--api-key-env", default=None, help="env var holding the key (default: preset's)")
    p.add_argument("--workers", type=int, default=16, help="concurrent requests for API backends")
    p.add_argument("--reasoning-effort", default=None, help="for thinking models, e.g. none/low (gemini default: low)")
    p.add_argument("--emotions", default="all",
                   help="'all' (171), 'core' (16), 'happy-min' (happy + 5 contrasts), or comma-separated list")
    p.add_argument("--topics-file", default=None, help="one topic per line (default: built-in topics)")
    p.add_argument("--story-prompt-file", default=None,
                   help="story prompt template with {n_stories},{topic},{emotion} (e.g. the paper's)")
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
    elif a.emotions == "happy-min":
        emotions = HAPPY_MIN
    else:
        emotions = [e.strip() for e in a.emotions.split(",")]

    topics = TOPICS
    if a.topics_file:
        with open(a.topics_file, encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    topics = topics[: a.n_topics]

    template = None
    if a.story_prompt_file:
        with open(a.story_prompt_file, encoding="utf-8") as f:
            template = f.read().strip()
    jobs = build_jobs(a.kind, topics, emotions if a.kind == "story" else None, a.n_per_call,
                      a.calls_per_topic, story_template=template)
    n_items = len(jobs) * a.n_per_call
    print(f"{len(jobs)} calls, ~{n_items} {a.kind} items")
    if a.dry_run:
        print("\n--- example prompt ---\n" + jobs[0].prompt)
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.backend == "anthropic":
        run_anthropic_batches(jobs, a.out, model=a.claude_model, effort=a.effort)
    elif a.backend != "hf":
        base_url, model, key_env = PRESETS.get(a.backend, (None, None, "OPENAI_API_KEY"))
        base_url = a.base_url or base_url
        if base_url is None:
            sys.exit("--backend openai-compatible needs --base-url")
        run_openai_compatible(jobs, a.out, model=a.api_model or model, base_url=base_url,
                              api_key_env=a.api_key_env or key_env, workers=a.workers,
                              reasoning_effort=a.reasoning_effort or REASONING_DEFAULTS.get(a.backend))
    else:
        if os.path.exists(a.out):
            sys.exit(f"{a.out} exists; delete it or choose another --out (hf backend appends).")
        run_hf(jobs, a.out, a.hf_model)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
