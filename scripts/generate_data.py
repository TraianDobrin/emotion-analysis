"""Step 1: have an LLM write emotion stories and neutral dialogues (JSONL). Resumable.

  python scripts/generate_data.py --kind story   --emotions happy-min --n-topics 30 --out data/happy_stories.jsonl
  python scripts/generate_data.py --kind neutral --n-topics 60 --n-per-call 6 --out data/neutral.jsonl

Needs data/paper_topics.txt and data/paper_story_prompt.txt (made by import_paper_assets.py)
and the API key of the chosen backend (default: GEMINI_API_KEY).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI  # noqa: E402

from emoprobe.emotions import EMOTIONS, HAPPY_MIN  # noqa: E402
from emoprobe.generate import PRESETS, build_jobs, run  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["story", "neutral"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--emotions", default="happy-min", help="'happy-min', 'all' (171), or a comma-separated list")
    p.add_argument("--n-topics", type=int, default=30)
    p.add_argument("--n-per-call", type=int, default=12, help="stories or dialogues per API call")
    p.add_argument("--topics-file", default="data/paper_topics.txt")
    p.add_argument("--story-prompt-file", default="data/paper_story_prompt.txt")
    p.add_argument("--backend", choices=list(PRESETS), default="gemini")
    p.add_argument("--model", default=None, help="override the backend's default model")
    p.add_argument("--reasoning-effort", default="low", help="for thinking models; 'none' to omit")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="print the job count and one prompt")
    a = p.parse_args()

    emotions = {"happy-min": HAPPY_MIN, "all": EMOTIONS}.get(a.emotions) or a.emotions.split(",")
    topics = read(a.topics_file).splitlines()[:a.n_topics]
    template = read(a.story_prompt_file) if a.kind == "story" else None
    jobs = build_jobs(a.kind, topics, a.n_per_call, emotions, template)
    print(f"{len(jobs)} calls, up to {len(jobs) * a.n_per_call} {a.kind} items")
    if a.dry_run:
        print(jobs[0].prompt)
        return

    base_url, model, key_env = PRESETS[a.backend]
    if not os.environ.get(key_env):
        sys.exit(f"Set {key_env} first.")
    client = OpenAI(api_key=os.environ[key_env], base_url=base_url)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    effort = None if a.reasoning_effort == "none" else a.reasoning_effort
    run(jobs, a.out, client, a.model or model, workers=a.workers, reasoning_effort=effort)


if __name__ == "__main__":
    main()
