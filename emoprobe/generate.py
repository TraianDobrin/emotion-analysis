"""Step 1: have an LLM write the labelled stories and neutral dialogues.

Any OpenAI-compatible chat API works (Gemini's free tier is what we used).
Output is JSONL, one row per story/dialogue:
  {"id", "kind": "story"|"neutral", "emotion", "topic_id", "topic", "text"}
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm.auto import tqdm

from .prompts import neutral_prompt, parse_items

# name: (base_url, default model, env var holding the key)
PRESETS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash", "GEMINI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "deepseek/deepseek-chat", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
}


@dataclass
class Job:
    """One API call: a prompt asking for several stories (or dialogues) at once."""
    id: str
    kind: str
    emotion: str | None
    topic_id: int
    topic: str
    prompt: str


def build_jobs(kind, topics, n_per_call, emotions=None, story_template=None):
    """One job per (emotion, topic) for stories, one per topic for neutral dialogues."""
    jobs = []
    for ti, topic in enumerate(topics):
        if kind == "neutral":
            jobs.append(Job(f"n-{ti:03d}", kind, None, ti, topic, neutral_prompt(topic, n_per_call)))
            continue
        for ei, emo in enumerate(emotions):
            prompt = story_template.format(n_stories=n_per_call, topic=topic, emotion=emo)
            jobs.append(Job(f"s-{ei:03d}-{ti:03d}", kind, emo, ti, topic, prompt))
    return jobs


def names_emotion(text, emotion):
    """True if the text uses the emotion word itself ('happy', 'happily', 'happiness').
    Whole words only, so 'sad' does not match 'saddle'."""
    e = re.escape(emotion.lower())
    stems = [e, e[:-1] + "i"] if emotion.endswith("y") else [e]
    return re.search(rf"\b(?:{'|'.join(stems)})(?:ly|ness|ed|er|est|s)?\b", text.lower()) is not None


def to_rows(job, text):
    """Split one response into rows. Stories that name their emotion are dropped;
    dialogue speakers are renamed to Human/Assistant as in the paper."""
    items = parse_items(text)
    if job.kind == "story":
        items = [t for t in items if not names_emotion(t, job.emotion)]
    else:
        items = [re.sub(r"^Person:", "Human:", re.sub(r"^AI:", "Assistant:", t, flags=re.M), flags=re.M)
                 for t in items]
    return [{"id": f"{job.id}-{k}", "kind": job.kind, "emotion": job.emotion,
             "topic_id": job.topic_id, "topic": job.topic, "text": t} for k, t in enumerate(items)]


def done_job_ids(path):
    """Jobs that already have rows in `path`, so an interrupted run can resume."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"].rsplit("-", 1)[0] for line in f if line.strip()}


def run(jobs, out_path, client, model, workers=16, reasoning_effort=None, max_retries=6):
    """Send jobs concurrently and append parsed rows to out_path. Re-running resumes."""
    done = done_job_ids(out_path)
    todo = [j for j in jobs if j.id not in done]
    print(f"{len(jobs) - len(todo)} jobs already done, {len(todo)} to run")
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def call(job):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=16000, temperature=1.0,
                    messages=[{"role": "user", "content": job.prompt}], **extra)
                return resp.choices[0].message.content or ""
            except Exception as e:  # rate limit, timeout, 5xx: wait 2, 4, ..., 32 s and retry
                print(f"{job.id}: {type(e).__name__}, retry {attempt + 1}/{max_retries}")
                time.sleep(min(60, 2 ** (attempt + 1)))
        return None

    n_rows = n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(out_path, "a", encoding="utf-8") as out:
        futures = {pool.submit(call, j): j for j in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            text = fut.result()
            if text is None:
                n_failed += 1
                continue
            for row in to_rows(futures[fut], text):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"wrote {n_rows} rows; {n_failed} jobs failed (re-run the same command to retry them)")
