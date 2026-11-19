"""Generating the labelled stories and neutral dialogues.

Two backends:
- "anthropic": Claude via the Message Batches API (50% price, async). Recommended
  for the real dataset; story quality matters more than the probed model's size.
- "hf": a local Hugging Face chat model. Fine for smoke tests, weak for real data.

Output is JSONL with one row per item:
  {"id", "kind": "story"|"neutral", "emotion", "topic_id", "topic", "text"}
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm.auto import tqdm

from .prompts import neutral_prompt, parse_items, story_prompt


@dataclass
class Job:
    custom_id: str
    kind: str
    emotion: str | None
    topic_id: int
    topic: str
    prompt: str


def build_jobs(kind: str, topics: list[str], emotions: list[str] | None, n_per_call: int,
               calls_per_topic: int = 1) -> list[Job]:
    jobs = []
    for ti, topic in enumerate(topics):
        for rep in range(calls_per_topic):
            if kind == "story":
                for ei, emo in enumerate(emotions):
                    jobs.append(Job(f"s-{ei:03d}-{ti:03d}-{rep}", kind, emo, ti, topic,
                                    story_prompt(topic, emo, n_per_call)))
            else:
                jobs.append(Job(f"n-{ti:03d}-{rep}", kind, None, ti, topic, neutral_prompt(topic, n_per_call)))
    return jobs


def names_emotion(text: str, emotion: str) -> bool:
    """True if the story uses the emotion word itself or a simple inflection of it
    ('happy', 'happily', 'happiness', 'calmly'). Whole words only, so 'sad' does not
    match 'saddle'. Synonyms (e.g. 'anger' for 'angry') are not caught."""
    e = re.escape(emotion.lower())
    alts = [e] + ([e[:-1] + "i"] if emotion.endswith("y") else [])
    pattern = r"\b(?:" + "|".join(alts) + r")(?:ly|ness|ed|er|est|s)?\b"
    return re.search(pattern, text.lower()) is not None


def _rows(job: Job, text: str, drop_named: bool = True) -> list[dict]:
    items = parse_items(text)
    if job.kind == "story" and drop_named:
        items = [t for t in items if not names_emotion(t, job.emotion)]
    return [
        {"id": f"{job.custom_id}-{k}", "kind": job.kind, "emotion": job.emotion,
         "topic_id": job.topic_id, "topic": job.topic, "text": item}
        for k, item in enumerate(items)
    ]


def _append(path: str, rows: list[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- Anthropic Message Batches -------------------------------------------------------
def run_anthropic_batches(jobs: list[Job], out_path: str, model: str = "claude-opus-5-5",
                          effort: str = "low", max_tokens: int = 16000, chunk: int = 10000,
                          poll_seconds: int = 60):
    """Submit jobs as message batches, wait, and write parsed items to out_path.

    Batch ids are stored in `<out_path>.batches.json`; re-running the same command
    resumes polling instead of resubmitting (and costing money twice).
    """
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    state_path = out_path + ".batches.json"
    by_id = {j.custom_id: j for j in jobs}

    if os.path.exists(state_path):
        with open(state_path) as f:
            batch_ids = json.load(f)
        print(f"Resuming {len(batch_ids)} existing batch(es) from {state_path}")
    else:
        batch_ids = []
        for i in range(0, len(jobs), chunk):
            reqs = [
                Request(
                    custom_id=j.custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=max_tokens,
                        output_config={"effort": effort},
                        messages=[{"role": "user", "content": j.prompt}],
                    ),
                )
                for j in jobs[i : i + chunk]
            ]
            batch = client.messages.batches.create(requests=reqs)
            batch_ids.append(batch.id)
            print(f"Created batch {batch.id} with {len(reqs)} requests")
        with open(state_path, "w") as f:
            json.dump(batch_ids, f)

    for bid in batch_ids:
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            print(f"{bid}: {b.processing_status}, processing={b.request_counts.processing}")
            time.sleep(poll_seconds)
        n_ok = n_bad = 0
        for res in client.messages.batches.results(bid):
            job = by_id.get(res.custom_id)
            if job is None:
                continue
            if res.result.type == "succeeded":
                msg = res.result.message
                if msg.stop_reason == "refusal":
                    n_bad += 1
                    continue
                text = "".join(b.text for b in msg.content if b.type == "text")
                _append(out_path, _rows(job, text))
                n_ok += 1
            else:
                n_bad += 1
        print(f"{bid}: {n_ok} succeeded, {n_bad} failed/refused (re-run failed ids separately if needed)")


# ---- OpenAI-compatible APIs (DeepSeek, OpenRouter, Groq, Together, Gemini, ...) --------
PRESETS = {
    # name: (base_url, default model, env var holding the key)
    "deepseek": ("https://api.deepseek.com", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "deepseek/deepseek-chat", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash", "GEMINI_API_KEY"),
}


def _done_job_ids(out_path: str) -> set[str]:
    """Job ids that already have rows in out_path, so an interrupted run can resume."""
    if not os.path.exists(out_path):
        return set()
    with open(out_path, encoding="utf-8") as f:
        return {json.loads(line)["id"].rsplit("-", 1)[0] for line in f if line.strip()}


def run_openai_compatible(jobs: list[Job], out_path: str, model: str, base_url: str,
                          api_key_env: str, workers: int = 16, max_tokens: int = 3000,
                          temperature: float = 1.0, max_retries: int = 5, client=None):
    """Run jobs concurrently against an OpenAI-compatible chat endpoint.

    Resumable: jobs whose rows are already in out_path are skipped. Jobs that fail
    after retries are reported and can be picked up by re-running the same command.
    """
    if client is None:
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"Set {api_key_env} to your API key first.")
        client = OpenAI(api_key=key, base_url=base_url)

    done = _done_job_ids(out_path)
    todo = [j for j in jobs if j.custom_id not in done]
    print(f"{len(jobs) - len(todo)} jobs already done, {len(todo)} to run")
    lock = threading.Lock()
    stats = {"items": 0, "failed": 0}

    def work(job: Job):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": job.prompt}],
                )
                return job, resp.choices[0].message.content or ""
            except Exception as e:  # rate limits, timeouts, 5xx: back off and retry
                if attempt == max_retries - 1:
                    print(f"{job.custom_id} failed: {type(e).__name__}: {e}")
                    return job, None
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, j) for j in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generating"):
            job, text = fut.result()
            with lock:
                if text is None:
                    stats["failed"] += 1
                    continue
                rows = _rows(job, text)
                _append(out_path, rows)
                stats["items"] += len(rows)
    print(f"wrote {stats['items']} items; {stats['failed']} jobs failed (re-run to retry them)")


# ---- Local Hugging Face model ----------------------------------------------------------
def run_hf(jobs: list[Job], out_path: str, model_name: str, max_new_tokens: int = 1200,
           temperature: float = 0.9):
    import torch

    from .model import load_model

    model, tok = load_model(model_name)
    for job in tqdm(jobs, desc="generating"):
        prompt = tok.apply_chat_template([{"role": "user", "content": job.prompt}], tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        _append(out_path, _rows(job, text))
