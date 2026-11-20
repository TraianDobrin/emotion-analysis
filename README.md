# llm-emotion-probes

Emotion vectors (linear probes) for open-weights LLMs, following Anthropic's
"Emotion concepts and their function in a large language model" (2026).
First goal: a "happy" probe whose score can be used as an RL reward.

## How it works

1. **Write stories.** An LLM (Gemini, free tier) writes short stories about a character
   feeling one emotion, using the paper's prompt and topics. Stories that name the
   emotion are dropped. It also writes emotionally neutral Human/Assistant dialogues.
2. **Read activations.** Each text goes through the model being probed (Qwen2.5-3B-Instruct).
   At every layer we average the residual stream over the tokens from token 50 on.
3. **Build vectors.** Per layer: the mean of each emotion's stories, minus the mean over
   emotions, with the top principal components of the neutral dialogues (50% of their
   variance; topic and style, not emotion) projected out.
4. **Check them.** Classify stories from unseen topics, then read the probe at the token
   where the Assistant starts replying: Tylenol dose, good vs bad news, steering.

## Run it

```bash
pip install -e .[dev]
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build

# paper's topics and story prompt from a local copy of the paper (not committed)
curl -L -o paper.html https://transformer-circuits.pub/2026/emotions/index.html
python scripts/import_paper_assets.py paper.html data/

# needs GEMINI_API_KEY
python scripts/generate_data.py --kind story   --emotions happy-min --n-topics 30 --out data/happy_stories.jsonl
python scripts/generate_data.py --kind neutral --n-topics 60 --n-per-call 6 --out data/neutral.jsonl

python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/happy_stories.jsonl --out acts/q3b_happy
python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/neutral.jsonl --out acts/q3b_neutral
python scripts/build_vectors.py --stories acts/q3b_happy --neutral acts/q3b_neutral --out probes/q3b_happy.pt
python scripts/validate.py --probe probes/q3b_happy.pt --stories acts/q3b_happy --neutral acts/q3b_neutral --steer
```

Results so far: [RESULTS.md](RESULTS.md).

## Use as a reward

```python
from emoprobe.model import load_model
from emoprobe.probe import EmotionProbe

model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")
probe = EmotionProbe(model, tok, "probes/q3b_happy.pt")
probe.reward(["You found a lost child and brought them home."], {"happy": 1.0})
```

## Files

| file | what |
|---|---|
| `emoprobe/emotions.py` | the paper's 171 emotion words, and the 6 used for happiness |
| `emoprobe/prompts.py` | neutral-dialogue prompt, response parser |
| `emoprobe/generate.py` | step 1: API calls, filtering, resume |
| `emoprobe/model.py` | load a model, find its layers |
| `emoprobe/activations.py` | step 2: hooks that record the residual stream |
| `emoprobe/vectors.py` | step 3: the vector recipe, held-out split |
| `emoprobe/probe.py` | `EmotionProbe`: scores and rewards |
| `emoprobe/steering.py` | add a vector during generation |
| `scripts/` | one command-line script per step |
