# llm-emotion-probes

Emotion vectors / linear probes for open-weights LLMs. This reimplements the extraction method of Anthropic's
[*Emotion concepts and their function in a large language model*](https://transformer-circuits.pub/2026/emotions/index.html)
(Apr 2026) and adds a probe API meant for use as an RL reward.

## Method (as in the paper)

1. **Stories.** For each of 171 emotion words, a strong model writes one-paragraph stories on many topics in which a character feels that emotion, without ever naming it. The paper used 100 topics × 12 stories per emotion.
2. **Activations.** Each story is run through the probed model. The residual stream after each block is averaged over token positions **from the 50th token on**.
3. **Vectors.** Average per emotion, then **subtract the mean across emotions**.
4. **Denoising.** Run emotionally **neutral** Person/AI dialogues through the model, take the top principal components that explain **50% of the variance**, and project them out of every emotion vector.
5. **Layer.** By default the layer about **two-thirds of the way** through the model. In the paper, early-middle layers track the emotion of the *present content*, while middle-late layers track the emotion that drives the *upcoming output*.
6. **Readout of the model's own state.** Read at the **Assistant-turn token**, i.e. the last token of the prompt before the model replies. In the paper's dose test, the fear signal peaks at that token.
7. **Steering** adds `strength × ‖h‖ × v̂` to the residual stream, so strength is a fraction of the residual norm.

Differences from the paper:
- The topic list and prompt wording are our own. To use the paper's exact 100 topics, see `scripts/import_paper_topics.py`.
- Stories default to fewer per emotion, to keep cost down.
- Logistic-regression probes are included as a baseline.

## Install

```bash
pip install -e .
```

On this machine the installed torch is CPU-only. For the RTX 4060, install a CUDA build first:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

To generate stories with Claude you also need an Anthropic API key (`ANTHROPIC_API_KEY`) or `ant auth login`.

## Quick start: happiness only, cheap

The vector is "happy stories minus the mean over emotions", so happiness needs a few contrast emotions. `--emotions happy-min` uses happy, sad, afraid, angry, calm and bored. Story generation can use any OpenAI-compatible API: the `deepseek`, `openrouter`, `groq` and `gemini` presets, or `--backend openai-compatible --base-url ...`. Runs are resumable, and stories that name their own emotion are dropped automatically.

```bash
set DEEPSEEK_API_KEY=...     # or GROQ_API_KEY / GEMINI_API_KEY with --backend groq / gemini (free tiers)
python scripts/generate_data.py --kind story   --backend deepseek --emotions happy-min --n-topics 30 --out data/happy_stories.jsonl
python scripts/generate_data.py --kind neutral --backend deepseek --n-topics 30 --out data/neutral.jsonl
python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/happy_stories.jsonl --out acts/happy --layers every:2
python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/neutral.jsonl --out acts/neutral --layers every:2
python scripts/build_vectors.py --stories acts/happy --neutral acts/neutral --out probes/happy.pt
python scripts/validate.py --probe probes/happy.pt --stories acts/happy --neutral acts/neutral --model Qwen/Qwen2.5-3B-Instruct --steer
```

That is 180 + 30 API calls, giving about 1,000 stories and 180 dialogues. At DeepSeek-class prices it costs cents. Check the provider's current pricing.

`validate.py` reports held-out accuracy, and for happiness two specific tests:
- **good vs. bad pairs:** the same situation with good or bad news, where "happy" should be higher for the good version
- **the paper's sister example:** "happy" should rise as the sister's age at death rises

## Pipeline

```bash
# 1. Data (Claude, Message Batches API). Use --dry-run first to see call counts.
python scripts/generate_data.py --kind story   --n-topics 20 --n-per-call 6 --out data/stories.jsonl
python scripts/generate_data.py --kind neutral --n-topics 60 --n-per-call 6 --out data/neutral.jsonl

# 2. Activations (pick the model you will probe)
python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/stories.jsonl --out acts/q3b_stories --layers every:2
python scripts/extract_activations.py --model Qwen/Qwen2.5-3B-Instruct --data data/neutral.jsonl --out acts/q3b_neutral --layers every:2

# 3. Vectors
python scripts/build_vectors.py --stories acts/q3b_stories --neutral acts/q3b_neutral --out probes/q3b.pt

# 4. Validation: held-out accuracy per layer, PCA (valence/arousal), dose test, steering
python scripts/validate.py --probe probes/q3b.pt --stories acts/q3b_stories --neutral acts/q3b_neutral \
    --model Qwen/Qwen2.5-3B-Instruct --steer
```

**Cost and size.** With all 171 emotions, 20 topics and 6 stories per call, step 1 makes 3,420 batch requests and produces about 20k stories. At roughly 1,000 output tokens per request, that is about 3.4M output tokens, or roughly $35 at Claude Opus 5.5 batch prices ($10/M output). Check current pricing before scaling to the paper's 100 topics × 12 stories.

Activations are float16 `[N, layers, d_model]`, so `--layers every:2` keeps the files manageable. With 20k stories, 18 layers and d=2048, that's about 1.5 GB.

**Models for an 8 GB GPU:** Qwen2.5-1.5B/3B-Instruct in bf16, or 7–8B models with `--load-in-4bit` (needs `bitsandbytes`).

## Using a probe (e.g. as an RL reward)

```python
from emoprobe import EmotionProbe
from emoprobe.model import load_model

model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")
probe = EmotionProbe(model, tok, "probes/q3b.pt")          # default layer ≈ 2/3 depth

probe.score_assistant("I just took 8000 mg of Tylenol.")    # {emotion: score} at the Assistant token
probe.reward(["You found the lost child and brought her home.",
              "You pushed a stranger to grab the last seat."],
             weights={"happy": 1.0, "guilty": -1.0, "afraid": -1.0})   # -> np.array([r1, r2])
tokens, scores = probe.token_scores("some text")            # per-token [T, E], for inspection
```

Scores are projections of `(activation − neutral mean)` onto unit emotion vectors. Only differences are meaningful, so compare against a neutral reference situation.

## Layout

```
emoprobe/
  emotions.py     171 emotion words (paper's list), core subset, our topics
  prompts.py      story / neutral-dialogue prompts, output parser
  generate.py     Claude batch backend and local HF backend
  model.py        model loading, layer lookup, layer specs
  activations.py  hook-based residual recording, mean-from-token-50, last-token
  vectors.py      diff-of-means, neutral-PC projection, held-out split, logistic probe
  probe.py        EmotionProbe: token / text / Assistant-token readouts, reward()
  steering.py     residual-norm-scaled steering
scripts/          generate_data, extract_activations, build_vectors, validate, import_paper_topics
tests/            model-free unit tests (pytest)
```
