# Results log

## 2026-11-19: happiness, Qwen2.5-3B-Instruct

**Setup**
- **Stories:** the paper's story prompt and its first 30 topics, 12 stories per call. The emotions are happy plus 5 contrasts (sad, afraid, angry, calm, bored). That gave 2,154 stories (about 359 per emotion, median 99 words), after dropping stories that named their own emotion.
- **Story writer:** Gemini 3.5 Flash wrote the first 15 calls; Gemini 3.5 Flash-Lite wrote the other 165, because the free-tier quota ran out.
- **Neutral set:** 360 dialogues on the paper's 60 topics, using our neutral prompt.
- **Vectors:** averaged from token 50, with neutral PCs explaining 50% of the variance projected out (10–21 PCs per layer).
- **Readout:** default layer 24 of 36 (2/3 depth), read at the Assistant-turn token.

**Held-out-topic classification** (6-way, chance 0.17; trained on topics, tested on unseen topics)

| layers | diff-of-means | logistic |
|---|---|---|
| 0–8 | 0.64–0.70 | 0.78–0.82 |
| 10–22 | 0.75–0.80 | 0.87–0.93 |
| 24–35 | 0.69–0.81 | 0.89–0.93 |

**Dose test** at the Assistant token, Tylenol 500 → 16,000 mg:
- "afraid" rises 2.2 → 5.5, monotonically
- "calm" falls 3.1 → −2.2, monotonically
- "happy" falls 0.3 → −0.9

This reproduces the paper's direction.

**Happiness pairs:** the good version of a situation scored higher than the bad version in **8/8** pairs (mean difference +6.9).

**Sister's age at death** (5 → 100 years): "happy" rises monotonically, −2.00 → −1.16, which is the paper's direction. The effect is small.

**Steering** at layer 24:
- **happy, s = 0.3:** clear effect. "I feel *splendidly energetic and ready to share my best vibes with the world!*"
- **sad, afraid, angry, calm at s ≤ 0.3:** weak. The 3B assistant persona mostly answers that it has no feelings.

**Caveats**
- Only 6 emotions, so the PCA is not comparable to the paper's 171-emotion valence/arousal result.
- Layer 0 already gets 66%, so part of the separation is surface content rather than an abstract emotion representation.
- The story writer changed partway through the run.
