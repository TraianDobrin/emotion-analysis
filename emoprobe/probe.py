"""EmotionProbe: read emotion-vector activations from a model, either over a text
or at the point where the Assistant is about to respond (the model's "own"
operative emotion, as in the paper's readouts at the Assistant turn token).

This is the object an RL environment wrapper would call to get an emotion reward.
"""

from __future__ import annotations

import numpy as np
import torch

from .activations import ResidualRecorder, last_token_activations
from .model import model_device

DEFAULT_SITUATION_TEMPLATE = (
    "Here is the situation you are in right now:\n\n{situation}\n\n"
    "How do you feel about this situation?"
)


def save_probe(path, *, model_name, layers, emotions, vectors, story_mean, neutral_mean=None,
               default_layer, start_token, n_components=None):
    torch.save(
        {
            "model_name": model_name,
            "layers": list(layers),
            "emotions": list(emotions),
            "vectors": torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)),
            "story_mean": torch.from_numpy(np.ascontiguousarray(story_mean, dtype=np.float32)),
            "neutral_mean": None if neutral_mean is None
            else torch.from_numpy(np.ascontiguousarray(neutral_mean, dtype=np.float32)),
            "default_layer": int(default_layer),
            "start_token": int(start_token),
            "n_components": list(n_components or []),
        },
        path,
    )


class EmotionProbe:
    """Projects residual activations at one layer onto unit-norm emotion vectors.

    Scores are dot products of (activation - baseline) with each unit vector, where
    the baseline is the mean neutral activation if available, else the mean story
    activation. Only differences between scores are meaningful; compare against
    a neutral reference situation when interpreting absolute values.
    """

    def __init__(self, model, tok, probe: str | dict, layer: int | None = None):
        d = torch.load(probe, weights_only=True) if isinstance(probe, str) else probe
        self.model, self.tok = model, tok
        self.meta = d
        self.emotions: list[str] = d["emotions"]
        self.layer = d["default_layer"] if layer is None else layer
        if self.layer not in d["layers"]:
            raise ValueError(f"layer {self.layer} not in probe layers {d['layers']}")
        li = d["layers"].index(self.layer)
        v = d["vectors"][:, li].float()
        self.unit = v / v.norm(dim=-1, keepdim=True)  # [E, D]
        base = d["neutral_mean"] if d.get("neutral_mean") is not None else d["story_mean"]
        self.baseline = base[li].float()  # [D]
        self._index = {e: i for i, e in enumerate(self.emotions)}

    # ---- helpers -----------------------------------------------------------------
    def _project(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float().cpu() - self.baseline) @ self.unit.T

    def vector(self, emotion: str) -> torch.Tensor:
        return self.unit[self._index[emotion]]

    def assistant_prompt(self, user_message: str, system: str | None = None) -> str:
        """The prompt up to the start of the Assistant's reply. With a chat model the
        last token is the end of the chat template's assistant header; for base models
        we use a 'Human/Assistant' format ending in 'Assistant:'."""
        if getattr(self.tok, "chat_template", None):
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": user_message}
            ]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prefix = f"{system}\n\n" if system else ""
        return f"{prefix}Human: {user_message}\n\nAssistant:"

    # ---- readouts ------------------------------------------------------------------
    @torch.no_grad()
    def token_scores(self, text: str) -> tuple[list[str], np.ndarray]:
        """Per-token emotion scores over a raw text. Returns (tokens, [T, E])."""
        enc = self.tok(text, return_tensors="pt").to(model_device(self.model))
        with ResidualRecorder(self.model, [self.layer]) as rec:
            self.model(**enc)
            h = rec.cache[self.layer][0]
        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"][0])
        return tokens, self._project(h).numpy()

    def score_text(self, text: str, start_token: int = 0) -> dict[str, float]:
        """Mean emotion scores over a text's tokens from `start_token` on."""
        _, s = self.token_scores(text)
        s = s[min(start_token, len(s) - 1):].mean(0)
        return dict(zip(self.emotions, s.tolist()))

    def score_assistant_batch(self, user_messages: list[str], system: str | None = None,
                              batch_size: int = 8) -> np.ndarray:
        """Emotion scores at the Assistant-turn token for each message. [N, E]."""
        prompts = [self.assistant_prompt(m, system) for m in user_messages]
        h = last_token_activations(self.model, self.tok, prompts, [self.layer], batch_size=batch_size)
        return self._project(torch.from_numpy(h[:, 0])).numpy()

    def score_assistant(self, user_message: str, system: str | None = None) -> dict[str, float]:
        s = self.score_assistant_batch([user_message], system)[0]
        return dict(zip(self.emotions, s.tolist()))

    def reward(self, situations: list[str], weights: dict[str, float],
               template: str = DEFAULT_SITUATION_TEMPLATE, system: str | None = None) -> np.ndarray:
        """Weighted emotion reward for a batch of situation descriptions, read at the
        Assistant-turn token. E.g. weights={"happy": 1.0, "afraid": -1.0}. [N]."""
        scores = self.score_assistant_batch([template.format(situation=s) for s in situations], system)
        w = np.zeros(len(self.emotions), dtype=np.float32)
        for e, x in weights.items():
            w[self._index[e]] = x
        return scores @ w
