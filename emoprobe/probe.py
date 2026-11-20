"""EmotionProbe: how strongly the model represents each emotion right before it replies.

This is what an RL environment would call to turn a situation into an emotion reward.
"""

import numpy as np
import torch

from .activations import activations

SITUATION_TEMPLATE = ("Here is the situation you are in right now:\n\n{situation}\n\n"
                      "How do you feel about this situation?")


class EmotionProbe:
    """Score = (activation at the Assistant-turn token - mean neutral activation) . unit emotion vector.

    Only differences between scores mean something (e.g. good vs bad version of a situation).
    """

    def __init__(self, model, tok, probe, layer=None):
        p = torch.load(probe, weights_only=True) if isinstance(probe, str) else probe
        self.model, self.tok, self.emotions = model, tok, p["emotions"]
        self.layer = p["default_layer"] if layer is None else layer
        li = p["layers"].index(self.layer)
        v = p["vectors"][:, li].float()
        self.unit = v / v.norm(dim=-1, keepdim=True)  # [E, D]
        self.baseline = p["neutral_mean"][li].float()  # [D]

    def vector(self, emotion):
        return self.unit[self.emotions.index(emotion)]

    def assistant_prompt(self, user_message):
        """The chat-formatted prompt, ending exactly where the Assistant's reply starts."""
        return self.tok.apply_chat_template([{"role": "user", "content": user_message}],
                                            tokenize=False, add_generation_prompt=True)

    def score(self, user_messages):
        """Emotion scores at the Assistant-turn token, one row per message: [N, E]."""
        prompts = [self.assistant_prompt(m) for m in user_messages]
        h = torch.from_numpy(activations(self.model, self.tok, prompts, [self.layer], mode="last")[:, 0])
        return ((h - self.baseline) @ self.unit.T).numpy()

    def reward(self, situations, weights):
        """Weighted emotion score per situation, e.g. weights={"happy": 1.0, "afraid": -0.5}."""
        scores = self.score([SITUATION_TEMPLATE.format(situation=s) for s in situations])
        w = np.array([weights.get(e, 0.0) for e in self.emotions], dtype=np.float32)
        return scores @ w
