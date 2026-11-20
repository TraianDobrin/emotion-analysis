"""Step 2: record the residual stream with forward hooks on the transformer blocks."""

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import decoder_layers, model_device


class ResidualRecorder:
    """`with ResidualRecorder(model, layers) as rec:` after a forward pass,
    rec.cache[layer] holds that block's output, shape [batch, tokens, dim]."""

    def __init__(self, model, layers):
        self.blocks, self.layers, self.cache, self.handles = decoder_layers(model), layers, {}, []

    def __enter__(self):
        for layer in self.layers:
            def hook(_module, _inp, out, layer=layer):
                self.cache[layer] = out[0] if isinstance(out, tuple) else out
            self.handles.append(self.blocks[layer].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def activations(model, tok, texts, layers, mode="mean", start_token=50, batch_size=8, max_length=512):
    """One vector per text and layer: float32 [N, len(layers), D].

    mode="mean": average over tokens from `start_token` on (the paper uses 50, by which
                 point the story's emotion is clear; short texts start at half their length).
    mode="last": the last token only (e.g. the token just before the Assistant replies).
    """
    device = model_device(model)
    tok.padding_side = "right"
    out = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float32)
    with ResidualRecorder(model, layers) as rec:
        for i in tqdm(range(0, len(texts), batch_size), disable=len(texts) <= batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length, add_special_tokens=(mode == "mean")).to(device)
            model(**enc)
            mask = enc["attention_mask"]
            lengths = mask.sum(1)
            if mode == "mean":
                start = (lengths // 2).clamp(min=1, max=start_token)
                positions = torch.arange(mask.shape[1], device=device)
                weights = (mask.bool() & (positions[None] >= start[:, None])).float()
            else:
                weights = torch.zeros_like(mask, dtype=torch.float)
                weights[torch.arange(len(lengths)), lengths - 1] = 1.0
            weights /= weights.sum(1, keepdim=True)
            for j, layer in enumerate(layers):
                h = rec.cache[layer].float()  # [B, T, D]
                out[i:i + len(lengths), j] = torch.einsum("bt,btd->bd", weights, h).cpu().numpy()
    return out
