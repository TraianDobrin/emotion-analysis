"""Recording residual-stream activations with forward hooks.

We hook the output of each decoder block rather than using output_hidden_states,
because HF applies the final norm to the last hidden state of some models and
because hooks let us keep only the layers we need.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import decoder_layers, model_device


class ResidualRecorder:
    """Context manager that stores the residual stream after the given blocks."""

    def __init__(self, model, layers):
        self.layers = list(layers)
        self.cache: dict[int, torch.Tensor] = {}
        self._blocks = decoder_layers(model)
        self._handles = []

    def _hook(self, layer):
        def fn(_module, _inp, out):
            self.cache[layer] = out[0] if isinstance(out, tuple) else out

        return fn

    def __enter__(self):
        for l in self.layers:
            self._handles.append(self._blocks[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.cache.clear()


@torch.no_grad()
def mean_activations(
    model,
    tok,
    texts: list[str],
    layers: list[int],
    start_token: int = 50,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Per-text mean residual activation, averaged over token positions >= start_token.

    Following the paper, averaging starts at the 50th token, by which point the
    emotional content should be apparent. Texts shorter than 2*start_token start
    averaging at half their length instead. Returns float16 [N, len(layers), D].
    """
    device = model_device(model)
    tok.padding_side = "right"
    out = None
    with ResidualRecorder(model, layers) as rec:
        for i in tqdm(range(0, len(texts), batch_size), desc="activations"):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

            mask = enc["attention_mask"]
            lengths = mask.sum(1)
            start = torch.clamp(torch.minimum(torch.full_like(lengths, start_token), lengths // 2), min=1)
            pos = torch.arange(mask.shape[1], device=device)[None, :]
            sel = (mask.bool() & (pos >= start[:, None])).float()  # [B, T]
            denom = sel.sum(1, keepdim=True).clamp(min=1)

            for j, l in enumerate(layers):
                h = rec.cache[l].float()  # [B, T, D]
                m = (h * sel[..., None]).sum(1) / denom
                if out is None:
                    out = np.zeros((len(texts), len(layers), h.shape[-1]), dtype=np.float16)
                out[i : i + len(batch), j] = m.cpu().numpy().astype(np.float16)
    return out


@torch.no_grad()
def last_token_activations(model, tok, texts: list[str], layers: list[int], batch_size: int = 8, max_length: int = 2048):
    """Residual activation at the final token of each text. Float32 [N, len(layers), D]."""
    device = model_device(model)
    tok.padding_side = "right"
    out = None
    with ResidualRecorder(model, layers) as rec:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                      add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)
            last = enc["attention_mask"].sum(1) - 1
            idx = torch.arange(len(batch), device=device)
            for j, l in enumerate(layers):
                h = rec.cache[l][idx, last].float()
                if out is None:
                    out = np.zeros((len(texts), len(layers), h.shape[-1]), dtype=np.float32)
                out[i : i + len(batch), j] = h.cpu().numpy()
    return out
