"""Activation steering with emotion vectors, for causal validation.

Strength is in units of the residual-stream norm at each position, as in the
paper (e.g. 0.05-0.1 for behavioural effects, ~0.5 for blatant effects).
"""

from contextlib import contextmanager

import torch

from .model import decoder_layers


@contextmanager
def steer(model, layer: int, vector: torch.Tensor, strength: float):
    unit = (vector / vector.norm()).detach()

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + strength * h.norm(dim=-1, keepdim=True) * unit.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 60, **kwargs) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id, **kwargs)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
