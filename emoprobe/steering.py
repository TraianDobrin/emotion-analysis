"""Steering: add an emotion vector to the residual stream and see if the output changes."""

from contextlib import contextmanager

import torch

from .model import decoder_layers, model_device


@contextmanager
def steer(model, layer, vector, strength):
    """Inside the block, layer `layer` outputs h + strength * |h| * unit(vector), at every token."""
    unit = vector / vector.norm()

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + strength * h.norm(dim=-1, keepdim=True) * unit.to(h.device, h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=30):
    """Greedy continuation of `prompt`."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model_device(model))
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
