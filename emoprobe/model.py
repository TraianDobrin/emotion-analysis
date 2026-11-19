"""Loading open-weights models and locating their residual stream."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_LAYER_PATHS = [
    "model.layers",                 # Llama, Qwen, Mistral, Gemma
    "model.language_model.layers",  # Gemma 3 / multimodal wrappers
    "model.model.layers",
    "transformer.h",                # GPT-2
    "gpt_neox.layers",              # Pythia
    "model.decoder.layers",         # OPT
]


def load_model(name: str, device: str | None = None, dtype: str = "auto", load_in_4bit: bool = False):
    """Return (model, tokenizer) in eval mode."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"dtype": torch_dtype}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    return model, tok


def decoder_layers(model):
    """The list of transformer blocks. Layer i's output is the residual stream after block i."""
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"Could not find decoder layers on {type(model).__name__}")


def num_layers(model) -> int:
    return len(decoder_layers(model))


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def parse_layers(spec: str, n_layers: int) -> list[int]:
    """'all' | 'every:K' | 'frac:0.66' | '10,14,18'."""
    spec = spec.strip()
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("every:"):
        k = int(spec.split(":")[1])
        layers = list(range(0, n_layers, k))
        if n_layers - 1 not in layers:
            layers.append(n_layers - 1)
        return layers
    if spec.startswith("frac:"):
        return [min(n_layers - 1, round(float(spec.split(":")[1]) * n_layers))]
    return [int(x) for x in spec.split(",")]
