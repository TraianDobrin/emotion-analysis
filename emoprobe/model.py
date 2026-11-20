"""Loading a Hugging Face model and finding its transformer blocks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, load_in_4bit=False):
    """Return (model, tokenizer): bf16 on GPU, fp32 on CPU, or 4-bit to fit larger models."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map="auto")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    return model.eval(), tok


def decoder_layers(model):
    """The list of transformer blocks; the output of block i is the residual stream at layer i."""
    for path in ["model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            pass
    raise ValueError(f"Could not find decoder layers on {type(model).__name__}")


def model_device(model):
    return next(model.parameters()).device
