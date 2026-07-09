# inference_utils.py
"""
"""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import KVCache


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.softmax(logits / temperature, dim=-1)


def nucleus_sampling(probs: torch.Tensor, p: float) -> int:
    """
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_probs = sorted_probs.masked_fill(cumulative_probs - sorted_probs > p, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_index = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_indices[sampled_index].item()


def greedy_sampling(probs: torch.Tensor) -> int:
    """
    """
    return torch.argmax(probs, dim=-1).item()


def generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    p: float = 0.9,
    eos_token_id: int | None = None,
    device: str = "cpu",
    sampling: str = "nucleus",   # "nucleus" 或 "greedy"
) -> list[int]:
    """
    """
    model.eval()
    tokens = list(prompt_tokens)    
    # KV_cache change_5
    kv_cache = KVCache(model.num_layers)

    with torch.no_grad():
        # prefill: for the first time before the loop, kv_cache yet still None we fill the cache and slice logits as last dim for next token generate.
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x, kv_cache=kv_cache)
        logits = logits[0, -1, :]
        for _ in range(max_new_tokens):
            # TODO: get rid of this garud with no context length limit
            if len(tokens) >= context_length:
                break
            probs = softmax_with_temperature(logits, temperature)
            next_token = nucleus_sampling(probs, p)
            tokens.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

            # decode: over ride x with the next token only into the loop rest information already in the KV cache.
            x = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits = model(x, kv_cache=kv_cache)
            logits = logits[0, -1, :]

    return tokens


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    p: float = 0.9,
    eos_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    prompt_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.encode(eos_token)[0] if eos_token else None

    output_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=temperature,
        p=p,
        eos_token_id=eos_token_id,
        device=device,
    )

    generated_tokens = output_tokens[len(prompt_tokens):]
    return tokenizer.decode(generated_tokens)



import pickle


def load_checkpoint(checkpoint_path: str, device: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    required_keys = ["model_state_dict", "config"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise KeyError(f"checkpoint missing keys: {missing}. Did you save config when training?")
    return ckpt


def build_model_from_config(model_cls, config: dict, device: str) -> torch.nn.Module:
    model = model_cls(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config["rope_theta"],
    )
    model.to(device)
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_cls,
    device: str,
) -> tuple[torch.nn.Module, dict]:
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt["config"]

    model = build_model_from_config(model_cls, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[load_model_from_checkpoint] loaded step={ckpt.get('step')} from {checkpoint_path}")
    return model, config


def load_tokenizer(vocab_path: str, tokenizer_cls, special_tokens: list[str]):
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)
    vocab, merges = data["vocab"], data["merges"]
    return tokenizer_cls(vocab, merges, special_tokens)


def load_model_and_tokenizer(
    checkpoint_path: str,
    vocab_path: str,
    model_cls,
    tokenizer_cls,
    special_tokens: list[str],
    device: str,
):
    model, config = load_model_from_checkpoint(checkpoint_path, model_cls, device)
    tokenizer = load_tokenizer(vocab_path, tokenizer_cls, special_tokens)
    return model, tokenizer, config