# generate.py
import sys, os, pickle, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import TransformerLM, Tokenizer


def load_model_and_tokenizer(checkpoint_path: str, vocab_path: str, device: str):
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)
    vocab, merges = data["vocab"], data["merges"]
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config["rope_theta"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"loaded checkpoint from step {ckpt['step']}")
    print(f"config: {config}")

    return model, tokenizer, config


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits / temperature, dim=-1)


def nucleus_sampling(probs: torch.Tensor, p: float) -> int:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative_probs - sorted_probs > p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled_index = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_indices[sampled_index].item()


def decode(model, prompt_tokens, max_new_tokens, context_length,
           temperature=1.0, p=0.9, eos_token_id=None, device="cpu"):
    model.eval()
    tokens = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-context_length:]
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)[0, -1, :]
            probs = softmax_with_temperature(logits, temperature)
            next_token = nucleus_sampling(probs, p)
            tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
    return tokens


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 改成你想测的那次训练的 run 文件夹 + step
    run_id = "run_20260705_143210"
    step = 4900
    checkpoint_path = os.path.join(BASE_DIR, "checkpoints", run_id, f"checkpoint_step{step}.pt")
    vocab_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txtvocab.pkl")

    model, tokenizer, config = load_model_and_tokenizer(checkpoint_path, vocab_path, DEVICE)

    prompt = "Long long time ago, there is a princess call snow white, and here is her saga: "
    prompt_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    output_tokens = decode(
        model, prompt_tokens,
        max_new_tokens=400,
        context_length=config["context_length"],
        temperature=0.8,
        p=0.9,
        eos_token_id=eos_token_id,
        device=DEVICE,
    )

    generated_text = tokenizer.decode(output_tokens[len(prompt_tokens):])
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")