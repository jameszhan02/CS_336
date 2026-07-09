# inference_utils.py
"""
推理相关的工具函数：温度采样、nucleus sampling、自回归生成。
命名约定：
  - generate(...)      : 自回归生成 token id 序列（模型只认识 id，不认识文字）
  - tokenizer.decode()  : 把 token id 序列转换回文字（在 Tokenizer 类里，不在这个文件）
"""

import torch
import torch.nn.functional as F


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: (..., vocab_size)  最后一维是 vocab，前面维度任意（方便以后支持 batch）
    temperature: < 1 更保守/确定，> 1 更随机/多样
    return: (..., vocab_size) 概率分布
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.softmax(logits / temperature, dim=-1)


def nucleus_sampling(probs: torch.Tensor, p: float) -> int:
    """
    单条序列版本。
    probs: (vocab_size,)
    p: nucleus 阈值
    return: 采样出的 token id（int）
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 排除自己后的前缀和超过 p，说明已经在长尾区，砍掉
    sorted_probs = sorted_probs.masked_fill(cumulative_probs - sorted_probs > p, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_index = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_indices[sampled_index].item()


def greedy_sampling(probs: torch.Tensor) -> int:
    """
    贪心解码，方便调试用（对比 nucleus sampling 的随机性，greedy 是确定性的）。
    probs: (vocab_size,)
    return: 概率最大的 token id
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
    sampling: str = "nucleus",   # "nucleus" 或 "greedy"，方便切换调试
) -> list[int]:
    """
    单条序列的自回归生成。返回的是 token id 列表，不是文字。
    文字需要之后调用 tokenizer.decode(生成的那一段) 才能得到。

    model: TransformerLM，forward(x) -> (batch, seq_len, vocab_size)
    prompt_tokens: tokenizer.encode(prompt) 得到的 token id 列表
    max_new_tokens: 最多生成多少个新 token
    context_length: 模型支持的最大上下文长度，超过要截断
    temperature: softmax 温度
    p: nucleus sampling 阈值
    eos_token_id: 遇到这个 token 就提前停止
    device: 运行设备
    sampling: "nucleus" 用概率采样（有随机性），"greedy" 用贪心（确定性，便于调试对比）
    return: 完整 token id 序列（包含 prompt 部分 + 新生成部分）
    """
    model.eval()
    tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 只取最近 context_length 个 token 喂给模型（超出上下文窗口的截断）
            context = tokens[-context_length:]
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

            logits = model(x)            # (1, seq_len, vocab_size)
            logits = logits[0, -1, :]    # (vocab_size,) 只要最后一个位置的预测

            probs = softmax_with_temperature(logits, temperature)

            if sampling == "greedy":
                next_token = greedy_sampling(probs)
            else:
                next_token = nucleus_sampling(probs, p)

            tokens.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

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
    """
    便捷封装：直接从文字 prompt 到文字输出，一步到位。
    内部做的事情：文字 -> encode -> generate(id 序列) -> 切掉 prompt -> decode 回文字
    """
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

# inference_utils.py 补充这部分
"""
补充：从 checkpoint 加载模型和 tokenizer，供 generate.py / batch 推理脚本复用。
"""

import pickle


def load_checkpoint(checkpoint_path: str, device: str) -> dict:
    """
    只做最基础的读取，返回原始 dict，方便需要 step / optimizer_state_dict 等信息时也能拿到。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    required_keys = ["model_state_dict", "config"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise KeyError(f"checkpoint missing keys: {missing}. Did you save config when training?")
    return ckpt


def build_model_from_config(model_cls, config: dict, device: str) -> torch.nn.Module:
    """
    根据 config 里的超参数重建一个结构一致的空模型（未加载权重）。
    model_cls: 传入 TransformerLM 这个类本身（不在这里 import，避免循环依赖/耦合）
    """
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
    """
    完整加载流程：读 checkpoint -> 用 config 重建空模型 -> 灌权重 -> eval 模式
    return: (model, config)
    """
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt["config"]

    model = build_model_from_config(model_cls, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[load_model_from_checkpoint] loaded step={ckpt.get('step')} from {checkpoint_path}")
    return model, config


def load_tokenizer(vocab_path: str, tokenizer_cls, special_tokens: list[str]):
    """
    从 vocab.pkl 加载 tokenizer（vocab/merges 不在 checkpoint 里，单独存）。
    tokenizer_cls: 传入 Tokenizer 这个类本身，同样避免这个文件耦合具体实现
    """
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
    """
    一步到位：同时加载模型和 tokenizer。generate.py 里最常用的入口。
    """
    model, config = load_model_from_checkpoint(checkpoint_path, model_cls, device)
    tokenizer = load_tokenizer(vocab_path, tokenizer_cls, special_tokens)
    return model, tokenizer, config