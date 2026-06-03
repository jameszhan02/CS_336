from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np
import math
import regex as re
from collections import defaultdict
import time
import torch.nn as nn

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    layer = Linear(d_in, d_out)
    layer.load_state_dict({"weight": weights}) # inhert form nn.Module
    return layer(in_features)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__() 
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embd = Embedding(vocab_size, d_model)
    embd.load_state_dict({"weight": weights})
    return embd(token_ids)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    # w1_out = run_silu(in_features @ w1_weight.T)
    # w3_out = in_features @ w3_weight.T
    # # apply "gate" to w1_out
    # gate_out = w1_out * w3_out
    # return gate_out @ w2_weight.T
    layer = SwiGLU(d_model, d_ff)
    layer.w1.weight.data = w1_weight
    layer.w2.weight.data = w2_weight
    layer.w3.weight.data = w3_weight
    return layer(in_features)
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(W1*x) ⊙ W3*x
        gate = torch.nn.functional.silu(self.w1(x))
        hidden = self.w3(x)
        # W2 * (gate ⊙ hidden)
        return self.w2(gate * hidden)

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    ## caludlate how colse the Q K is 
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    # control each scroes wont be too big when you doing soft max major part goning to be who big mainly.
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    # ??TODO: this is the question (if V represent "mean" does softmax in some case is losing information?) 
    # | multi head sort of fixing this issue by learning different parttern.
    # but mutihead 
    weights = torch.softmax(scores, dim=-1) 

    return weights @ V


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_k = d_model // num_heads
    batch = in_features.shape[:-2]
    seq = in_features.shape[-2]
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T


    Q = Q.view(*batch, seq, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch, seq, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch, seq, num_heads, d_k).transpose(-3, -2)

    # torch.tril control over with giving triangular lower.
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=in_features.device))
    out = run_scaled_dot_product_attention(Q, K, V, mask=mask)
    out = out.transpose(-3, -2).contiguous().view(*batch, seq, d_model)
    return out @ o_proj_weight.T


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_k = d_model // num_heads
    batch = in_features.shape[:-2]
    seq = in_features.shape[-2]
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    Q = Q.view(*batch, seq, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch, seq, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch, seq, num_heads, d_k).transpose(-3, -2)

    Q_rope = run_rope(d_k, theta, max_seq_len, Q, torch.arange(0, seq)) 
    K_rope = run_rope(d_k, theta, max_seq_len, K, torch.arange(0, seq)) 

    # torch.tril control over with giving triangular lower.
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=in_features.device))
    out = run_scaled_dot_product_attention(Q_rope, K_rope, V, mask=mask)
    out = out.transpose(-3, -2).contiguous().view(*batch, seq, d_model)
    return out @ o_proj_weight.T


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dim_indices / d_k))
        
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * inv_freq  # (max_seq_len, d_k//2)
        
        cos = torch.cos(angles)  # (max_seq_len, d_k//2)
        sin = torch.sin(angles)  # (max_seq_len, d_k//2)
        
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k//2)
        
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    block = TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len)
    block.load_state_dict(weights)
    return block(in_features)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.attn = nn.Module()
        self.attn.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out = run_multihead_self_attention_with_rope(
            self.d_model, self.num_heads, self.max_seq_len, self.theta,
            self.attn.q_proj.weight,
            self.attn.k_proj.weight,
            self.attn.v_proj.weight,
            self.attn.output_proj.weight,
            x_norm
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    model = TransformerLM(vocab_size, context_length, d_model,
                          num_layers, num_heads, d_ff, rope_theta)
    
    model.load_state_dict(weights)
    
    return model(in_indices)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, 
                 num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.rope_theta = rope_theta
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length)
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        
        return self.lm_head(x)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    layer = RMSNorm(d_model, eps)
    layer.load_state_dict({"weight": weights})
    return layer(in_features)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)  # 初始化为1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        
        x = x.to(torch.float32)
        
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        out = x_norm * self.weight
        
        return out.to(orig_dtype)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    ## also consider the range need to make sure y (x window shift by 1 is also fullfilled)
    random_range = len(dataset) - context_length
    start_indices = np.random.randint(0, random_range, size=batch_size).tolist()
    # [] is the fundatmetal python data type where we need convert to numpy type to do the farther process
    x = np.stack([dataset[i : i + context_length] for i in start_indices])        # (batch_size, context_length)
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices]) # (batch_size, context_length)
    # steps that convert numpy data type into the tensor type    
    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return (x, y)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # subtract max for numerical stability || prevent overflow issue 
    ## keepdim is the STAT data per "row", through boardcast(pytorch feature)
    x = in_features - in_features.max(dim=dim, keepdim=True).values
    # opeation that apply to every single element x -> e^x
    exp_x = torch.exp(x)
    
    # similar with step 1, is doing some sort of STAT and boardcast
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    softmax_inputs = run_softmax(inputs, -1)

    # shape: (batch_size,)  ← one probability per example
    batch_size = inputs.shape[0]

    # ✅ stable: one formula
    max_vals = inputs.max(dim=-1, keepdim=True).values
    # TODO: inference this one line foumlar on paper
    log_softmax = (inputs - max_vals) - torch.log(torch.exp(inputs - max_vals).sum(dim=-1, keepdim=True))

    # then just index and negate
    cross_entropy = -log_softmax[torch.arange(batch_size), targets]
    return cross_entropy.mean()

    ## !!!!!! =================. Unstable call (Cross Entropy) ======================
    # # ex: torch.arange(32)   # → [0, 1, 2, 3, 4, ... 31]
    # correct_probs = softmax_inputs[torch.arange(batch_size), targets]
    # ## !!! softmax_inputs[torch.arange(batch_size), targets] -> Is exactly equivalent to below
    # # [softmax_inputs[0][2],    # row 0, col 2
    # # softmax_inputs[1][0],    # row 1, col 0
    # # softmax_inputs[2][1]]    # row 2, col 1
    # cross_entropy = -torch.log(correct_probs)  # -log for each example
    # return cross_entropy.mean()                # average across batch


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # convert to list so we can iterate twice
    parameters = list(parameters)
    ## TODO: get comfortable with syntax here 
    # step 1: collect all grads
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    # step 2: combined L2 norm across ALL gradients
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    # step 3: clip if exceeds max
    if total_norm > max_l2_norm:
        ratio = max_l2_norm / total_norm
        for g in grads:
            g.mul_(ratio)
    # parameters_length = math.sqrt((parameters.grad ** 2).sum())
    # ratio = 1
    # if parameters_length > max_l2_norm:
    #     ratio = max_l2_norm / parameters_length
    # return parameters * ratio


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p] # each p is a unque reference point to memopry address so is unquie for the key for state as well
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data) # weight for entire component ex: 4 x 8
                    state["v"] = torch.zeros_like(p.data)  

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t) # adjust value: 
                v_hat = v / (1 - beta2 ** t)
                # eps is for divide 0 issue
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr) # p.data = p.data - lr * (v_hat.sqrt().add_(eps))

                p.data.mul_(1 - lr * weight_decay)

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    
    else:
        return min_learning_rate



def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model": model.state_dict(),         
        "optimizer": optimizer.state_dict(), 
        "iteration": iteration               
    }
    torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])         
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"] 

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)
##  ================================= Start of calss tokenizer ================================= 
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab                    # int → bytes
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = tok_bytes
                self.bytes_to_id[tok_bytes] = new_id
        
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        import json
        with open(vocab_filepath, "r") as f:
            vocab_raw = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in vocab_raw.items()}
        
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                a, b = line.strip().split()
                merges.append((a.encode("latin-1"), b.encode("latin-1")))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        GPT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        ids = []
        
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pat = "|".join(re.escape(tok) for tok in sorted_special)
            chunks = re.split(f"({special_pat})", text)  
        else:
            chunks = [text]
        
        for chunk in chunks:
            if not chunk:
                continue
            
            if chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue
            
            words = re.findall(GPT_PAT, chunk)
            for word in words:
                tokens = [bytes([b]) for b in word.encode("utf-8")]
                
                while len(tokens) > 1:
                    best_rank = float("inf")
                    best_i = None
                    for i in range(len(tokens) - 1):
                        rank = self.merge_rank.get((tokens[i], tokens[i+1]), float("inf"))
                        if rank < best_rank:
                            best_rank = rank
                            best_i = i
                    
                    if best_i is None or best_rank == float("inf"):
                        break
                    
                    tokens = tokens[:best_i] + [tokens[best_i] + tokens[best_i+1]] + tokens[best_i+2:]
                
                ids.extend(self.bytes_to_id[tok] for tok in tokens)
        
        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            ids = self.encode(text)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
# =================================  END of tokenizer class ================================= 

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    t0 = time.perf_counter()
    GPT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # read file form the inputpath
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        # read file form the inputpath
    if special_tokens: 
        special_pat = "|".join(re.escape(tok) for tok in special_tokens) # escape to make sure | -> \| so when split apply to this will know is plain text
        chunks = re.split(special_pat, text)
    else:
        chunks = [text]  # if special token is empty

    words = []
    for chunk in chunks:
        words += re.findall(GPT_PAT, chunk)
    
    word_freqs = defaultdict(int)
    for word in words:
        key = tuple(bytes([b]) for b in word.encode("utf-8"))
        word_freqs[key] += 1


    ## END of pre-tokenizer 
    t1 = time.perf_counter()
    # inital vocab
    vocab = {}
    idx = 0

    # load special tokens into the vocab
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1
    # add 0 - 255 into vocab in UTF-8 
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    # BPE merge loop
    merges = []
    num_merges = vocab_size - len(vocab) # how many merge we allow in current load

    # for _ in range(num_merges):
    #     pair_freqs = defaultdict(int)  # in each loop we need a pair_dict to record each pair comb. and freq
    #     for word, freq in word_freqs.items():  #  ex. word: (h,e,l,l,o) freq: 5
    #         for i in range(len(word) - 1):
    #             pair_freqs[(word[i], word[i+1])] += freq
    #     if not pair_freqs:
    #         break

    #     # 
    #     best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p)) # max for tuple will compare form left to right 

    #     merged = best_pair[0] + best_pair[1] # bestpair only been return in key string nonlike JS, determined by python for structure
    #     new_word_freqs = defaultdict(int)
    #     for word, freq in word_freqs.items():
    #         new_word = []
    #         i = 0
    #         while i < len(word):
    #             if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
    #                 new_word.append(merged)
    #                 i += 2
    #             else:
    #                 new_word.append(word[i])
    #                 i += 1
    #         new_word_freqs[tuple(new_word)] += freq
    #     word_freqs = new_word_freqs

    #     merges.append(best_pair)
    #     vocab[idx] = merged
    #     idx += 1

    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i+1])] += freq

    for _ in range(num_merges):
        if not pair_freqs:
            break

        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        merged = best_pair[0] + best_pair[1]

        new_word_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    if i > 0:
                        pair_freqs[(new_word[-1], word[i])]      -= freq
                        pair_freqs[(new_word[-1], merged)]        += freq
                    if i + 2 < len(word):
                        pair_freqs[(word[i+1], word[i+2])]       -= freq
                        pair_freqs[(merged,     word[i+2])]       += freq
                    pair_freqs[best_pair] -= freq

                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] += freq

        word_freqs = new_word_freqs
        merges.append(best_pair)
        vocab[idx] = merged
        idx += 1

    t2 = time.perf_counter()
    return vocab, merges