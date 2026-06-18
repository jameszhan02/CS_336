from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have same length, "
            f"got {len(prompt_strs)} vs {len(output_strs)}"
        )

    batch_size = len(prompt_strs)

    # empty batch case
    if batch_size == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {
            "input_ids": empty,
            "labels": empty,
            "response_mask": empty,
        }

    # pad token fallback
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # -----------------------------
    # 1. tokenize + keep raw seq
    # -----------------------------
    all_combined_ids = []
    all_prompt_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        combined_ids = prompt_ids + output_ids

        all_combined_ids.append(combined_ids)
        all_prompt_lens.append(len(prompt_ids))

    # -----------------------------
    # 2. pad (NO shift yet)
    # -----------------------------
    max_len = max(len(x) for x in all_combined_ids)

    full = torch.full(
        (batch_size, max_len),
        pad_id,
        dtype=torch.long
    )

    for i, ids in enumerate(all_combined_ids):
        full[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # -----------------------------
    # 3. shift 
    # -----------------------------
    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()

    # -----------------------------
    # 4. response mask
    # -----------------------------
    response_mask = torch.zeros(
        (batch_size, max_len - 1),
        dtype=torch.long
    )

    for i, prompt_len in enumerate(all_prompt_lens):
        # after shift:
        # prompt boundary moves by -1
        start = max(prompt_len - 1, 0)

        # response length = total_len - prompt_len
        end = start + (len(all_combined_ids[i]) - prompt_len)

        response_mask[i, start:end] = 1

    # for each track(answer) we create them as training data 
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy of next-token distribution (over vocab dim).

    Args:
        logits: (batch_size, sequence_length, vocab_size)

    Returns:
        entropies: (batch_size, sequence_length)
    """
    # if logits.ndim != 3:
    #     raise ValueError(f"logits must have shape (B, T, V), got {tuple(logits.shape)}")
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dim, got {tuple(logits.shape)}")

    # log_probs = logits - logsumexp(logits)
    # for nomalizetion, so when exp value wont overflow
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)   # (B, T, 1)
    log_probs = logits - log_z                              # (B, T, V)
    probs = torch.exp(log_probs)                            # (B, T, V)

    # H(p) = - sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)              # (B, T)
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"input_ids and labels must be (B, T). Got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have same shape. Got {tuple(input_ids.shape)} vs {tuple(labels.shape)}"
        )

    # forward
    logits = model(input_ids=input_ids).logits  # (B, T, V) || Just one single time forward

    # stable log-probs over vocab
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # get log_probs for that monent for right label. |ANSWERS Likelyhood|
    gathered = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1))  # (B, T, 1) | 
    token_log_probs = gathered.squeeze(-1)  # (B, T)

    out: Dict[str, torch.Tensor] = {"log_probs": token_log_probs}

    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)  # (B, T)

    return out


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    rewards = []
    format_rewards = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, ground_truth)
        rewards.append(result["reward"])
        format_rewards.append(result["format_reward"])
    
    raw_rewards = torch.tensor(rewards)
    
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(format_rewards) / len(format_rewards),
    }
    
    return raw_rewards, metadata


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    # reshape (n_prompts, group_size)
    n_prompts = len(raw_rewards) // group_size
    rewards = raw_rewards.reshape(n_prompts, group_size)

    if baseline == "mean":
        group_mean = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - group_mean
    else:
        raise NotImplementedError(f"baseline={baseline} not supported")

    if advantage_normalizer == "std":
        group_std = rewards.std(dim=1, keepdim=True)
        ## this is convert what ever data distribution into mean 0 std 1.
        advantages = advantages / (group_std + advantage_eps)
    else:
        raise NotImplementedError(f"advantage_normalizer={advantage_normalizer} not supported")
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward":  raw_rewards.std().item(),
        "max_reward":  raw_rewards.max().item(),
        "min_reward":  raw_rewards.min().item(),
    }

    return advantages.flatten(), metadata


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    # this function is intend to convert reward into "loss" in order to do backward
    # raw_rewards_or_advantages -> a batch_size list tell you reward for each element
    # policy_log_probs -> for each token the pi_theta(token) log_prob that the current token appear chance
    # importance_reweighting_method how we adjust grad base on the 'new' model during training.
    # old_log_probs -> the log_probs is what status
    # cliprange ｜ params for "grpo" | "gspo".
    # response_mask where is the response locate

    # make sure |raw_rewards_or_advantages| is the right shape
    advantages = raw_rewards_or_advantages.reshape(-1, 1)
    if importance_reweighting_method == "none":
        per_token_loss = -(advantages * policy_log_probs)
        return per_token_loss, {}
    elif importance_reweighting_method == "noclip":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -(advantages * ratio)
        return per_token_loss, {} 
    elif importance_reweighting_method == "grpo":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        grpo_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        unclip_loss = -(advantages * ratio)
        clip_loss = -(advantages * grpo_ratio)
        per_token_loss = torch.maximum(unclip_loss, clip_loss)
        return per_token_loss, {} 
    elif importance_reweighting_method == "gspo":
        seq_log_ratio = torch.sum((policy_log_probs - old_log_probs) * response_mask, dim=-1) / torch.sum(response_mask, dim=-1)
        ratio = torch.exp(seq_log_ratio).reshape(-1, 1)
        gspo_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        unclip_loss = -(advantages * ratio)
        clip_loss = -(advantages * gspo_ratio)
        per_token_loss = torch.maximum(unclip_loss, clip_loss)
        # expand back to the correct shape with [batch_size, seq]
        per_token_loss = per_token_loss.expand(-1, policy_log_probs.shape[-1])  # (2, 4)
        return per_token_loss, {} 
    raise NotImplementedError


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    if(loss_normalization == "constant"):
        raise NotImplementedError
    pre_token_with_mask = per_token_policy_gradient_loss * mask
    batch_loss = torch.sum(pre_token_with_mask, dim=-1) / torch.sum(mask, dim=-1)
    loss = torch.mean(batch_loss, dim=-1) 
    return loss


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    
    # ---- 1. reward ----
    raw_rewards, reward_metadata = run_compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    # ---- 2. advantage ----
    advantages, adv_metadata = run_compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )

    # ---- 3. tokenize ----
    tokenized = run_tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    rollout_batch_size = input_ids.shape[0]
    microbatch_size = rollout_batch_size // gradient_accumulation_steps

    total_loss = 0.0
    last_pg_metadata = {}

    optimizer.zero_grad()

    for i in range(gradient_accumulation_steps):
        start = i * microbatch_size
        end = start + microbatch_size

        micro_input_ids = input_ids[start:end]
        micro_labels = labels[start:end]
        micro_response_mask = response_mask[start:end]
        micro_advantages = advantages[start:end]
        micro_old_log_probs = (
            old_log_probs[start:end] if old_log_probs is not None else None
        )

        # ---- 4. forward: get current log_probs ----
        micro_input_ids = micro_input_ids.to(next(model.parameters()).device)
        micro_labels = micro_labels.to(next(model.parameters()).device)
        log_probs_out = run_get_response_log_probs(
            model, micro_input_ids, micro_labels, return_token_entropy=False
        )
        policy_log_probs = log_probs_out["log_probs"]

        # ---- 5. per-token loss ----
        per_token_loss, pg_metadata = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages=micro_advantages,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=micro_old_log_probs,
            cliprange=cliprange,
            response_mask=micro_response_mask,
        )

        # ---- 6. into scale loss ----
        loss = run_aggregate_loss_across_microbatch(
            per_token_loss, micro_response_mask, loss_normalization, normalization_constant
        )

        # ---- 7. mean the loss + back ward ----
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()
        last_pg_metadata = pg_metadata

    # ---- 8. grad clipping ----
    grad_norm_before_clip = None
    if max_grad_norm is not None:
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )

    # ---- 9. update ----
    optimizer.step()
    optimizer.zero_grad()  
    avg_loss = total_loss / gradient_accumulation_steps

    metadata = {
        **reward_metadata,
        **adv_metadata,
        **last_pg_metadata,
        "grad_norm_before_clip": grad_norm_before_clip,
    }

    return torch.tensor(avg_loss), metadata


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
