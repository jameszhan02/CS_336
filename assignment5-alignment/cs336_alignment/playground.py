"""
单卡 (RTX 3070 Ti, 8GB) GRPO 训练脚本骨架。

架构：vLLM 只用来做推理（生成rollout），训练用PyTorch原生backward。
两者交替占用同一张GPU，权重同步靠"存checkpoint -> vLLM重新加载"，
不使用NCCL/IPC实时同步（单卡场景下这两者要么不支持、要么实现成本过高）。

时间线：
  for grpo_step in range(num_steps):
      1. 把当前policy存成HF checkpoint
      2. 启动vLLM，加载这个checkpoint，生成rollout，关闭vLLM
      3. 用这些rollout跑 run_grpo_train_step (PyTorch backward)

=== OOM修复说明 (相对上一版的3处改动) ===
1. AdamW -> bitsandbytes.optim.PagedAdamW8bit
   普通AdamW在bf16下，exp_avg+exp_avg_sq两个state各占~2GB，
   加上参数+梯度，光"地基"就吃满8GB，根本没给activation/vLLM留空间。
   8bit state把这部分压缩到~0.5GB×2，腾出大约3GB。
2. 新增 move_optimizer_state()，并在model.to("cpu")/to(DEVICE)前后都调用它
   model.to(device) 只搬模型的parameter/buffer，不会动optimizer.state里的
   exp_avg/exp_avg_sq——这些tensor是独立存在optimizer.state这个dict里的。
   之前代码里"model挪到CPU给vLLM腾显存"那步，这部分根本没挪，
   vLLM起来的时候GPU上还卡着这两块，这是导致OOM最直接的原因。
3. 开启 gradient_checkpointing
   用少量重计算换activation显存，8GB卡上基本是免费的优化。

运行前需要: pip install bitsandbytes --break-system-packages
"""

import gc
import json
import os
import random
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设你的 vllm_helpers.py 和 adapters.py (含 run_grpo_train_step 等) 都在同一目录
from vllm_helpers import VLLMServer
from adapters import run_grpo_train_step  # 你之前实现的训练函数
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# ============ 配置（来自 handout 的 default 超参数） ============

MODEL_ID = "allenai/OLMo-2-0425-1B"
CHECKPOINT_DIR = "./grpo_checkpoints"
DEVICE = "cuda:0"

N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024

NUM_ROLLOUT_STEPS = 200
LEARNING_RATE = 1e-5

ROLLOUT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 256
GROUP_SIZE = 8
N_PROMPTS_PER_ROLLOUT_BATCH = ROLLOUT_BATCH_SIZE // GROUP_SIZE  # 32

GRADIENT_ACCUMULATION_STEPS = 32
MICROBATCH_SIZE = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS  # 8，对8GB显存友好

SAMPLING_TEMPERATURE = 1.0
SAMPLING_MAX_TOKENS = 512
MAX_GRAD_NORM = 1.0

# vLLM 显存配置：8GB卡，必须给训练模型留够空间
# 训练阶段vLLM是关闭的，所以这个值在"采样阶段"可以设高一点
# 注：optimizer state也搬到CPU之后，这里其实可以适当调高(比如0.6~0.7)
VLLM_GPU_MEMORY_UTILIZATION = 0.5

# r1_zero prompt 模板路径（跟你之前eval脚本里用的一致）
R1_ZERO_PROMPT_PATH = "prompts/r1_zero.prompt"
GSM8K_TRAIN_PATH = "../data/gsm8k/train.jsonl"


def reward_fn(response: str, ground_truth: str) -> dict:
    """直接用 handout 提供的 r1_zero_reward_fn，不自己写打分逻辑。

    handout明确说明：即使返回了format_reward，训练时也只用 reward 这个字段，
    不要把 format_reward 混进 advantage 计算。
    """
    return r1_zero_reward_fn(response, ground_truth)


def load_prompts_and_ground_truths(n_prompts: int, prompt_template: str, examples_pool: list[dict]) -> tuple[list[str], list[str]]:
    """从给定的GSM8K样本池里随机采样 n_prompts 条，套上 r1_zero 模板。"""
    sampled = random.sample(examples_pool, n_prompts)

    prompts = [prompt_template.format(question=ex["question"]) for ex in sampled]
    ground_truths = [ex["answer"].split("####")[-1].strip() for ex in sampled]
    return prompts, ground_truths


def free_gpu_memory():
    """训练/推理切换时手动清理显存，8GB卡上很关键"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str):
    """把optimizer.state里的tensor(exp_avg/exp_avg_sq等)搬到指定device。

    model.to(device) 不会动这些tensor——它们存在optimizer.state这个
    独立的dict里，不是model的parameter/buffer，traversal覆盖不到。
    这是单卡场景下"model挪CPU给vLLM腾显存"必须配套做的一步，
    漏掉这一步，GPU上的optimizer state会一直占着，是最容易被忽略的OOM来源。
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with open(R1_ZERO_PROMPT_PATH) as f:
        prompt_template = f.read()

    # 按 handout 要求，只用前 N_TRAIN_EXAMPLES 条作为训练池
    with open(GSM8K_TRAIN_PATH) as f:
        all_train_examples = [json.loads(line) for line in f]
    train_pool = all_train_examples[:N_TRAIN_EXAMPLES]

    # ---- 初始化训练用的模型和tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # 8GB显存上，bf16比fp32省一半显存
    ).to(DEVICE)
    model.gradient_checkpointing_enable()  # 用计算换activation显存，几乎免费
    model.train()

    # handout 指定的 optimizer 配置：betas=(0.9, 0.95), weight_decay=0.0
    # 改用8bit Adam：exp_avg/exp_avg_sq从bf16(各~2GB)压到int8(各~0.5GB)，
    # 在8GB卡上这是单项收益最大的改动。PagedAdamW8bit遇到真不够时
    # 还会自动往CPU溢出，相当于多一层保险。
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    for rollout_step in range(NUM_ROLLOUT_STEPS):
        print(f"\n=== Rollout step {rollout_step} ===")

        # ---------------------------------------------------------
        # 阶段1：存checkpoint，启动vLLM采样，关闭vLLM
        # ---------------------------------------------------------
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{rollout_step}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

        # 训练模型 + optimizer state 都挪到CPU，给vLLM腾显存
        # （只挪model不挪optimizer state，是上一版OOM的直接原因）
        model.to("cpu")
        move_optimizer_state(optimizer, "cpu")
        free_gpu_memory()

        prompts, ground_truths = load_prompts_and_ground_truths(
            N_PROMPTS_PER_ROLLOUT_BATCH, prompt_template, train_pool
        )
        repeated_prompts = [p for p in prompts for _ in range(GROUP_SIZE)]
        repeated_ground_truths = [g for g in ground_truths for _ in range(GROUP_SIZE)]

        server = VLLMServer(
            model_id=ckpt_path,
            host="127.0.0.1",
            port=8000,
            gpu=0,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
        server.start()

        completions = server.generate_completions(
            prompts=repeated_prompts,
            sampling_params={
                "temperature": SAMPLING_TEMPERATURE,
                "max_tokens": SAMPLING_MAX_TOKENS,
                "n": 1,  # 已经在repeated_prompts里手动复制了group_size份，这里设1
                "seed": rollout_step,
            },
        )
        rollout_responses = [c.text for c in completions]

        server.stop()  # 关闭vLLM进程，释放显存
        free_gpu_memory()

        # ---------------------------------------------------------
        # 阶段2：把训练模型 + optimizer state 都挪回GPU，跑GRPO训练
        # ---------------------------------------------------------
        model.to(DEVICE)
        move_optimizer_state(optimizer, DEVICE)
        model.train()

        loss, metadata = run_grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            reward_fn=reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=GROUP_SIZE,
            baseline="mean",
            advantage_eps=1e-6,
            advantage_normalizer="std",
            importance_reweighting_method="none",  # 单卡每步都重新采样，新旧policy一致，不需要重要性采样
            old_log_probs=None,
            cliprange=None,
            loss_normalization="sequence",
            normalization_constant=None,
        )

        print(f"loss={loss.item():.4f}  metadata={metadata}")

    # ---- 保存最终模型 ----
    final_path = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n训练完成，最终模型存在: {final_path}")


if __name__ == "__main__":
    main()