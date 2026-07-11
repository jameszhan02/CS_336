# Batch Decoding / Inference TODO

## Goal

把当前单 prompt inference 扩展成 batch inference。

建议路线：

1. 先做等长 prompt batch，不处理 padding。
2. 再做 left padding 的变长 prompt batch。
3. 最后把变长 batch 和 KV cache 结合起来。

不要一开始直接做最复杂版本，否则 padding mask、RoPE position、KV cache length 会同时混在一起，debug 很难。

---

## Phase 1: 等长 Prompt Batch Baseline

### TODO 1.1 新增 batch generate 入口

当前单条 generate 输入类似：

- `prompt_tokens: list[int]`

batch 版输入改成概念上的：

- `prompt_tokens_batch: list[list[int]]`

先限制：

- batch 内所有 prompt 长度相同
- 不需要 padding
- 不需要 padding mask

### TODO 1.2 维护 batch 状态

batch decoding 需要维护：

- `B`: batch size
- `outputs`: 每条样本自己的 token list
- `finished`: shape `[B]`，记录每条是否已经遇到 EOS
- `next_tokens`: shape `[B]`，每一步采样出的 token

检查点：

- 某条样本遇到 EOS 后，不应该继续追加真实生成 token。
- 其他未结束样本仍然继续生成。

### TODO 1.3 batch 输入模型

等长 prompt 可以直接堆成：

- `input_ids`: `[B, T]`

模型输出：

- `logits`: `[B, T, vocab_size]`

下一 token logits 取：

- `logits[:, -1, :]`

检查点：

- `logits[:, -1, :]` shape 应该是 `[B, vocab_size]`

### TODO 1.4 batch sampling

把 sampling 从单个 vocab distribution 扩展到 batch：

- 输入 probs shape: `[B, vocab_size]`
- 输出 next token shape: `[B]`

可以先用循环对每一行调用现有 sampling 逻辑，等正确后再考虑向量化。

---

## Phase 2: 等长 Prompt Batch + KV Cache

### TODO 2.1 prefill batch prompt

输入：

- `input_ids`: `[B, T]`
- `kv_cache = None`

调用模型 cache 路径后得到：

- `logits`: `[B, T, vocab_size]`
- `kv_cache`: 每层 K/V cache

采样：

- 用 `logits[:, -1, :]` 采样第一批 next tokens

### TODO 2.2 incremental batch decode

后续每一步输入：

- `last_tokens`: `[B, 1]`
- `kv_cache`: 上一步返回的 cache

模型输出：

- `logits`: `[B, 1, vocab_size]`
- updated `kv_cache`

采样：

- 使用 `logits[:, -1, :]`

### TODO 2.3 cache shape 检查

如果你的 attention layout 是 `[B, H, T, D]`，每层 cache 应该是：

- `K_cache[layer]`: `[B, num_heads, cached_seq_len, head_dim]`
- `V_cache[layer]`: `[B, num_heads, cached_seq_len, head_dim]`

每生成一个 token，`cached_seq_len` 应加 1。

检查点：

- prefill prompt 长度为 `T`，cache length 应为 `T`
- 生成 1 步后，cache length 应为 `T + 1`
- 生成 2 步后，cache length 应为 `T + 2`

---

## Phase 3: 变长 Prompt Batch With Left Padding

### TODO 3.1 构造 left-padded input_ids

对不同长度 prompt 做 left padding。

例子：

- prompt 1: `[A, B, C]`
- prompt 2: `[D, E, F, G, H]`

left-padded 后：

- `[PAD, PAD, A, B, C]`
- `[D,   E,   F, G, H]`

需要决定：

- `pad_token_id`

如果没有专门 PAD token，可以临时用 `<|endoftext|>`，但要通过 mask 阻止它作为真实上下文被 attend。

### TODO 3.2 构造 attention_mask

attention mask 标记哪些位置是真实 token：

- `1` / `True`: 真实 token
- `0` / `False`: PAD

例子：

- `[0, 0, 1, 1, 1]`
- `[1, 1, 1, 1, 1]`

shape:

- `[B, T_max]`

这个 mask 是 key padding mask 的来源。

### TODO 3.3 构造 position_ids

RoPE position 应该让真实 token 从 0 开始计数。

例子：

- input: `[PAD, PAD, A, B, C]`
- position: `[0,   0,   0, 1, 2]`

- input: `[D, E, F, G, H]`
- position: `[0, 1, 2, 3, 4]`

PAD 位置的 position 可以先设成 0，因为 PAD 应该被 attention mask 屏蔽。

shape:

- `[B, T_max]`

### TODO 3.4 prefill logits 仍然取最后一列

left padding 的好处是每条样本的最后有效 token 都在最后一列。

因此 prefill 后可以取：

- `logits[:, -1, :]`

但前提是：

- attention mask 正确屏蔽 PAD
- position ids 对真实 token 正确

---

## Phase 4: 修改模型 forward 接口

### TODO 4.1 TransformerLM.forward 支持可选参数

当前概念接口：

- `model(in_indices, kv_cache=None)`

扩展为概念上的：

- `token_positions=None`
- `attention_mask=None`
- `kv_cache=None`

默认不传时，训练路径保持不变：

- `model(train_x)` 仍然只返回 logits

### TODO 4.2 token_positions 向下传递

`TransformerLM.forward` 需要把 `token_positions` 传给每个 `TransformerBlock`。

如果 `token_positions is None`：

- 使用默认位置 `0 ... T-1`

如果传入了 batch position：

- 使用 shape `[B, T]` 的位置

### TODO 4.3 attention_mask 向下传递

`TransformerLM.forward` 需要把 `attention_mask` 或 key padding mask 传给每个 block / attention。

注意：

- 训练时可以不传，只使用 causal mask。
- batch left padding 时必须传，否则真实 token 会 attend 到 PAD。

---

## Phase 5: 修改 TransformerBlock

### TODO 5.1 block forward 接收 token_positions

block 不需要自己推导 batch 内每条样本的位置。

它应该接收上层传下来的：

- `token_positions`

然后传给 attention。

### TODO 5.2 block forward 接收 attention_mask

block 接收：

- `attention_mask`

然后传给 attention，用于构造最终 attention mask。

### TODO 5.3 KV cache 路径不要只依赖 cache.seq_len 推导位置

单条或等长 batch 时，用 `cache.seq_len()` 推导位置可以工作。

变长 batch 时，每条样本真实长度不同，应该由 generate 维护并传入：

- incremental `token_positions`: `[B, 1]`

---

## Phase 6: 修改 Attention Mask 逻辑

### TODO 6.1 保留 causal mask

训练和 prefill 阶段仍然需要 causal mask：

- query 不能看未来 key

### TODO 6.2 添加 padding mask

变长 batch left padding 时，还需要 key padding mask：

- query 不能 attend 到 PAD key

最终 mask 概念上是：

- `final_mask = causal_mask AND key_padding_mask`

### TODO 6.3 mask shape 检查

attention scores 常见 shape：

- `[B, H, query_len, key_len]`

final mask 应该能 broadcast 到这个 shape。

常见目标 shape：

- `[B, 1, query_len, key_len]`

### TODO 6.4 incremental 阶段的 padding mask

如果 cache 里包含 left padding 产生的 PAD K/V，那么 incremental 阶段仍然需要知道历史 key 哪些是 PAD。

因此 generate 需要维护：

- `cache_attention_mask`: `[B, cached_key_len]`

prefill 后例子：

- `[0, 0, 1, 1, 1]`
- `[1, 1, 1, 1, 1]`

生成一步后 append `1`：

- `[0, 0, 1, 1, 1, 1]`
- `[1, 1, 1, 1, 1, 1]`

attention 用它来屏蔽 PAD keys。

---

## Phase 7: RoPE Position Handling

### TODO 7.1 prefill 使用 batch position_ids

left padding prefill 时，不要直接用 padded column index。

错误：

- `[PAD, PAD, A, B, C]`
- positions: `[0, 1, 2, 3, 4]`

正确概念：

- `[PAD, PAD, A, B, C]`
- positions: `[0, 0, 0, 1, 2]`

### TODO 7.2 incremental 使用 per-sample current position

如果 batch prompt 长度是：

- `[3, 5]`

prefill 后第一次 incremental token 的 positions 应该是：

- `[[3], [5]]`

生成一步后变成：

- `[[4], [6]]`

不要用一个 batch 共享的 scalar position。

### TODO 7.3 finished 样本的 position

如果某条样本已经 EOS：

- 可以继续喂 EOS/PAD 以保持 batch shape
- 但不要继续把采样结果 append 到该样本输出
- position 是否递增取决于你的实现策略

最简单策略：

- finished 样本仍然走模型，但输出忽略
- 保持 position 递增，避免 shape/state 分支复杂

---

## Phase 8: Sampling And EOS

### TODO 8.1 batch top-p sampling

先用简单循环实现：

- 对 batch 中每一行 logits/probs 独立 sampling
- 得到 `next_tokens: [B]`

### TODO 8.2 finished mask

如果 `finished[i] == True`：

- 不再追加真实生成 token
- 下一步输入可以固定为 `eos_token_id` 或 `pad_token_id`

### TODO 8.3 停止条件

整个 batch 可以在以下条件停止：

- 所有 `finished` 都为 True
- 或达到 `max_new_tokens`

---

## Phase 9: Correctness Checks

### TODO 9.1 等长 batch vs 单条逐个跑

准备两个等长 prompts。

比较：

- batch generate 的第 i 条输出
- 单独 generate 第 i 条输出

在固定随机种子或 greedy sampling 下，应该一致。

### TODO 9.2 left padding vs 单条跑

准备不同长度 prompts。

比较：

- left-padded batch prefill 的最后 logits
- 单条 prompt prefill 的最后 logits

应该接近。

如果不接近，优先检查：

- position ids
- padding mask
- causal mask

### TODO 9.3 KV cache incremental 等价性

对每条样本比较：

- full forward `prompt + generated_so_far` 的最后 logits
- cache forward 只输入 last token 的 logits

应该接近。

### TODO 9.4 cache shape 检查

每一层检查：

- K/V batch size 是否等于 B
- cached seq length 是否按预期增加
- K/V device 是否和 model/input 一致

---

## Recommended Implementation Order

1. 新建 `batch_generate`，只支持等长 prompt，不使用 KV cache。
2. 给 batch sampling 和 EOS 状态跑通。
3. 加等长 prompt batch KV cache。
4. 加 left padding input 构造。
5. 加 `attention_mask` 和 `position_ids`。
6. 修改 model/block/attention 接口透传 mask 和 position。
7. 加变长 batch prefill correctness check。
8. 最后加变长 batch + KV cache。

---

## Most Likely Bugs

- left padding 后仍然用 `0..T-1` 作为 RoPE position。
- causal mask 正确，但忘了屏蔽 PAD key。
- prefill 能跑，但 incremental 阶段忘了维护 `cache_attention_mask`。
- batch 内某条 EOS 后仍然不断 append token。
- 用 cache tensor length 当作每条样本真实 position。
- mask shape 不能 broadcast 到 `[B, H, query_len, key_len]`。
- batch sampling accidentally 对整个 batch 做一次 multinomial，而不是每条样本各采样一次。

