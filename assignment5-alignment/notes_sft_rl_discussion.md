# Side Notes — SFT data usage & RL loss/backward

> Running notes from a discussion. These are conceptual study notes, not solution code.

## Part 1 — How the SFT "training data" is used

After tokenizing, each example holds three aligned tensors:

- `input_ids` — the full prompt + response token sequence
- `labels` — the next-token targets (a shifted version of `input_ids`)
- `response_mask` — 1 on response tokens, 0 on prompt tokens

### Key idea
An SFT training step is **structurally identical to pretraining**:

1. **Forward pass** — run `input_ids` through the model *once* → logits `(batch, seq_len, vocab)`. One pass, all positions at once (teacher forcing). Not token-by-token.
2. **Per-token cross-entropy** between logits and `labels`.
3. **Apply `response_mask`** — the *only* real addition vs. pretraining. Zero out loss on prompt tokens so gradients come only from response tokens.
4. **Backprop + optimizer step.**

### "Inference" — two meanings (don't conflate)
- **(1) Single forward pass** → this is what a training step uses. Same as pretraining.
- **(2) Autoregressive generation** (token → sample → append → repeat) → only happens *later*, at eval / sampling. Plays **no role** in computing the SFT loss.

### Bug-hunting checklist
- **Shift alignment**: wherever the off-by-one shift between `input_ids` and `labels` happens, make sure `response_mask` uses the *same* indexing as `labels`. Classic off-by-one bug.
- **Denominator**: when averaging masked loss, divide by sum of mask (# response tokens) vs. total length → different gradient magnitudes. Match what the assignment specifies.
- **Toy check**: 3-token prompt + 2-token response, print `input_ids`, `labels`, `response_mask` as columns. Mask should be 1 exactly on response-token label positions.

---

## Part 2 — RL (GRPO): reward → loss → backward, with numbers

### Step 0: What is `log π`, and why log? (primer)

**π = the policy = the model viewed as a probability distribution.** `π_θ` = policy with params θ. At each position the model sees the prefix (state `s`) and outputs a distribution over the vocab; `π(token | prefix)` = the probability it assigns to that one token (action `a`). It comes straight from softmax over the logits.

**`log π` = log of that probability.** One number per token. Example (Step 6, pos 3): model gave `p("5") = 0.25` → `log π = log(0.25) = −1.386`.
- Always ≤ 0 (prob ≤ 1). Confident (p→1) → `log π → 0`. Unsure (p→0) → very negative.
- Read it as: *how confident the model was it would emit this exact token.*
- Function of **θ** (carries gradient). The advantage does not.

**Where the `(batch, seq_len)` tensor comes from:**
`model forward → logits (b, seq, vocab) → log_softmax over vocab → gather the entry for the ACTUAL token at each position → log π (b, seq)`. Keep only the real token's log-prob, not all vocab.

**Whole-sequence:** `π(response) = ∏ₜ π(tokₜ)` ⇒ `log π(response) = Σₜ log π(tokₜ)`. Product → sum.

#### Why work in log at all? Core reason: **log turns multiplication into addition.**
A response is many tokens, so its probability is a *product* of per-token probs. Multiplying many probabilities is miserable:

1. **Underflow.** Each prob < 1, so the product vanishes. `0.5¹⁰ ≈ 0.001`; `0.5¹⁰⁰ ≈ 8×10⁻³¹` → floats round to 0 → `log(0) = −∞` → training dies. In log-space: `100·log(0.5) = −69.3`, perfectly ordinary. **No underflow ever.**
2. **Gradients.** Product-rule over 100 factors entangles every token with every other. `log` makes it a sum ⇒ `∇log π(resp) = Σₜ ∇log π(tokₜ)`, each token's gradient **independent**, then added. (This is exactly the per-token-summed loss of Step 6 — it only looks clean because of log.)
3. **Free to do.** `log` is monotonic: bigger `π` ⟺ bigger `log π`. Maximizing log-prob *is* maximizing prob — same optimum, zero downside.

| raw `π` | `log π` |
|---------|---------|
| product of many terms | sum of many terms |
| underflows → `−inf` | stable numbers |
| entangled product-rule gradients | independent additive gradients |
| — | same optimum (log monotonic) |

That's why the function receives `policy_log_probs` already in log-space.

### Step 1: Rollout + reward
Prompt: `"What is 2+3?"`, sample a group of G = 4 responses:

| Response | Output | Correct? | Reward r |
|----------|--------|----------|----------|
| 1 | `"5"` | ✓ | 1.0 |
| 2 | `"6"` | ✗ | 0.0 |
| 3 | `"5"` | ✓ | 1.0 |
| 4 | `"4"` | ✗ | 0.0 |

### Step 2: Reward → advantage (group-normalized)
- mean = 0.5, std = 0.5
- `A = (r - mean) / std`

| Response | r | Advantage A |
|----------|---|-------------|
| 1 | 1.0 | +1.0 |
| 2 | 0.0 | −1.0 |
| 3 | 1.0 | +1.0 |
| 4 | 0.0 | −1.0 |

Advantage is broadcast to **every response token** of that response. `response_mask` decides which tokens count.

### Step 3: Per-token loss
$$L = -\,A \cdot \log \pi_\theta(\text{token})$$
Minimize it. A > 0 ⇒ push token prob up; A < 0 ⇒ push it down.

Concrete (response 1, token `"5"`, vocab `{"5","6","4"}`):
- logits `z = [2.0, 1.0, 0.5]`
- softmax `p = [0.629, 0.231, 0.140]`
- `log π("5") = -0.464`
- `L = -(+1.0)(-0.464) = +0.464`

### Step 4: Backward (softmax gradient)
$$\frac{\partial \log \pi_i}{\partial z_j} = \delta_{ij} - p_j
\quad\Rightarrow\quad
\frac{\partial L}{\partial z} = -A\,(\text{onehot}_i - p)$$

Numbers (A = +1, sampled index 0):
$$\frac{\partial L}{\partial z} = -1.0\cdot([1,0,0]-[0.629,0.231,0.140]) = [-0.371,\ +0.231,\ +0.140]$$

Illustrative-only step on the logits (lr = 0.1) — see ⚠️ note below:
$$z_{new} = [2.0,1.0,0.5] - 0.1\cdot[-0.371,0.231,0.140] = [2.037,\ 0.977,\ 0.486]$$

Logit for `"5"` ↑, others ↓ ⇒ prob of `"5"` increases. Wrong response (A = −1) flips signs ⇒ token prob decreases. **Reward sign → gradient direction.**

> ⚠️ **What is `z`, and what actually gets updated?**
> - `z = [2.0, 1.0, 0.5]` are **logits** — the model's *output* over the vocab at one position, *before* softmax. **Not parameters.** They're activations, recomputed every forward pass from `z = (hidden state) · (output embedding matrix)`. Never stored.
> - **Parameters `θ`** = the weight matrices/embeddings. These are what's stored and what the optimizer updates.
> - `z` is **downstream** of `θ`: logits depend on params but aren't params.
> - The `z_new = z - lr·∂L/∂z` line above is a **pedagogical shortcut** to show *direction* ("logit for `5` goes up"). Mechanically you do **not** edit logits in place.
> - **Real update:** `θ_new = θ - lr·∂L/∂θ`. The `∂L/∂z = [-0.371, +0.231, +0.140]` is an **intermediate** in backprop; it chains further back through the output matrix and the whole net to reach `∂L/∂θ`. On the *next* forward pass the new `θ` produces a new `z` with the higher logit for `5`. That re-emergence is where the probability actually shifts.

### Step 5: In practice
You do **not** hand-compute the softmax gradient. Build the scalar loss `~ -(A * logπ)` over masked response tokens, call `.backward()`, optimizer does the update across *all* params. Autograd reproduces the numbers above.

### Step 6: Multi-token response — backward over a sequence

The 1-token case hid the most important structural fact. Now make the response **3 tokens**: `["The", "sum", "5"]`, correct ⇒ **A = +1.0 broadcast to all 3 tokens**. Toy 3-word vocab.

**Forward chain (where these numbers come from):**
`hidden h_t → logits z_t = h_t·W_out → softmax(z_t) = p_t → logπ = log p(target)`
The model emits **logits**; `p_t` is *derived* by softmax. Realistic = confidence *varies* per position (common word confident, answer digit unsure):

| t | target token | logits `z_t` | softmax `p_t` | `p(target)` | `log π` | `L_t = −A·logπ` |
|---|--------------|--------------|---------------|-------------|---------|------------------|
| 1 | `The` (idx 0) | [1.95, 0.69, 0.00] | [0.70, 0.20, 0.10] | 0.70 | −0.357 | 0.357 |
| 2 | `sum` (idx 1) | [0.00, 0.29, 0.00] | [0.30, 0.40, 0.30] | 0.40 | −0.916 | 0.916 |
| 3 | `5` (idx 2)   | [0.47, 0.34, 0.00] | [0.40, 0.35, 0.25] | 0.25 | −1.386 | 1.386 |

> **Two facts about logits:**
> 1. **In code you never do softmax→log→pick.** Use `log_softmax(z)` / `cross_entropy(z, target)` directly on **logits** — separate softmax+log underflows (`log(0) = −inf`). Your log-probs come straight from the logits tensor.
> 2. **Logits are only defined up to an additive constant:** `softmax(z + c) = softmax(z)`. So `[1.95,0.69,0.0]` and `[101.95,100.69,100.0]` give identical `p`. Only the *gaps* between logits carry information.

- Sum form:  `L_total = 0.357 + 0.916 + 1.386 = 2.659`
- Average form: `L_total = 2.659 / 3 = 0.886`  (divide by # response tokens = 3)

**Per-position logit gradients** `∂L/∂z_t = −A(onehot_t − p_t)`:
- `∂L/∂z_1 = −1·([1,0,0] − [0.70,0.20,0.10]) = [−0.30, +0.20, +0.10]`
- `∂L/∂z_2 = −1·([0,1,0] − [0.30,0.40,0.30]) = [+0.30, −0.60, +0.30]`
- `∂L/∂z_3 = −1·([0,0,1] − [0.40,0.35,0.25]) = [+0.40, +0.35, −0.75]`

Each position pushes *its own* target logit ↑ and the others ↓ — **independent at the logit level** (position `t`'s loss depends only on position `t`'s logits).

**Lesson from the varied numbers:** gradient magnitude is **largest where the model was least confident** in the correct token. Token `5` (p = 0.25) has the biggest loss (1.386) and the biggest logit-gradient (−0.75). With A > 0, the update pushes *hardest* on the low-probability tokens along a good response — i.e. it learns most from the parts it was getting wrong. As `p(target) → 1`, both `logπ → 0` and the gradient `(onehot − p) → 0`: a token the model is already certain about contributes almost nothing. (When all `logπ` look equal, that's a toy artifact, not reality.)

**The key multi-token fact:**
$$L_{total} = \sum_{t\in\text{response}} L_t
\qquad\Rightarrow\qquad
\frac{\partial L_{total}}{\partial \theta} = \sum_{t\in\text{response}} \frac{\partial L_t}{\partial z_t}\cdot\frac{\partial z_t}{\partial \theta}$$

- The transformer **weights θ are shared across all time steps**, so every token contributes a gradient term and they **accumulate into the same parameters**. Gradient of a sum = sum of gradients.
- Because of causal attention, `z_t` depends on θ via the hidden states for the whole prefix `0..t` — so each `∂z_t/∂θ` already spans the prefix. Autograd handles all of it.
- **One forward pass** (teacher forcing) produces all 3 positions' logits → **one summed/averaged scalar loss** → **one `.backward()`** computes the total gradient. You do *not* run backward per token.
- `response_mask` makes the sum run over **response tokens only**; prompt tokens contribute 0.

**What happens AFTER the intermediate `∂L/∂z_t` (the missing link):**

The three `∂L/∂z_t` rows are **not** summed together as vectors — they live at different positions. Stack them into one tensor, the starting point of backward:
```
∂L/∂logits = [ [−0.30, +0.20, +0.10]    ← pos 1
               [+0.30, −0.60, +0.30]    ← pos 2
               [+0.40, +0.35, −0.75] ]  ← pos 3   shape (seq_len, vocab)
```
Each row propagates further back on its own. The **sum happens at the parameters**, via weight sharing. Since `z_t = h_t·W_out` with the *same* `W_out` at every position:
$$\frac{\partial L}{\partial W_{out}} = \sum_t h_t^\top \otimes \frac{\partial L}{\partial z_t}$$
(each term = position `t`'s hidden ⊗ its logit-gradient row → a `W_out`-shaped matrix; sum over `t`). That accumulation lands on **parameters**, not logits.

**Sequence of events:**
1. **One** `.backward()` — loss → `(seq_len, vocab)` logit-grad tensor → back through `W_out` → all transformer layers → embeddings.
2. **Weight-sharing accumulation** — every shared weight (used at all positions + coupled across positions by causal attention) sums its per-position contributions into that param's `.grad`.
3. **One** `optimizer.step()` — update *all* params at once: `θ ← θ − lr·θ.grad`.

So: "sum them up?" = yes, but over **parameter** gradients (weight sharing), not over the logit-grad vectors. "Full backward?" = yes, exactly one, carrying all positions together. "Update all at once?" = yes, single optimizer step.

**Why this connects to length normalization (wrinkle 1):** a 3-token response contributes 3 terms; a 50-token response contributes 50. If you *sum*, long responses dominate the gradient just by being long. If you *average per response*, each response gets equal say regardless of length. That choice is exactly this Σ vs. mean.

### Wrinkles to pin down in the real implementation
1. **Length normalization** — sum vs. average over response tokens; average by per-response length or a fixed constant? Affects long-vs-short weighting.
2. **Masking denominator** — divide by `response_mask.sum()` vs. total length.
3. **Plain REINFORCE vs. clip/ratio (PPO-style)** — does the variant use current-policy logπ directly, or a ratio against an old policy?
4. **Microbatching** — gradient accumulation over microbatches; how it interacts with the averaging constant in (1).

### Self-test
Take the 4-response example (each response = 1 token), compute total loss and predict which logits go up/down for all four *before* running code.

---

## Open threads / to discuss next
- [ ] Why length-normalization choice matters for training stability
- [ ] How masking denominator interacts with microbatch gradient accumulation
- [ ] REINFORCE vs. PPO clip term — when the ratio matters
