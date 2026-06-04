# CS336 Assignment 1 Summary

## Transformer model from scratch [Model]

### Tokenizer (BPE)

- start with UTF-8 byte level tokens
- merge tokens base on co-occurrence

---

## Loss - Cross Entropy

- for the “correct answer” calculate -log(perdict_res)
    - if we predict correct as 0.99 → -log(0.99) = 0.01 [Loss is small]
    - if we predict correct as 0.01 (very wrong) = -log(0.01) = 4.6 [large loss]

---

## Optimizer - AdamW

- Evaluate Process: SGD → Momentum → RMSProp → Adam → AdamW

- w = (1 - lambda * learning_rate) w - learning_rate * (V^t / sqar(G^t + a_small_number))

- V^t: = beta(a ratio) * V^t-1 + (1 - beta) g_t (current step grad)

- G^t: = beta * G^t-1 + (1 - beta) g_t^2
    - square cause we only care how "long" it goes should we scale the original not the sign it self

- Weight Decay:
    - (1 - lambda * learning_rate) w
    - [too make sure weight is as small as possible, general knowledge]
    - {but Why small weight will have such effect?}

- !! in real practical:
    - m_hat = m / (1 - beta1 ** t)
    - m_hat is an adjust value so in early step m are able to scale itself to bigger to push the training and stay almost same as m later in the traing since beta < 0, while t get bigger tend to 0.

---

## Activation function | && | FNN - SwiGLU

- SiLU = input * softmax(input)
    - a more smooth ReLU

- hidden (nomal linear layer)
    - W * x

- SiLU as gate
    - siLU * hidden
    - @ W_3 projection to the target dim as output of SwiGLU

---

## Transformer Block

- Multi-head attention
    - TODO: detail for implementation of Multi-head attention in plain English text.

- RMSNorm

- SwiGLU

---

## Transformer Overview

### Model - Training Loop

### Forward Overview

- x is like [batch seq_length]
    - with last dim idx of our vocab

- then after embedding is
    - [batch seq_length embedding_dim(d_model)]

- then went into # of transformer block
    - shape wont change
    - is just doing projection and "combine meaning" sort of stuff

- self.ln_final = RMSNorm(d_model)
    - same shape
    - just generalize output

- output
    - [batch seq_length vocab_size]
    - the last dim is logits

---

## Decoder

take the output form model last layer (logits)

- logits is a vocab_size numbers
- we need to apply softmax to it in order to get the "probability"

### Temperature

- during the softmax, apply a const number to logits

```python
logits / temperature
```

- default is 1, so logits remain unchanged
- temp < 1 will make output more stable
- temp > 1 will be more "random"

### Top-p Sampling

- set a threshold (ex. 0.9)
- pick largest probability form softmax result
- continue until sum went over the threshold we set
- so we get a smallest set that contribute the target rate

### Generation Stop Conditions

keep generate tokens base on the model prediction until:

1. find the special token for stop

OR

2. hit the max_tokens that allowed to generate
