import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import KVCache, AdamW, run_get_batch, run_transformer_lm, TransformerLM, run_cross_entropy, run_save_checkpoint, run_train_bpe, Tokenizer
import numpy as np
import torch

import torch.nn.functional as F
import pickle


# do the tokenizer first 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# bin_path = os.path.join(BASE_DIR, "data", "owt_valid_tokens.bin")
train_input_path = os.path.join(BASE_DIR, "data", "TinyStories_small.txt")
train_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txt.bin")
valid_input_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt")
valid_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt.bin")

vocab_size = 10000
CONTEXT_LENGTH = 256
d_model = 256
num_layers = 4
num_heads = 4
d_ff = 672
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
rope_theta = 10000



vocab_path = os.path.join(BASE_DIR, "data", "vocab.pkl")

if not os.path.exists(train_bin_path):
    print("step 1: starting BPE training...")
    vocab, merges = run_train_bpe(
        input_path=train_input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    with open(vocab_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    myTokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    print("step 2: BPE done, encoding text...")
    with open(train_input_path, "r") as f:
        train_text = f.read()
    train_token_ids = myTokenizer.encode(train_text)
    with open(valid_input_path, "r") as f:
        valid_text = f.read()
    valid_token_ids = myTokenizer.encode(valid_text)
    print(f"step 3: encoding done, {len(train_token_ids)} train tokens | {len(valid_token_ids)} valid tokens")
    train_token_array = np.array(train_token_ids, dtype=np.int32)
    train_token_array.tofile(train_bin_path)
    valid_token_array = np.array(valid_token_ids, dtype=np.int32)
    valid_token_array.tofile(valid_bin_path)
    print(f"step 4: saved to {train_bin_path} {valid_bin_path}")
else:
    print("token file already exists, loading vocab...")
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)
    vocab, merges = data["vocab"], data["merges"]
    myTokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    print("done tokenizer")


model = TransformerLM(vocab_size, CONTEXT_LENGTH, d_model,
                    num_layers, num_heads, d_ff, rope_theta)
model.to(DEVICE)
optimizer = AdamW(model.parameters())

def train(model, optimizer, train_path, val_path, max_steps, log_interval, eval_interval, save_interval, save_path):
    os.makedirs(save_path, exist_ok=True) 
    # TODO: change these to var
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 80

    # train_path = "" # fill in the real path 
    # val_path = ""
    train_data = np.memmap(train_path, dtype=np.int32, mode="r") 
    val_data = np.memmap(val_path,dtype=np.int32, mode="r") 

    for step in range(max_steps):
        model.train()
        # 1. get a batch
        train_x, train_y = run_get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        # 2. forward pass + loss
        logits = model(train_x)
        batch, seq_len, vocab = logits.shape
        loss = run_cross_entropy(
            logits.reshape(batch * seq_len, vocab),
            train_y.reshape(batch * seq_len)
        )
        
        # 3. backprop + optimizer step
        optimizer.zero_grad()   # clear gradients from last step
        loss.backward()         # compute gradients
        optimizer.step()        # update weights
        # 4. every N steps — log train loss
        if step % log_interval == 0:
            print(f"step {step} | train loss: {loss.item():.4f}")
        # 5. every N steps — check validation loss
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():   # don't compute gradients during eval
                val_x, val_y = run_get_batch(val_data, TEST_BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
                val_logits = model(val_x)
                batch, seq_len, vocab = val_logits.shape
                val_loss = run_cross_entropy(
                    val_logits.reshape(batch * seq_len, vocab),
                    val_y.reshape(batch * seq_len)
                )
            print(f"step {step} | val loss: {val_loss.item(F):.4f}")
            model.train()  # switch back to train mode
        
        # 6. every N steps — save checkpoint
        if step % save_interval == 0:
            file_name = f"checkpoint_step{step}.pt"
            final_out_path = save_path + '/' + file_name
            run_save_checkpoint(model, optimizer, step, final_out_path)
            print(f"step {step} | checkpoint saved to {final_out_path}")




    

def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: (vocab_size,)
    return: (vocab_size,) 
    """
    return F.softmax(logits / temperature, dim=-1)


def nucleus_sampling(probs: torch.Tensor, p: float) -> int:
    """
    probs: (vocab_size,)
    p: nucleus 
    return: 
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True) # sort by the prob from softmax layer | return sort res and orginal idx
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # total prob cum with idx
    
    sorted_probs[cumulative_probs - sorted_probs > p] = 0.0 # for top p rule we should keep util hit limit and set all rest prob to 0
    
    sorted_probs = sorted_probs / sorted_probs.sum() # re-cal the current prob
    
    sampled_index = torch.multinomial(sorted_probs, num_samples=1).item() # sampling form sorted_probs as distribution.
    
    return sorted_indices[sampled_index].item()


def decode(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    p: float = 0.9,
    eos_token_id: int = None,
    device: str = "cpu"
) -> list[int]:
    """
    model:  TransformerLM
    prompt_tokens: tokenizer.encode(prompt) token ids
    max_new_tokens: max generate token
    temperature: < 1 more stable | > 1 more random
    p: nucleus sampling threshold
    eos_token_id: <|endoftext|> stop at end of text token id
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
        for _ in range(max_new_tokens): # generate at least till hit the max token limit 
            probs = softmax_with_temperature(logits, temperature)
            next_token = nucleus_sampling(probs, p)
            tokens.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

            # decode: over ride x with the next token only into the loop rest information already in the KV cache.
            x = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits = model(x, kv_cache=kv_cache)
            logits = logits[0, -1, :]
            
            # context = tokens[-CONTEXT_LENGTH:] # only fetch as long as the tokens not exceed the context tokens.
            # x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            # # x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            # logits = model(x)           # (1, sequence_length, vocab_size)
            # # only get the last seq_length position logits as the input for next token probs
            # logits = logits[0, -1, :]   # (vocab_size,)

            # # 3. temperature softmax
            # probs = softmax_with_temperature(logits, temperature)

            # # 4. nucleus sampling 
            # next_token = nucleus_sampling(probs, p)

            # # 5. append
            # tokens.append(next_token)

            # # 6. ending token condition
            # if eos_token_id is not None and next_token == eos_token_id:
            #     break

    return tokens



print("step 4: starting training loop...")
train(
    model, optimizer,
    train_path=train_bin_path,
    val_path=valid_bin_path,
    max_steps=5000,
    log_interval=100,
    eval_interval=50,
    save_interval=100,
    save_path="./checkpoints"
)



prompt = "The quick brown fox"
prompt_tokens = myTokenizer.encode(prompt)

eos_token_id = myTokenizer.encode("<|endoftext|>")[0]

output_tokens = decode(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=400,
    temperature=0.8,
    p=0.9,
    eos_token_id=eos_token_id,
    device=DEVICE
)

generated_tokens = output_tokens[len(prompt_tokens):]
generated_text = myTokenizer.decode(generated_tokens)

print(f"\nPrompt: {prompt}")
print(f"Generated: {generated_text}")
