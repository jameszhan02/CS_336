import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests')) # so current file could visit other files under ../tests folder
from adapters import find_chunk_boundaries, run_train_bpe_mp, AdamW, run_get_batch, run_transformer_lm, TransformerLM, run_cross_entropy, run_save_checkpoint, run_train_bpe, Tokenizer
import pickle
import time
import torch
import numpy as np
import mmap
import multiprocessing as mp

import torch.nn.functional as F
# Logger
import json
import time
import datetime
import os

import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainLogger:
    def __init__(self, log_dir: str, config: dict):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self.config_path = os.path.join(log_dir, "config.json")

        config_with_meta = dict(config)
        config_with_meta["start_time"] = datetime.datetime.now().isoformat()
        with open(self.config_path, "w") as f:
            json.dump(config_with_meta, f, indent=2)

        self._f = open(self.metrics_path, "a")

    def log(self, step: int, **kwargs):
        entry = {"step": step, "time": time.time(), **kwargs}
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush() 

    def close(self):
        self._f.close()
###

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("|BASE URL : ", BASE_DIR , "|")

# BPE_TRAIN_FILE_NAME = "owt_valid.txt"
BPE_TRAIN_FILE_NAME = "TinyStoriesV2-GPT4-train.txt"
train_input_path = os.path.join(BASE_DIR, "data", BPE_TRAIN_FILE_NAME)
valid_input_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt")

train_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txt.bin")
valid_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt.bin")


t0 = time.perf_counter()
vocab_size = 10000
saving_path = BPE_TRAIN_FILE_NAME + "vocab.pkl"
vocab_path = os.path.join(BASE_DIR, "data", saving_path)
if not os.path.exists(vocab_path):
    print("step 1: starting BPE training...")
    # TODO: implement run_train_bpe | mutiprocess
    vocab, merges = run_train_bpe_mp(
        input_path=train_input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    with open(vocab_path, "wb") as f: # wb stand for write binary
        pickle.dump({"vocab": vocab, "merges": merges}, f) # pickle is a tool that save py obj into files and then able to restore obj form files.
else: 
    print("token file already exists, loading vocab...")
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)
    vocab, merges = data["vocab"], data["merges"]

myTokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

print("longest token id:", longest_id)
print("longest token byte length:", len(longest_token))
print("longest token bytes:", longest_token)
## END of pre-tokenizer 
t1 = time.perf_counter()
print(f"BPE total time spent: {t1 - t0:.4f} seconds") # TODO: python native implementation seems like could not do perfect at this task, this implement save later to complete with "Tiny CC" task.


print("Step2: Endcoding training valid files ...")


# ======================= encoding part ========================
_worker_tokenizer = None  # 每个子进程自己持有一份 tokenizer

def _init_worker(vocab, merges, special_tokens):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer(vocab, merges, special_tokens)  # 换成你自己的类

def _encode_chunk(args: tuple[str, int, int]) -> np.ndarray:
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="ignore")
    del raw

    token_ids = _worker_tokenizer.encode(text)
    del text

    arr = np.array(token_ids, dtype=np.int32)
    del token_ids
    return arr


def encode_file_to_bin(
    input_path: str,
    output_bin_path: str,
    vocab: dict,
    merges: list,
    special_tokens: list[str],
    num_workers: int = 4,
) -> int:
    split_token = special_tokens[0].encode("utf-8") if special_tokens else None

    if split_token:
        boundaries = find_chunk_boundaries(input_path, num_workers, split_token)
    else:
        file_size = os.path.getsize(input_path)
        boundaries = [0, file_size]

    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    args = [(input_path, s, e) for s, e in ranges]

    total_tokens = 0
    with open(output_bin_path, "wb") as out_f:
        if len(ranges) == 1:
            arr = _encode_chunk(args[0])
            arr.tofile(out_f)
            total_tokens += len(arr)
        else:
            with mp.Pool(
                num_workers,
                initializer=_init_worker,
                initargs=(vocab, merges, special_tokens),
            ) as pool:
                for arr in pool.imap(_encode_chunk, args):
                    arr.tofile(out_f)      
                    total_tokens += len(arr)
                    del arr              

    return total_tokens


print("step 2: BPE done, encoding text...")

if os.path.exists(train_bin_path) and os.path.exists(valid_bin_path):
    print("bin files already exist, skipping encoding...")
    n_train = os.path.getsize(train_bin_path) // np.dtype(np.int32).itemsize
    n_valid = os.path.getsize(valid_bin_path) // np.dtype(np.int32).itemsize
else:
    n_train = encode_file_to_bin(train_input_path, train_bin_path, vocab, merges, ["<|endoftext|>"], num_workers=8)
    n_valid = encode_file_to_bin(valid_input_path, valid_bin_path, vocab, merges, ["<|endoftext|>"], num_workers=8)

print(f"step 3/4: encoded & saved | {n_train} train tokens | {n_valid} valid tokens")
# check out the compression ratio
with open(valid_input_path, "r", encoding="utf-8") as f:
        text = f.read()
        docs = text.split("<|endoftext|>")
        docs_10 = docs[:10]

my_tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
total_bytes = 0
total_tokens = 0

print("first doc sample print: ", docs_10[0])

for doc in docs_10:
    ids = my_tokenizer.encode(doc)
    total_bytes += len(doc.encode("utf-8"))
    total_tokens += len(ids)
ratio = total_bytes / total_tokens
print(f"compression ratio: {ratio:.4f} bytes/token")


# hypterParamters
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

learning_rate = 1e-3

SEED = 42
set_seed(SEED)

model = TransformerLM(vocab_size, CONTEXT_LENGTH, d_model,
                    num_layers, num_heads, d_ff, rope_theta)
model.to(DEVICE)
optimizer = AdamW(model.parameters())


def train(model, optimizer, train_path, val_path, max_steps, log_interval, eval_interval, save_interval, save_path, config):
    os.makedirs(save_path, exist_ok=True) 
    # TODO: change these to var
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 80 # for valid

    logger = TrainLogger(log_dir=save_path, config=config)

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
            train_loss_val = loss.item()
            print(f"step {step} | train loss: {train_loss_val:.4f}")
            logger.log(step=step, train_loss=train_loss_val)  # 写进 metrics.jsonl
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
            val_loss_val = val_loss.item()
            print(f"step {step} | val loss: {val_loss_val:.4f}")
            logger.log(step=step, val_loss=val_loss_val)  # 写进 metrics.jsonl
            model.train()  # switch back to train mode
        
        # 6. every N steps — save checkpoint
        if step % save_interval == 0:
            file_name = f"checkpoint_step{step}.pt"
            final_out_path = save_path + '/' + file_name
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "config": config,
            }, final_out_path)
            print(f"step {step} | checkpoint saved to {final_out_path}")
    # save the last cyheck point 
    final_step = max_steps - 1
    final_out_path = os.path.join(save_path, f"checkpoint_step{final_step}_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": final_step,
        "config": config,
    }, final_out_path)
    print(f"training done | final checkpoint saved to {final_out_path}")

    logger.close()
    

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

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-CONTEXT_LENGTH:]
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            # x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            logits = model(x)           # (1, sequence_length, vocab_size)
            logits = logits[0, -1, :]   # (vocab_size,)

            # 3. temperature softmax
            probs = softmax_with_temperature(logits, temperature)

            # 4. nucleus sampling 
            next_token = nucleus_sampling(probs, p)

            # 5. append
            tokens.append(next_token)

            # 6. ending token condition
            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens



print("step 4: starting training loop...")

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(BASE_DIR, "checkpoints", f"run_{run_id}")
config = {
    "vocab_size": vocab_size,
    "context_length": CONTEXT_LENGTH,
    "d_model": d_model,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "d_ff": d_ff,
    "rope_theta": rope_theta,
    "batch_size": 32,
    "test_batch_size": 80,
    "max_steps": 5000,
    "log_interval": 100,
    "eval_interval": 100,
    "save_interval": 1000,
    "device": DEVICE,
    "run_id": run_id,   
    "lr" : learning_rate
}

train(
    model, optimizer,
    train_path=train_bin_path,
    val_path=valid_bin_path,
    max_steps=5000,
    log_interval=100,
    eval_interval=100,
    save_interval=100,
    save_path=save_path, 
    config=config,   
)

prompt = "Long long time ago, there is a princess call snow white, and here is her saga: "
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

