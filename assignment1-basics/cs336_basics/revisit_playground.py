import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests')) # so current file could visit other files under ../tests folder
from adapters import run_train_bpe
import pickle
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("|BASE URL : ", BASE_DIR , "|")

BPE_TRAIN_FILE_NAME = "TinyStoriesV2-GPT4-valid.txt"
train_input_path = os.path.join(BASE_DIR, "data", BPE_TRAIN_FILE_NAME)
valid_input_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt")

# train_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txt.bin")
# valid_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt.bin")


t0 = time.perf_counter()
vocab_size = 10000
saving_path = BPE_TRAIN_FILE_NAME + "vocab.pkl"
vocab_path = os.path.join(BASE_DIR, "data", saving_path)
if not os.path.exists(vocab_path):
    print("step 1: starting BPE training...")
    vocab, merges = run_train_bpe(
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


longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

print("longest token id:", longest_id)
print("longest token byte length:", len(longest_token))
print("longest token bytes:", longest_token)
## END of pre-tokenizer 
t1 = time.perf_counter()
print(f"BPE total time spent: {t1 - t0:.4f} seconds") # TODO: python native implementation seems like could not do perfect at this task, this implement save later to complete with "Tiny CC" task.