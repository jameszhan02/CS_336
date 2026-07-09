import sys, os, datetime, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import AdamW, TransformerLM
import torch
import pandas as pd
import matplotlib.pyplot as plt

from train_util import set_seed, train   

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txt.bin")
valid_bin_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-valid.txt.bin")

vocab_size = 10000
CONTEXT_LENGTH = 256
d_model = 256
num_layers = 4
num_heads = 4
d_ff = 672
rope_theta = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

SEED = 42
LR_CANDIDATES = [1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2]

run_dirs = []  

for lr in LR_CANDIDATES:
    set_seed(SEED)

    model = TransformerLM(vocab_size, CONTEXT_LENGTH, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_lr{lr}"
    save_path = os.path.join(BASE_DIR, "checkpoints", f"run_{run_id}")
    run_dirs.append((lr, save_path))  

    # batch size limit roughly 120 ? 
    config = {
        "vocab_size": vocab_size, "context_length": CONTEXT_LENGTH, "d_model": d_model,
        "num_layers": num_layers, "num_heads": num_heads, "d_ff": d_ff, "rope_theta": rope_theta,
        "batch_size": 128, "test_batch_size": 80, "max_steps": 2000,
        "log_interval": 50, "eval_interval": 50, "save_interval": 500,
        "device": DEVICE, "run_id": run_id, "lr": lr, "seed": SEED,
    }

    print(f"\n===== sweeping lr = {lr} =====")
    train(
        model, optimizer,
        train_path=train_bin_path, val_path=valid_bin_path,
        max_steps=config["max_steps"], log_interval=config["log_interval"],
        eval_interval=config["eval_interval"], save_interval=config["save_interval"],
        save_path=save_path, config=config,
        context_length=CONTEXT_LENGTH, device=DEVICE,
    )


print("\n===== sweep done, plotting comparison =====")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr, save_path in run_dirs:
    metrics_path = os.path.join(save_path, "metrics.jsonl")
    records = [json.loads(line) for line in open(metrics_path)]
    df = pd.DataFrame(records)

    train_df = df.dropna(subset=["train_loss"])
    val_df = df.dropna(subset=["val_loss"])

    axes[0].plot(train_df["step"], train_df["train_loss"], label=f"lr={lr}")
    axes[1].plot(val_df["step"], val_df["val_loss"], label=f"lr={lr}")

axes[0].set_title("Train Loss")
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss")
axes[0].legend()

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("step")
axes[1].set_ylabel("loss")
axes[1].legend()

plt.tight_layout()

plot_path = os.path.join(BASE_DIR, "checkpoints", f"lr_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(plot_path)
print(f"comparison plot saved to {plot_path}")
plt.show()