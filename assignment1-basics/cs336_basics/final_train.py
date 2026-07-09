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
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344
rope_theta = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

SEED = 42
BEST_LR = 3e-3   
MAX_STEPS = 5000  

set_seed(SEED)

model = TransformerLM(vocab_size, CONTEXT_LENGTH, d_model, num_layers, num_heads, d_ff, rope_theta)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=BEST_LR)

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_final_lr{BEST_LR}"
save_path = os.path.join(BASE_DIR, "checkpoints", f"run_{run_id}")

config = {
    "vocab_size": vocab_size, "context_length": CONTEXT_LENGTH, "d_model": d_model,
    "num_layers": num_layers, "num_heads": num_heads, "d_ff": d_ff, "rope_theta": rope_theta,
    "batch_size": 128, "test_batch_size": 80, "max_steps": MAX_STEPS,
    "log_interval": 100, "eval_interval": 100, "save_interval": 500,
    "device": DEVICE, "run_id": run_id,
    "lr": BEST_LR,                          
    "min_lr": BEST_LR * 0.1,                
    "warmup_steps": int(0.05 * MAX_STEPS),  
    "seed": SEED,
}

print(f"\n===== formal training run | lr={BEST_LR} =====")
train(
    model, optimizer,
    train_path=train_bin_path, val_path=valid_bin_path,
    max_steps=config["max_steps"], log_interval=config["log_interval"],
    eval_interval=config["eval_interval"], save_interval=config["save_interval"],
    save_path=save_path, config=config,
    context_length=CONTEXT_LENGTH, device=DEVICE,
)


print("\n===== training done, plotting loss curve =====")

metrics_path = os.path.join(save_path, "metrics.jsonl")
records = [json.loads(line) for line in open(metrics_path)]
df = pd.DataFrame(records)

train_df = df.dropna(subset=["train_loss"])
val_df = df.dropna(subset=["val_loss"])

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_df["step"], train_df["train_loss"], label="train loss", color="tab:blue")
ax.plot(val_df["step"], val_df["val_loss"], label="val loss", color="tab:orange")

ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title(f"Training Loss Curve (lr={BEST_LR})")
ax.legend()

config_text = "\n".join([f"{k}: {v}" for k, v in config.items() if k not in ["run_id", "device"]])
ax.text(
    1.02, 0.5, config_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"),
)

plt.tight_layout()

plot_path = os.path.join(save_path, "loss_curve.png")
plt.savefig(plot_path, bbox_inches="tight")
print(f"loss curve saved to {plot_path}")
plt.show()

final_val_loss = val_df["val_loss"].iloc[-1]
print(f"\nfinal val loss: {final_val_loss:.4f}")