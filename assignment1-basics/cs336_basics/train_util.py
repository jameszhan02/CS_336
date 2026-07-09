# train_utils.py
import os
import json
import time
import random
import datetime
import numpy as np
import torch
import math
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import run_get_batch, run_cross_entropy


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

def get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step > max_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + cosine_decay * (max_lr - min_lr)
def train(model, optimizer, train_path, val_path, max_steps, log_interval,
          eval_interval, save_interval, save_path, config, context_length, device):
    os.makedirs(save_path, exist_ok=True)
    BATCH_SIZE = config["batch_size"]
    TEST_BATCH_SIZE = config["test_batch_size"]

    # cos lr sechdueler
    max_lr = config["lr"]
    min_lr = config.get("min_lr", max_lr * 0.1)  
    warmup_steps = config.get("warmup_steps", int(0.05 * max_steps))  

    logger = TrainLogger(log_dir=save_path, config=config)

    train_data = np.memmap(train_path, dtype=np.int32, mode="r")
    val_data = np.memmap(val_path, dtype=np.int32, mode="r")

    for step in range(max_steps):
        model.train()
        current_lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups: # change lr during steps
            param_group['lr'] = current_lr

        train_x, train_y = run_get_batch(train_data, BATCH_SIZE, context_length, device)
        logits = model(train_x)
        batch, seq_len, vocab = logits.shape
        loss = run_cross_entropy(
            logits.reshape(batch * seq_len, vocab),
            train_y.reshape(batch * seq_len)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            train_loss_val = loss.item()
            print(f"step {step} | train loss: {train_loss_val:.4f}")
            logger.log(step=step, train_loss=train_loss_val)

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = run_get_batch(val_data, TEST_BATCH_SIZE, context_length, device)
                val_logits = model(val_x)
                batch, seq_len, vocab = val_logits.shape
                val_loss = run_cross_entropy(
                    val_logits.reshape(batch * seq_len, vocab),
                    val_y.reshape(batch * seq_len)
                )
            val_loss_val = val_loss.item()
            print(f"step {step} | val loss: {val_loss_val:.4f}")
            logger.log(step=step, val_loss=val_loss_val)
            model.train()

        if step % save_interval == 0:
            file_name = f"checkpoint_step{step}.pt"
            final_out_path = os.path.join(save_path, file_name)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "config": config,
            }, final_out_path)
            print(f"step {step} | checkpoint saved to {final_out_path}")

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