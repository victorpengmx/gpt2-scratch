import os
import time
import json

class Logger:
    def __init__(self, out_dir, is_master=True):
        self.is_master = is_master
        self.out_dir = out_dir
        self.log_file = os.path.join(out_dir, "log.txt")
        self.sample_dir = os.path.join(out_dir, "generations")

        if self.is_master:
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(self.sample_dir, exist_ok=True)
            # Clear previous log file
            with open(self.log_file, "w") as f:
                pass

    def log_train(self, step, train_loss, grad_norm, lr, dt_ms, tokens_per_sec):
        if not self.is_master:
            return
        
        log_str = (f"step {step} | train_loss: {train_loss:.6f} | "
                   f"grad_norm: {grad_norm:.4f} | lr: {lr:.4e} | "
                   f"dt: {dt_ms:.2f} ms | tok/s: {tokens_per_sec:.0f}")
        print(log_str)
        with open(self.log_file, "a") as f:
            f.write(f"train_loss: {train_loss:.6f} | grad_norm: {grad_norm:.4f} | lr: {lr:.4e} | tok/s: {tokens_per_sec:.0f}\n")

    def log_val(self, step, val_loss, perplexity):
        if not self.is_master:
            return

        log_str = (f"step {step} | val_loss: {val_loss:.4f} | perplexity: {perplexity:.4f}")
        print(log_str)
        with open(self.log_file, "a") as f:
            f.write(f"val_loss: {val_loss:.4f} | perplexity: {perplexity:.4f} | ")

    def log_sample(self, step, rank, i, prompt, decoded_text):
        if not self.is_master:
            return

        print(f"step {step} | rank {rank} sample {i}: {decoded_text}")

        # Save to file for each step
        out_path = os.path.join(self.sample_dir, f"step_{step}.jsonl")
        os.makedirs(self.sample_dir, exist_ok=True)
        record = {
            "sample_id": i,
            "rank": rank,
            "prompt": prompt,
            "generated": decoded_text
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_checkpoint(self, checkpoint_path):
        if not self.is_master:
            return
        print(f"Saving checkpoint to {checkpoint_path}")

    def log_start_step(self, step):
        if not self.is_master:
            return
        with open(self.log_file, "a") as f:
            f.write(f"step {step} | ")

    def log_message(self, msg):
        if not self.is_master:
            return
        print(msg)
