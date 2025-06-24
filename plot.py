import os
import re
import matplotlib.pyplot as plt

log_path = os.path.join("logs", "log.txt")
report_dir = "report"
os.makedirs(report_dir, exist_ok=True)

# Data containers
steps = []
train_losses = []
val_steps = []
val_losses = []
perplexities = []
lrs = []

# Regex patterns
step_pattern = re.compile(r"step (\d+)")
train_loss_pattern = re.compile(r"train_loss: ([\d\.]+)")
val_loss_pattern = re.compile(r"val_loss: ([\d\.]+)")
perplexity_pattern = re.compile(r"perplexity: ([\d\.]+)")
lr_pattern = re.compile(r"lr: ([\d\.e\-]+)")

with open(log_path, "r") as f:
    for line in f:
        step_match = step_pattern.search(line)
        if step_match:
            step = int(step_match.group(1))

            train_loss_match = train_loss_pattern.search(line)
            lr_match = lr_pattern.search(line)

            if train_loss_match and lr_match:
                train_loss = float(train_loss_match.group(1))
                lr = float(lr_match.group(1))

                steps.append(step)
                train_losses.append(train_loss)
                lrs.append(lr)

            # Only present in eval steps
            val_loss_match = val_loss_pattern.search(line)
            perplexity_match = perplexity_pattern.search(line)

            if val_loss_match and perplexity_match:
                val_loss = float(val_loss_match.group(1))
                perplexity = float(perplexity_match.group(1))

                val_steps.append(step)
                val_losses.append(val_loss)
                perplexities.append(perplexity)

# ----------- Plot 1: val_loss vs step ----------- 
plt.figure()
plt.plot(val_steps, val_losses, marker='o')
plt.title("Validation Loss vs Steps")
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.grid()
plt.savefig(os.path.join(report_dir, "val_loss.png"))

# ----------- Plot 2: perplexity vs step -----------
plt.figure()
plt.plot(val_steps, perplexities, marker='o')
plt.title("Perplexity vs Steps")
plt.xlabel("Step")
plt.ylabel("Perplexity")
plt.yscale('log')
plt.ylim(top=100.0)
plt.grid()
plt.savefig(os.path.join(report_dir, "perplexity.png"))

# ----------- Plot 3: learning rate vs step -----------
plt.figure()
plt.plot(steps, lrs)
plt.title("Learning Rate vs Steps")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid()
plt.savefig(os.path.join(report_dir, "learning_rate.png"))

# ----------- Plot 4: train_loss + val_loss + baseline + checkpoint markers -----------

# Specify model size
sz = "124M"
loss_baseline = {
    "124M": 3.2924,
}[sz]

plt.figure(figsize=(10, 6))

plt.plot(steps, train_losses, label="Training Loss", color="blue")

plt.plot(val_steps, val_losses, label="Validation Loss", color="orange", marker='o')

# Add horizontal baseline line
plt.axhline(y=loss_baseline, color="red", linestyle="--",
            label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")

# Add vertical checkpoint lines
checkpoint_interval = 2000
max_step = steps[-1]
for ckpt_step in range(checkpoint_interval, max_step + checkpoint_interval, checkpoint_interval):
    plt.axvline(x=ckpt_step, color="gray", linestyle=":", alpha=0.5)

plt.title("Training and Validation Loss vs Steps")
plt.xlabel("Steps")
plt.ylabel("Loss")
# plt.yscale('log')
plt.ylim(top=5.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "loss_vs_steps.png"))

print(f"Plots saved to {report_dir}/")
