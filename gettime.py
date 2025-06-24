import os
import torch

# Directory where your checkpoint files are stored
log_dir = 'logs'

# Collect all .pt files
checkpoint_files = [f for f in os.listdir(log_dir) if f.endswith('.pt')]
checkpoint_files.sort()  # optional: sort for consistent ordering

# Extract train_wall_time from each checkpoint
results = []

for filename in checkpoint_files:
    path = os.path.join(log_dir, filename)
    checkpoint = torch.load(path, map_location='cuda', weights_only=False)
    step = checkpoint.get('step', 'N/A')
    train_wall_time = checkpoint.get('train_wall_time', 'N/A')
    results.append((step, train_wall_time))

# Print results
for step, train_wall_time in results:
    print(f"Step: {step}, Train Wall Time: {train_wall_time:.2f} sec")
