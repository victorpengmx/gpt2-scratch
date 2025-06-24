import os
import requests
import tiktoken
import numpy as np

# Configuration
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_dir = "datasets/tinyshakespeare"
train_fraction = 0.9
encoding = "gpt2"

os.makedirs(data_dir, exist_ok=True)
input_file = os.path.join(data_dir, "input.txt")

# Download the dataset
if not os.path.exists(input_file):
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved to {input_file}")
else:
    print(f"{input_file} already exists. Skipping download.")

# Read raw text
with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

n = len(data)
train_data = data[:int(n * train_fraction)]
val_data = data[int(n * train_fraction):]

# Encode with GPT-2 tokenizer
enc = tiktoken.get_encoding(encoding)
train_ids = np.array(enc.encode(train_data), dtype=np.uint16)
val_ids = np.array(enc.encode(val_data), dtype=np.uint16)

# Save encoded binary files
train_bin_path = os.path.join(data_dir, "train", "train.bin")
val_bin_path = os.path.join(data_dir, "val", "val.bin")
os.makedirs(os.path.dirname(train_bin_path), exist_ok=True)
os.makedirs(os.path.dirname(val_bin_path), exist_ok=True)

train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)

# Save metadata
meta = {
    "vocab_size": enc.n_vocab,
    "encoding": encoding
}
import pickle
with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Train tokens: {len(train_ids)} saved to {train_bin_path}")
print(f"Val tokens:   {len(val_ids)} saved to {val_bin_path}")
