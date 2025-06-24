import os
import tiktoken
import torch
import numpy as np
from model import GPT, GPTConfig
from torch.nn import functional as F

# Sampling function
# Supports temperature scaling, top-k, and top-p filtering
def sample_logits(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(1, indices, values)
        logits = mask

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))
        logits.scatter_(1, sorted_indices, sorted_logits)

    # Convert to probabilities and sample next tokens
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# Main generation function
# Accepts context and generates up to max_tokens by iterative sampling
def generate_text(model, tokenizer, context, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    tokens = tokenizer.encode(context)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(tokens)
            logits = logits[:, -1, :]
            next_token = sample_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if model generates end-of-text token
            if next_token.item() == tokenizer.eot_token:
                break

        # Truncate context window if exceeds model limit
        if tokens.size(1) > context_window:
            tokens = tokens[:, -context_window:]

    return tokenizer.decode(tokens[0].tolist())

# Load model
out_dir = "logs"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initializer model with architecture configuration
model = GPT(GPTConfig())

# Load model weights from saved checkpoint
ckpt_path = os.path.join(out_dir, 'model_18000.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.to(device)

tokenizer = tiktoken.get_encoding('gpt2')
context_window = 1024  # GPT2 context length

# Store full dialogue history to enable context continuation
history = []

# Main conversation loop
while True:
    user_input = input("Prompt > ")
    history.append(f"User: {user_input}")

    # Build context from recent history
    full_context = "\n".join(history)
    full_context_tokens = tokenizer.encode(full_context)

    # Truncate if context exceeds limit
    if len(full_context_tokens) > context_window - 100:
        # keep most recent tokens
        full_context_tokens = full_context_tokens[-(context_window - 100):]
        full_context = tokenizer.decode(full_context_tokens)

    # Generate response
    response = generate_text(model, tokenizer, full_context, temperature=0.9, top_k=50, max_new_tokens=100)
    # Extract only newly generated text
    generated_part = response[len(full_context):].strip()

    print(f"Model > {generated_part}")
    history.append(f"Model: {generated_part}")
