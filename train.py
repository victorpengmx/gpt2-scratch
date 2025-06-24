import os
import time
import tiktoken
import torch
import math
import torch.distributed as dist
import numpy as np

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from log import Logger
from model import GPT, GPTConfig

prompts = [
    "The history of AI begins with ",
    "Explain how photosynthesis works.",
    "Once upon a time, there was a dragon who ",
    "Hello, I'm a language model, ",
    "A conversation between Alice and Bob:"
]

# =====================================================
# Hyperparameters and Config
# =====================================================
data_root = "datasets/edu_fineweb10B"
out_dir = "logs"
init_from = "scratch" # "scratch" to train from scratch, "resume" to resume from checkpoint
ckpt_to_load_from = "model_00200.pt"
seed = 1337
eval_interval = 500
total_batch_size = 2**19 # ~ 0.5m tokens
B = 32 # mini batch size
T = 1024 # sequence len
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
mixed_precision_dtype = torch.bfloat16

# =====================================================
# Data Loader
# =====================================================
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# DDP launch for 2 GPUs
# torchrun --standalone --nproc_per_node=2 train_ddp.py
# =====================================================
# DDP setup
# =====================================================
# torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "need CUDA for ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

# =====================================================
# Set random seeds for reproducibility
# =====================================================
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision('high')

# =====================================================
# Prepare data loaders
# =====================================================
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

enc = tiktoken.get_encoding('gpt2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================
# Learning rate scheduler (cosine decay with warmup)
# =====================================================
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    elif it > max_steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff =  0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

# =====================================================
# Model setup
# =====================================================
logger = Logger(out_dir=out_dir, is_master=master_process)
model = GPT(GPTConfig())
step = 0
total_train_time = 0

if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
elif init_from == "resume":
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, ckpt_to_load_from)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    step = checkpoint['step'] + 1
    total_train_time += checkpoint['train_wall_time']

model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
model = torch.compile(model)
raw_model = model.module if DDP else model

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint['optimizer'])

# =====================================================
# Training Loop
# =====================================================
while True:
    t0 = time.time()
    logger.log_start_step(step)

    # Evaluate
    if step % eval_interval == 0 or step == max_steps-1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            perplexity = torch.exp(val_loss_accum)
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            logger.log_val(step, val_loss=val_loss_accum.item(), perplexity=perplexity.item())
            
            # write model checkpoints
            if step > 0 and (step % (eval_interval * 4) == 0 or step == max_steps-1):
                checkpoint_path = os.path.join(out_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'train_wall_time': total_train_time
                }
                logger.log_checkpoint(checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

    # once in a while generate from the model
    if (step > 0 and (step % eval_interval == 0) or step == max_steps-1):
        model.eval()
        num_return_sequences = 1  # 1 sample per prompt
        max_length = 64

        for prompt in prompts:
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)

            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                logger.log_sample(step, ddp_rank, i, prompt, decoded)

    # Training step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=mixed_precision_dtype): # automatic mixed precision
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()

    # metrics
    t1 = time.time()
    total_train_time += (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    logger.log_train(step, train_loss=loss_accum.item(), grad_norm=norm, lr=lr,
                dt_ms=(t1-t0)*1000, tokens_per_sec=tokens_per_sec)

    step += 1
    if step > max_steps:
        logger.log_message("Reached max iterations, stopping training.")
        break

if ddp:
    destroy_process_group()