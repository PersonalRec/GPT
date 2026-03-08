#GPT2 128M Long version

from dataclasses import dataclass
import torch, math
import torch.nn as nn
from torch.nn import functional as F
import time
import inspect
import os
import csv
from evals import hellaswag, mmlu, arc
import tiktoken
import numpy as np
from gpt_model import GPTConfig, GPT


# ========================================= Training parameters ==================================================================

torch.set_float32_matmul_precision('high')

use_compile = True # Using of torch.compile() to speed up the training process
checkpointer_steps = 10000

# Evaluation benchmarks to run (set to False to skip)
run_hellaswag = True
run_mmlu = False
run_arc = False

# Gradient accumulation parameters
total_batch_size = 524288 # 2**19, ~0.5M in number of tokens
B = 16 # Fits on RTX 5090 with 32 GB VRAM
T = 1024 # sequence of length (context window size) for GPT-2, 2048 for GPT-3

max_lr = 6e-4 * 4
min_lr = max_lr * 0.1
warmup_steps = 715 // 2 # 375e6 / 2**19 (warmup during 375M tokens) / (0.5M tokens in a batch) = 715 steps
max_steps = 20000 * 2 # 10 billion tokens, 10e10 / 0.5e6 (0.5M tokens in a batch) = 20,000 steps per epoch. For the new 100BT dataset we multiply steps by 10

weight_decay = 0.1
eval_steps = 500 # evaluate the model every 250 steps

# Set the torch seed parameter
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# per-GPU: B * grad_accum_steps sequences per step, 3x for fwd+bwd
gpu_peak_flops = 209.5e12  # RTX 5090 bf16 tensor core TFLOPS

# T4: 65 TFLOPS, No BF16 tensor support — use dtype=torch.float16 instead
# RTX 3090 BF16:  142e12 (142 TFLOPS)
# RTX 5090 BF16:  209.5e12 (209.5 TFLOPS)

# ========================================= Data Loader ==================================================================
# Improved DataLoaderLite with proper shuffling for multi-epoch training

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root='edu_fineweb', seed=1337):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        
        # Initialize random number generator for shuffling
        self.rng = np.random.default_rng(seed)
        self.base_seed = seed
        
        # Get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        # Memory-map all shards for efficient access (doesn't load into RAM)
        self.mmap_shards = [np.load(f, mmap_mode='r') for f in self.shards]
        self.shard_lengths = [m.shape[0] for m in self.mmap_shards]
        
        # Build global window index and initialize pointer
        self._build_index()
        self.ptr = 0
    
    def _build_index(self):
        """
        Build a global list of all (shard_id, start_offset) windows across all shards.
        Each window represents one valid training example of length T+1 (T inputs + 1 target).
        """
        all_indices = []
        for shard_id, length in enumerate(self.shard_lengths):
            # Calculate number of non-overlapping windows in this shard
            # We need T+1 tokens per window (T for input, 1 for final target)
            num_windows = (length - 1) // self.T  # -1 because we need one extra token for targets
            if num_windows <= 0:
                continue
            
            # Create array of starting positions for each window
            starts = (np.arange(num_windows) * self.T).astype(np.int64)
            shard_ids = np.full_like(starts, shard_id, dtype=np.int64)
            pairs = np.stack([shard_ids, starts], axis=1)
            all_indices.append(pairs)
        
        all_indices = np.concatenate(all_indices, axis=0)
        
        if self.split == "train":
            # Shuffle globally so batches contain windows from different shards/positions
            self.rng.shuffle(all_indices)
            # Split across DDP ranks: each GPU gets every num_processes-th window
            # This ensures each GPU sees unique, non-overlapping data
            self.index = all_indices[self.process_rank::self.num_processes]
        else:
            # Validation: no shuffle, deterministic across all ranks
            # Each rank still processes its own slice for parallel evaluation
            self.index = all_indices[self.process_rank::self.num_processes]
        
        if master_process:
            print(f"{self.split}: {len(all_indices)} total windows, {len(self.index)} windows for this rank")
    
    def __len__(self):
        """Number of full batches available in the dataset for this rank."""
        return len(self.index) // self.B
    
    def next_batch(self):
        """
        Return one batch of shape (B, T) for inputs x and targets y.
        Reads from memory-mapped shards for efficiency.
        """
        B, T = self.B, self.T
        
        # Get B windows from the shuffled index
        if self.ptr + B > len(self.index):
            # If we don't have enough windows left, wrap around (start new epoch)
            self.ptr = 0
            if self.split == "train":
                # Reshuffle for the new epoch
                self._build_index()
        
        rows = self.index[self.ptr : self.ptr + B]
        self.ptr += B
        
        # Gather tokens from memory-mapped shards
        xs, ys = [], []
        for shard_id, start in rows:
            # Read T+1 tokens: T for input, last one shifts to create target
            tokens = self.mmap_shards[shard_id][start : start + T + 1].astype(np.int64)
            tokens = torch.from_numpy(tokens)
            xs.append(tokens[:-1])  # Input: first T tokens
            ys.append(tokens[1:])   # Target: last T tokens (shifted by 1)
        
        x = torch.stack(xs)
        y = torch.stack(ys)
        return x, y
    
    def reset(self, new_seed=None):
        """
        Reset the data loader to the beginning.
        Optionally provide a new seed to get different shuffling (for new epochs).
        """
        if new_seed is not None:
            self.rng = np.random.default_rng(new_seed)
        elif self.split == "train":
            # Auto-increment seed for different shuffling each reset
            self.base_seed += 1
            self.rng = np.random.default_rng(self.base_seed)
        
        self._build_index()
        self.ptr = 0

# ================================ DDP Training Settings ==============================================================

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set DDP (distributed data parallel)
# torchrun command sets the env variable RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get("RANK", -1)) != -1 # Checks if environment variable RANK exists, if it exists → DDP mode (multi-GPU), if not → Single GPU/CPU mode

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl') # Sets up GPU-to-GPU communication, all GPUs can now sync gradients
    ddp_rank = int(os.environ['RANK']) # Global process ID
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # GPU ID on this machine
    ddp_world_size = int(os.environ['WORLD_SIZE']) # Total # of processes
    device = f'cuda:{ddp_local_rank}' # Each process → its own GPU
    torch.cuda.set_device(device) # # Pin this process to that GPU
    master_process = ddp_rank == 0 # Only RANK 0 (master) should print logs, save checkpoints, write logs, validate on test set

else:
    # vanilla, non-ddp
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect_device

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using the device: {device}")

device_type = 'cuda' if device.startswith('cuda') else 'cpu' # for autocast

# =========================================== Assert training parameters ====================================================


assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# ================================ Instantiate the data loader ==============================================================

data_root = "edu_fineweb"  # Directory containing the tokenized shards
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, 
                               split='train', data_root=data_root, seed=1337)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, 
                             split='val', data_root=data_root, seed=1337)

# ================================ Instantiate the model ====================================================================

model = GPT(GPTConfig(mlp_type="swiglu"))
model.to(device)

if use_compile:
    model = torch.compile(model)
# Wrap with DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# ================================================ Learning rate Scheduler =================================================

def get_lr(it):
    # 1) linear warmup for warmup_iters step
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# ================================================ MFU Calculator ==============================================================
# Simple "6N" approximation (Kaplan et al. / OpenAI scaling laws):
# Each token requires ~6N FLOPs for a full forward + backward pass (1x fwd + 2x bwd).
# MFU = (actual FLOPs) / (peak FLOPs of the GPU)

num_params = sum(p.numel() for p in raw_model.parameters())
flops_per_token = 6 * num_params  # forward + backward FLOPs per token
if master_process:
    print(f"model parameters: {num_params:,} | flops per token (6N): {flops_per_token:,}")

# ======================================= Configure the optimizer and a logger ===============================================

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, verbose=master_process)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.csv")
# Write CSV header
csv_columns = ["step", "train_loss", "val_loss", "hellaswag", "mmlu", "arc", "lr", "grad_norm", "tokens_per_sec", "mfu", "dt_ms"]
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_columns)

# TensorBoard writer (master process only)
if master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

# ======================================= Optimization loop =================================================================

enc = tiktoken.get_encoding('gpt2') # iniatilize the encoder for token generation

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Initialize per-step metrics (None = not computed this step)
    step_val_loss = None
    step_hellaswag = None
    step_mmlu = None
    step_arc = None

    # Once in a while evaluate our validation loss and benchmarks
    if step % eval_steps == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20 # accumulate gradients over 20 steps
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) # average the accumulated gradients across all gpus
        
        step_val_loss = val_loss_accum.item()

        # write logs and save model checkpoints
        if master_process:
            print(f"validation loss: {step_val_loss:.4f}")
            tb_writer.add_scalar("loss/val", step_val_loss, step)
            if step > 0 and (step % checkpointer_steps == 0 or last_step):
                # write model checkpoints (includes optimizer state for resuming pre-training)
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': step_val_loss,
                    'rng_state': {
                        'python': torch.random.get_rng_state(),
                        'numpy': np.random.get_state(),
                    },
                }
                if device_type == "cuda":
                    checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state(device)
                torch.save(checkpoint, checkpoint_path)
        
        # Evaluate benchmarks (each handles DDP internally and prints results on master)
        
        if run_hellaswag:
            step_hellaswag = hellaswag.evaluate(model, device, device_type, ddp, ddp_rank, ddp_world_size)
            if master_process:
                tb_writer.add_scalar("eval/hellaswag", step_hellaswag, step)

        if run_mmlu:
            step_mmlu = mmlu.evaluate(model, device, device_type, ddp, ddp_rank, ddp_world_size)
            if master_process:
                tb_writer.add_scalar("eval/mmlu", step_mmlu, step)

        if run_arc:
            step_arc = arc.evaluate(model, device, device_type, ddp, ddp_rank, ddp_world_size)
            if master_process:
                tb_writer.add_scalar("eval/arc", step_arc, step)

    # Training loop

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    # Gradient accumulation for around 0.5M tokens
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # get the next batch
        x, y = x.to(device), y.to(device) # move the tensors to the device
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # Cast our dtype to bfloat16
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize the loss
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # require_backward_grad_sync will only turn on on the last micro_step to sync the gradients
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the losses across all gpu processes

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # calculates the global norm of the parameters. Prevents too big gradient shock. Global gradient clipping.
   
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    # Calculate statistics
    t1 = time.time()
    dt = (t1 - t0) # times difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    mfu = (flops_per_token * tokens_processed) / (dt * gpu_peak_flops * ddp_world_size)

    # Write/show the statistics
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | mfu: {mfu:.2%}")
        # Write a single CSV row with all metrics for this step
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                f"{loss_accum.item():.6f}",
                f"{step_val_loss:.4f}" if step_val_loss is not None else "",
                f"{step_hellaswag:.4f}" if step_hellaswag is not None else "",
                f"{step_mmlu:.4f}" if step_mmlu is not None else "",
                f"{step_arc:.4f}" if step_arc is not None else "",
                f"{lr:.6e}",
                f"{norm:.4f}",
                f"{tokens_per_sec:.2f}",
                f"{mfu:.6f}",
                f"{dt*1000:.2f}",
            ])
        tb_writer.add_scalar("loss/train", loss_accum.item(), step)
        tb_writer.add_scalar("training/lr", lr, step)
        tb_writer.add_scalar("training/grad_norm", norm, step)
        tb_writer.add_scalar("training/tokens_per_sec", tokens_per_sec, step)
        tb_writer.add_scalar("training/mfu", mfu, step)
    
    # Generate from the model on the last step
    if last_step and master_process:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model, ")
        tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (4, 8)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)

        with torch.no_grad():
            while xgen.size(1) < max_length:
            # forward the model to get the logits
                # Pass the entire sequence so far through the model.
                # Get predictions for every position in the sequence.
                logits, loss = model(xgen) # (B, T, vocab_size)
                # Take the logits at the last token's position. The model predicts "what comes next" after the last token.
                logits = logits[:, -1, :] # (B, vocab_size)
                # Convert logits to the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default). Only consider the 50 most likely tokens.
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                # probs = [0.001, 0.35, 0.002, 0.30, 0.001, ...]  # 50,257 values
                # topk_probs = [0.35, 0.30, 0.15, ...]  # Top 50 highest
                # topk_indices = [1, 3, 42, ...]  # Their token IDs
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # Select a token from the top-k probabilities. Randomly pick one of the 50 tokens. Higher probability → more likely to be chosen.
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # Gather the corresponding indices. Get actual ID of the sampled token.
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # Append the new token id to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

        # Print the generated text
        for i in range(num_return_sequences): # Loops through each sequence in the batch (B sequences)
            tokens = xgen[i, :max_length].tolist() # Gets sequence i, first max_length tokens. Converts tensor to Python list.
            decoded = enc.decode(tokens) # Converts token IDs back to text.
            print(">", decoded)


# Close TensorBoard writer
if master_process:
    tb_writer.close()

# Destroy the process group after train end
if ddp:
    destroy_process_group()