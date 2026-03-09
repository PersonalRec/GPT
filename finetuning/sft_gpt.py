"""
SFT post-training of the GPT 128M on the Databricks-Dolly-15K Dataset

Usage:
    1. Download the dataset:   python prepare_dolly.py
    2. Single GPU:             python sft_gpt.py
    2. Multi-GPU:              torchrun --standalone --nproc_per_node=2 sft_gpt.py
    3. Tensorboard (locally):  tensorboard --logdir log/tensorboard
"""
import os
import sys

import torch, math
from torch.nn import functional as F
import time
import csv
import tiktoken
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)


from gpt_model import GPT, GPTConfig

# ========================================= Define model generation prompts  ==============================================================

generation_prompts = [
    'What is best Playstation or Xbox',
    'How do I start running?',
    'What is the best tv series in the world',
    'Which of these are rappers? Eminem, Michael Jackson, Rihanna, 50 Cent',
    'Which products apple sell?',
    'What is investment banking?'
]

# ========================================= SFT Post-Training parameters ==================================================================

torch.set_float32_matmul_precision('high')

# Paths (resolved relative to this script's location, so it works from any cwd)
checkpoint_path = os.path.join(SCRIPT_DIR, "..", "results", "200226", "model_39999.pt")
data_root = os.path.join(SCRIPT_DIR, 'data/dolly_15k')

use_compile = True # Using of torch.compile() to speed up the training process

# Gradient accumulation parameters
B = 8  # micro-batch size
T = 1024 # sequence of length (context window size) for GPT-2, 2048 for GPT-3
gradient_accum_steps = 16
total_batch_size = B * T * gradient_accum_steps 


max_lr = 2e-5 
min_lr = max_lr * 0.1
warmup_steps = 40
constant_lr = False  # if True, use a fixed max_lr throughout training (no warmup, no decay)

epochs = 5
steps_per_epoch = 8692 // (B * gradient_accum_steps)
max_steps = steps_per_epoch * epochs
# print(f"steps_per_epoch = {steps_per_epoch}")
# sys.exit()

weight_decay = 0.1 

eval_steps = 10            # Evaluate every N steps
checkpointer_steps = steps_per_epoch  # Save every epoch
generate_steps = 10       # Generate sample text every N steps

# Set the torch seed parameter
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# ========================================= Data Loader ==================================================================
# Improved DataLoaderLite with proper shuffling for multi-epoch training

class SFTLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root='data/dolly_15k', seed=1337):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        
        # Initialize random number generator for shuffling
        self.rng = np.random.default_rng(seed)
        self.base_seed = seed
        
        # Get token/mask shard filename pairs written by prepare_dolly.py
        token_files = sorted(
            f for f in os.listdir(data_root)
            if f.startswith(f"{split}_tokens_") and f.endswith(".npy")
        )
        mask_files = sorted(
            f for f in os.listdir(data_root)
            if f.startswith(f"{split}_masks_") and f.endswith(".npy")
        )
        assert len(token_files) > 0, f"no token shards found for split {split}"
        assert len(token_files) == len(mask_files), f"token/mask shard count mismatch for split {split}"
        if master_process:
            print(f"found {len(token_files)} token/mask shard pairs for split {split}")
        
        # Memory-map token and mask shards separately; each row is one SFT example of length T+1
        self.token_shards = [np.load(os.path.join(data_root, f), mmap_mode='r') for f in token_files]
        self.mask_shards = [np.load(os.path.join(data_root, f), mmap_mode='r') for f in mask_files]
        self.shard_lengths = [shard.shape[0] for shard in self.token_shards]

        for token_shard, mask_shard in zip(self.token_shards, self.mask_shards):
            assert token_shard.shape == mask_shard.shape, "token and mask shard shapes must match"
            assert token_shard.shape[1] == self.T + 1, f"expected examples of width {self.T + 1}"
        
        # Build global example index and initialize pointer
        self._build_index()
        self.ptr = 0
    
    def _build_index(self):
        """
        Build a global list of all (shard_id, row_id) examples across all shards.
        Each row already stores one full training example of length T+1.
        """
        all_indices = []
        for shard_id, num_rows in enumerate(self.shard_lengths):
            if num_rows <= 0:
                continue

            row_ids = np.arange(num_rows, dtype=np.int64)
            shard_ids = np.full_like(row_ids, shard_id, dtype=np.int64)
            pairs = np.stack([shard_ids, row_ids], axis=1)
            all_indices.append(pairs)
        
        all_indices = np.concatenate(all_indices, axis=0)
        
        if self.split == "train":
            # Shuffle globally so batches contain examples from different shards/rows
            self.rng.shuffle(all_indices)
            self.index = all_indices[self.process_rank::self.num_processes]
        else:
            self.index = all_indices[self.process_rank::self.num_processes]
        
        if master_process:
            print(f"{self.split}: {len(all_indices)} total examples, {len(self.index)} examples for this rank")
    
    def __len__(self):
        """Number of full batches available in the dataset for this rank."""
        return len(self.index) // self.B
    
    def next_batch(self):
        B = self.B
        if self.ptr + B > len(self.index):
            self.ptr = 0
            if self.split == "train":
                self._build_index()
        rows = self.index[self.ptr : self.ptr + B]
        self.ptr += B
        xs, ys, masks = [], [], []
        for shard_id, row_id in rows:
            tokens = self.token_shards[shard_id][row_id].astype(np.int64)   # (T+1,)
            mask = self.mask_shards[shard_id][row_id].astype(np.float32)    # (T+1,)
            tokens = torch.from_numpy(tokens)
            mask = torch.from_numpy(mask)
            x = tokens[:-1]         # (T,)
            y = tokens[1:]          # (T,)
            loss_mask = mask[1:]    # (T,)
            xs.append(x)
            ys.append(y)
            masks.append(loss_mask)
        x = torch.stack(xs)         # (B, T)
        y = torch.stack(ys)         # (B, T)
        loss_mask = torch.stack(masks)  # (B, T)
        return x, y, loss_mask

    
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

# ================================ SFT masked loss calculation =========================================================

def compute_masked_loss(logits, targets, loss_mask):
    # logits: (B, T, vocab_size)
    # targets: (B, T)
    # loss_mask: (B, T), 1 for assistant tokens, 0 otherwise

    token_losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)), # (B, T, vocab_size) --> (B*T, vocab_size), all batch and time positions are flattened into one long list of predictions
        targets.view(-1), # (B, T) --> (B*T)
        reduction='none', # return one loss value per token position, we get a list of losses for each token [2.1, 1.7, 3.2, 0.7, etc.]
    ).view_as(targets) # reshape (B*T) --> (B, T)

    loss_mask = loss_mask.to(token_losses.dtype) # Cast the mask to the same dtype

    masked_loss = (token_losses * loss_mask).sum() # Apply the mask to prompt tokens losses (zero them out) and sum the other losses to one value
    num_active = loss_mask.sum().clamp_min(1.0) # get number of predicted tokens, prevent division by zero

    loss = masked_loss / num_active # Compute final mean masked loss

    return loss


def compute_masked_loss_stats(logits, targets, loss_mask):
    # Returns the summed masked loss and number of active target tokens
    # so validation can average over exact token counts, not over batches.
    token_losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none',
    ).view_as(targets)

    loss_mask = loss_mask.to(token_losses.dtype)
    masked_loss_sum = (token_losses * loss_mask).sum()
    num_active = loss_mask.sum().clamp_min(1.0)
    return masked_loss_sum, num_active

# ================================ DDP Training Settings ==============================================================

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set DDP (distributed data parallel)
# torchrun command sets the env variable RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get("RANK", -1)) != -1 # Checks if environment variable RANK exists, if it exists → DDP mode (multi-GPU), if not → Single GPU/CPU mode

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available()
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

# ========================================= Assert training parameters ==============================================================

assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"\nPost-training batch config:")
    print(f"  micro-batch size B={B}, sequence length T={T}")
    print(f"  total batch size: {total_batch_size:,} tokens")
    print(f"  gradient accumulation steps: {grad_accum_steps}")

# ========================================= Data Loaders =====================================================================

train_loader = SFTLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
                               split='train', data_root=data_root, seed=42)
val_loader = SFTLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
                             split='val', data_root=data_root, seed=42)


# ========================================= Load Pre-trained Checkpoint =======================================================

if master_process:
    print(f"\nLoading pre-trained checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the saved config and rebuild the model
saved_config = checkpoint['config']
if master_process:
    print(f"Model config: {saved_config}")
    print(f"Pre-training step: {checkpoint['step']}")
    print(f"Pre-training val loss: {checkpoint['val_loss']:.4f}")

# Create model from saved config and load weights
model = GPT(saved_config)
# Strip '_orig_mod.' prefix if checkpoint was saved from a torch.compile()-wrapped model
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'

if any(k.startswith(unwanted_prefix) for k in state_dict):
    state_dict = {k.removeprefix(unwanted_prefix): v for k, v in state_dict.items()}
result = model.load_state_dict(state_dict, strict=False)

if master_process:
    if result.missing_keys:
        print(f"  Note: missing keys (will use init weights): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Note: unexpected keys (ignored): {result.unexpected_keys}")
model.to(device)

if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")


if use_compile and device_type == 'cuda':
    model = torch.compile(model)

# Wrap with DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# ================================================ Learning rate Scheduler =================================================

def get_lr(it):
    if constant_lr:
        return max_lr
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

# ======================================= Configure the optimizer and a logger ===============================================

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, verbose=master_process)

log_dir = os.path.join(SCRIPT_DIR, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.csv")

csv_columns = ["step", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec", "dt_ms"]
if master_process:
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

# TensorBoard writer (master process only)
if master_process:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

# ========================================= Text Generation Helper ============================================================

enc = tiktoken.get_encoding('gpt2')

def generate_text(model, prompt, max_new_tokens=128, top_k=50, device=device):
    """Generate text from a prompt using top-k sampling. Stops at <|endoftext|>."""
    model.eval()
    eot = enc._special_tokens['<|endoftext|>']
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = tokens if tokens.size(1) <= raw_model.config.block_size else tokens[:, -raw_model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            if xcol.item() == eot:
                break
            tokens = torch.cat((tokens, xcol), dim=1)

    return enc.decode(tokens[0].tolist())

# ======================================= Optimization loop =================================================================


for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    step_val_loss = None

    # Once in a while evaluate our validation loss and benchmarks
    if step % eval_steps == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_sum = torch.zeros((), device=device)
            val_token_count = torch.zeros((), device=device)
            val_loss_steps = max(1, len(val_loader))  # use all available val batches (dataset is small)
            for _ in range(val_loss_steps):
                x, y, loss_mask = val_loader.next_batch()
                x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(x)
                    batch_loss_sum, batch_token_count = compute_masked_loss_stats(logits, y, loss_mask)
                val_loss_sum += batch_loss_sum.detach()
                val_token_count += batch_token_count.detach()
        if ddp:
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        
        step_val_loss = (val_loss_sum / val_token_count).item()

        # write logs and save model checkpoints
        if master_process:
            print(f"validation loss: {step_val_loss:.4f}")
            tb_writer.add_scalar("loss/val", step_val_loss, step)
            if step > 0 and (step % checkpointer_steps == 0 or last_step):
                # write model checkpoints (includes optimizer state for resuming post-training)
                ckpt_out_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                ckpt_out = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': step_val_loss,
                    'base_checkpoint': checkpoint_path,  # reference to original pre-trained model
                    'rng_state': {
                        'python': torch.random.get_rng_state(),
                        'numpy': np.random.get_state(),
                    },
                }
                if device_type == "cuda":
                    ckpt_out['rng_state']['cuda'] = torch.cuda.get_rng_state(device)
                torch.save(ckpt_out, ckpt_out_path)
        
    # Training loop

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    # Gradient accumulation for around 0.5M tokens
    for micro_step in range(grad_accum_steps):
        x, y, loss_mask = train_loader.next_batch()
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # Cast our dtype to bfloat16
            logits, _ = model(x)
            loss = compute_masked_loss(logits, y, loss_mask)
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

    # Write/show the statistics
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        # Write a single CSV row with all metrics for this step
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                f"{loss_accum.item():.6f}",
                f"{step_val_loss:.4f}" if step_val_loss is not None else "",
                f"{lr:.6e}",
                f"{norm:.4f}",
                f"{tokens_per_sec:.2f}",
                f"{dt*1000:.2f}",
            ])
        tb_writer.add_scalar("loss/train", loss_accum.item(), step)
        tb_writer.add_scalar("training/lr", lr, step)
        tb_writer.add_scalar("training/grad_norm", norm, step)
        tb_writer.add_scalar("training/tokens_per_sec", tokens_per_sec, step)
    
    if master_process and (step % generate_steps == 0 or last_step):
        prompts = generation_prompts if last_step else [generation_prompts[step % len(generation_prompts)]]
        for prompt in prompts:
            generated = generate_text(model, prompt, max_new_tokens=64, device=device)
            print(f"  [gen] \"{prompt}\" → {generated}")


# ========================================= Cleanup ===========================================================================

# Close TensorBoard writer
if master_process:
    tb_writer.close()


if ddp:
    destroy_process_group()
