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

@dataclass
class GPTConfig():
    block_size: int = 1024 # maximum sequence length (context length)
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 tokens + 1 <|endoftext|> token = 50257 which is inefficient in terms of cuda processing, new vocab size 50304 could be easily divided by a lot of numbers
    n_layer: int = 12 # number of layers (transformer blocks: each block has attention + MLP + RMSNorms)
    n_head: int = 12 # number of heads per transformer blocks. Each head sees 768 ÷ 12 = 64 dimensions. Different heads can learn different attention patterns.
    n_embd: int = 768 # embedding dimension

    # additional options
    use_rope: bool = True      # use RoPE embedding for training
    rope_base: float = 10000.0
    mlp_type: str = "swiglu"   # MLP activation type: "gelu" or "swiglu"


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
B = 32 # ~16 GB of memory, ideally maximize to B = 32 (for ~28 GB in 32GB RTX 5090) or B = 64 (in more modern architectures e.g. A100 40/80GB)
T = 1024 # sequence of length (context window size) for GPT-2, 2048 for GPT-3

max_lr = 6e-4 * 4
min_lr = max_lr * 0.1
warmup_steps = 715 // 2 # 375e6 / 2**19 (warmup during 375M tokens) / (0.5M tokens in a batch) = 715 steps
max_steps = 20000 * 10 # 10 billion tokens, 10e10 / 0.5e6 (0.5M tokens in a batch) = 20,000 steps per epoch. For the new 100BT dataset we multiply steps by 10

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

# ================================================================================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, max_position_embeddings=1024):
        """
        Args:
            dim: head dimension (e.g., 64 for a model with 512 embed and 8 heads)
            base: controls the frequency range (10000 is standard from the paper)
            max_position_embeddings: maximum sequence length to precompute
        """
        super().__init__()

        self.dim = dim

        # Compute inverse frequencies for each dimension pair
        # Higher frequencies = faster rotation = captures local patterns
        # Lower frequencies = slower rotation = captures global patterns
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) #  Shape: (dim/2,) e.g., (32,) for dim=64

        # Compute position × frequency matrix
        # For each position m and frequency f, compute: angle = m * f
        t = torch.arange(max_position_embeddings, dtype=torch.float) # Shape: (max_pos,) e.g., (2048,)
        
        # This gives us the rotation angle for: - each position (row), - each dimension pair (column)
        freqs = torch.einsum("i,j->ij", t, inv_freq) # freqs[m, i] = position_m * inv_freq_i, Shape: (max_pos, dim/2) e.g., (2048, 32)

        # Duplicate frequencies for the cos/sin trick
        emb = torch.cat((freqs, freqs), dim=-1) # Shape: (max_pos, dim) e.g., (2048, 64)

        # Precompute and cache cos/sin values
        # Both shapes: (1, 1, max_pos, dim)
        # Will broadcast to (B, n_head, T, dim)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2] # First half
        x2 = x[..., x.shape[-1] // 2 :] # Second half
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        """
        Apply RoPE to q and k.

        Args:
            q, k: (B, n_head, T, head_dim)

        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        if seq_len is None:
            seq_len = q.size(-2)
        # It is critical to match the dtype/device of q. Otherwise tensors
        # become fp32 and Flash Attention may be disabled.
        cos = self.cos_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        q2 = (q * cos) + (self._rotate_half(q) * sin)
        k2 = (k * cos) + (self._rotate_half(k) * sin)
        return q2, k2


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Safety check: embedding dimension must be divisible by number of heads
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in batches
        # Single linear layer that creates Query, Key, Value all at once. More efficient than 3 separate layers. 
        # Will be split into Q, K, V later. 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # we multiply by 3 because later we will split it into 3 matricies: Q, K, V

        # Output projection after attention is computed. 
        # Projects concatenated multi-head output back to embedding dimension [B, T, 768]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) 
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # # Creates the causal mask (lower triangular matrix). Saved with model but NOT a trainable parameter.
        # # Not used with flash-attention!
        # self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
        #                     .view(1, 1, config.block_size, config.block_size))
        
        # RoPE configuration
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            rope_base = getattr(config, "rope_base", 10000.0)
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=rope_base,
                max_position_embeddings=config.block_size,
            )
        else:
            self.rotary_emb = None
    
    def forward(self, x): # Multi-headed attention
        B, T, C = x.size() # B: batch size (how many sequences), T: sequence length (number of tokens), C: channels, embedding dimensionality (n_embd)
        # calculate key, query, values for all heads in batch and move head forward to be the batch
        # nh is the number of heads, hs is head size, and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the transformer

        # Projects input to Q, K, V all at once. [B, T, 768] → [B, T, 2304]. Contains concatenated Q, K, V
        qkv = self.c_attn(x)

        # Splits the concatenated qkv tenor into three tensors
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshapes for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Classical attention implementation
        # # attention (materializes the large (T, T) matrix for all queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # Applies causal mask
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # # Normalizes attention scores to probabilities
        # att = F.softmax(att, dim=-1)
        # # Weighted sum of values. Each token gets a weighted average of all values it's allowed to attend to.
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)

        # apply RoPE to q, k
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, seq_len=T)

        # Flash-attention implementation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassembles multi-head outputs, concatenates side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)

        return y
 
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        mlp_type = getattr(config, "mlp_type", "gelu")
        self.mlp_type = mlp_type

        # SwiGLU MLP
        if mlp_type == "swiglu":
            # Set inner dim ~ 8/3 * d so that parameter count matches 4d-GELU
            inner_dim = int(4 * config.n_embd * 2 / 3) # expand the matrix 4 times to use it as a "thinking space", then reduce it to 2/3 to match the GeLU params count
            # Round up to multiple of 256 for GPU efficiency
            inner_dim = ((inner_dim + 255) // 256) * 256
            self.inner_dim = inner_dim
            # value and gate
            self.c_fc = nn.Linear(config.n_embd, 2 * inner_dim) # convolutional fully-connected --> expand from n_embd to 2 * inner_dim
            self.c_proj = nn.Linear(inner_dim, config.n_embd) # convolutional projection --> compress back from 2 * inner_dim to n_embd
            self.c_proj.NANOGPT_SCALE_INIT = 1

        else:
            # Standard GELU MPL
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # expand the embedding dimension 4 times
            self.gelu = nn.GELU(approximate='tanh') # apply GeLU non-linearity
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # compress back to original size
            self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        if self.mlp_type == "swiglu": # [B, T, n_embd]
            x_in = self.c_fc(x) # expand the input from [B, T, n_embd] --> [B, T, inner_dim * 2]
            # x_gate: Passed through Swish, produces values 0→1ish that control "how much" information we pass through
            # x_up: The actual information being passed through
            x_gate, x_up = x_in.chunk(2, dim=-1) # [B, T, inner_dim]
            x = F.silu(x_gate) * x_up # [B, T, inner_dim]
            x = self.c_proj(x) # [B, T, n_embd]
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)


        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Normalize → Attention → Add residual
        x = x + self.mlp(self.ln_2(x))  # Normalize → MLP → Add residual

        # Parallel attention + MLP instead of sequential implementation. Do mot forget to remove ln_2 in the initialization. Otherwise --> dead weight.
        # x = x + self.attn(self.ln_1(x)) + self.mlp(self.ln_1(x))

        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embedding [B, T, 768]
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers (a list of transformer blocks)
            dpn = nn.RMSNorm(config.n_embd), # added another normalization step for the DualPatchNorm
            ln_f = nn.RMSNorm(config.n_embd) # layer norm final after all transformer blocks
        ))
        
        # Only add wpe if NOT using RoPE
        if not getattr(config, "use_rope", True):
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd) # wpe --> learned word positional embedding tensor


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language modelling head, projects embeddings back to vocabulary logits [B, T, 50304]

        # Weights sharing scheme. We share the same embeddings tensor for the input (wte) and output (lm_head) of the transformer to save parameters and improve training
        self.transformer.wte.weight = self.lm_head.weight

        # Init params. Recursively walks through every module in the model (Linear layers, Embeddings, LayerNorms, etc.), applies _init_weights.
        self.apply(self._init_weights)
    
    def _init_weights(self, module): # Weights & biases initialization according to the original GPT-2 paper.
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # Fix variance growth in residual paths. Compensate for the accumulation across layers
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
            if module.bias is not None: # biases if exist are all initialized as zeros
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx of shape (B, T). B - number of batches, T - size of a batch
        B, T = idx.size() #  input token IDs

        # Checks sequence isn't longer than context window
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T},  block size is only {self.config.block_size}" 
        
        # Forward the token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        if getattr(self.config, "use_rope", True):
            # RoPE handles positional encoding in the attention layer
            x = tok_emb
        
        else:
            # Use learned absolute position embeddings. We create new position indices
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # input: token positions, output: position embeddings for these tokens, shape: (T, n_embd)
            pos_emb = self.transformer.wpe(pos)  # input: token positions, output: position embeddings for these tokens, shape: (T, n_embd)
            x = tok_emb + pos_emb # add token embeddings and position embeddings
        
        x = self.transformer.dpn(x) # DualPatchNorm: Apply RMSnorm after positional embedding

        # Forward the blocks of transformer
        # Each block processes the input sequentially. So we pass the input through e.g. 12 blocks in sequence.
        # Each block builds on what previous blocks learned. Early blocks: simple patterns (syntax, word relationships).
        # Middle blocks: medium complexity (phrases, local context). Late blocks: abstract patterns (semantics, global context).
        # Each block applies: 
        # 1. Normalize → Self-Attention → Add residual
        # 2. Normalize → MLP → Add residual
        # Shape of the data stays the same, just the values change.
        for block in self.transformer.h:
            x = block(x)

        # Normalizes the output from all transformer blocks. Stabilizes values before making predictions.
        x = self.transformer.ln_f(x)

        # Projects from embedding space to vocabulary space
        # For each token position, produces a score for every possible next token
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Initialize loss as None (for inference mode)
        loss = None

        # Training mode: targets contains the correct next tokens. Inference mode: targets=None, so loss stays None.
        if targets is not None:
            # 1. Reshape logits. Flattens batch and sequence dimensions, example: [2, 10, 50257] → [20, 50257]. Treats each position as an independent prediction.
            # 2. Reshape targets. Flattens to match logits, example: [2, 10] → [20]
            # 3. Compute cross-entropy: F.cross_entropy(predictions, ground_truth). Returns a single scalar loss value.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # weight decay prevents overfitting by penalizing large weights to improve generalization and stabilize the training process
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        # eps --> small param to prevent division by 0, betas --> Momentum parameters, beta1 --> gradient momentum, beta2--> squared gradient momentum
        # fused --> Combines multiple operations into single GPU kernel, ~20-30% faster on CUDA
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

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

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)

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