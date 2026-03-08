#GPT2 128M Long version

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect


@dataclass
class GPTConfig():
    block_size: int = 1024 # maximum sequence length (context length)
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 tokens + 1 <|endoftext|> token = 50257 which is inefficient in terms of cuda processing, new vocab size 50304 could be easily divided by a lot of numbers
    n_layer: int = 30 # number of layers (transformer blocks: each block has attention + MLP + RMSNorms)
    n_head: int = 8 # number of heads per transformer blocks. Each head sees 512 ÷ 8 = 64 dimensions. Different heads can learn different attention patterns.
    n_embd: int = 512 # embedding dimension

    # additional options
    use_rope: bool = True      # use RoPE embedding for training
    rope_base: float = 10000.0
    mlp_type: str = "swiglu"   # MLP activation type: "gelu" or "swiglu"


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
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose):
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
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"using fused AdamW: {use_fused}")
        # eps --> small param to prevent division by 0, betas --> Momentum parameters, beta1 --> gradient momentum, beta2--> squared gradient momentum
        # fused --> Combines multiple operations into single GPU kernel, ~20-30% faster on CUDA
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer