# Training Parameters - GPT-2 Long Architecture

**Training Date:** 2026-02-20  
**Run ID:** 200226

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Model Type | GPT-2 long (deeper layout) |
| Layers (`n_layer`) | 30 |
| Attention Heads (`n_head`) | 8 |
| Embedding Dimension (`n_embd`) | 512 |
| Context Length (`block_size`) | 1024 tokens |
| Vocabulary Size (`vocab_size`) | 50304 (padded from 50257) |
| Total Parameters | ~128M |

### Architecture Details
- **Attention:** Flash Attention (scaled_dot_product_attention)
- **Activation Function:** SwiGLU (gated activation)
- **Positional Encoding:** RoPE (Rotary Position Embedding)
- **Normalization:** RMSNorm
- **Dual PatchNorm:** RMSNorm applied after (tok_embed + pos_embed)
- **Weight Tying:** Token embedding weights shared with LM head

### Key Changes Over 050226 Run
1. **Longer/deeper architecture:** 30 layers × 512 embd × 8 heads instead of 12 × 768 × 12
2. **Higher learning rate:** 2.4e-3 vs. 1.8e-3

---

## Training Configuration

### Batch Size & Gradient Accumulation

| Parameter | Value |
|-----------|-------|
| Micro-batch Size (`B`) | 16 |
| Sequence Length (`T`) | 1024 tokens |
| Total Batch Size | 524,288 tokens (~0.5M) |
| Gradient Accumulation Steps | Calculated: `524288 / (B × T × num_gpus)` |

### Optimizer Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused version) |
| Weight Decay | 0.1 |
| Beta1 | 0.9 |
| Beta2 | 0.95 |
| Epsilon | 1e-8 |
| Gradient Clipping | 1.0 (global norm) |

### Learning Rate Schedule

| Parameter | Value |
|-----------|-------|
| Max Learning Rate | 2.4e-3 (4x base GPT-2 LR) |
| Min Learning Rate | 2.4e-4 (10% of max) |
| Warmup Steps | 357 steps (~187M tokens) |
| Total Steps | 40,000 steps (~2 epochs) |
| Schedule Type | Linear warmup + Cosine decay |

### Training Steps & Evaluation

| Parameter | Value |
|-----------|-------|
| Total Training Steps | 40,000 |
| Validation Frequency | Every 250 steps |
| Validation Steps | 20 batches |
| HellaSwag Eval Frequency | Every 250 steps |
| Checkpoint Frequency | Every 5,000 steps + last step |

---

## Data Configuration

### Dataset
- **Name:** FineWeb-Edu 10BT
- **Source:** HuggingFace
- **Size:** ~10 billion tokens
- **Tokenizer:** GPT-2 BPE (tiktoken)
- **Splits:** Train & Validation shards
- **Training Epochs:** 2 (full dataset seen twice)

### Data Loading
- **Global shuffling:** All windows shuffled across shards before training
- **Multi-epoch support:** Proper reshuffling between epochs
- **Memory-mapped shards:** Efficient loading without loading entire dataset into RAM

---

## Hardware & Distributed Training

### Compute Resources

| Parameter | Value |
|-----------|-------|
| GPU Type | NVIDIA RTX 5090 (32GB VRAM) |
| Number of GPUs | 2 |
| Platform | Vast.ai (rented) |
| Training Duration | ~18 hours |
| Estimated Cost | ~$11–15 USD |

### Distributed Setup

| Parameter | Value |
|-----------|-------|
| Framework | DDP (DistributedDataParallel) |
| Backend | NCCL |
| Precision | bfloat16 (autocast) |
| Compile | `torch.compile` enabled |
| matmul precision | 'high' (TF32) |
| GPU Peak FLOPS | 209.5 TFLOPS (RTX 5090 bf16) |

---

## Initialization & Regularization

### Weight Initialization
- **Linear layers:** Normal distribution (mean=0.0, std=0.02)
- **Residual projections:** Scaled by `(2 * n_layer)^(-0.5)` ≈ 0.00408
- **Embeddings:** Normal distribution (mean=0.0, std=0.02)
- **Biases:** Zeros

### Regularization
- Weight decay applied to 2D+ parameters (weights, embeddings)
- No weight decay for biases and norm parameters
- Gradient clipping at norm 1.0

---

## Training Results

### Final Metrics (step 39,999)

| Metric | Value | Reference (GPT-2 124M) | Previous Run (050226) |
|--------|-------|------------------------|-----------------------|
| Validation Loss | 2.9441 | ~3.29 | ~2.99 |
| HellaSwag Accuracy | 0.3354 (33.5%) | 0.294 (29.4%) | 0.320 (32.0%) |

### Notes
- HellaSwag target for GPT-3 124M: 0.337 — this run reaches 0.3354, essentially matching it
- Training on only 10B tokens; GPT-3 124M used ~300B tokens

---

## Reproducibility

### Random Seeds
- **Manual Seed:** 1337
- **CUDA Seed:** 1337
- **Generation Seed:** 42 + rank (for distributed generation)
