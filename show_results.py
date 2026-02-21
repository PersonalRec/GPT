"""
Combined Training Results Visualization

Compares the training runs:
- 230126: Baseline GPT-2 reproduction
- 050226: Improved version with RoPE, SwiGLU, and RMSNorm
- 200226: Improved version, deep & narrow architecture

Saves the resulting plot to results/combined_comparison.png

Supports both legacy .txt log format and the newer .csv log format.
"""
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sz = "124M"

loss_baseline = {
    # https://github.com/openai/gpt-2/blob/master/model_card.md
    "124M": 3.29,
    "350M": 2.90,
    "774M": 2.70,
    "1558M": 2.60,
}[sz]

hella2_baseline = {
    # https://github.com/openai/gpt-2/blob/master/model_card.md
    "124M": 0.294,
    "350M": 0.430,
    "774M": 0.485,
    "1558M": 0.638,
}[sz]

hella3_baseline = {
    # https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling (Table 2)
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.511,
    "1558M": 0.598,
}[sz]


def load_log_txt(log_path):
    """Load legacy .txt log format: '{step} {stream} {value}' per line.
    Returns a dict of streams: {'train': {step: val}, 'val': {...}, 'hella': {...}}
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    streams = {}
    for line in lines:
        parts = line.strip().split(" ")
        if len(parts) >= 3:
            k, stream, v = parts[0], parts[1], parts[2]
            if stream not in streams:
                streams[stream] = {}
            streams[stream][int(k)] = float(v)
    return streams


def load_log_csv(log_path):
    """Load new CSV log format with columns:
    step, train_loss, val_loss, hellaswag, mmlu, arc, lr, grad_norm, tokens_per_sec, mfu, dt_ms
    Returns a dict of streams matching the legacy format for compatibility.
    """
    df = pd.read_csv(log_path)

    streams = {}

    # Train loss (present on every row)
    if "train_loss" in df.columns:
        train_data = df[["step", "train_loss"]].dropna(subset=["train_loss"])
        streams["train"] = dict(zip(train_data["step"].astype(int), train_data["train_loss"].astype(float)))

    # Validation loss (only on eval steps)
    if "val_loss" in df.columns:
        val_data = df[["step", "val_loss"]].dropna(subset=["val_loss"])
        if not val_data.empty:
            streams["val"] = dict(zip(val_data["step"].astype(int), val_data["val_loss"].astype(float)))

    # Eval benchmarks (only on eval steps)
    for col, key in [("hellaswag", "hella"), ("mmlu", "mmlu"), ("arc", "arc")]:
        if col in df.columns:
            col_data = df[["step", col]].dropna(subset=[col])
            if not col_data.empty:
                streams[key] = dict(zip(col_data["step"].astype(int), col_data[col].astype(float)))

    # Additional metrics (unique to CSV format)
    for col in ["lr", "grad_norm", "tokens_per_sec", "mfu", "dt_ms"]:
        if col in df.columns:
            col_data = df[["step", col]].dropna(subset=[col])
            if not col_data.empty:
                streams[col] = dict(zip(col_data["step"].astype(int), col_data[col].astype(float)))

    return streams


def load_log(log_path):
    """Auto-detect log format by file extension and load accordingly."""
    if log_path.endswith(".csv"):
        return load_log_csv(log_path)
    else:
        return load_log_txt(log_path)


# Define log files to load
log_files = {
    "gpt-2 (baseline)": "logs/log124M_40B_230126/log.txt",
    "gpt-improved (RoPE + SwiGLU)": "logs/log124M_050226/log.txt",
    "gpt-improved-v2 (deep & narrow)": "results/200226/log.csv",
}

# Load and parse both log files
all_runs = {}
for run_name, log_path in log_files.items():
    print(f"Loading {log_path}...")
    streams = load_log(log_path)
    all_runs[run_name] = streams

    # print the streams
    print(f"{run_name}:")
    for k, v in sorted(streams.items()):
        print(f"  {k}: {len(v)} entries")
    print()

# plot the streams
plt.figure(figsize=(16, 6))

# Plot 1: Loss
plt.subplot(121)

# Define colors for each run
colors = {
    "gpt-2 (baseline)": {'train': '#1f77b4', 'val': '#ff7f0e'},
    "gpt-improved (RoPE + SwiGLU)": {'train': '#2ca02c', 'val': '#d62728'},
    "gpt-improved-v2 (deep & narrow)": {'train': '#9467bd', 'val': '#e377c2'},
}

for run_name, streams in all_runs.items():
    for prefix, label in [("train", "train"), ("val", "val")]:
        if prefix in streams:
            xs, ys = zip(*sorted(streams[prefix].items()))
            plt.plot(xs, ys,
                    label=f'{run_name} {label}',
                    color=colors[run_name][prefix],
                    alpha=0.8,
                    linewidth=1.5)

if sz in ["124M"]:
    plt.axhline(y=loss_baseline, color='r', linestyle='--',
                label=f"OpenAI GPT-2 ({sz}) val loss", linewidth=2)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.yscale('log')
ax1 = plt.gca()
ax1.yaxis.set_major_locator(ticker.FixedLocator(np.arange(2.4, 4.01, 0.1)))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True, which='both', axis='both', alpha=0.3, linestyle='-')
plt.ylim(top=4.0)
plt.legend(loc='upper right', fontsize=10)
plt.title("Training & Validation Loss", fontsize=14)
plt.grid(True, alpha=0.3, axis='both')

# Plot 2: HellaSwag
plt.subplot(122)

hella_colors = {
    "gpt-2 (baseline)": '#1f77b4',
    "gpt-improved (RoPE + SwiGLU)": '#2ca02c',
    "gpt-improved-v2 (deep & narrow)": '#9467bd',
}

for run_name, streams in all_runs.items():
    if "hella" in streams:
        xs, ys = zip(*sorted(streams["hella"].items()))
        plt.plot(xs, ys,
                label=f"{run_name}",
                color=hella_colors[run_name],
                alpha=0.8,
                linewidth=1.5)

if sz in ["124M"]:
    plt.axhline(y=hella2_baseline, color='r', linestyle='--',
                label=f"OpenAI GPT-2 ({sz})", linewidth=2)
if sz in ["124M"]:
    plt.axhline(y=hella3_baseline, color='g', linestyle='--',
                label=f"OpenAI GPT-3 ({sz})", linewidth=2)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
ax2 = plt.gca()
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.legend(loc='lower right', fontsize=10)
plt.title("HellaSwag Eval", fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_path = os.path.join("results", "combined_comparison.png")
os.makedirs("results", exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_path}")
