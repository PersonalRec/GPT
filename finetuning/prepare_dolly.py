"""
Download Dolly-15K, tokenize it with the GPT-2 tokenizer, and write fixed-length
token/mask shards for supervised fine-tuning.

Usage:
    python prepare_dolly.py
"""

import json
import os

import numpy as np
import tiktoken
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "dolly_15k")

SEED = 1337
BLOCK_SIZE = 1024
SEQ_LEN = BLOCK_SIZE + 1
VAL_SIZE = 0.05
SHARD_SIZE = 2048
MIN_RESPONSE_TOKENS = 16

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def format_dolly_prompt(instruction, context=""):
    if context:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            "### Response:\n"
        )

    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def encode_example(example):
    prompt = format_dolly_prompt(example["instruction"], example["context"])
    response = example["response"].strip()

    prompt_tokens = enc.encode(prompt)
    response_tokens = enc.encode(response)
    full_tokens = prompt_tokens + response_tokens + [eot]

    full_tokens = full_tokens[:SEQ_LEN]
    response_start = len(prompt_tokens)
    response_len = len(full_tokens) - response_start
    if response_len < MIN_RESPONSE_TOKENS:
        return None

    loss_mask = np.zeros(len(full_tokens), dtype=np.uint8)
    loss_mask[response_start:] = 1

    pad_len = SEQ_LEN - len(full_tokens)
    if pad_len > 0:
        full_tokens = full_tokens + [eot] * pad_len
        loss_mask = np.pad(loss_mask, (0, pad_len), constant_values=0)

    return np.asarray(full_tokens, dtype=np.uint16), loss_mask


def write_split(split_name, dataset_split):
    tokens_buffer = []
    masks_buffer = []
    shard_idx = 0
    skipped = 0
    kept = 0

    for example in dataset_split:
        encoded = encode_example(example)
        if encoded is None:
            skipped += 1
            continue

        tokens, masks = encoded
        tokens_buffer.append(tokens)
        masks_buffer.append(masks)
        kept += 1

        if len(tokens_buffer) == SHARD_SIZE:
            flush_shard(split_name, shard_idx, tokens_buffer, masks_buffer)
            shard_idx += 1
            tokens_buffer, masks_buffer = [], []

    if tokens_buffer:
        flush_shard(split_name, shard_idx, tokens_buffer, masks_buffer)

    return {"kept": kept, "skipped": skipped, "shards": shard_idx + (1 if tokens_buffer else 0)}


def flush_shard(split_name, shard_idx, tokens_buffer, masks_buffer):
    tokens_path = os.path.join(OUTPUT_DIR, f"{split_name}_tokens_{shard_idx:03d}.npy")
    masks_path = os.path.join(OUTPUT_DIR, f"{split_name}_masks_{shard_idx:03d}.npy")
    np.save(tokens_path, np.stack(tokens_buffer))
    np.save(masks_path, np.stack(masks_buffer))
    print(f"wrote {tokens_path} and {masks_path} with {len(tokens_buffer)} examples")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("downloading Dolly-15K from Hugging Face...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    split_dataset = dataset.train_test_split(test_size=VAL_SIZE, seed=SEED, shuffle=True)

    train_stats = write_split("train", split_dataset["train"])
    val_stats = write_split("val", split_dataset["test"])

    metadata = {
        "dataset": "databricks/databricks-dolly-15k",
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "sequence_length": SEQ_LEN,
        "val_size": VAL_SIZE,
        "shard_size": SHARD_SIZE,
        "min_response_tokens": MIN_RESPONSE_TOKENS,
        "train": train_stats,
        "val": val_stats,
    }
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"train examples kept: {train_stats['kept']} | skipped: {train_stats['skipped']}")
    print(f"val examples kept: {val_stats['kept']} | skipped: {val_stats['skipped']}")
    print(f"metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
