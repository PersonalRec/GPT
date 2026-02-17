"""
Downloads and evaluates ARC-Challenge (AI2 Reasoning Challenge) in Python.
https://github.com/allenai/arc-solvers

Uses teacher-forcing (completion style) evaluation, matching the HellaSwag/MMLU pattern:
for each question, we feed context + each answer completion through the model,
compute loss over the completion region, and pick the answer with the lowest loss.

ARC-Challenge contains ~1,170 test questions from grade-school science exams.
Questions have 3-5 answer choices (not always 4).

gpt2 (124M): ~21-25% on ARC-Challenge (near random)
gpt2-xl (1558M): ~27-30%

Dataset format (JSON):
{
    "id": "Mercury_SC_415702",
    "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
    "choices": {"text": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"], "label": ["A", "B", "C", "D"]},
    "answerKey": "A"
}
"""

import os
import json
import requests
import tiktoken
import torch
from torch.nn import functional as F

from evals import evaluate_benchmark

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "arc")

enc = tiktoken.get_encoding("gpt2")

# HuggingFace datasets-server API for ARC-Challenge
HF_API_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "allenai/ai2_arc"
HF_CONFIG = "ARC-Challenge"
HF_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def download(split):
    """Downloads ARC-Challenge data for a given split from HuggingFace."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, f"ARC-Challenge-{split}.jsonl")
    if not os.path.exists(data_filename):
        hf_split = HF_SPLIT_MAP[split]
        print(f"Downloading ARC-Challenge {hf_split} split from HuggingFace...")
        rows = []
        offset = 0
        limit = 100
        while True:
            resp = requests.get(HF_API_URL, params={
                "dataset": HF_DATASET,
                "config": HF_CONFIG,
                "split": hf_split,
                "offset": offset,
                "length": limit,
            })
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("rows", [])
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        # Convert HuggingFace format to the ARC JSONL format expected by render_example:
        # HF has flat {"question": str, "choices": {...}} while the code expects
        # {"question": {"stem": str, "choices": {...}}}
        with open(data_filename, "w") as f:
            for item in rows:
                row = item["row"]
                example = {
                    "id": row["id"],
                    "question": {
                        "stem": row["question"],
                        "choices": row["choices"],
                    },
                    "answerKey": row["answerKey"],
                }
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(rows)} examples to {data_filename}")
    return data_filename


def render_example(example):
    """
    Given an ARC example dict, render it as torch tensors:
    - tokens (context + completion, of size num_choices x N)
    - mask (is 1 in the region of the candidate completion)
    - label (the index of the correct completion)

    ARC questions can have 3-5 choices, so tensor row count varies.
    Uses teacher-forcing style matching HellaSwag/MMLU.
    """
    question = example["question"]["stem"]
    choice_texts = example["question"]["choices"]["text"]
    choice_labels = example["question"]["choices"]["label"]
    answer_key = example["answerKey"]

    # Find the correct answer index
    # Some ARC questions use numeric labels ("1","2","3","4") instead of letters
    label = None
    for i, cl in enumerate(choice_labels):
        if cl == answer_key:
            label = i
            break
    if label is None:
        # Fallback: try matching by position
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                     "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        label = label_map.get(answer_key, 0)

    num_choices = len(choice_texts)

    # Build the prompt
    ctx = f"{question}\n"
    for i in range(num_choices):
        ctx += f"{choice_labels[i]}. {choice_texts[i]}\n"
    ctx += "Answer:"

    # Tokenize context
    ctx_tokens = enc.encode(ctx)

    # Build completions: " A. answer_text", " B. answer_text", etc.
    tok_rows = []
    mask_rows = []
    for i in range(num_choices):
        end_text = f" {choice_labels[i]}. {choice_texts[i]}"
        end_tokens = enc.encode(end_text)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    # Collate into tensors (variable number of choices)
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((num_choices, max_len), dtype=torch.long)
    mask = torch.zeros((num_choices, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


def iterate_examples(split):
    """Yields examples from ARC-Challenge for a given split."""
    data_filename = download(split)
    with open(data_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def _render_for_eval(example):
    """Adapter matching the evaluate_benchmark interface."""
    return render_example(example)


def evaluate(model, device, device_type, ddp, ddp_rank, ddp_world_size):
    """
    Evaluate the model on ARC-Challenge test set.

    Returns:
        accuracy (float) -- normalized accuracy
    """
    return evaluate_benchmark(
        model=model,
        device=device,
        device_type=device_type,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        iterate_fn=lambda: iterate_examples("test"),
        render_fn=_render_for_eval,
        benchmark_name="ARC-Challenge",
    )


# Standalone CLI evaluation using a HuggingFace model
if __name__ == "__main__":
    import argparse
    from transformers import GPT2LMHeadModel
    from evals import get_most_likely_row

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2",
                        help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="the device to use")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(args.model_type)
    model.to(args.device)

    num_correct = 0
    num_total = 0
    for example in iterate_examples("test"):
        tokens, mask, label = render_example(example)
        tokens = tokens.to(args.device)
        mask = mask.to(args.device)

        with torch.no_grad():
            logits = model(tokens).logits
        pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct += int(pred_norm == label)

        if num_total % 50 == 0:
            print(f"ARC-Challenge {num_total} acc={num_correct / num_total:.4f}")

    acc = num_correct / num_total
    print(f"\nARC-Challenge accuracy: {num_correct}/{num_total} = {acc:.4f}")
