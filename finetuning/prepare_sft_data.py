"""
Download and prepare multiple SFT datasets for GPT-2 fine-tuning.
All datasets are converted to a unified format, filtered for English,
filtered for length (≤ BLOCK_SIZE tokens), and written as fixed-length
token/mask shards.

Supported datasets:
    - databricks/databricks-dolly-15k
    - yahma/alpaca-cleaned
    - HuggingFaceTB/smol-smoltalk
    - allenai/tulu-3-sft-personas-instruction-following
    - timdettmers/openassistant-guanaco
    - imone/OpenOrca_FLAN
    - Open-Orca/SlimOrca          (system+human→gpt ShareGPT format, ~518K examples)
    - openai/gsm8k                (grade school math chain-of-thought, ~7.5K train examples)
    - lmsys/lmsys-chat-1m         (real-world LLM conversations; requires HF license agreement)
    - garage-bAInd/Open-Platypus  (curated reasoning & instruction dataset)

Usage:
    python prepare_sft_data.py                          # prepare all datasets (15K cap each)
    python prepare_sft_data.py --datasets dolly alpaca   # prepare specific ones
    python prepare_sft_data.py --max-per-dataset 0       # no cap (use all available)
    python prepare_sft_data.py --max-per-dataset 5000    # smaller cap per dataset
"""

import argparse
import json
import multiprocessing as mp
import os

# Skip the HF Hub network check for cached datasets to avoid 60s+ ETag timeout
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import re
import time
from functools import partial
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset, enable_progress_bar
enable_progress_bar()
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
DetectorFactory.seed = 0   # make langdetect deterministic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SEED = 1337
BLOCK_SIZE = 1024
SEQ_LEN = BLOCK_SIZE + 1
VAL_SIZE = 0.05
SHARD_SIZE = 2048
MIN_RESPONSE_TOKENS = 32

def load_hf_token():
    """Load HF_TOKEN from the .env file in the project root and set it as an env var.

    Setting HF_TOKEN in the environment is sufficient for load_dataset() to
    authenticate — no network round-trip is needed at startup.
    """
    env_path = os.path.join(PROJECT_ROOT, ".env")
    token = None

    # Try .env file first
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    # Fall back to environment variable already set in the shell
    if not token:
        token = os.environ.get("HF_TOKEN")

    if not token:
        print("ERROR: HF_TOKEN not found.")
        print(f"  Looked in: {env_path}")
        print(f"  Also checked: HF_TOKEN environment variable")
        print(f"  Please add HF_TOKEN=hf_... to your .env file or run:")
        print(f"    export HF_TOKEN=hf_...")
        raise SystemExit(1)

    if not token.startswith("hf_"):
        print(f"ERROR: HF_TOKEN looks malformed (expected 'hf_...'): {token[:8]}...")
        raise SystemExit(1)

    # Expose to child processes and huggingface_hub without a network round-trip
    os.environ["HF_TOKEN"] = token
    print(f"HuggingFace token loaded (token: ...{token[-4:]})")
    return token
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "sft_mix")


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


# ========================= Language detection ===============================================

# Unicode ranges that indicate a non-English script.
# Any character falling into one of these ranges immediately disqualifies the text.
_NON_LATIN_RANGES = [
    (0x0400, 0x052F),   # Cyrillic + Cyrillic Supplement
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),   # Syriac
    (0x0900, 0x097F),   # Devanagari
    (0x0980, 0x09FF),   # Bengali
    (0x0A00, 0x0A7F),   # Gurmukhi
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0B00, 0x0B7F),   # Oriya
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
    (0x0E00, 0x0E7F),   # Thai
    (0x0E80, 0x0EFF),   # Lao
    (0x1000, 0x109F),   # Myanmar
    (0x10A0, 0x10FF),   # Georgian
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
]


def _has_non_latin_script(text: str) -> bool:
    """Return True if any character belongs to a non-Latin script block."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _NON_LATIN_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def is_likely_english(text: str) -> bool:
    """
    Three-stage English filter:

    1. Block any non-Latin script characters (Cyrillic, CJK, Arabic, Devanagari …).
       Even a single such character disqualifies the example.
    2. Reject if non-ASCII Latin ratio is too high (catches heavy accented text that
       is technically Latin-script but not English).
    3. Use langdetect to reject Latin-script non-English languages (Spanish, French …).
    """
    if not text or len(text.strip()) < 10:
        return False

    # Stage 1 – non-Latin script characters
    if _has_non_latin_script(text):
        return False

    # Stage 2 – ASCII ratio (remaining chars are Latin; high non-ASCII = likely
    # Romance/Germanic non-English)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.90:
        return False

    # Stage 3 – language detection on a representative sample
    sample = text[:2000]
    try:
        return detect(sample) == "en"
    except LangDetectException:
        return False


# ========================= Prompt formatting ===============================================

def format_prompt(instruction, context=""):
    """Unified Alpaca-style prompt template used across all datasets."""
    if context and context.strip():
        return (
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{context.strip()}\n\n"
            "### Response:\n"
        )
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        "### Response:\n"
    )


# ========================= Dataset converters ==============================================
# Each _convert_X(ex) takes one raw HuggingFace example and returns a unified dict
# {"instruction", "context", "response", "source"}, or None to skip the example.

def _convert_dolly(ex):
    return {"instruction": ex["instruction"], "context": ex.get("context", ""),
            "response": ex["response"], "source": "dolly"}


def _convert_alpaca(ex):
    return {"instruction": ex["instruction"], "context": ex.get("input", ""),
            "response": ex["output"], "source": "alpaca"}


def _convert_smol_smoltalk(ex):
    msgs = ex.get("messages", [])
    user_msg = asst_msg = None
    for m in msgs:
        if m["role"] == "user" and user_msg is None:
            user_msg = m["content"]
        elif m["role"] == "assistant" and user_msg is not None:
            asst_msg = m["content"]
            break
    if not (user_msg and asst_msg):
        return None
    return {"instruction": user_msg, "context": "", "response": asst_msg,
            "source": "smol-smoltalk"}


def _convert_tulu3(ex):
    msgs = ex.get("messages", [])
    if len(msgs) < 2:
        return None
    user_msg = msgs[0].get("content", "") if msgs[0]["role"] == "user" else ""
    asst_msg = msgs[1].get("content", "") if msgs[1]["role"] == "assistant" else ""
    if not (user_msg and asst_msg):
        return None
    return {"instruction": user_msg, "context": "", "response": asst_msg, "source": "tulu3"}


def _convert_guanaco(ex):
    text = ex["text"]
    human_match = re.search(r"### Human:\s*(.*?)### Assistant:\s*", text, re.DOTALL)
    if not human_match:
        return None
    instruction = human_match.group(1).strip()
    after_first_asst = text[human_match.end():]
    next_human = after_first_asst.find("### Human:")
    response = (after_first_asst[:next_human].strip() if next_human != -1
                else after_first_asst.strip())
    if not (instruction and response):
        return None
    return {"instruction": instruction, "context": "", "response": response,
            "source": "guanaco"}


def _convert_openorca(ex):
    instruction = ex["instruction"].strip()
    system = ex.get("system", "").strip()
    response = ex["response"].strip()
    full_instruction = f"{system}\n\n{instruction}" if system and len(system) > 10 else instruction
    if not (full_instruction and response):
        return None
    return {"instruction": full_instruction, "context": "", "response": response,
            "source": "openorca"}


def _convert_slimorca(ex):
    convs = ex.get("conversations", [])
    system_msg = human_msg = gpt_msg = ""
    for turn in convs:
        role = turn.get("from", "")
        value = (turn.get("value") or "").strip()
        if role == "system" and not system_msg:
            system_msg = value
        elif role == "human" and not human_msg:
            human_msg = value
        elif role == "gpt" and human_msg and not gpt_msg:
            gpt_msg = value
            break
    if not (human_msg and gpt_msg):
        return None
    full_instruction = (f"{system_msg}\n\n{human_msg}" if system_msg and len(system_msg) > 10
                        else human_msg)
    return {"instruction": full_instruction, "context": "", "response": gpt_msg,
            "source": "slimorca"}


def _convert_gsm8k(ex):
    question = (ex.get("question") or "").strip()
    answer = (ex.get("answer") or "").strip()
    if not (question and answer):
        return None
    return {"instruction": question, "context": "", "response": answer, "source": "gsm8k"}


def _convert_lmsys_chat(ex):
    convs = ex.get("conversation", [])
    human_msg = asst_msg = ""
    for turn in convs:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if role == "user" and not human_msg:
            human_msg = content
        elif role == "assistant" and human_msg and not asst_msg:
            asst_msg = content
            break
    if not (human_msg and asst_msg):
        return None
    return {"instruction": human_msg, "context": "", "response": asst_msg,
            "source": "lmsys-chat"}


def _convert_open_platypus(ex):
    instruction = (ex.get("instruction") or "").strip()
    response = (ex.get("output") or "").strip()
    context = (ex.get("input") or "").strip()
    if not (instruction and response):
        return None
    return {"instruction": instruction, "context": context, "response": response,
            "source": "open-platypus"}


# ========================= Registry ========================================================
# Maps dataset name -> (raw_loader_fn, converter_fn)
# raw_loader_fn() calls load_dataset() and returns the HF Dataset (triggers download with
# HF's own progress bars). converter_fn(raw_ex) converts one example or returns None to skip.

DATASET_REGISTRY = {
    "dolly":         (lambda: load_dataset("databricks/databricks-dolly-15k", split="train"),
                      _convert_dolly),
    "alpaca":        (lambda: load_dataset("yahma/alpaca-cleaned", split="train"),
                      _convert_alpaca),
    "smol-smoltalk": (lambda: load_dataset("HuggingFaceTB/smol-smoltalk", split="train"),
                      _convert_smol_smoltalk),
    "tulu3":         (lambda: load_dataset("allenai/tulu-3-sft-personas-instruction-following",
                                           split="train"), _convert_tulu3),
    "guanaco":       (lambda: load_dataset("timdettmers/openassistant-guanaco", split="train"),
                      _convert_guanaco),
    "openorca":      (lambda: load_dataset("imone/OpenOrca_FLAN", split="train"),
                      _convert_openorca),
    "slimorca":      (lambda: load_dataset("Open-Orca/SlimOrca", split="train"),
                      _convert_slimorca),
    "gsm8k":         (lambda: load_dataset("openai/gsm8k", "main", split="train"),
                      _convert_gsm8k),
    "lmsys-chat":    (lambda: load_dataset("lmsys/lmsys-chat-1m", split="train").filter(
                          lambda x: x["language"] == "English",
                          num_proc=max(1, os.cpu_count() - 1),
                      ), _convert_lmsys_chat),
    "open-platypus": (lambda: load_dataset("garage-bAInd/Open-Platypus", split="train"),
                      _convert_open_platypus),
}


# ========================= Encoding ========================================================

def encode_example(example):
    """Convert a unified example dict to padded token + mask arrays of length SEQ_LEN."""
    prompt = format_prompt(example["instruction"], example["context"])
    response = example["response"].strip()

    prompt_tokens = enc.encode(prompt, disallowed_special=())
    response_tokens = enc.encode(response, disallowed_special=())
    full_tokens = prompt_tokens + response_tokens + [eot]

    # Skip if total length exceeds our context window
    if len(full_tokens) > SEQ_LEN:
        return None

    response_start = len(prompt_tokens)
    response_len = len(full_tokens) - response_start
    if response_len < MIN_RESPONSE_TOKENS:
        return None

    # Build loss mask: 1 for response tokens only
    loss_mask = np.zeros(len(full_tokens), dtype=np.uint8)
    loss_mask[response_start:] = 1

    # Pad to fixed length
    pad_len = SEQ_LEN - len(full_tokens)
    if pad_len > 0:
        full_tokens = full_tokens + [eot] * pad_len
        loss_mask = np.pad(loss_mask, (0, pad_len), constant_values=0)

    return np.asarray(full_tokens, dtype=np.uint16), loss_mask


# ========================= Shard writing ===================================================

def flush_shard(split_name, shard_idx, tokens_buf, masks_buf, sources_buf, output_dir):
    t_path = os.path.join(output_dir, f"{split_name}_tokens_{shard_idx:03d}.npy")
    m_path = os.path.join(output_dir, f"{split_name}_masks_{shard_idx:03d}.npy")
    s_path = os.path.join(output_dir, f"{split_name}_sources_{shard_idx:03d}.npy")
    np.save(t_path, np.stack(tokens_buf))
    np.save(m_path, np.stack(masks_buf))
    np.save(s_path, np.array(sources_buf, dtype="U30"))
    print(f"  wrote {t_path} ({len(tokens_buf)} examples)")


def write_split(split_name, examples, output_dir):
    tokens_buf, masks_buf, sources_buf = [], [], []
    shard_idx = 0

    for tokens, mask, source in examples:
        tokens_buf.append(tokens)
        masks_buf.append(mask)
        sources_buf.append(source)
        if len(tokens_buf) == SHARD_SIZE:
            flush_shard(split_name, shard_idx, tokens_buf, masks_buf, sources_buf, output_dir)
            shard_idx += 1
            tokens_buf, masks_buf, sources_buf = [], [], []

    if tokens_buf:
        flush_shard(split_name, shard_idx, tokens_buf, masks_buf, sources_buf, output_dir)

    total = shard_idx * SHARD_SIZE + len(tokens_buf)
    return shard_idx + (1 if tokens_buf else 0), total


# ========================= Parallel worker =================================================

def _worker_init():
    """Called once per worker process — makes langdetect deterministic in every process."""
    DetectorFactory.seed = 0


def _process_one(raw_ex, *, converter):
    """
    Multiprocessing worker: convert → language-filter → encode one raw HF example.
    Returns (tokens, mask, source) on success, or (None, reason_str) when skipped.
    Must be a top-level function so it can be pickled by mp.Pool (spawn method on macOS).
    """
    example = converter(raw_ex)
    if example is None:
        return None, "fmt"
    text = example["instruction"] + " " + example["response"]
    if not is_likely_english(text):
        return None, "lang"
    encoded = encode_example(example)
    if encoded is None:
        return None, "len"
    tokens, mask = encoded
    return (tokens, mask, example["source"]), None


# ========================= Main ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for GPT fine-tuning")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Which datasets to include")
    parser.add_argument("--max-per-dataset", type=int, default=15000,
                        help="Max examples to keep per dataset (after filtering). Set to 0 for unlimited.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--nprocs", type=int, default=max(8, os.cpu_count() - 1),
                        help="Number of worker processes for parallel encoding (default: all cores - 1)")
    args = parser.parse_args()

    # Authenticate with HuggingFace before downloading anything
    load_hf_token()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- Collect and encode all examples ----
    all_encoded = []
    stats = {}
    t_total_start = time.perf_counter()

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Loading: {ds_name}")
        print(f"{'='*60}")
        kept, skipped_lang, skipped_len, skipped_fmt = 0, 0, 0, 0
        ds_encoded = []
        t_ds_start = time.perf_counter()

        raw_loader, converter = DATASET_REGISTRY[ds_name]

        print(f"  Downloading {ds_name} ...")
        ds = raw_loader()          # HF shows its own download progress bar here
        total_raw = len(ds)
        print(f"  {total_raw:,} raw examples — processing ...")

        cap = args.max_per_dataset or None
        worker_fn = partial(_process_one, converter=converter)

        with mp.Pool(args.nprocs, initializer=_worker_init) as pool:
            pbar = tqdm(
                pool.imap(worker_fn, ds, chunksize=32),
                total=total_raw,
                desc=f"  {ds_name:<20}",
                unit=" ex",
                dynamic_ncols=True,
            )
            for result, skip_reason in pbar:
                if result is None:
                    if skip_reason == "fmt":
                        skipped_fmt += 1
                    elif skip_reason == "lang":
                        skipped_lang += 1
                    else:
                        skipped_len += 1
                else:
                    ds_encoded.append(result)
                    kept += 1

                if (kept + skipped_lang + skipped_len + skipped_fmt) % 200 == 0:
                    pbar.set_postfix(kept=kept, skip_lang=skipped_lang,
                                     skip_len=skipped_len, refresh=False)

                if cap and kept >= cap:
                    pool.terminate()
                    break

            pbar.set_postfix(kept=kept, skip_lang=skipped_lang, skip_len=skipped_len)
            pbar.close()

        ds_elapsed = time.perf_counter() - t_ds_start
        stats[ds_name] = {
            "kept": kept,
            "skipped_not_english": skipped_lang,
            "skipped_too_long_or_short": skipped_len,
            "elapsed_s": round(ds_elapsed, 1),
        }
        print(f"  done: kept={kept:,}  skip_lang={skipped_lang:,}  "
              f"skip_len={skipped_len:,}  skip_fmt={skipped_fmt:,}  time={ds_elapsed:.1f}s")
        all_encoded.extend(ds_encoded)

    print(f"\n{'='*60}")
    print(f"Total encoded examples: {len(all_encoded)}")
    print(f"{'='*60}")

    # ---- Shuffle and split into train / val ----
    indices = np.arange(len(all_encoded))
    rng.shuffle(indices)

    val_count = max(1, int(len(all_encoded) * VAL_SIZE))
    val_indices = set(indices[:val_count].tolist())

    train_examples = [all_encoded[i] for i in range(len(all_encoded)) if i not in val_indices]
    val_examples   = [all_encoded[i] for i in val_indices]

    # Shuffle train again
    train_order = list(range(len(train_examples)))
    rng.shuffle(train_order)
    train_examples = [train_examples[i] for i in train_order]

    print(f"\nTrain: {len(train_examples)} | Val: {len(val_examples)}")

    # ---- Write shards ----
    print("\nWriting train shards...")
    train_shards, train_total = write_split("train", train_examples, args.output_dir)
    print("\nWriting val shards...")
    val_shards, val_total = write_split("val", val_examples, args.output_dir)

    # ---- Write metadata ----
    total_elapsed = time.perf_counter() - t_total_start

    metadata = {
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "sequence_length": SEQ_LEN,
        "val_size": VAL_SIZE,
        "shard_size": SHARD_SIZE,
        "min_response_tokens": MIN_RESPONSE_TOKENS,
        "datasets_used": args.datasets,
        "per_dataset_stats": stats,
        "train": {"examples": len(train_examples), "shards": train_shards},
        "val": {"examples": len(val_examples), "shards": val_shards},
        "total_elapsed_s": round(total_elapsed, 1),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Metadata written to {meta_path}")
    print(f"Train: {len(train_examples)} examples in {train_shards} shards")
    print(f"Val:   {len(val_examples)} examples in {val_shards} shards")

    # Print per-dataset breakdown with timing
    print("\nPer-dataset breakdown:")
    print(f"  {'dataset':<20}  {'kept':>6}  {'skip_lang':>9}  {'skip_len':>8}  {'time':>7}")
    print(f"  {'-'*57}")
    for ds_name, s in stats.items():
        print(f"  {ds_name:<20}  {s['kept']:>6d}  {s['skipped_not_english']:>9d}  "
              f"{s['skipped_too_long_or_short']:>8d}  {s['elapsed_s']:>6.1f}s")
    print(f"  {'-'*57}")
    total_kept = sum(v["kept"] for v in stats.values())
    tm, ts = divmod(int(total_elapsed), 60)
    print(f"  {'TOTAL':<20}  {total_kept:>6d}  {'':>9}  {'':>8}  {tm:02d}m {ts:02d}s")


if __name__ == "__main__":
    main()