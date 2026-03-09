"""
Analyze the prepared SFT shards in data/sft_mix/.

Reads token + mask .npy shards, decodes tokens back to text using the GPT-2
tokenizer, and produces:

  data/sft_mix/analysis_train.csv   – full pandas-compatible CSV  (train split)
  data/sft_mix/analysis_val.csv     – full pandas-compatible CSV  (val split)
  data/sft_mix/analysis_preview.txt – human-readable text preview

Usage:
    python analyze_sft_data.py                # analyse all train + val shards
    python analyze_sft_data.py --max 500      # only first N examples per split
    python analyze_sft_data.py --split val    # only val split
"""

import argparse
import json
import os
import re
import numpy as np
import tiktoken
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "sft_mix"

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


# ---------------------------------------------------------------------------

def decode_tokens(token_ids: np.ndarray) -> str:
    """Decode a 1-D uint16 token array back to a UTF-8 string, ignoring EOT padding."""
    ids = token_ids.tolist()
    # Strip trailing EOT padding
    while ids and ids[-1] == eot:
        ids.pop()
    return enc.decode(ids)


def parse_example(text: str) -> dict[str, str]:
    """
    Parse an Alpaca-style decoded string into its three logical fields:
      instruction, input (optional, empty string if absent), response.

    Template variants handled:
      ### Instruction: ... ### Input: ... ### Response: ...
      ### Instruction: ... ### Response: ...
    """
    # ---- instruction ----
    inst_m = re.search(r"### Instruction:\n(.*?)(?=### (?:Input|Response):)", text, re.DOTALL)
    instruction = inst_m.group(1).strip() if inst_m else text.strip()

    # ---- optional input / context ----
    inp_m = re.search(r"### Input:\n(.*?)### Response:", text, re.DOTALL)
    context = inp_m.group(1).strip() if inp_m else ""

    # ---- response ----
    resp_m = re.search(r"### Response:\n(.*)", text, re.DOTALL)
    response = resp_m.group(1).strip() if resp_m else ""

    return {"instruction": instruction, "input": context, "response": response}


def split_prompt_response(text: str):
    """Return (prompt_text, response_text) — kept for token-count helpers."""
    marker = "### Response:\n"
    idx = text.find(marker)
    if idx == -1:
        return text, ""
    split = idx + len(marker)
    return text[:split], text[split:]


def token_count(token_ids: np.ndarray) -> int:
    """Count non-padding tokens."""
    ids = token_ids.tolist()
    n = len(ids)
    while n > 0 and ids[n - 1] == eot:
        n -= 1
    return n


def prompt_token_count(mask: np.ndarray) -> int:
    """Tokens where mask == 0 (before response start), excluding padding."""
    # First '1' in mask marks start of response
    ones = np.where(mask == 1)[0]
    return int(ones[0]) if len(ones) else int(np.sum(mask == 0))


def response_token_count(mask: np.ndarray) -> int:
    return int(np.sum(mask == 1))


# ---------------------------------------------------------------------------

def load_shards(split: str, max_examples: int | None):
    """Yield (tokens_row, mask_row, source) for every example in the split.
    Falls back to 'unknown' when no *_sources_* shard exists (older shards)."""
    shard_idx = 0
    seen = 0
    while True:
        t_path = DATA_DIR / f"{split}_tokens_{shard_idx:03d}.npy"
        m_path = DATA_DIR / f"{split}_masks_{shard_idx:03d}.npy"
        s_path = DATA_DIR / f"{split}_sources_{shard_idx:03d}.npy"
        if not t_path.exists():
            break
        tokens  = np.load(t_path)                                      # (N, SEQ_LEN)
        masks   = np.load(m_path)                                      # (N, SEQ_LEN)
        sources = np.load(s_path) if s_path.exists() else np.array(   # (N,)
            ["unknown"] * len(tokens), dtype="U30"
        )
        for t_row, m_row, src in zip(tokens, masks, sources):
            if max_examples and seen >= max_examples:
                return
            yield t_row, m_row, str(src)
            seen += 1
        shard_idx += 1


def build_dataframe(split: str, max_examples: int | None) -> pd.DataFrame:
    rows = []
    for i, (t_row, m_row, source) in enumerate(load_shards(split, max_examples)):
        full_text = decode_tokens(t_row)
        fields    = parse_example(full_text)

        rows.append({
            "index":           i,
            "source":          source,
            "instruction":     fields["instruction"],
            "input":           fields["input"],
            "response":        fields["response"],
            "total_tokens":    token_count(t_row),
            "prompt_tokens":   prompt_token_count(m_row),
            "response_tokens": response_token_count(m_row),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def sample_one_per_dataset(df: pd.DataFrame, n_extra: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return one random row per unique source, then pad with random rows up to n_extra total."""
    if df.empty:
        return df
    sources = df["source"].unique()
    picked_idx: list[int] = []
    for src in sorted(sources):
        rows = df[df["source"] == src]
        picked_idx.append(int(rng.choice(rows.index)))
    # Fill remaining slots with random rows not already picked
    remaining = df.drop(index=picked_idx)
    extra_needed = max(0, n_extra - len(picked_idx))
    if extra_needed and not remaining.empty:
        extra = remaining.sample(n=min(extra_needed, len(remaining)),
                                 random_state=42)
        picked_idx.extend(extra.index.tolist())
    return df.loc[picked_idx].reset_index(drop=True)


def write_txt_preview(df_train: pd.DataFrame, df_val: pd.DataFrame,
                      n: int, out_path: Path):
    sep  = "=" * 80
    sep2 = "-" * 80
    rng  = np.random.default_rng(42)

    with open(out_path, "w", encoding="utf-8") as f:
        # ---- metadata summary ----
        meta_path = DATA_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as mf:
                meta = json.load(mf)
            f.write(f"{sep}\n  DATASET METADATA\n{sep}\n")
            f.write(f"  Block size       : {meta['block_size']}\n")
            f.write(f"  Sequence length  : {meta['sequence_length']}\n")
            f.write(f"  Shard size       : {meta['shard_size']}\n")
            f.write(f"  Min response tok : {meta['min_response_tokens']}\n")
            f.write(f"  Train examples   : {meta['train']['examples']}  "
                    f"({meta['train']['shards']} shards)\n")
            f.write(f"  Val examples     : {meta['val']['examples']}  "
                    f"({meta['val']['shards']} shards)\n\n")

            f.write(f"  Per-dataset breakdown:\n")
            f.write(f"  {'Dataset':<22} {'kept':>7} {'skip_lang':>10} {'skip_len':>10}\n")
            f.write(f"  {'-'*52}\n")
            for ds, s in meta["per_dataset_stats"].items():
                f.write(f"  {ds:<22} {s['kept']:>7,}  "
                        f"{s['skipped_not_english']:>9,}  "
                        f"{s['skipped_too_long_or_short']:>9,}\n")
            f.write("\n")

        # ---- aggregate stats ----
        for label, df in [("TRAIN", df_train), ("VAL", df_val)]:
            if df.empty:
                continue
            f.write(f"{sep}\n  {label} SPLIT — TOKEN STATISTICS ({len(df):,} examples)\n{sep}\n")
            for col in ["total_tokens", "prompt_tokens", "response_tokens"]:
                s = df[col]
                f.write(f"  {col:<20}  "
                        f"mean={s.mean():6.1f}  "
                        f"median={s.median():6.1f}  "
                        f"min={s.min():4d}  "
                        f"max={s.max():4d}\n")
            f.write("\n")
            # per-source counts
            if "source" in df.columns and df["source"].nunique() > 1:
                f.write(f"  Examples per source dataset (in this sample):\n")
                for src, cnt in df["source"].value_counts().items():
                    f.write(f"    {src:<22} {cnt:>6,}\n")
                f.write("\n")

        # ---- sample examples (one per dataset + extras) ----
        for label, df in [("TRAIN", df_train), ("VAL", df_val)]:
            if df.empty:
                continue
            sample = sample_one_per_dataset(df, n_extra=n, rng=rng)
            f.write(f"{sep}\n  {label} — {len(sample)} EXAMPLES (one per source dataset)\n{sep}\n\n")
            for _, row in sample.iterrows():
                source_tag = f"[{row.get('source', 'unknown')}]"
                f.write(f"{source_tag}  #{row['index']}  "
                        f"total={row['total_tokens']} tok  "
                        f"(prompt={row['prompt_tokens']} | response={row['response_tokens']})\n")
                f.write(f"{sep2}\n")
                f.write(f"INSTRUCTION:\n{row['instruction']}\n\n")
                if row.get("input"):
                    f.write(f"INPUT:\n{row['input']}\n\n")
                f.write(f"RESPONSE:\n{row['response']}\n")
                f.write(f"{sep}\n\n")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse SFT shards")
    parser.add_argument("--max",    type=int, default=None,
                        help="Max examples to load per split (default: all)")
    parser.add_argument("--split",  choices=["train", "val", "both"], default="both")
    parser.add_argument("--preview-n", type=int, default=10,
                        help="Number of examples to include in the .txt preview (default: 10)")
    args = parser.parse_args()

    load_train = args.split in ("train", "both")
    load_val   = args.split in ("val",   "both")

    df_train = pd.DataFrame()
    df_val   = pd.DataFrame()

    if load_train:
        print(f"Loading train shards (max={args.max or 'all'}) …")
        df_train = build_dataframe("train", args.max)
        out = DATA_DIR / "analysis_train.csv"
        df_train.to_csv(out, index=False)
        print(f"  → {len(df_train):,} examples  saved to {out}")

    if load_val:
        print(f"Loading val shards (max={args.max or 'all'}) …")
        df_val = build_dataframe("val", args.max)
        out = DATA_DIR / "analysis_val.csv"
        df_val.to_csv(out, index=False)
        print(f"  → {len(df_val):,} examples  saved to {out}")

    txt_out = DATA_DIR / "analysis_preview.txt"
    print(f"Writing text preview ({args.preview_n} examples/split) → {txt_out}")
    write_txt_preview(df_train, df_val, args.preview_n, txt_out)

    print("\nDone.")
    if load_train and not df_train.empty:
        print("\n--- Train token stats ---")
        print(df_train[["total_tokens","prompt_tokens","response_tokens"]].describe().round(1).to_string())
    if load_val and not df_val.empty:
        print("\n--- Val token stats ---")
        print(df_val[["total_tokens","prompt_tokens","response_tokens"]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
