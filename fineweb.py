"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py                 # downloads 10BT (default)
$ python fineweb.py --size 100BT    # downloads 100BT
$ python fineweb.py --test           # tiny dataset for testing (e.g. TensorBoard)
Will save shards to the "edu_fineweb" directory.
"""

import argparse
import os

# Fixing of the HuggingFace issue with slow dataset downloading 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable legacy fast transfer
os.environ["HF_HUB_DISABLE_XET"] = "1"         # Disable Xet backend (v1.0+)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes (default is 10s)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"       # Metadata timeout


import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# Try to load .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# ------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Download a tiny dataset subset for quick testing")
parser.add_argument("--size", type=str, default="10BT", choices=["10BT", "100BT"], help="Dataset size to download: 10BT or 100BT (default: 10BT)")
args, _ = parser.parse_known_args()

shard_size = int(1e8) # 100M tokens per shard

local_dir = "edu_fineweb"

if args.test:
    remote_name = None
    print("TEST MODE: downloading tiny dataset subset")
else:
    remote_name = f"sample-{args.size}"
    print(f"Downloading FineWeb-Edu {args.size} dataset")

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer (at module level so worker processes can access it)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def download_test_dataset():
    """Download tiny Shakespeare and save as shards in the same format as FineWeb."""
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading tiny Shakespeare dataset...")
    text = urllib.request.urlopen(url).read().decode('utf-8')
    tokens = enc.encode_ordinary(text)
    tokens = np.array([eot] + tokens, dtype=np.uint16)
    print(f"Tokenized: {len(tokens)} tokens")

    # Split 90/10 into train/val and save as shards
    split_idx = int(len(tokens) * 0.9)
    for split, data in [("val", tokens[split_idx:]), ("train", tokens[:split_idx])]:
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_000000")
        write_datafile(filename, data)
        print(f"Saved {split} shard: {len(data)} tokens -> {filename}.npy")


# Guard for multiprocessing on macOS (spawn method)
if __name__ == '__main__':

    # Test mode: use tiny Shakespeare instead of FineWeb
    if args.test:
        download_test_dataset()
        exit()

    # Get HuggingFace token from environment variable if available
    # This enables faster downloads for authenticated users
    hf_token = os.environ.get('HF_TOKEN', None)
    if hf_token:
        print(f"Using HuggingFace token for authenticated download (faster)")
    else:
        print("No HF_TOKEN found. Downloads may be slower. Set HF_TOKEN env variable for faster downloads.")

    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", token=hf_token, num_proc=8) # Parallel processing after download

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()) # use all CPU cores
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
