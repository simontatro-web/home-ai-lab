"""Turn a folder of .txt files into token files for training.

    # cleaned corpus (normalise whitespace, drop tiny docs, exact dedup)
    python training/prepare_data.py --src corpus/ --out data/clean --train-tokenizer

    # raw corpus with the same tokenizer and the SAME validation set as clean,
    # so losses from the two datasets can be compared fairly
    python training/prepare_data.py --src corpus/ --out data/raw --no-clean --val-from data/clean

Each .txt file is one document. Writes into --out:
    train.bin, val.bin   uint16 token ids, documents separated by <|endoftext|>
    val_hashes.json      fingerprints of the validation documents
    meta.json            vocab size, token counts, cleaning stats
The tokenizer is saved to / loaded from --tokenizer (default model/tokenizer.json).
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import BPETokenizer  # noqa: E402


def clean(doc):
    doc = doc.replace("\r\n", "\n")
    doc = re.sub(r"[ \t]+", " ", doc)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    return doc.strip()


def fingerprint(doc):
    return hashlib.sha1(clean(doc).lower().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of .txt documents (searched recursively)")
    ap.add_argument("--out", required=True, help="output folder, e.g. data/clean")
    ap.add_argument("--tokenizer", default="model/tokenizer.json")
    ap.add_argument("--train-tokenizer", action="store_true", help="train a new tokenizer first")
    ap.add_argument("--vocab-size", type=int, default=50_000)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=100)
    ap.add_argument("--no-clean", action="store_true", help="skip cleaning and dedup (raw baseline)")
    ap.add_argument("--val-from", help="reuse this dataset's val.bin and keep its documents out of training")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(
        os.path.join(root, f) for root, _, files in os.walk(a.src) for f in files if f.endswith(".txt")
    )
    if not paths:
        sys.exit(f"no .txt files found under {a.src}")

    held_out = set()
    if a.val_from:
        with open(os.path.join(a.val_from, "val_hashes.json")) as f:
            held_out = set(json.load(f))

    docs, seen = [], set()
    stats = {"files": len(paths), "too_short": 0, "duplicates": 0, "held_out_for_val": 0}
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            doc = f.read()
        if fingerprint(doc) in held_out:
            stats["held_out_for_val"] += 1
            continue
        if not a.no_clean:
            doc = clean(doc)
            if len(doc) < a.min_chars:
                stats["too_short"] += 1
                continue
            # dedup AFTER cleaning, so formatting differences don't hide copies
            h = fingerprint(doc)
            if h in seen:
                stats["duplicates"] += 1
                continue
            seen.add(h)
        docs.append(doc)
    stats["kept"] = len(docs)
    print(f"documents: {stats}")

    random.Random(a.seed).shuffle(docs)

    if a.train_tokenizer:
        budget, sample = int(a.tokenizer_sample_mb * 1e6), []
        for d in docs:
            sample.append(d)
            budget -= len(d)
            if budget <= 0:
                break
        print(f"training tokenizer (vocab {a.vocab_size}) on {sum(map(len, sample)) / 1e6:.1f} MB ...")
        tok = BPETokenizer.train("\n\n".join(sample), vocab_size=a.vocab_size - 1, verbose=True)
        os.makedirs(os.path.dirname(a.tokenizer) or ".", exist_ok=True)
        tok.save(a.tokenizer)
    else:
        tok = BPETokenizer.load(a.tokenizer)

    eot = tok.vocab_size  # one extra id after the BPE vocab marks document boundaries
    vocab_size = eot + 1
    assert vocab_size <= 65_536, "uint16 token files need vocab_size <= 65536"

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    if a.val_from:
        with open(os.path.join(a.val_from, "meta.json")) as f:
            if json.load(f)["vocab_size"] != vocab_size:
                sys.exit("--val-from dataset used a different tokenizer")
        shutil.copyfile(os.path.join(a.val_from, "val.bin"), os.path.join(a.out, "val.bin"))
        shutil.copyfile(os.path.join(a.val_from, "val_hashes.json"), os.path.join(a.out, "val_hashes.json"))
        counts["val"] = os.path.getsize(os.path.join(a.out, "val.bin")) // 2
        splits = {"train": docs}
    else:
        n_val = max(1, int(len(docs) * a.val_frac))
        splits = {"val": docs[:n_val], "train": docs[n_val:]}
        with open(os.path.join(a.out, "val_hashes.json"), "w") as f:
            json.dump([fingerprint(d) for d in splits["val"]], f)
    for name, split_docs in splits.items():
        ids = []
        for d in split_docs:
            ids.extend(tok.encode(d))
            ids.append(eot)
        np.array(ids, dtype=np.uint16).tofile(os.path.join(a.out, f"{name}.bin"))
        counts[name] = len(ids)
        print(f"{name}: {len(split_docs)} docs, {len(ids):,} tokens")

    meta = {"vocab_size": vocab_size, "eot_id": eot, "tokenizer": a.tokenizer,
            "cleaned": not a.no_clean, "tokens": counts, "stats": stats}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
