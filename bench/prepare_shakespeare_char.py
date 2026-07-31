"""Fetch the tinyshakespeare corpus -- run on the cluster, not the laptop.

Downloads tinyshakespeare (~1 MB), builds train.bin / val.bin (uint16) and
meta.pkl with vocab_size=65. Levels L7 and L11 of bench/profile_suite.py need
this for their held-out cross-entropy and next-token accuracy columns; without
it they fall back to synthetic tokens, where the loss is uninformative because
there is nothing to learn.

Usage, from the repo root:

    export PYTHONPATH="$(pwd):$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
    python bench/prepare_shakespeare_char.py

Note the vocabulary is 65 symbols. Running a level at a large --V trains a
50257-wide head on 65 of them, which is fine for measuring engine cost and
meaningless as a language-modelling result.

Uses stdlib urllib only (no requests).
"""

from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path

import numpy as np

# Repo root = parent of bench/; bins land where Batcher looks by default.
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "shakespeare_char"
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_path = OUT / "input.txt"

    if not input_path.exists():
        print(f"Downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        input_path.write_text(text, encoding="utf-8")
    else:
        print(f"Using existing {input_path}")

    data = input_path.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(set(data))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")
    if vocab_size != 65:
        raise SystemExit(
            f"Expected vocab_size=65 for this bench (got {vocab_size}). "
            "Do not proceed — Batcher asserts data.max() < V with V=65."
        )

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    n = len(data)
    train_ids = np.array(encode(data[: int(n * 0.9)]), dtype=np.uint16)
    val_ids = np.array(encode(data[int(n * 0.9) :]), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids.tofile(OUT / "train.bin")
    val_ids.tofile(OUT / "val.bin")
    with open(OUT / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

    print(f"Wrote {OUT / 'train.bin'}, {OUT / 'val.bin'}, {OUT / 'meta.pkl'}")


if __name__ == "__main__":
    main()
