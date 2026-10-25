"""
prepare.py - Turn raw text into numbers the model can learn from.
=================================================================

A neural network cannot read letters. It only understands numbers (tensors).
So before training we must:

  1. Get some text (by default: the complete works of Shakespeare, ~1 MB).
  2. Build a *vocabulary*: the list of every distinct "token" that appears.
     Here a token is simply ONE CHARACTER. This is called a character-level
     tokenizer. Real GPTs use sub-word tokens (BPE, see WALKTHROUGH.md), but
     characters make every step visible and need no extra libraries.
  3. Give every token an integer id:  'a' -> 39, 'b' -> 40, ...
  4. Convert the whole text into one long list of ids ("encoding").
  5. Split it into a training part (90%) and a validation part (10%).
     The model learns ONLY from the training part. We measure the loss on
     the validation part to check that it generalises instead of memorising.
  6. Save everything to the data/ folder so train.py can load it quickly.

Usage:
    python prepare.py                    # download and use tiny-shakespeare
    python prepare.py --input my.txt     # use your own text file instead
"""

import argparse
import json
import os
import urllib.request

import numpy as np

# Andrej Karpathy's "tiny shakespeare" dataset: ~40,000 lines of plays.
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_DIR = "data"


class CharTokenizer:
    """
    The simplest possible tokenizer: one character = one token.

    stoi = "string to integer" lookup table   {'a': 39, 'b': 40, ...}
    itos = "integer to string" lookup table   {39: 'a', 40: 'b', ...}

    encode("hi")      -> [46, 47]
    decode([46, 47])  -> "hi"
    """

    def __init__(self, chars):
        # Sort so the mapping is deterministic (same text -> same ids every time).
        self.chars = sorted(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        # set(text) gives every distinct character that occurs in the text.
        return cls(set(text))

    def encode(self, s):
        # Characters never seen during prepare.py have no id; we silently skip
        # them so that a prompt with e.g. an emoji does not crash sample.py.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        # JSON (not pickle) so you can open the file and look at the vocabulary.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


def main():
    parser = argparse.ArgumentParser(description="Prepare a text dataset for mini-GPT.")
    parser.add_argument("--input", help="path to a UTF-8 text file (default: tiny-shakespeare)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="fraction of the text kept aside for validation")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- 1. Get the raw text -------------------------------------------------
    if args.input:
        path = args.input
    else:
        path = os.path.join(DATA_DIR, "input.txt")
        if not os.path.exists(path):
            print(f"Downloading tiny-shakespeare to {path} ...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)

    # Always say encoding="utf-8": on Windows the default is a legacy code page
    # that can mangle or reject non-English characters.
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {path}")

    # ---- 2+3. Build the vocabulary ------------------------------------------
    tok = CharTokenizer.from_text(text)
    print(f"Vocabulary size: {tok.vocab_size} distinct characters")
    print("Vocabulary:", repr("".join(tok.chars)))

    # ---- 4. Encode the entire text into integers ----------------------------
    # uint16 holds ids up to 65,535 - plenty for a character vocabulary - and
    # uses half the disk/RAM of int32.
    ids = np.array(tok.encode(text), dtype=np.uint16)

    # Show a tiny example so you can see what encoding does.
    sample = text[:20]
    print(f"Example: {sample!r} -> {tok.encode(sample)}")

    # ---- 5. Train / validation split ----------------------------------------
    # We split at one point (not randomly) so the validation text is a
    # contiguous chunk the model never sees during training.
    n_val = int(len(ids) * args.val_fraction)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"Train tokens: {len(train_ids):,}   Val tokens: {len(val_ids):,}")

    # ---- 6. Save -------------------------------------------------------------
    # .bin = raw bytes of the array; train.py reads it back with np.memmap.
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    tok.save(os.path.join(DATA_DIR, "vocab.json"))
    print(f"Saved train.bin, val.bin and vocab.json to {DATA_DIR}/")


if __name__ == "__main__":
    main()
