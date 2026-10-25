"""
sample.py - Use a trained model to generate text.
=================================================

Loads out/ckpt.pt (written by train.py, which also stores the vocabulary
the model was trained with), encodes your prompt into token ids, lets the model continue it,
and decodes the ids back into text.

Usage:
    python sample.py
    python sample.py --prompt "ROMEO:" --max_new_tokens 500
    python sample.py --temperature 0.5 --top_k 10      # safer, more repetitive
    python sample.py --temperature 1.2                 # wilder
    python sample.py --num_samples 3 --seed 7
"""

import argparse
import os
import sys

import torch

from model import GPT, GPTConfig
from prepare import CharTokenizer


def main():
    p = argparse.ArgumentParser(description="Generate text with a trained mini-GPT.")
    p.add_argument("--prompt", default="\n", help="text the model should continue")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40, help="0 = no top-k filtering")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ckpt", default=os.path.join("out", "ckpt.pt"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # Generated text may contain characters the Windows console code page
    # cannot show; force UTF-8 so printing never crashes.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- rebuild the model exactly as it was during training ----------------
    # The checkpoint stores the architecture (model_args) next to the weights,
    # so we can construct an identical empty model and pour the weights in.
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()  # turn dropout off: we want the model's real predictions
    print(f"Loaded {args.ckpt} (step {ckpt['iter']}, val loss {ckpt['val_loss']:.4f})",
          file=sys.stderr)

    # Use the vocabulary saved inside the checkpoint: ids only mean something
    # relative to the vocabulary the model was trained with. (Older
    # checkpoints without it fall back to data/vocab.json.)
    if "chars" in ckpt:
        tok = CharTokenizer(ckpt["chars"])
    else:
        tok = CharTokenizer.load(os.path.join("data", "vocab.json"))

    # ---- text -> ids -> model -> ids -> text --------------------------------
    ids = tok.encode(args.prompt)
    if not ids:
        sys.exit("The prompt contains no characters from the training vocabulary.")
    # The model expects a batch dimension: shape (1, T).
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                           top_k=args.top_k or None)
        print(tok.decode(y[0].tolist()))
        if i < args.num_samples - 1:
            print("-" * 60)


if __name__ == "__main__":
    main()
