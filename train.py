"""
train.py - Teach the GPT to predict the next character.
=======================================================

The training loop, in one sentence: show the model random chunks of text,
measure how wrong its next-token predictions are (the loss), compute how each
weight should change to reduce that loss (backpropagation), and change the
weights a tiny bit in that direction (the optimizer step). Repeat thousands of
times.

Usage:
    python train.py                    # default model, ~10.8M params (wants a GPU)
    python train.py --preset tiny      # small model that trains on a CPU in minutes
    python train.py --max_iters 2000 --n_layer 4      # override any setting

Run prepare.py first so that data/train.bin, data/val.bin and data/vocab.json exist.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

from model import GPT, GPTConfig
from prepare import CharTokenizer

# -----------------------------------------------------------------------------
# Settings ("hyperparameters")
# -----------------------------------------------------------------------------
DEFAULTS = dict(
    # --- model size (see GPTConfig in model.py) ---
    n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
    # --- optimisation ---
    batch_size=64,          # how many sequences are processed in parallel per step
    max_iters=2500,         # total number of optimizer steps (val loss bottoms out around here)
    learning_rate=1e-3,     # peak step size
    min_lr=1e-4,            # learning rate at the end of the cosine decay
    warmup_iters=100,       # steps to ramp the learning rate up from 0
    weight_decay=0.1,       # pulls weights gently towards 0 (regularisation)
    grad_clip=1.0,          # cap the gradient norm to avoid rare explosive updates
    # --- bookkeeping ---
    eval_interval=250,      # how often to measure train/val loss
    eval_iters=200,         # how many batches to average when measuring loss
    out_dir="out",
    seed=1337,
    device="auto",          # "auto", "cuda", "cpu" or "mps"
)

# Presets override some defaults in one go.
PRESETS = {
    # ~0.2M parameters. Learns words and line structure on a laptop CPU.
    "tiny": dict(n_layer=4, n_head=4, n_embd=64, block_size=64, dropout=0.0,
                 batch_size=32, max_iters=2000, learning_rate=2e-3, min_lr=2e-4,
                 eval_interval=200, eval_iters=50),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train mini-GPT.")
    p.add_argument("--preset", choices=PRESETS.keys(), help="start from a named preset")
    # Every key in DEFAULTS becomes a command-line flag, e.g. --n_layer 4.
    for key, value in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(value), default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    # Explicit flags win over the preset.
    cfg.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    return cfg


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple silicon
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def get_batch(data, block_size, batch_size, device):
    """
    Pick batch_size random windows of block_size+1 tokens from the data.

    Example with block_size = 8, text "First Citizen":
        x = "First Ci"      (input)
        y = "irst Cit"      (target = input shifted by one)

    One window gives block_size training examples at once:
        after "F"         predict "i"
        after "Fi"        predict "r"
        ...
        after "First Ci"  predict "t"
    The causal mask in the model is what makes all of these predictions
    independent: position t only sees tokens 0..t.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":
        # pin_memory + non_blocking lets the copy to the GPU overlap with work.
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device, autocast):
    """
    The loss of a single batch is noisy. Average over many batches to get a
    trustworthy number. model.eval() switches dropout off; model.train()
    switches it back on.
    """
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
            with autocast:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(it, cfg):
    r"""
    Learning-rate schedule: linear warm-up, then cosine decay.

        lr
        ^   /\
        |  /   \_
        | /       \__
        |/           \______  min_lr
        +-----------------------> iteration
         warmup

    Warm-up: at the start the weights are random and gradients are large and
    erratic, so we start with tiny steps. Decay: near the end, smaller steps
    let the model settle into a good minimum instead of bouncing around it.
    """
    if it < cfg["warmup_iters"]:
        return cfg["learning_rate"] * (it + 1) / cfg["warmup_iters"]
    progress = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # goes 1 -> 0
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def main():
    # The Windows console may not be UTF-8; avoid crashes when printing text.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = parse_args()
    device = pick_device(cfg["device"])

    # Fixing the random seed makes runs reproducible (same init, same batches).
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---- data ---------------------------------------------------------------
    # np.memmap reads the file lazily from disk instead of loading it all into
    # RAM. Irrelevant for 1 MB, essential for gigabytes.
    train_data = np.memmap(os.path.join("data", "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join("data", "val.bin"), dtype=np.uint16, mode="r")
    tok = CharTokenizer.load(os.path.join("data", "vocab.json"))
    splits = {"train": train_data, "val": val_data}

    # ---- model --------------------------------------------------------------
    model_args = dict(vocab_size=tok.vocab_size, block_size=cfg["block_size"],
                      n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                      n_embd=cfg["n_embd"], dropout=cfg["dropout"])
    model = GPT(GPTConfig(**model_args)).to(device)
    print(f"Device: {device}   Parameters: {model.num_params() / 1e6:.2f}M")

    # ---- mixed precision ----------------------------------------------------
    # On a GPU we let matrix multiplies run in bfloat16 (16-bit floats): about
    # twice as fast and half the memory, with negligible accuracy loss. The
    # weights themselves stay in float32. On CPU we keep plain float32.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = torch.autocast(device_type="cpu", enabled=False)  # a no-op

    # ---- optimizer ----------------------------------------------------------
    # AdamW: Adam (keeps a running average of each weight's gradient and its
    # magnitude, giving every weight its own adaptive step size) + decoupled
    # weight decay. Weight decay is only applied to 2-D weight matrices, not to
    # biases and LayerNorm parameters, which is the GPT-2/3 convention.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.99),
    )

    # ---- sanity check -------------------------------------------------------
    # An untrained model should guess almost uniformly at random, so its loss
    # should be close to ln(vocab_size). Wider models land a little above it
    # (e.g. 4.30 instead of 4.17) because their tiny random logits are not
    # exactly zero. If it is far off (say 10, or 1), something is wrong.
    losses = estimate_loss(model, splits, cfg, device, autocast)
    print(f"Initial loss {losses['train']:.4f}  (expected about ln({tok.vocab_size}) = "
          f"{math.log(tok.vocab_size):.4f})")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    # ---- the training loop --------------------------------------------------
    for it in range(cfg["max_iters"] + 1):

        # 1. Set the learning rate for this step.
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. Every so often: measure loss on both splits and save a checkpoint.
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            losses = estimate_loss(model, splits, cfg, device, autocast)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
                  f" | lr {lr:.2e} | {time.time() - t0:.1f}s")
            # Keep the weights with the best VALIDATION loss. If train loss
            # keeps falling while val loss rises, the model is overfitting
            # (memorising the training text) and later checkpoints are worse.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "model_args": model_args,
                    # Store the vocabulary too, so the checkpoint still decodes
                    # correctly after prepare.py is run on a different text.
                    "chars": tok.chars,
                    "iter": it,
                    "val_loss": best_val,
                    "train_config": cfg,
                }, os.path.join(cfg["out_dir"], "ckpt.pt"))
        if it == cfg["max_iters"]:
            break

        # 3. Forward pass: run the model on a batch and compute the loss.
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        with autocast:
            _, loss = model(x, y)

        # 4. Backward pass: compute d(loss)/d(weight) for every weight.
        #    set_to_none=True clears the previous step's gradients (cheaper
        #    than filling them with zeros). Gradients ACCUMULATE in PyTorch, so
        #    forgetting this line is a classic bug.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 5. Gradient clipping: if the overall gradient is unusually large,
        #    scale it down so one bad batch cannot wreck the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        # 6. Update the weights.
        optimizer.step()

    print(f"Done. Best val loss {best_val:.4f}, checkpoint in {cfg['out_dir']}/ckpt.pt")

    # A quick taste of what the model learned.
    model.eval()
    start = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
    print("-" * 60)
    print(tok.decode(model.generate(start, 300, temperature=0.8, top_k=40)[0].tolist()))


if __name__ == "__main__":
    main()
