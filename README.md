# mini-GPT

A small, heavily commented GPT in PyTorch, built for learning how GPT works.
It trains on Shakespeare (or any text file) at the character level, and a
10.7M-parameter model trains in about 3 minutes on a modern GPU.

Start with **[WALKTHROUGH.md](WALKTHROUGH.md)**. It explains every part of the
model step by step and covers how to train it and how to use it.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install numpy

.venv\Scripts\python prepare.py          # download and encode the data
.venv\Scripts\python train.py            # train (add --preset tiny on a CPU)
.venv\Scripts\python sample.py --prompt "ROMEO:"
```

| File | What it does |
|---|---|
| `prepare.py` | Character tokenizer and data preparation |
| `model.py` | The GPT model: embeddings, causal self-attention, MLP, blocks, generation |
| `train.py` | The training loop |
| `sample.py` | Text generation from a trained checkpoint |
