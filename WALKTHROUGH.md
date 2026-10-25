# mini-GPT: a walkthrough

This project is a complete, working GPT: the model itself is about 330 lines of
heavily commented PyTorch (about 830 lines across four files). It trains on Shakespeare (or any text you give it) and learns to write
text in the same style. Architecturally it is the same thing as GPT-2. It is
just much smaller and works on characters instead of sub-words.

Read this file top to bottom with the source code open next to it.

| File | What it does |
|---|---|
| [prepare.py](prepare.py) | Downloads text, builds the character vocabulary, encodes the text into integers |
| [model.py](model.py) | The GPT itself: embeddings, attention, MLP, blocks, generation |
| [train.py](train.py) | The training loop: batches, loss, backprop, AdamW, LR schedule, checkpoints |
| [sample.py](sample.py) | Loads a trained checkpoint and generates text from a prompt |

---

## Part 1: Quick start

All commands are run from the project folder. They use the virtual
environment's Python directly (`.venv\Scripts\python`), so you never need to
"activate" it. PowerShell often blocks `Activate.ps1` anyway.

### 1. Set up (one time)

The `.venv` folder already exists with PyTorch 2.11 (CUDA 12.8) and NumPy
installed. To recreate it from scratch:

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```powershell
.venv\Scripts\python -m pip install numpy
```

Check that the GPU is visible:

```powershell
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Prepare the data

```powershell
.venv\Scripts\python prepare.py
```

This downloads tiny-shakespeare (1.1 million characters) into `data/` and
writes `train.bin`, `val.bin` and `vocab.json`.

### 3. Train

```powershell
.venv\Scripts\python train.py
```

On an RTX 5080 the default 10.7M-parameter model trains 2500 steps in about
3 minutes and reaches a validation loss of about 1.46. If you have no GPU, use
the tiny preset (0.2M parameters, about 30 seconds on a CPU, val loss about
1.8):

```powershell
.venv\Scripts\python train.py --preset tiny
```

### 4. Generate

```powershell
.venv\Scripts\python sample.py --prompt "ROMEO:" --max_new_tokens 500
```

---

## Part 2: The big idea

A GPT is a machine that answers exactly one question:

> Given the text so far, what is the probability of each possible next token?

Everything else is built on top of that question.

- **Training:** take real text, ask the model to predict each next token, and
  measure how much probability it gave to the token that actually came next.
  Then adjust the weights so that probability goes up. Do this millions of
  times.
- **Generation:** give it a prompt, get the probabilities for the next token,
  pick one at random according to those probabilities, append it, and repeat.

ChatGPT, Claude and the rest are this same idea scaled up enormously (billions
of parameters, trillions of tokens), followed by extra training to make them
follow instructions. The core you are about to read is the same.

"GPT" stands for:

- **Generative:** it generates text.
- **Pre-trained:** it learns from raw text with no labels. The text is its own
  label, because the "answer" for each position is simply the next character.
- **Transformer:** the neural-network architecture, built from attention and
  MLP blocks.

---

## Part 3: Step by step through the code

### Step 1: Tokenization ([prepare.py](prepare.py))

Neural networks work on numbers, so text has to become numbers first.

We use the simplest possible tokenizer: every distinct character gets an id.
Shakespeare contains 65 distinct characters:

```
"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
```

So `'\n'` is 0, `' '` is 1, `'A'` is 13, `'a'` is 39, and so on.
`prepare.py` prints an example:

```
'First Citizen:\nBefor' -> [18, 47, 56, 57, 58, 1, 15, 47, 58, ...]
```

The entire text becomes one long array of 1.1 million integers. The first 90%
is used for training. The last 10% is the **validation set**: text the model
never trains on, which we use to check whether it has learned general patterns
or just memorized the training text.

> **How real GPTs differ.** GPT-2/3/4 use *Byte-Pair Encoding* (BPE). It starts
> from bytes and repeatedly merges the most frequent adjacent pair into a new
> token, so common words like `" the"` become a single token and rare words are
> split into pieces. The vocabulary has about 50,000 to 200,000 tokens. Each
> token then carries more meaning, so the model can see more text within the
> same context length. The model code does not change at all; only
> `vocab_size` gets bigger. To try it, the `tiktoken` library gives you GPT-2's
> tokenizer in two lines.

### Step 2: Batches ([train.py](train.py) `get_batch`)

We never feed the whole text at once. Each training step picks 64 random
windows of `block_size + 1` = 257 characters. From each window:

```
x = window[0:256]     the input
y = window[1:257]     the target: the same text shifted left by one
```

With a tiny `block_size` of 8 and the text "First Citizen":

```
x = F  i  r  s  t  ␣  C  i
y = i  r  s  t  ␣  C  i  t
```

That one window holds **8 training examples**: after "F" predict "i", after
"Fi" predict "r", ..., after "First Ci" predict "t". The model makes all 8
predictions in a single forward pass. The causal mask (Step 5) makes sure each
prediction only uses the characters before it.

Shapes: `x` and `y` are both `(B, T)` = `(64, 256)` integers.

### Step 3: Embeddings ([model.py](model.py) `GPT.__init__` / `forward`)

An id like 47 means nothing on its own. The **token embedding** is a table
with one row of 384 learned numbers per token. Looking up id 47 returns row
47, so each character becomes a 384-dimensional vector. During training these
vectors move so that characters that behave similarly end up close together
(for example, all the vowels).

Attention (Step 5) has no built-in sense of order: it compares vectors by
content, not by where they sit. (The causal mask leaks a little positional
information, because position 0 sees 1 token and position 100 sees 101, but
that is a weak, indirect signal.) The **position embedding** gives the model
order explicitly: it is a second table with one learned vector per position
(0 to 255), and it is *added* to the token vector.

```python
x = tok_emb(idx) + pos_emb(arange(T))     # (B, T, C) = (64, 256, 384)
```

From here on, every position carries a vector of `C = n_embd = 384` numbers.
This stream of vectors is called the **residual stream**.

### Step 4: The Transformer block ([model.py](model.py) `Block`)

The model is 6 identical blocks stacked on top of each other. Each one does:

```python
x = x + attention(layernorm(x))    # tokens exchange information
x = x + mlp(layernorm(x))          # each token processes what it gathered
```

Three ideas are packed into those two lines.

- **Residual connections (`x + ...`).** Each layer does not replace `x`; it
  adds a small correction to it. Picture the residual stream as a shared
  notepad that each layer reads and writes notes on. This also gives the
  gradient a direct path back to the first layer, which is why deep
  transformers can be trained at all.
- **LayerNorm.** It rescales each vector to mean 0 and variance 1, followed by
  a learned scale and shift. This keeps the numbers in a stable range layer
  after layer. Putting it *before* each sub-layer ("pre-norm", as GPT-2 does)
  makes training more stable.
- **Communicate, then compute.** Attention mixes information *across*
  positions. The MLP works on *each position separately*. Alternating the two
  is the whole Transformer.

### Step 5: Self-attention, the heart of it ([model.py](model.py) `CausalSelfAttention`)

This is the most important part. Here it is one step at a time, with shapes
(B=64, T=256, C=384, 6 heads of size 64).

**5a. Queries, keys, values.** Each position's vector is projected (by learned
linear layers) into three new vectors:

- **query** `q`: what am I looking for?
- **key** `k`: what do I contain?
- **value** `v`: what will I pass along if someone attends to me?

```python
q, k, v = self.qkv(x).split(C, dim=2)            # each (B, T, C)
```

**5b. Split into heads.** The 384 channels are cut into 6 heads of 64. Each
head does its own independent attention, so one head might learn "look at the
previous character" while another learns "look at the start of the current
word".

```python
q = q.view(B, T, 6, 64).transpose(1, 2)          # (B, 6, T, 64)
```

**5c. Scores.** Every query is compared with every key using a dot product.
If they point in the same direction, the score is large: "I found what I was
looking for."

```python
att = q @ k.transpose(-2, -1) / sqrt(64)         # (B, 6, T, T)
```

`att[b, h, i, j]` = how relevant position j is to position i, according to
head h. Dividing by `sqrt(head_size)` keeps the scores from growing with the
vector size. Without it, the softmax in 5e would become nearly one-hot and its
gradients would vanish.

**5d. Causal mask.** Position i must not see positions after i, because those
are exactly what it is trying to predict. We set those scores to minus
infinity:

```
         key j →
query i  0    -inf -inf -inf
   ↓     0    0    -inf -inf
         0    0    0    -inf
         0    0    0    0
```

**5e. Softmax.** Each row turns into positive weights that sum to 1, and the
minus-infinity entries become exactly 0. Now each position has a recipe for
how much of each earlier position to take.

**5f. Weighted sum of values.**

```python
y = att @ v                                      # (B, 6, T, 64)
```

Each position's output is a blend of the value vectors of the positions it
attended to.

**5g. Merge the heads and project.** Concatenate the 6 heads back into 384
channels and pass them through one more linear layer (`proj`) that mixes the
heads together.

That's all there is to attention. In production code, steps 5c to 5f are
replaced by a single call, `F.scaled_dot_product_attention(q, k, v,
is_causal=True)`. It computes the same thing with a fused "FlashAttention"
kernel that never stores the T×T matrix. Try swapping it in as an exercise and
check that the loss stays the same.

### Step 6: The MLP ([model.py](model.py) `MLP`)

```python
x = Linear(384 → 1536) → GELU → Linear(1536 → 384)
```

This is applied to each position on its own. It widens the vector 4×, applies
a non-linearity (GELU, a smooth ReLU), and narrows it back. Without the
non-linearity, the whole network would collapse into one big linear function.
The MLPs hold about two-thirds of the parameters and are thought to store much
of the model's "knowledge".

### Step 7: From vectors to predictions ([model.py](model.py) `forward`)

After the last block comes one final LayerNorm, and then `lm_head`, a linear
layer from 384 to 65, which produces **logits**: one score per possible next
character, at every position.

```python
logits = lm_head(ln_f(x))                        # (B, T, 65)
```

**Weight tying:** `lm_head` shares its weight matrix with the token
embedding. The vector that *represents* "e" on the way in is the same vector
that *predicts* "e" on the way out.

### Step 8: The loss ([model.py](model.py) `forward`)

Softmax turns logits into probabilities. **Cross-entropy loss** is
`-log(probability given to the correct next token)`, averaged over all
64 × 256 positions.

| Model's probability for the right answer | Loss |
|---|---|
| 1.0 (certain and correct) | 0.00 |
| 0.5 | 0.69 |
| 1/65 (random guess) | 4.17 |
| 0.001 | 6.91 |

That gives you a built-in sanity check. At initialization all weights are
small, so every logit is about 0, the distribution is uniform, and the loss
should be about **ln(65) ≈ 4.17**. `train.py` prints this at startup. For the
tiny model:

```
Initial loss 4.1783  (expected about ln(65) = 4.1744)
```

For the default model it prints 4.30. That is still fine. Each logit is a sum
of 384 small random products, so its spread is about 0.02·√384 ≈ 0.4 rather
than exactly 0, and that adds about 0.4²/2 ≈ 0.08 to the loss. If the number
were 10 or 0.5, you would know there is a bug before wasting any training
time.

### Step 9: Learning ([train.py](train.py) training loop)

Each step does four things:

```python
_, loss = model(x, y)               # 1. forward: compute predictions and loss
optimizer.zero_grad()               # 2. clear the old gradients
loss.backward()                     # 3. backprop: d(loss)/d(weight) for all ~10.7M weights
optimizer.step()                    # 4. move every weight a little downhill
```

**Backpropagation** (`loss.backward()`) uses the chain rule to work out, for
every weight, which direction of change would reduce the loss. PyTorch does
this automatically, because it recorded every operation during the forward
pass.

**AdamW** is the optimizer that applies those changes. Plain gradient descent
uses `w -= lr * grad`. Adam also keeps a running average of each weight's
gradient (momentum) and of its squared size, so every weight gets its own
adaptive step size. The "W" means weight decay: a gentle pull of every weight
toward zero, which discourages memorization.

Supporting tricks in the loop:

- **Learning-rate schedule** (`get_lr`). A linear warm-up over the first 100
  steps, while gradients are still wild, then a cosine decay from 1e-3 down
  to 1e-4 so the model settles into a good minimum at the end.
- **Gradient clipping.** If one batch produces a huge gradient, it is scaled
  down to norm 1.0 so it cannot wreck the weights.
- **Dropout (0.2).** During training, 20% of activations are randomly zeroed.
  This forces the network to build redundant features and reduces
  overfitting. `model.eval()` switches it off for evaluation and generation.
- **bfloat16 autocast.** On the GPU, matrix multiplies run in 16-bit floats
  for roughly twice the speed. The weights themselves stay 32-bit.

### Step 10: Watching training

Every 250 steps, `train.py` prints the loss averaged over 200 batches of each
split:

Here is a real run of the default settings on an RTX 5080:

```
step     0 | train loss 4.3032 | val loss 4.2994 | lr 1.00e-05 | 5.7s
step   250 | train loss 1.9466 | val loss 2.0592 | lr 9.91e-04 | 35.1s
step   500 | train loss 1.5346 | val loss 1.7241 | lr 9.40e-04 | 51.4s
step  1000 | train loss 1.2620 | val loss 1.5262 | lr 7.22e-04 | 84.3s
step  1500 | train loss 1.1334 | val loss 1.4685 | lr 4.34e-04 | 117.3s
step  1750 | train loss 1.0772 | val loss 1.4550 | lr 3.00e-04 | 133.7s   <- best val
step  2000 | train loss 1.0357 | val loss 1.4591 | lr 1.93e-04 | 150.1s
step  2500 | train loss 0.9825 | val loss 1.4758 | lr 1.00e-04 | 182.7s
```

How to read these numbers:

- **Both falling:** the model is learning.
- **Train falling, val flat or rising:** **overfitting**. The model is
  memorizing the training text. Only the checkpoint with the best *val* loss
  is saved, so you keep the best version automatically. (The short sample
  printed at the end of `train.py` comes from the final weights; `sample.py`
  uses the saved best ones.)

You can see overfitting clearly with `--max_iters 5000`. Train loss drops to
0.60 while val loss climbs back up to 1.73. One megabyte of text is simply
not enough for a 10.7M-parameter model to keep improving, so it starts
learning the training text by heart. The cures are more data, a smaller
model, more dropout, or stopping early.
- **Val loss around 1.5** on Shakespeare gives clearly Shakespeare-shaped text
  with mostly real words. Around 1.8 to 2.3 (the tiny CPU model) gives the
  right "shape" (names, line breaks, punctuation) but many invented words.

Real output from the default model (`sample.py --prompt "ROMEO:" --seed 1`):

```
ROMEO:
If I being so, I shall not be your honour,
How you comes here something talk of a more
More dead-heinous than your remedy, sit in hand,
If you were bound, we will not be a thousand man.

LORD ROSS:
Stay, stay not, my lord, by us: I do beseech you,
```

It has learned spelling, the play format, verse-like line lengths, and
archaic phrasing, all from raw characters. It has not learned meaning; that
takes far more data and parameters.

Roughly what the model learns over time:

1. Loss 4.17 → 3.3: which characters are common (lots of spaces and e's).
2. → 2.5: common pairs of letters ("th", "he"), and that lines are short.
3. → 2.0: whole common words, capitals after newlines, `NAME:` headers.
4. → 1.5: grammar-like structure, plausible phrases, the dialogue format.

### Step 11: Generation ([model.py](model.py) `generate`, [sample.py](sample.py))

```python
for _ in range(max_new_tokens):
    logits = model(idx[:, -block_size:])[:, -1, :]   # look at the last position only
    logits = logits / temperature
    keep only the top_k largest logits
    probs = softmax(logits)
    next  = sample one id from probs
    idx   = append(idx, next)
```

- **Only the last position matters.** The forward pass predicts for every
  position, but we only need the prediction after the final character.
- **Cropping to `block_size`.** The position table has only 256 entries, so
  the model can never see more than the last 256 characters. That is its
  *context window*.
- **Temperature.** Divide the logits before the softmax. At 0.5 the
  distribution gets sharper (safer, more repetitive text). At 1.0 you sample
  the model's true distribution. At 1.5 it gets flatter (more chaotic).
- **Top-k.** Throw away everything except the k most likely characters. This
  stops rare bad choices from derailing the text.
- **Sampling, not argmax.** Always taking the single most likely character
  quickly falls into loops ("the the the"). Sampling keeps the text varied.

> **Speed note.** This loop recomputes attention over the whole context for
> every new token. Real systems use a **KV cache**: the keys and values of
> earlier tokens never change, so they are stored and reused. Each new token
> then costs one row of attention instead of a full T×T matrix.

---

## Part 4: Training in practice

### Settings you can change

Every setting in `DEFAULTS` at the top of `train.py` is also a command-line
flag:

```powershell
.venv\Scripts\python train.py --n_layer 8 --n_embd 512 --n_head 8 --max_iters 8000
```

| Setting | Default | Effect of making it bigger |
|---|---|---|
| `n_layer` | 6 | Deeper: more steps of reasoning, slower |
| `n_embd` | 384 | Wider vectors: more capacity; must be divisible by `n_head` |
| `n_head` | 6 | More, smaller attention heads |
| `block_size` | 256 | Longer context; attention cost grows as T² |
| `batch_size` | 64 | Smoother gradients, more memory |
| `learning_rate` | 1e-3 | Faster learning, until it becomes unstable (loss spikes or NaN) |
| `dropout` | 0.2 | Less overfitting, slower learning; use 0.0 for big datasets |
| `max_iters` | 2500 | Longer training (overfits on Shakespeare beyond this) |

If you run out of GPU memory, lower `batch_size` or `block_size`.

### Training on your own text

Any UTF-8 `.txt` file works: your own writing, song lyrics, source code, a
book from Project Gutenberg. Aim for at least about 1 MB; less than that and
the model just memorizes.

```powershell
.venv\Scripts\python prepare.py --input path\to\my_text.txt
```

```powershell
.venv\Scripts\python train.py
```

`prepare.py` overwrites `data/`, and `train.py` overwrites `out/ckpt.pt`. To
keep several models, pass `--out_dir out_mytext` to `train.py` and
`--ckpt out_mytext\ckpt.pt` to `sample.py`. Each checkpoint stores the
vocabulary it was trained with, so old models keep working after you prepare
a different text. Only *training* reads `data/`, so run `prepare.py` again
before training on the other text.

### Using the model

```powershell
.venv\Scripts\python sample.py --prompt "JULIET:" --temperature 0.8 --num_samples 3
```

| Flag | Meaning |
|---|---|
| `--prompt` | Text to continue. Characters not in the vocabulary are skipped. |
| `--max_new_tokens` | How many characters to generate |
| `--temperature` | Randomness (see Step 11) |
| `--top_k` | Sample only from the k most likely characters (0 = all) |
| `--num_samples` | How many independent continuations to print |
| `--seed` | Fix the randomness to get the same output again |

---

## Part 5: Experiments to deepen your understanding

1. **Break the sanity check.** In `_init_weights`, change `std=0.02` to
   `std=1.0`. The initial loss will be far above 4.17. Why?
2. **Remove the causal mask.** Comment out the `masked_fill` line. Training
   loss will fall dramatically, because the model can now see the answer.
   Generation will then be garbage. This shows exactly why the mask exists.
3. **Remove the position embedding.** Replace `tok + pos` with `tok`. The
   loss gets worse, but less than you might expect, because the causal mask
   lets the model infer some position information by itself. Interesting in
   its own right.
4. **Remove the residuals.** Change `x = x + self.attn(...)` to
   `x = self.attn(...)` in `Block`. Training becomes much harder.
5. **Look at attention.** Store `att` on `self` inside
   `CausalSelfAttention.forward`, run one prompt through the model, and plot
   `att[0, head]` with matplotlib (`.venv\Scripts\python -m pip install
   matplotlib`). You will see heads that attend to the
   previous character, to spaces, and so on.
6. **Use the fast kernel.** Replace steps 5c to 5f with
   `y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
   dropout_p=self.attn_dropout.p if self.training else 0.0)`. The loss should
   be about the same (not bit-identical) and training noticeably faster.
7. **Scale up.** Try `--n_layer 8 --n_embd 512 --n_head 8` and watch for
   overfitting. Then try `--dropout 0.3`.
8. **Swap the tokenizer.** Use `tiktoken` (GPT-2 BPE) instead of
   `CharTokenizer`. That means changing `prepare.py`, plus the places in
   `train.py` and `sample.py` that build a `CharTokenizer`, and setting
   `vocab_size` to 50257 (ids still fit in `uint16`). Be warned: a
   50257 × 384 embedding adds about 19M parameters, and on 1 MB of text the
   model will overfit very quickly. Use a bigger dataset.

### Where to go next

- Andrej Karpathy, *"Let's build GPT: from scratch, in code, spelled out"*
  (YouTube). This project follows the same path.
- **nanoGPT** / **nanochat** (GitHub, karpathy): the same ideas, scaled up to
  reproduce GPT-2.
- *"Attention Is All You Need"* (Vaswani et al., 2017): the original
  Transformer paper.
- *"Language Models are Unsupervised Multitask Learners"* (the GPT-2 paper).
