"""
model.py - A small GPT (Generative Pre-trained Transformer), written out in full.
==================================================================================

What does a GPT do?
-------------------
It is a function that takes a sequence of tokens and, for EVERY position in
that sequence, outputs a probability distribution over what the NEXT token is.

    input :  "T  o     b  e"
    target:  "o     b  e  ,"      (the same text shifted left by one)

Training = nudging the weights so the correct next token gets a higher
probability. Generation = repeatedly sampling a next token and appending it.

The architecture, from bottom to top:

    token ids            (B, T)            B = batch size, T = sequence length
       |
    token embedding  +  position embedding  -> (B, T, C)    C = n_embd
       |
    [ Block ] x n_layer      each Block = LayerNorm -> Self-Attention -> + residual
       |                                 LayerNorm -> MLP            -> + residual
    final LayerNorm
       |
    lm_head (Linear C -> vocab_size)      -> logits (B, T, vocab_size)

Tensor shapes are written in comments as (B, T, C) etc. Following the shapes
is the single best way to understand this file.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # how many distinct tokens exist (set from data/vocab.json)
    block_size: int = 256     # max context length: how many tokens the model can "see" at once
    n_layer: int = 6          # how many Transformer blocks are stacked
    n_head: int = 6           # how many attention heads per block
    n_embd: int = 384         # width of every token's vector ("channels", C)
    dropout: float = 0.2      # fraction of activations randomly zeroed during training (regularisation)


# -----------------------------------------------------------------------------
# 1. Causal self-attention: how tokens talk to each other
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    Self-attention lets each position gather information from earlier positions.

    Every token vector x is projected into three vectors:
        query q : "what am I looking for?"
        key   k : "what do I contain?"
        value v : "what will I hand over if you pick me?"

    The attention score between position i and j is the dot product q_i . k_j
    (high when what i is looking for matches what j contains). After a softmax
    the scores become weights that sum to 1, and position i's output is the
    weighted sum of the values v_j.

    "Causal" means position i may only look at positions j <= i. Otherwise the
    model could cheat during training by peeking at the very token it must
    predict.

    "Multi-head" means we do this n_head times in parallel with smaller vectors
    (head_size = n_embd / n_head). Each head can learn a different kind of
    relationship (e.g. "previous character", "start of the word", "matching
    quote mark"). Their outputs are concatenated back to n_embd.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # One Linear layer computes q, k and v for all heads at once
        # (3 * n_embd outputs) - faster than three separate layers, same maths.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # After the heads are concatenated, mix their information together.
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # The causal mask: a lower-triangular matrix of ones.
        #   1 0 0 0
        #   1 1 0 0      row i = query position, column j = key position
        #   1 1 1 0      a 0 means "i is NOT allowed to look at j"
        #   1 1 1 1
        # register_buffer = part of the module (moves to GPU with it) but is
        # not a trainable parameter.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (n_embd)

        # ---- project to queries, keys, values -------------------------------
        q, k, v = self.qkv(x).split(C, dim=2)  # each (B, T, C)

        # ---- split channels into heads --------------------------------------
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # Moving the head dimension next to the batch dimension lets one matrix
        # multiply process every head of every sequence in parallel.
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # ---- attention scores -----------------------------------------------
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # att[b, h, i, j] = how much position i attends to position j.
        # Dividing by sqrt(head_size) keeps the scores' variance around 1;
        # without it, large dot products push softmax into near one-hot
        # outputs whose gradients are almost zero.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # ---- causal mask ----------------------------------------------------
        # Set "future" scores to -infinity; softmax turns e^-inf into exactly 0.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # ---- softmax: scores -> weights that sum to 1 along each row --------
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # ---- weighted sum of values -----------------------------------------
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # ---- glue the heads back together -----------------------------------
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        # .contiguous() is needed because transpose only changes the "view" of
        # memory, and .view() requires the memory to be laid out in order.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # NOTE: everything from "attention scores" to here is exactly what
        # F.scaled_dot_product_attention(q, k, v, is_causal=True) computes in
        # one fused, much faster and more memory-efficient kernel. Real code
        # uses that; we spell it out so you can see every step.

        return self.resid_dropout(self.proj(y))


# -----------------------------------------------------------------------------
# 2. MLP (feed-forward network): how each token "thinks" on its own
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Attention moves information BETWEEN positions. The MLP then processes each
    position INDEPENDENTLY: expand to 4x the width, apply a non-linearity,
    shrink back. Much of the model's "knowledge" is believed to live in these
    weights.
    """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU is a smooth version of ReLU (max(0, x)). Without a non-linearity
        # stacking Linear layers would collapse into one Linear layer.
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):          # x: (B, T, C)
        x = self.fc(x)             # (B, T, 4C)
        x = self.gelu(x)
        x = self.proj(x)           # (B, T, C)
        return self.dropout(x)


# -----------------------------------------------------------------------------
# 3. Transformer block = attention + MLP, each wrapped in LayerNorm + residual
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """
    x = x + Attention(LayerNorm(x))     <- communicate
    x = x + MLP(LayerNorm(x))           <- compute

    Residual connection ("x + ..."): each sub-layer only ADDS a correction to
    x instead of replacing it. Think of x as a shared "residual stream" that
    every layer reads from and writes into. It also gives gradients a direct
    highway back to the early layers, which is what makes deep networks
    trainable.

    LayerNorm rescales each token vector to mean 0 / variance 1 (then applies a
    learned scale and shift). It keeps the numbers in a healthy range. GPT-2
    and later put it BEFORE each sub-layer ("pre-norm"), which trains more
    stably than the original Transformer's "post-norm".
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------------------------------------------------------------
# 4. The full GPT
# -----------------------------------------------------------------------------
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding: a lookup table with one learned vector per token.
        # Row 39 of this matrix IS the model's representation of token 39.
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding: one learned vector per position 0..block_size-1.
        # Attention by itself has no notion of order ("dog bites man" and "man
        # bites dog" would look identical), so we add "I am at position t" to
        # every token vector.
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language-model head: turns each final vector into one score ("logit")
        # per vocabulary entry. Higher logit = model thinks that token is more
        # likely to come next.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and the output head share one
        # matrix. Intuition: "which vector represents token t" and "which
        # vector predicts token t" should be similar. It also saves parameters.
        self.lm_head.weight = self.tok_emb.weight

        # Initialise weights (see _init_weights).
        self.apply(self._init_weights)
        # GPT-2 trick: scale down the init of layers that write into the
        # residual stream, because n_layer * 2 of them are summed together.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        # Small random normal weights (std 0.02, as in GPT-2) and zero biases.
        # With small weights the initial logits are all near 0, so the first
        # prediction is roughly uniform over the vocabulary: the loss starts at
        # about ln(vocab_size). That is a great sanity check - see train.py.
        # (Not exactly: each logit is a sum of n_embd small random products,
        # with std about 0.02 * sqrt(n_embd) = 0.4 for n_embd=384, which adds
        # roughly 0.4^2 / 2 = 0.08 to the starting loss.)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # Position embeddings are usually excluded from the headline count.
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) integer token ids
        targets: (B, T) the correct next token for every position, or None

        Returns (logits, loss). loss is None when no targets are given.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence of {T} tokens exceeds block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)       # (T,)  = 0, 1, ..., T-1
        tok = self.tok_emb(idx)                         # (B, T, C)
        pos = self.pos_emb(pos)                         # (T, C) - broadcast over the batch
        x = self.drop(tok + pos)                        # (B, T, C)

        for block in self.blocks:
            x = block(x)                                # (B, T, C)
        x = self.ln_f(x)

        logits = self.lm_head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy = -log(probability the model gave to the correct
            # token), averaged over every position of every sequence.
            #   probability 1.0  -> loss 0
            #   probability 1/65 -> loss ln(65) = 4.17  (random guessing)
            # F.cross_entropy wants (N, vocab) and (N,), so flatten B and T.
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()  # no gradients needed while generating: saves memory and time
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation: predict one token, append it, repeat.

        idx:          (B, T) the prompt as token ids
        temperature:  divides the logits before softmax.
                      < 1.0 -> sharper, safer, more repetitive
                      > 1.0 -> flatter, more random, more "creative"
        top_k:        if set, only the k most likely tokens may be sampled.
        """
        for _ in range(max_new_tokens):
            # The model can only see block_size tokens, so crop to the last ones.
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)                  # (B, T, vocab)
            # We only care about the prediction made at the LAST position.
            logits = logits[:, -1, :] / temperature     # (B, vocab)

            if top_k is not None:
                # Find the k-th largest logit and remove everything below it.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)           # (B, vocab), sums to 1
            # Sample (not argmax!) so the output is varied, like rolling a
            # weighted die.
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)      # (B, T+1)
        return idx
