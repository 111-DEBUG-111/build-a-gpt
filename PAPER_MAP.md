# From the paper to our code

A section-by-section map of *Attention Is All You Need* (Vaswani et al., 2017)
onto this repository — including the parts we deliberately skip, and why.

Read this after step 10. It turns the paper from an intimidating document into
a table of contents for things you have already built.

---

## Where each piece of the paper lives

All paths below are relative to this directory, the one holding the lessons.

| Paper | What it is | Our code | Built in |
|---|---|---|---|
| §3.1, Fig. 1 | Overall encoder–decoder architecture | `lib/gpt_model.py` → `GPT` (decoder half only) | step 9 |
| §3.1 | "Add & Norm", residual + layer norm | `Block.forward` | step 8 |
| §3.2.1, Eq. 1 | Scaled Dot-Product Attention | `AttentionHead.forward` | step 5 |
| §3.2.1 | The `1/√d_k` scaling | `scores / (head_size ** 0.5)` | step 5 |
| §3.2.2, Eq. 2 | Multi-Head Attention | `MultiHeadAttention` | step 7 |
| §3.2.2 | `W_O`, the output projection | `MultiHeadAttention.project` | step 7 |
| §3.2.3 | Masking to prevent leftward information flow | `masked_fill(tril == 0, -inf)` | steps 4, 5 |
| §3.3, Eq. 2 | Position-wise Feed-Forward Network | `FeedForward` | step 8 |
| §3.3 | Inner dimension `d_ff = 4 × d_model` | `nn.Linear(d_model, 4 * d_model)` | step 8 |
| §3.4 | Learned input embeddings | `GPT.word_embed` | step 2 |
| §3.5, Eq. 3 | Sinusoidal positional encoding | `sinusoidal_table()` | step 6 |
| §3.5 | "We also experimented with learned embeddings" | `GPT.pos_embed` | step 6 |
| §5.4 | Residual dropout, `P_drop = 0.1` | `nn.Dropout(dropout)` | step 9 |
| Table 3 | Ablations over heads and dimensions | ablation table | steps 7, 8 |

## The equations, decoded

**Equation 1 — Scaled Dot-Product Attention**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

```python
scores  = q @ k.transpose(-2, -1)          # Q Kᵀ        match questions to labels
scores  = scores / (head_size ** 0.5)      # / √d_k      keep softmax gentle
scores  = scores.masked_fill(mask, -inf)   #             no peeking ahead (§3.2.3)
weights = F.softmax(scores, dim=-1)        # softmax     into percentages
out     = weights @ v                      # ... V       blend the contents
```

Built and motivated in `05_self_attention.py`.

**Equation 2 — Multi-Head Attention**

```
MultiHead(Q, K, V) = Concat(head₁, …, head_h) W^O
      where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

The per-head `W^Q, W^K, W^V` are the three `nn.Linear` layers inside
`AttentionHead`; `W^O` is `MultiHeadAttention.project`. Built in
`07_multi_head.py`.

**Equation 3 — Positional Encoding**

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

Implemented as `sinusoidal_table()` in `06_position.py`, which also plots it
and trains against the learned alternative.

---

## What we left out, and why

### The encoder, and cross-attention (§3.1, left half of Figure 1)

The biggest omission, and the one most likely to confuse someone moving from
this course to the paper.

The 2017 paper is about **translation**, so it has two stacks:

- an **encoder** that reads the whole French sentence, with *unmasked*
  attention — every word may look at every other word, including later ones,
  because the input is already complete;
- a **decoder** that writes English one word at a time, with masked
  self-attention (what we built) *plus* a second attention layer whose queries
  come from the English being written and whose keys and values come from the
  encoder's output. That is **cross-attention**, and it is how the translation
  gets across the gap.

GPT is **decoder-only**. There is no separate thing to read — the prompt and
the continuation are the same sequence — so the encoder and cross-attention
are simply not needed. Dropping them leaves the masked self-attention stack
this course builds.

Three-way summary, worth a slide:

| Family | Attention | Good at |
|---|---|---|
| Encoder-only (BERT) | unmasked, sees both directions | understanding, classification |
| Decoder-only (GPT, ours) | masked, past only | generating text |
| Encoder–decoder (the paper, T5) | both, plus cross-attention | translation, summarising |

### Training details we skipped (§5.1–5.4)

| Skipped | What it does | Why we skipped it |
|---|---|---|
| Learning-rate warm-up, `lr ∝ min(n^-0.5, n·warmup^-1.5)` | Ramps the rate up then decays it | Needed for post-norm stability; our pre-norm model trains fine on a flat rate |
| Label smoothing `ε = 0.1` | Stops the model becoming over-confident | Costs a little loss for a little BLEU; a distraction here |
| Beam search | Explores several continuations at decode time | We sample instead, which is what GPT-style models do |
| Byte-pair encoding | Splits rare words into sub-word pieces | Our 500-word vocabulary covers our language exactly |
| Adam with `β₂ = 0.98` | Optimiser tuning | `AdamW` defaults are fine at this scale |

### Where the paper has since been improved

Worth telling an audience, because it stops them treating a 2017 paper as the
final word:

- **Pre-norm instead of post-norm.** The paper does `LayerNorm(x + sublayer(x))`;
  GPT-2 onward does `x + sublayer(LayerNorm(x))`, which keeps the residual
  path clean and trains deep models without warm-up. We use pre-norm — see
  step 8 for the side-by-side.
- **RoPE** (rotary position embeddings) has largely replaced both of the
  paper's positional schemes in current models.
- **FlashAttention** computes exactly the same Equation 1 without ever storing
  the full T×T score matrix, which is what makes long contexts affordable.
- **Grouped-query attention** shares keys and values across heads to shrink
  the memory needed at generation time.

None of these change the idea. They change the accounting.

---

## Reading the paper after this course

A suggested order, now that you have built it:

1. **§3.2.1** — you wrote this; check the notation matches your mental model
2. **Figure 1** — find the two lines of `Block.forward` in the diagram
3. **§3.2.2** — multi-head, then look back at your step 7 head profiles
4. **§3.5** — positional encoding, and your step 6 plot
5. **§4** — "Why Self-Attention", the paper's own argument for the whole idea;
   it is the most readable section and it is best read *last*, once you know
   what is being argued for
6. **§3.1** — the encoder and cross-attention, the part we skipped
