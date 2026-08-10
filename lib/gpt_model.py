"""
gpt_model.py  --  the finished model, in one file

This is the takeaway. Every class below was built up piece by piece in steps
2 through 8; here they are assembled with nothing left out and nothing extra.
Roughly 120 lines of actual code, and it is the same architecture as GPT-2,
GPT-3 and GPT-4 -- those differ from this only in size, data and training.

Reading order, if you want to follow the flow of a single word:

    GPT.forward       the whole pipeline, top to bottom
    Block             communicate (attention), then compute (feed-forward)
    MultiHeadAttention   several heads in parallel
    AttentionHead     the mechanism the paper is named after
    FeedForward       per-word thinking time
    GPT.generate      how text actually gets written, one word at a time

Used by 09_gpt.py (which trains it) and 10_see_attention.py (which looks
inside it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import grammar
import data

# Defaults. Step 9 passes its own values; these are here so the model can be
# built with GPT() and no arguments.
D_MODEL = 96          # width of a word vector          (paper: 512)
N_HEADS = 4           # attention heads per block       (paper: 8)
N_BLOCKS = 3          # how many blocks stacked         (paper: 6)
BLOCK_SIZE = data.BLOCK_SIZE   # context length
DROPOUT = 0.1         # regularisation                  (paper: 0.1)


class AttentionHead(nn.Module):
    """One head of causal self-attention.  (Step 5.)"""

    def __init__(self, d_model, head_size, block_size, dropout):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)

        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = self.dropout(weights) @ v
        return (out, weights) if return_weights else out


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated and mixed.  (Step 7.)"""

    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        if return_weights:
            outs, w = zip(*[h(x, return_weights=True) for h in self.heads])
            return self.dropout(self.project(torch.cat(outs, dim=-1))), w
        outs = [h(x) for h in self.heads]
        return self.dropout(self.project(torch.cat(outs, dim=-1)))


class FeedForward(nn.Module):
    """Per-word thinking time: expand, non-linearity, compress.  (Step 8.)"""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communicate, then compute. Both residual, both pre-normed.  (Step 8.)"""

    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.feedforward = FeedForward(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_weights=False):
        if return_weights:
            attended, w = self.attention(self.norm1(x), return_weights=True)
            x = x + attended
            return x + self.feedforward(self.norm2(x)), w
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class GPT(nn.Module):
    """
    A complete decoder-only Transformer. This is the real thing -- the same
    architecture as GPT-2, GPT-3 and GPT-4, differing only in size.
    """

    def __init__(self, vocab_size=grammar.VOCAB_SIZE, d_model=D_MODEL,
                 n_heads=N_HEADS, n_blocks=N_BLOCKS,
                 block_size=BLOCK_SIZE, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, block_size, dropout) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.predict = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.word_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.predict(self.norm(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def attention_maps(self, idx):
        """Run the model and hand back every block's attention weights."""
        B, T = idx.shape
        x = self.word_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        maps = []
        for block in self.blocks:
            x, w = block(x, return_weights=True)
            maps.append(w)
        return maps

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0, top_k=None,
                 stop_at_full_stop=True):
        """
        Write new text, one word at a time.

        The loop is the same one from step 0: guess, append, feed it back in.
        Two dials control how the guess is turned into a choice.
        """
        self.eval()
        idx = prompt_ids
        for _ in range(max_new_tokens):
            # Never feed in more than the model's context length.
            window = idx[:, -self.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :]                     # only the last position

            # TEMPERATURE. Divide the logits before softmax.
            #   < 1  sharpens  -> safe, repetitive, predictable
            #   = 1  unchanged -> the model's honest opinion
            #   > 1  flattens  -> surprising, creative, eventually gibberish
            logits = logits / temperature

            # TOP-K. Keep only the k best candidates and zero out the rest.
            # Even a good model puts a tiny probability on hundreds of absurd
            # words; added up, that tail gets sampled more often than you'd
            # think. Top-k simply refuses to consider it.
            if top_k is not None:
                kth_best = logits.topk(top_k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if stop_at_full_stop and data.ID_TO_WORD[int(next_id)] == ".":
                break
        self.train()
        return idx
