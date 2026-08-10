"""
STEP 7  --  Multi-head attention: asking several questions at once

    Run me:  python3 07_multi_head.py   (about 20 seconds)

Our model works now. But it has exactly one attention head, which means every
word gets to ask exactly one question of its past, and one set of weights has
to serve every purpose at once.

Consider what a word genuinely needs to know in our language. Standing at
'quietly', the useful questions are at least:

    "who is the subject of this sentence?"        -> look at 'cat'
    "what word is immediately behind me?"         -> look at 'hungry'
    "have we had a verb yet, or is one due?"      -> scan everything

Those want completely different attention weights. One head must average them
into a single compromise that serves none of them well.

THE FIX
-------
Run several attention heads side by side, each with its own Q, K and V
matrices, and glue their outputs together. Each head is free to specialise.

THE ANALOGY
-----------
One expert reading a contract can only concentrate on one thing at a time.
Hire four -- a lawyer, an accountant, an engineer, a proofreader -- let them
all read the same document simultaneously, then staple their reports together.
You get four expert opinions for roughly the cost of one generalist, because
they work in parallel.

That last clause is not decoration. The heads never talk to each other, so
they are computed in the same batched matrix multiply. Multi-head attention
costs almost nothing extra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import grammar
import data
from display import title, section, bar, matrix

torch.manual_seed(1337)

title(7, "Multi-head attention: several questions at once")

D_MODEL = 32
BLOCK = data.BLOCK_SIZE


class AttentionHead(nn.Module):
    """Unchanged from step 5, except it can now report its weights."""

    def __init__(self, d_model, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return (out, weights) if return_weights else out


# ---------------------------------------------------------------------------
# THE KEY DESIGN DECISION: SPLIT, DON'T MULTIPLY
# ---------------------------------------------------------------------------
section("The budget trick: heads split the width, they don't add to it")

N_HEADS = 4
HEAD_SIZE = D_MODEL // N_HEADS

print(f"""  d_model    = {D_MODEL}     the width of a word vector
  n_heads    = {N_HEADS}
  head_size  = {HEAD_SIZE}      because {D_MODEL} / {N_HEADS} = {HEAD_SIZE}

  This is the detail people misread. Four heads do NOT make the model four
  times wider. Each head is given a quarter of the width, and their outputs
  are concatenated back to {HEAD_SIZE} x {N_HEADS} = {D_MODEL}. Same size in, same size out,
  essentially the same number of parameters.

  So multi-head attention is not "more compute for more power". It is a
  reorganisation of compute you were already spending: instead of one
  {D_MODEL}-wide question, ask {N_HEADS} independent {HEAD_SIZE}-wide ones. The paper does
  exactly this -- d_model 512, 8 heads, 64 each (section 3.2.2).""")


class MultiHeadAttention(nn.Module):
    """
    n_heads attention heads in parallel, concatenated, then mixed.

    The paper's equation 2:
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O
    """

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, head_size, block_size) for _ in range(n_heads)]
        )
        # W_O in the paper. Each head worked in its own little subspace with no
        # knowledge of the others; this final layer lets their findings mix and
        # be translated back into the model's shared vocabulary of features.
        # Without it you'd have four reports stapled together but never read
        # side by side.
        self.project = nn.Linear(d_model, d_model)

    def forward(self, x, return_weights=False):
        if return_weights:
            outs, weights = zip(*[h(x, return_weights=True) for h in self.heads])
            return self.project(torch.cat(outs, dim=-1)), weights
        outs = [h(x) for h in self.heads]
        return self.project(torch.cat(outs, dim=-1))


section("Shapes, concretely")

mha = MultiHeadAttention(D_MODEL, N_HEADS, BLOCK)
x = torch.randn(1, 6, D_MODEL)
with torch.no_grad():
    out, per_head = mha(x, return_weights=True)

print(f"  input                    {tuple(x.shape)}")
for i in range(N_HEADS):
    print(f"    head {i} produces        (1, 6, {HEAD_SIZE})")
print(f"  concatenated             (1, 6, {D_MODEL})")
print(f"  after the W_O projection {tuple(out.shape)}   <- same shape we started with")
print("""
  Same shape in, same shape out. That is deliberate and it's what makes the
  next step possible: if a component's output looks exactly like its input,
  you can stack it on top of itself as many times as you like.""")

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
section("Training with 4 heads")


class MultiHeadModel(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.word_embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)
        self.pos_embed = nn.Embedding(BLOCK, D_MODEL)
        self.attention = MultiHeadAttention(D_MODEL, n_heads, BLOCK)
        self.predict = nn.Linear(D_MODEL, grammar.VOCAB_SIZE)

    def forward(self, idx, return_weights=False):
        B, T = idx.shape
        x = self.word_embed(idx) + self.pos_embed(torch.arange(T))
        if return_weights:
            x, weights = self.attention(x, return_weights=True)
            return self.predict(x), weights
        return self.predict(self.attention(x))


def train(model, steps=3000, lr=1e-2, quiet=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_data, _ = data.load()
    for step in range(steps + 1):
        x, y = data.get_batch(train_data, batch_size=64)
        loss = F.cross_entropy(model(x).view(-1, grammar.VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not quiet and step % 1000 == 0:
            print(f"    step {step:>5}   loss {loss.item():7.4f}   "
                  f"{bar(1 - loss.item() / 6.5, 26)}")
    return loss.item()


@torch.no_grad()
def generate(model, max_new=18):
    idx = data.encode(["."]).unsqueeze(0)
    out = []
    for _ in range(max_new):
        probs = F.softmax(model(idx[:, -BLOCK:])[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        w = data.ID_TO_WORD[int(nxt)]
        if w == ".":
            break
        out.append(w)
    return out


scores = {}
for n_heads in [1, 4]:
    print(f"\n  {n_heads} head{'s' if n_heads > 1 else ''}"
          f" (head_size {D_MODEL // n_heads}):")
    torch.manual_seed(1337)
    model = MultiHeadModel(n_heads)
    final_loss = train(model)
    samples = [generate(model) for _ in range(1000)]
    scores[n_heads] = (final_loss, grammar.score_sentences(samples))
    if n_heads == 4:
        four_head_model = model

print(f"""
      1 head,  32 wide    {scores[1][1]:5.1%}    loss {scores[1][0]:.2f}
      4 heads, 8 wide     {scores[4][1]:5.1%}    loss {scores[4][0]:.2f}

  Same parameter budget, carved up differently.""")

# ---------------------------------------------------------------------------
# WHAT DID THE HEADS ACTUALLY LEARN?
# ---------------------------------------------------------------------------
section("Looking inside: did the heads specialise?")

sentence = ["the", "hungry", "cat", "quietly", "chased", "a", "small", "mouse"]
with torch.no_grad():
    _, head_weights = four_head_model(data.encode(sentence).unsqueeze(0),
                                      return_weights=True)

short = [w[:6] for w in sentence]
for h, w in enumerate(head_weights):
    print(f"\n  HEAD {h} -- row = the word doing the looking, column = what it looks at")
    matrix(w[0], row_labels=short, col_labels=short, fmt="{:>7.2f}")

print("""
  Read a row: it's where that word sent its attention. Some patterns you may
  spot (they vary from run to run, which is itself worth pointing out to an
  audience -- nothing here is hand-designed):

    * a head with a strong diagonal        = "mostly I care about myself"
    * a head with weight just below the
      diagonal                             = "look at the word right behind me"
    * a head that piles weight on one
      early column                         = "keep an eye on the subject"
    * a head that spreads evenly           = "give me the general gist"

  Nobody assigned those jobs. Four heads started from four different random
  initialisations, and the pressure to predict the next word pushed them into
  different specialities, because doing the same job four times is a waste and
  gradient descent finds that out.

  This is the origin of a whole research field: mechanistic interpretability,
  which reverse-engineers real models by reading their attention heads. Some
  heads in real GPT models have famous names -- "induction heads" that spot
  repeated patterns, heads that track quotation marks, heads that resolve
  pronouns. Same idea, bigger model.

WHAT'S NEXT
-----------
  Every model we've built has the same gap in it: attention GATHERS
  information from other words, but nothing ever sits and THINKS about what it
  gathered. Gather, then immediately guess. No processing in between.

    Next:  python3 08_the_block.py
""")
