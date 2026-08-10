"""
STEP 8  --  The Transformer block: thinking time, highways and thermostats

    Run me:  python3 08_the_block.py   (about 2 minutes)

We have attention with multiple heads and positions. What's left is the other
half of the paper's Figure 1 -- the three components that turn one attention
layer into something you can stack twenty of:

    1. a FEED-FORWARD network     private thinking time after gathering
    2. RESIDUAL connections       so gradients survive the trip down
    3. LAYER NORM                 so the numbers stay in a sane range

None of the three is glamorous and none gets a mention in the paper's title.
All three are load-bearing: remove any one and deep Transformers stop
training. Attention gets the headline; these keep the building up.

FIRST, A NEW SCOREBOARD
-----------------------
From here on, grammar percentage is a blunt instrument -- we're at 93.7% and
the remaining errors are rare flukes. So we switch to the loss, and we now
have something precious to compare it against: because we invented this
language, we can calculate the lowest loss ANY model could possibly achieve.
See grammar.entropy_per_token(). That number is the finish line.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import grammar
import data
from display import title, section, bar

torch.manual_seed(1337)

title(8, "The Transformer block")

D_MODEL = 32
N_HEADS = 4
BLOCK = data.BLOCK_SIZE
FLOOR = grammar.entropy_per_token()

print(f"\n  The finish line: no model can ever score below {FLOOR:.4f}")
print(f"  (our step 7 model reached about 3.86 on held-out text)")


# ---------------------------------------------------------------------------
# Attention, carried over unchanged from steps 5-7.
# ---------------------------------------------------------------------------
class AttentionHead(nn.Module):
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, head_size, block_size) for _ in range(n_heads)]
        )
        self.project = nn.Linear(d_model, d_model)

    def forward(self, x, return_weights=False):
        if return_weights:
            outs, w = zip(*[h(x, return_weights=True) for h in self.heads])
            return self.project(torch.cat(outs, dim=-1)), w
        return self.project(torch.cat([h(x) for h in self.heads], dim=-1))


# ---------------------------------------------------------------------------
# PIECE 1: THE FEED-FORWARD NETWORK
# ---------------------------------------------------------------------------
section("Piece 1: the feed-forward network -- thinking time")

print(f"""  Every model so far does this: gather information from other words, then
  immediately guess the next word. There is no step in between where a word
  processes what it just collected.

  It's a committee that gathers evidence and votes instantly, with no
  discussion. The feed-forward network is the discussion.

      Linear({D_MODEL} -> {4 * D_MODEL})   expand: room to spread the thought out
      ReLU                 keep the positive parts, discard the rest
      Linear({4 * D_MODEL} -> {D_MODEL})   compress: back to normal size

  WHY EXPAND THEN SHRINK?
  Compare it to writing. To think a problem through you scribble across three
  pages, then boil it down to one paragraph. The wide middle layer is the
  scratch paper -- room to represent many possible features at once -- and the
  second Linear is the summary. The paper uses d_model 512 with a 2048-wide
  middle (section 3.3): the same 4x ratio, which has stuck around ever since.

  WHY DOES ReLU MATTER? Without it the two Linears would collapse into one
  matrix -- multiplying by A then B is just multiplying by AB, so you'd have a
  wide, expensive layer that can only do what a single narrow one could.
  ReLU (keep positives, zero the negatives) is a non-linearity, and it's what
  lets a network learn curved, conditional relationships rather than only
  proportional ones.

  AND THE DETAIL PEOPLE MISS
  This runs on each position SEPARATELY. The word at position 3 is processed
  without ever seeing position 4. That's deliberate and it gives the block a
  clean division of labour:

      attention     the ONLY place where words look at each other
      feed-forward  private thinking, one word at a time

  Every bit of communication between words in a Transformer happens in
  attention. Everywhere else, words are alone.""")


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# PIECE 2: RESIDUAL CONNECTIONS
# ---------------------------------------------------------------------------
section("Piece 2: residual connections -- the highway")

print("""  One character's difference:

      x = attention(x)          without
      x = x + attention(x)      with

  Instead of REPLACING the word's vector, each layer suggests an EDIT to it.

  THE ANALOGY: a manuscript passed between editors. Without residuals, each
  editor rewrites the whole thing from scratch and hands it on -- twenty
  editors later the original is unrecognisable, and any of them can destroy it
  single-handedly. With residuals, each editor writes notes in the margin. The
  original text always survives, and every layer's contribution is additive.

  This also gives a nice way to think about what a Transformer is doing: the
  vector flowing down the middle is a "residual stream", a running summary of
  the word that each layer reads from and writes to.

  But the real reason is about training, and you can measure it:""")

DEPTH = 20


def gradient_at_first_layer(use_residual):
    """Push a signal through 20 layers, then measure the gradient at the bottom."""
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(DEPTH)])
    x = torch.randn(8, 32)
    h = x
    for layer in layers:
        h = h + torch.tanh(layer(h)) if use_residual else torch.tanh(layer(h))
    h.sum().backward()
    return layers[0].weight.grad.abs().mean().item()


without = gradient_at_first_layer(False)
with_res = gradient_at_first_layer(True)
print(f"""
  Gradient reaching layer 1 of a {DEPTH}-layer network:

      without residuals :  {without:.3e}
      with residuals    :  {with_res:.3e}
      ratio             :  {with_res / without:,.0f}x stronger

  Without residuals the learning signal is squeezed a little at every layer on
  its way back down, and after twenty layers there is essentially nothing left
  -- the early layers receive no instruction and never improve. This is the
  "vanishing gradient" problem, and it is why neural networks were stuck at a
  handful of layers for decades.

  Residuals fix it because addition passes gradient straight through,
  untouched. The '+' is an express lane from the loss back to layer 1. This
  idea (ResNet, 2015) is what made deep learning actually deep, and the
  Transformer inherits it.""")

# ---------------------------------------------------------------------------
# PIECE 3: LAYER NORM
# ---------------------------------------------------------------------------
section("Piece 3: layer normalisation -- the thermostat")

print("""  After each layer adds its contribution, the numbers can drift -- growing
  steadily larger, or collapsing towards zero. Both wreck training. LayerNorm
  resets each word's vector to mean 0 and standard deviation 1:

      x_normalised = (x - mean) / standard_deviation

  THE ANALOGY: grading on a curve. Raw exam marks drift wildly between
  papers -- one test averages 30, another 80. Normalising each to the same
  mean and spread makes them comparable and keeps any single one from
  dominating. LayerNorm does that to activations, after every sub-layer.

  It also has two learnable knobs (a scale and a shift) so the model can undo
  the normalisation where it turns out to be unhelpful. nn.LayerNorm gives you
  those automatically.""")

demo = torch.randn(4, 8) * 15 + 40
print(f"\n  before: mean {demo.mean():6.2f}   std {demo.std():5.2f}")
normed = nn.LayerNorm(8)(demo)
print(f"  after : mean {normed.mean():6.2f}   std {normed.std():5.2f}")

print("""
  A HONEST FOOTNOTE ON ORDERING
  The paper puts the norm AFTER each sub-layer (its Figure 1 says "Add & Norm"):

      x = LayerNorm(x + attention(x))              "post-norm", the 2017 paper

  Almost everything since -- GPT-2 onward -- puts it BEFORE instead:

      x = x + attention(LayerNorm(x))              "pre-norm", what we use

  Pre-norm keeps the residual highway completely clean: nothing sits between
  the '+' signs, so gradients flow all the way down unmodified. Post-norm needs
  a careful learning-rate warm-up to train deep models at all. This is a real
  example of the field improving a paper after publication, and it's worth
  telling your audience -- papers are snapshots, not scripture.""")


# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------
section("The block, assembled")

print("""
    class Block(nn.Module):
        def forward(self, x):
            x = x + self.attention(self.norm1(x))    # words talk to each other
            x = x + self.feedforward(self.norm2(x))  # each word thinks alone
            return x

  Two lines. Communicate, then compute. Both wrapped in residuals, both
  normalised on the way in. Shape in equals shape out -- so you can stack it.
""")


class Block(nn.Module):
    """
    One Transformer block: communicate (attention), then compute (FFN).

    The two flags exist only so the experiment below can switch pieces off and
    measure what each one is worth. A real implementation is just the two
    lines shown above.
    """

    def __init__(self, d_model, n_heads, block_size,
                 use_ffn=True, use_residual=True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, block_size)
        self.feedforward = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_ffn = use_ffn
        self.use_residual = use_residual

    def forward(self, x):
        attended = self.attention(self.norm1(x))
        x = x + attended if self.use_residual else attended

        if self.use_ffn:
            thought = self.feedforward(self.norm2(x))
            x = x + thought if self.use_residual else thought
        return x


class BlockModel(nn.Module):
    def __init__(self, n_blocks, use_ffn=True, use_residual=True):
        super().__init__()
        self.word_embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)
        self.pos_embed = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, N_HEADS, BLOCK, use_ffn, use_residual)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(D_MODEL)
        self.predict = nn.Linear(D_MODEL, grammar.VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        x = self.word_embed(idx) + self.pos_embed(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.predict(self.norm(x))


# ---------------------------------------------------------------------------
# TRAIN AND ABLATE
# ---------------------------------------------------------------------------
section("What is each piece actually worth?")


def train(model, steps=2000, lr=1e-2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_data, val_data = data.load()
    for step in range(steps + 1):
        x, y = data.get_batch(train_data, batch_size=64)
        loss = F.cross_entropy(model(x).view(-1, grammar.VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Measure on held-out text the model never trained on.
    with torch.no_grad():
        losses = []
        for _ in range(50):
            x, y = data.get_batch(val_data, batch_size=64)
            losses.append(F.cross_entropy(
                model(x).view(-1, grammar.VOCAB_SIZE), y.reshape(-1)).item())
    return sum(losses) / len(losses)


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


print("  Each row trains from scratch, so this takes a couple of minutes.\n")

experiments = [
    ("1 block, attention only (no FFN)", dict(n_blocks=1, use_ffn=False)),
    ("1 block, full",                    dict(n_blocks=1)),
    ("3 blocks, full",                   dict(n_blocks=3)),
    ("6 blocks, full",                   dict(n_blocks=6)),
    ("6 blocks, NO residuals",           dict(n_blocks=6, use_residual=False)),
]

print(f"  {'configuration':<34} {'val loss':>9}  {'above floor':>11}  {'grammar':>8}")
print("  " + "-" * 68)
results = {}
for name, kwargs in experiments:
    torch.manual_seed(1337)
    model = BlockModel(**kwargs)
    val_loss = train(model)
    score = grammar.score_sentences([generate(model) for _ in range(1000)])
    results[name] = (val_loss, score)
    print(f"  {name:<34} {val_loss:9.4f}  {val_loss - FLOOR:+11.4f}  {score:8.1%}")

print(f"  {'-' * 68}")
print(f"  {'THE FLOOR (best possible)':<34} {FLOOR:9.4f}  {0.0:+11.4f}  {1.0:8.1%}")

deep_with = results["6 blocks, full"][0]
deep_without = results["6 blocks, NO residuals"][0]

print(f"""
  Four things to draw out of that table:

    1. The feed-forward network earns its place. Gathering information without
       any time to process it was leaving performance behind.

    2. Residuals are not optional once you go deep. Six blocks with them:
       {deep_with:.2f}. The same six without: {deep_without:.2f} -- catastrophically worse,
       exactly as the gradient measurement predicted. Notice this only showed
       up at depth; at one or two blocks you would never have known. That is
       why residuals were invented for very deep networks and why the paper
       (with six blocks per stack) needs them.

    3. Depth past three blocks buys us nothing here. That's not a bug, it's
       our language being simple: there is only so much structure to learn, and
       three blocks already capture it. Real language keeps rewarding depth for
       dozens of layers, which is why GPT-3 has 96 of them.

    4. Grammar score and loss are not the same thing. A model can sit close to
       the loss floor and still emit an ungrammatical sentence now and then,
       because leaking a sliver of probability onto an impossible word costs
       almost nothing in loss but does occasionally get sampled. Which metric
       matters depends on what you actually want -- a lesson that generalises
       well beyond this course.

  We are around {results['3 blocks, full'][0] - FLOOR:.2f} above the floor. Every architectural piece from
  the paper is now in place, so what remains is scale and training: a wider
  model, trained longer. That's step 9.

    Next:  python3 09_gpt.py
""")
