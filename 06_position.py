"""
STEP 6  --  Position: telling the model where the words are

    Run me:  python3 06_position.py   (about 15 seconds, and saves a picture)

Step 5 built attention and it barely beat a lookup table. The accusation we
made was that attention is blind to word order. This step proves that
accusation, fixes it, and the score roughly quadruples.

THE ANALOGY
-----------
Attention treats your sentence like a BAG of words tipped onto a table. It can
see every word and judge how relevant each one is -- but there are no
positions on a table. "cat" is just lying there. Nothing records that it came
third.

So we do the obvious thing: before tipping the words out, we stamp a number on
each one. Word vector plus position vector. Now "cat, which was third" is a
different thing from "cat, which was first", and attention can finally learn
relationships like "look at whoever is directly behind me".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import grammar
import data
from display import title, section, bar, matrix

torch.manual_seed(1337)

title(6, "Position: telling the model where the words are")

D_MODEL = 32


# ---------------------------------------------------------------------------
# The attention head from step 5, unchanged.
# ---------------------------------------------------------------------------
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


# ---------------------------------------------------------------------------
# THE PROOF
#
# Don't assert that attention is order-blind. Demonstrate it, exactly.
# ---------------------------------------------------------------------------
section("Proof: attention genuinely cannot see word order")

head = AttentionHead(D_MODEL, D_MODEL, data.BLOCK_SIZE)
embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)

original = ["the", "hungry", "cat", "quietly", "chased"]
shuffled = ["quietly", "the", "cat", "hungry", "chased"]   # last word held fixed

with torch.no_grad():
    out_a = head(embed(data.encode(original).unsqueeze(0)))[0, -1]
    out_b = head(embed(data.encode(shuffled).unsqueeze(0)))[0, -1]

print(f"  sentence A: {' '.join(original)}")
print(f"  sentence B: {' '.join(shuffled)}   (same words, scrambled)")
print("\n  Attention's output at the final position -- the vector it will use")
print("  to predict the next word:\n")
print(f"    from A: {[round(v, 4) for v in out_a[:6].tolist()]} ...")
print(f"    from B: {[round(v, 4) for v in out_b[:6].tolist()]} ...")
print(f"\n    identical? {torch.allclose(out_a, out_b, atol=1e-6)}")
print(f"    largest difference anywhere: {(out_a - out_b).abs().max():.10f}")

print("""
  Not "similar". Bit-for-bit identical, to the limits of floating point.

  And the reason is no mystery once you look at the last line of attention:
  out = weights @ v is a SUM. Addition doesn't care what order you add things
  in. 2+3+5 and 5+2+3 are the same number, and so are these vectors.

  So our model is currently incapable of distinguishing "the cat chased a
  mouse" from "mouse a chased cat the". It's been doing well at all only
  because it can still tell WHICH words are present, not where they sit.""")

# ---------------------------------------------------------------------------
# FIX 1: LEARNED POSITION EMBEDDINGS
# ---------------------------------------------------------------------------
section("Fix: give every position its own vector, and add it on")

print("""  We already have a table of 500 word vectors. Now add a second table of 16
  POSITION vectors -- one for "I am the 1st word", one for "I am the 2nd", and
  so on up to our context length. Then:

      x = word_vector + position_vector

  and hand that to attention.

  "Why ADD them? Doesn't that scramble the two signals together?"
  ---------------------------------------------------------------
  It's the question everyone asks, and it's a good one. Two answers:

    * Space is cheap up here. In 32 dimensions there is ample room for the
      model to keep "which word" and "which slot" in different directions and
      read them back out separately. It sounds lossy; in practice it isn't.

    * The alternative, glueing them side by side, makes every vector wider and
      therefore every matrix in the model bigger, for no measured gain.

  Addition is the cheap option that works, which is usually why something is
  done this way.""")


class PositionalModel(nn.Module):
    """Step 5's model, plus a position table."""

    def __init__(self, use_sinusoidal=False):
        super().__init__()
        self.word_embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)
        self.use_sinusoidal = use_sinusoidal
        if use_sinusoidal:
            self.register_buffer("pos_table", sinusoidal_table(data.BLOCK_SIZE, D_MODEL))
        else:
            self.pos_embed = nn.Embedding(data.BLOCK_SIZE, D_MODEL)
        self.attention = AttentionHead(D_MODEL, D_MODEL, data.BLOCK_SIZE)
        self.predict = nn.Linear(D_MODEL, grammar.VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        words = self.word_embed(idx)                          # (B, T, C)
        if self.use_sinusoidal:
            positions = self.pos_table[:T]                    # (T, C)
        else:
            positions = self.pos_embed(torch.arange(T))       # (T, C)
        x = words + positions                                 # <-- the whole fix
        x = self.attention(x)
        return self.predict(x)


# ---------------------------------------------------------------------------
# FIX 2: THE PAPER'S VERSION -- SINUSOIDAL
# ---------------------------------------------------------------------------
def sinusoidal_table(block_size, d_model):
    """
    The positional encoding from section 3.5 of "Attention Is All You Need":

        PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

    Every pair of columns is a wave. The first pair oscillates fast, and each
    pair after that is slower than the last, all the way down to waves so slow
    they barely move across the whole sequence.

    THE ANALOGY: it's a clock face, or an old car odometer. The seconds hand
    spins quickly, the minutes hand slowly, the hours hand slower still. Read
    all the hands together and you get a unique reading for every moment --
    and, crucially, readings for nearby moments look similar. That "nearby
    positions get similar codes" property is the entire point; it's what lets
    the model generalise a relationship like "three words back".
    """
    position = torch.arange(block_size).unsqueeze(1)                  # (T, 1)
    i = torch.arange(0, d_model, 2)                                   # 0,2,4,...
    frequency = torch.exp(-math.log(10000.0) * i / d_model)           # one per pair

    table = torch.zeros(block_size, d_model)
    table[:, 0::2] = torch.sin(position * frequency)                  # even columns
    table[:, 1::2] = torch.cos(position * frequency)                  # odd columns
    return table


section("The paper's version: sinusoidal encodings")

table = sinusoidal_table(data.BLOCK_SIZE, D_MODEL)
print("  The first 8 positions, first 8 of the 32 columns:\n")
matrix(table[:8, :8], row_labels=[f"pos {i}" for i in range(8)],
       col_labels=[f"c{i}" for i in range(8)])

print("""
  Read DOWN a column and you see a wave. The leftmost columns wiggle quickly;
  by the right-hand columns the values barely change from row to row.

  Now the property that makes it useful -- how similar is each position's code
  to position 0's?""")

normed = table / table.norm(dim=1, keepdim=True)
sims = (normed @ normed[0])
print()
for p in range(8):
    print(f"    position 0 vs position {p}:  {sims[p]:+.3f}  {bar((sims[p] + 1) / 2, 30)}")

print("""
  Smoothly decreasing with distance. Position 1 is nearly position 0; position
  7 is much less so. The model gets a genuine sense of "near" and "far" for
  free, rather than having to learn every position from scratch.

  LEARNED vs SINUSOIDAL -- which should you use?
  ----------------------------------------------
  The paper tried both and reports (section 3.5) that they scored about the
  same, choosing sinusoidal on the theory that it might extrapolate to
  sequences longer than any seen in training -- a learned table simply has no
  row for position 5000 if you only ever trained to 512.

  In practice almost everything since has used learned embeddings or a later
  invention called RoPE. We train both below so you can compare them yourself.
  Expect them to land close together, with the ordering wobbling if you change
  the random seed -- at this size the gap between them is smaller than the
  run-to-run noise, which is itself a useful lesson about reading benchmark
  tables.""")

# Save a picture -- worth putting on a slide.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    big = sinusoidal_table(64, 64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im = axes[0].imshow(big, aspect="auto", cmap="RdBu")
    axes[0].set_xlabel("dimension"); axes[0].set_ylabel("position")
    axes[0].set_title("Sinusoidal positional encoding")
    fig.colorbar(im, ax=axes[0])
    for d in [0, 4, 12, 24]:
        axes[1].plot(big[:, d], label=f"dimension {d}")
    axes[1].set_xlabel("position"); axes[1].set_title("Individual columns are waves")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=110)
    print("\n  Saved a picture of it to positional_encoding.png")
except Exception as e:                                    # matplotlib is optional
    print(f"\n  (skipped the plot: {e})")

# ---------------------------------------------------------------------------
# DOES THE ORDER-BLINDNESS GO AWAY?
# ---------------------------------------------------------------------------
section("Re-running the proof, now with positions")

check = PositionalModel()
with torch.no_grad():
    a = check(data.encode(original).unsqueeze(0))[0, -1]
    b = check(data.encode(shuffled).unsqueeze(0))[0, -1]
print(f"  largest difference between A and B now: {(a - b).abs().max():.4f}")
print("  Not zero any more. The model can tell the two sentences apart.")

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
section("Training")


def train(model, steps=3000, lr=1e-2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_data, _ = data.load()
    for step in range(steps + 1):
        x, y = data.get_batch(train_data, batch_size=64)
        loss = F.cross_entropy(model(x).view(-1, grammar.VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"    step {step:>5}   loss {loss.item():7.4f}   "
                  f"{bar(1 - loss.item() / 6.5, 26)}")
    return loss.item()


@torch.no_grad()
def generate(model, max_new=18):
    idx = data.encode(["."]).unsqueeze(0)
    out = []
    for _ in range(max_new):
        logits = model(idx[:, -data.BLOCK_SIZE:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        w = data.ID_TO_WORD[int(nxt)]
        if w == ".":
            break
        out.append(w)
    return out


results = {}
for name, sinusoidal in [("learned", False), ("sinusoidal", True)]:
    print(f"\n  {name} positional encoding:")
    torch.manual_seed(1337)
    model = PositionalModel(use_sinusoidal=sinusoidal)
    final_loss = train(model)
    samples = [generate(model) for _ in range(1000)]
    results[name] = (final_loss, grammar.score_sentences(samples), samples)

# ---------------------------------------------------------------------------
section("Results")

print("  Sentences from the model with learned positions:\n")
for s in results["learned"][2][:8]:
    print("    " + " ".join(s))

learned_score = results["learned"][1]
sin_score = results["sinusoidal"][1]

print(f"""
      step 3, bigram                        22.6%    loss 3.95
      step 4, flat average                   2.1%    loss 4.35
      step 5, attention, no positions       21.5%    loss 3.96
      step 6, attention + learned positions {learned_score:5.1%}    loss {results['learned'][0]:.2f}
      step 6, attention + sinusoidal        {sin_score:5.1%}    loss {results['sinusoidal'][0]:.2f}

  There it is. The biggest single jump in the course, from one added line:

      x = word_vector + position_vector

  Two lessons worth stating out loud:

    1. Attention was never the problem. It was starving for information it was
       never given. The mechanism from step 5 is unchanged -- we just told it
       where the words were and it took off.

    2. This is why the paper needs section 3.5 at all. A recurrent network
       reads words one at a time, so it knows the order automatically, for
       free. Attention looks at everything simultaneously -- which is exactly
       what makes it fast and parallel -- and the price of that parallelism is
       that you must hand it position explicitly. Positional encoding is the
       bill for attention's speed.

WHAT'S NEXT
-----------
  Our model asks one question per word. But "who is the subject?" and "how far
  back was the last verb?" are different questions, and a single set of
  attention weights has to compromise between them.

  So let's ask several questions at once.

    Next:  python3 07_multi_head.py
""")
