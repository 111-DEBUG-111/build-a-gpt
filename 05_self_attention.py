"""
STEP 5  --  Self-attention

    Run me:  python3 05_self_attention.py   (about 10 seconds)

Step 4 ended with a machine that was almost right:

    scores -> block the future -> softmax -> weighted sum of the past

and one thing that was clearly wrong: the scores were all zero, so every past
word got equal weight, and the result was a useless blur.

This step fills in the scores. That's all attention is. The mask, the softmax
and the weighted sum are already built and do not change.

THE ANALOGY: A LIBRARY
----------------------
You walk into a library with a question. Every book has a label on its spine.
You compare your question to each spine, and the closer the match, the more of
that book you read.

    QUERY  = the question you walk in with
    KEY    = the label on a book's spine
    VALUE  = what's actually written inside the book

Three things, and they're genuinely different. A book's spine ("Roman
History") is not the same as its contents, and neither is the same as your
question ("when did Caesar die?"). You match question against SPINES, but you
read CONTENTS.

Now the twist that makes it "self"-attention: here every word plays all three
roles at once. Each word walks in with a question, and each word also sits on
the shelf with a label and contents. Every word asks every earlier word "are
you relevant to me?", and builds its answer out of whoever says yes.

    'quietly' asks:  "I'm an adverb, I need to know who's doing the action"
    'cat' answers:   "I'm the subject noun" -- strong match, high weight
    'the' answers:   "I'm a determiner"     -- weak match, low weight

Nobody wrote those questions. Q, K and V are produced by three matrices that
start as random noise and are learned by gradient descent, exactly like every
other weight in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import grammar, data
from lib.display import title, section, bar, matrix

torch.manual_seed(1337)

title(5, "Self-attention: letting the model choose the weights")

# ---------------------------------------------------------------------------
# THE THREE PROJECTIONS
# ---------------------------------------------------------------------------
section("Every word produces a query, a key and a value")

B, T, C = 1, 5, 32           # 1 chunk, 5 words, 32 numbers per word
HEAD_SIZE = 16

x = torch.randn(B, T, C)     # pretend these are 5 embedded words

query = nn.Linear(C, HEAD_SIZE, bias=False)
key = nn.Linear(C, HEAD_SIZE, bias=False)
value = nn.Linear(C, HEAD_SIZE, bias=False)

q = query(x)                 # (B, T, 16)  what each word is looking for
k = key(x)                   # (B, T, 16)  what each word advertises
v = value(x)                 # (B, T, 16)  what each word will hand over

print(f"  input x     {tuple(x.shape)}   the words")
print(f"  queries q   {tuple(q.shape)}   'what am I looking for?'")
print(f"  keys    k   {tuple(k.shape)}   'what do I offer?'")
print(f"  values  v   {tuple(v.shape)}   'what do I contribute?'")
print("""
  Three separate matrices, three separate meanings. Beginners often ask why we
  don't just use x for all three -- the answer is that "what I'm looking for"
  and "what I have to offer" are different questions, and forcing one vector
  to answer both makes the model much weaker.""")

# ---------------------------------------------------------------------------
# THE SCORES
# ---------------------------------------------------------------------------
section("Matching every query against every key")

scores = q @ k.transpose(-2, -1)        # (B, T, T)

print("  scores = q @ k-transposed  ->  shape", tuple(scores.shape))
print("""
  Entry [i, j] is the dot product of word i's query with word j's key: how
  well does what word i wants match what word j offers?

  The dot product is the whole matching mechanism, and it's worth one sentence:
  it is large when two vectors point the same way, near zero when they're
  unrelated, negative when they oppose. So "high dot product" literally means
  "these two are aligned", which is exactly what "relevant" should mean.

  Untrained, these scores are noise:""")
matrix(scores[0], row_labels=[f"q{i}" for i in range(T)],
       col_labels=[f"k{i}" for i in range(T)])

# ---------------------------------------------------------------------------
# WHY DIVIDE BY sqrt(head_size)?
# ---------------------------------------------------------------------------
section("The mysterious square root")

print("""  The paper divides the scores by the square root of the key dimension:

      Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V
                                          ^^^^^^^^^

  It looks like an arbitrary fudge factor. It isn't, and you can see why in
  about ten seconds of arithmetic.

  A dot product adds up d_k separate multiplications. Add up more random
  numbers and the total naturally spreads out further -- the variance grows
  with d_k. So the bigger your vectors, the more extreme your scores get:""")

for d in [4, 16, 64, 256]:
    a, b = torch.randn(4000, d), torch.randn(4000, d)
    raw = (a * b).sum(dim=1)
    print(f"    d_k = {d:>4}:  scores spread about {raw.std():6.2f}"
          f"   -> after /sqrt(d_k): {(raw / d ** 0.5).std():.2f}")

print("""
  Now: why does a wide spread hurt? Because softmax exaggerates. Feed it
  gentle scores and you get a sensible blend; feed it extreme scores and it
  collapses onto a single winner:""")

gentle = torch.tensor([1.0, 0.5, -0.3, 0.2, -0.9])
extreme = gentle * 12
print(f"\n    gentle scores  {gentle.tolist()}")
print(f"      -> softmax   {[round(p, 3) for p in F.softmax(gentle, -1).tolist()]}")
print(f"\n    same scores x12")
print(f"      -> softmax   {[round(p, 3) for p in F.softmax(extreme, -1).tolist()]}")

print("""
  The second one is a one-hot vector in all but name: one word gets ~100% and
  the rest get nothing. That's bad for two reasons. The obvious one is that
  attention stops blending and starts picking a single word. The subtle and
  more damaging one is that softmax gradients go to zero when it saturates, so
  the model stops learning almost immediately -- and you're stuck there.

  Dividing by sqrt(d_k) keeps the scores in the gentle range at the start of
  training, so softmax stays soft and gradients keep flowing. That's the whole
  reason. One line of defence against a dead model.""")

scores = scores / (HEAD_SIZE ** 0.5)

# ---------------------------------------------------------------------------
# MASK, SOFTMAX, WEIGHTED SUM -- all unchanged from step 4
# ---------------------------------------------------------------------------
section("Mask and softmax: exactly as in step 4")

tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)

print("  The attention weights. Compare with step 4's flat average:\n")
matrix(weights[0], row_labels=[f"pos {i}" for i in range(T)],
       col_labels=[f"p{i}" for i in range(T)])
print("""
  Still lower-triangular -- no peeking. Still each row sums to 1. But the
  numbers are no longer all equal, and (once trained) they depend on the
  actual words in front of us. This matrix is different for every sentence.
  THAT is the entire difference between step 4 and step 5.""")

out = weights @ v
print(f"\n  output = weights @ v  ->  {tuple(out.shape)}")
print("  Note we blend the VALUES, not the raw input. Match on spines, read contents.")

# ---------------------------------------------------------------------------
# THE WHOLE THING AS ONE CLASS
# ---------------------------------------------------------------------------
section("The complete attention head")

print('''
    class AttentionHead(nn.Module):
        def forward(self, x):
            q = self.query(x)                          # what I want
            k = self.key(x)                            # what I offer
            v = self.value(x)                          # what I give

            scores = q @ k.transpose(-2,-1)            # match wants to offers
            scores = scores / (head_size ** 0.5)       # keep softmax gentle
            scores = scores.masked_fill(mask, -inf)    # no peeking ahead
            weights = F.softmax(scores, dim=-1)        # into percentages

            return weights @ v                         # blend the values

  Nine lines. That is the mechanism the entire paper is named after, and you
  have now seen every one of them derived from "the flat average was too dumb".
''')


class AttentionHead(nn.Module):
    """One attention head. This is the real implementation, used from here on."""

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
# TRAIN IT
# ---------------------------------------------------------------------------
section("Training a model with one attention head")

D_MODEL = 32


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)
        self.attention = AttentionHead(D_MODEL, D_MODEL, data.BLOCK_SIZE)
        self.predict = nn.Linear(D_MODEL, grammar.VOCAB_SIZE)

    def forward(self, idx):
        x = self.embed(idx)
        x = self.attention(x)
        return self.predict(x)


train_data, val_data = data.load()
model = AttentionModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

print("  step      loss")
for step in range(3001):
    x, y = data.get_batch(train_data, batch_size=64)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, grammar.VOCAB_SIZE), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"  {step:>5}   {loss.item():7.4f}   {bar(1 - loss.item() / 6.5, 30)}")


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


samples = [generate(model) for _ in range(1000)]
score = grammar.score_sentences(samples)
print("\n  Sentences from the attention model:")
for s in samples[:6]:
    print("    " + " ".join(s))

print(f"""
  Grammatical: {score:6.1%}  {bar(score)}

      step 3, bigram (1 word of context)   22.6%    loss 3.95
      step 4, flat average                  2.1%    loss 4.35
      step 5, attention                    {score:5.1%}    loss {loss.item():.2f}

  Attention completely undid the damage the flat average did -- same context,
  same mask, same softmax, and the only change is that the model now decides
  how much each past word matters. The loss is the best we've seen.

  And yet the grammar score is stuck at roughly where the one-word bigram was.
  We built the machine the paper is named after and it has not, so far, beaten
  a lookup table. Don't paper over that. Ask why.

WHY IT'S STUCK
--------------
  Look very carefully at the attention head, and answer this: where does it
  use the ORDER of the words?

  It doesn't. Nowhere. Queries and keys are computed from word vectors alone,
  and the vector for 'cat' is the same whether 'cat' is first or last. Attention
  matches purely on content, so as far as this model is concerned:

      the cat chased a mouse    ==    mouse a chased cat the

  which means it cannot express the single most useful relationship in all of
  language: "the word immediately before me". It has rich context and no idea
  where any of it is.

  Next step we prove that claim -- literally, by shuffling a sentence and
  watching the output not change -- and then fix it. It is the largest single
  jump in the whole course.

    Next:  python3 06_position.py
""")
