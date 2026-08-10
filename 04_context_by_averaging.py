"""
STEP 4  --  Looking at all the previous words (the crude version)

    Run me:  python3 04_context_by_averaging.py   (about 10 seconds)

This is the most important step in the course. If your audience understands
this file, step 5 -- attention itself -- is a one-line change.

THE SITUATION
-------------
Our model can only see one word back, and that's what's holding it at 22%.
We want it to see everything that came before. So here is the dumbest possible
way to do that:

    take all the previous words' vectors and average them.

That's it. "the hungry cat" becomes the average of the vectors for 'the',
'hungry' and 'cat'. One vector summarising everything so far.

Fair warning, because it's the whole point of this step: this is going to
FAIL. Our score will get worse, not better. Sit with that when it happens --
the specific way it fails tells us exactly what attention has to be, and a
failed experiment you understand is worth more than a success you don't.

What we build here is nonetheless the right machine. In step 5 we change ONE
thing about it and get attention.

Keep this sentence in your pocket for the whole file:

    ATTENTION IS THIS AVERAGE, WITH THE MODEL CHOOSING THE WEIGHTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import grammar, data
from lib.display import title, section, bar, matrix

torch.manual_seed(1337)

title(4, "Looking at all previous words: the crude version")

# ---------------------------------------------------------------------------
# THE RULE WE MUST NEVER BREAK
# ---------------------------------------------------------------------------
section("The one unbreakable rule: no peeking ahead")

print("""  We're training the model to predict the next word. The correct next word
  is sitting right there in the training data. So if position 3 is allowed to
  look at position 4, the model doesn't learn anything -- it just reads the
  answer off the page, scores perfectly in training, and is useless the moment
  you ask it to write something new.

  It's an exam where the answer sheet is stapled to the question paper.

  So: position t may look at positions 0..t, and never beyond. This is called
  CAUSAL masking (causes come before effects) or a "decoder" restriction. GPT
  models are "decoder-only" transformers, and this rule is exactly what that
  phrase means.""")

# ---------------------------------------------------------------------------
# VERSION 1: the obvious loop
# ---------------------------------------------------------------------------
section("Version 1: average the past, with a for-loop")

B, T, C = 1, 5, 2
x = torch.tensor([[[1., 0.], [0., 2.], [3., 1.], [1., 1.], [2., 4.]]])

averaged = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        previous = x[b, :t + 1]                 # positions 0..t INCLUSIVE
        averaged[b, t] = previous.mean(dim=0)

print("  Five word-vectors (pretend d_model is 2 so we can read them):")
matrix(x[0], row_labels=[f"pos {i}" for i in range(T)])
print("\n  Each position replaced by the average of itself and everything before:")
matrix(averaged[0], row_labels=[f"pos {i}" for i in range(T)])
print("""
  Position 0 sees only itself. Position 4 sees all five. Nobody sees the
  future. Check the arithmetic by hand -- row 1 is (1+0)/2 and (0+2)/2.""")

# ---------------------------------------------------------------------------
# VERSION 2: the same thing as one matrix multiply
# ---------------------------------------------------------------------------
section("Version 2: the identical result, as a matrix multiply")

weights = torch.tril(torch.ones(T, T))          # lower triangle of 1s
weights = weights / weights.sum(dim=1, keepdim=True)

print("  Build this weight matrix -- row t says how much each position feeds")
print("  into position t:\n")
matrix(weights, row_labels=[f"pos {i}" for i in range(T)],
       col_labels=[f"p{i}" for i in range(T)])

print("""
  Read row 2: "position 2 is one third position 0, one third position 1, one
  third position 2, and zero of positions 3 and 4." The zeros in the upper
  triangle are the no-peeking rule, made concrete.

  Now multiply:""")

averaged_v2 = weights @ x
print(f"\n  weights @ x  gives the same numbers: "
      f"{torch.allclose(averaged, averaged_v2)}")
matrix(averaged_v2[0], row_labels=[f"pos {i}" for i in range(T)])

print("""
  Same answer, no loop. This matters enormously in practice: a GPU can do that
  one matrix multiply for thousands of positions at once, while a loop crawls
  through them one at a time. "Averaging the past" and "multiplying by a
  triangular matrix" are the same operation, and the second one is fast.""")

# ---------------------------------------------------------------------------
# VERSION 3: the same thing again, via softmax  <-- THE BRIDGE
# ---------------------------------------------------------------------------
section("Version 3: the same thing AGAIN, this time built with softmax")

scores = torch.zeros(T, T)                                  # nobody has an opinion yet
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float("-inf"))       # block the future
weights_v3 = F.softmax(scores, dim=-1)

print("  Start with a table of scores that are all ZERO -- meaning 'every")
print("  previous position is equally interesting'. Then block out the future")
print("  by setting those entries to minus infinity:\n")
matrix(scores, row_labels=[f"pos {i}" for i in range(T)],
       col_labels=[f"p{i}" for i in range(T)], fmt="{:>6}")

print("\n  ...and softmax each row. (e^0 = 1, and e^-inf = 0, so the blocked")
print("  positions get exactly zero weight and the rest share equally.)\n")
matrix(weights_v3, row_labels=[f"pos {i}" for i in range(T)],
       col_labels=[f"p{i}" for i in range(T)])

print(f"\n  Identical to version 2: {torch.allclose(weights, weights_v3)}")

print("""
  Stop here. This is the moment.

  We just took a plain average and rebuilt it as: SCORES -> mask -> softmax ->
  weighted sum. Which is a ridiculous amount of machinery for "add them up and
  divide by five"... unless the scores stop being zero.

  Right now every score is 0, so every past word gets equal weight. But when
  you're deciding what word comes after "the hungry cat quietly", those words
  are NOT equally useful. 'cat' tells you a lot. 'the' tells you almost
  nothing. A flat average drowns the signal in noise.

  So: what if the model computed those scores itself, from the words, freshly
  for every sentence?

  That's attention. Everything in step 5 is about how the scores get filled in.
  The mask, the softmax, the weighted sum -- all of that is already built,
  right here, and it doesn't change.""")

# ---------------------------------------------------------------------------
# DOES IT ACTUALLY HELP? Let's train it.
# ---------------------------------------------------------------------------
section("Now let's train it -- and watch it fail")

D_MODEL = 32


class AveragingModel(nn.Module):
    """Embed -> average everything so far -> predict the next word."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)
        self.predict = nn.Linear(D_MODEL, grammar.VOCAB_SIZE)
        # register_buffer = "save this with the model, but it isn't learned"
        self.register_buffer("mask", torch.tril(torch.ones(data.BLOCK_SIZE,
                                                           data.BLOCK_SIZE)))

    def forward(self, idx):
        B, T = idx.shape
        vectors = self.embed(idx)                          # (B, T, C)

        scores = torch.zeros(T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                # (T, T)

        context = weights @ vectors                        # (B, T, C)
        return self.predict(context)                       # (B, T, vocab)


train_data, val_data = data.load()
model = AveragingModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

print("  step      loss")
for step in range(3001):
    x, y = data.get_batch(train_data, batch_size=64)
    logits = model(x)
    # Flatten (B,T,vocab) to (B*T,vocab) because cross_entropy wants a list of
    # independent predictions, and each position IS an independent prediction.
    loss = F.cross_entropy(logits.view(-1, grammar.VOCAB_SIZE), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"  {step:>5}   {loss.item():7.4f}   {bar(1 - loss.item() / 6.5, 30)}")


@torch.no_grad()
def generate(model, max_new=18):
    """Feed in a full stop, then keep appending whatever the model predicts."""
    idx = data.encode(["."]).unsqueeze(0)                 # (1, 1)
    out = []
    for _ in range(max_new):
        logits = model(idx[:, -data.BLOCK_SIZE:])         # never exceed context
        probs = F.softmax(logits[:, -1, :], dim=-1)       # only the LAST position
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        w = data.ID_TO_WORD[int(nxt)]
        if w == ".":
            break
        out.append(w)
    return out


samples = [generate(model) for _ in range(1000)]
score = grammar.score_sentences(samples)
print("\n  Sentences from the averaging model:")
for s in samples[:6]:
    print("    " + " ".join(s))

print(f"""
  Grammatical: {score:6.1%}  {bar(score)}

      step 0, counting        22.3%      loss  --
      step 3, neural bigram   22.6%      loss  3.95
      step 4, averaging        {score:4.1%}      loss  {loss.item():.2f}

  WORSE. Considerably worse, on both measures. We gave the model MORE
  information and it got dumber. Read those sentences again -- they ramble on
  and never finish, and the words no longer even fit together in pairs, which
  is the one thing the bigram model could do.

WHY IT FAILED -- and this is the clue we came for
-------------------------------------------------
  An average is a blur.

  When the model is at position 8 and needs the next word, the single most
  useful fact available is what word is at position 8. The bigram model had
  that fact perfectly. Our averaging model has it smeared into one ninth of a
  vector, mixed in with 'the' and 'a' and every other filler word that came
  before. It can no longer reliably tell what the previous word even was.

  So we traded precision for breadth, and it was a terrible trade.

  Notice what this rules out. The problem is NOT that context is useless --
  the sentence really does depend on words from far back. The problem is that
  a flat average is a stupid way to combine them. What the model actually
  needs, when standing at "the hungry cat quietly ___", is something like:

      0.55  on 'quietly'   (the word right before me)
      0.30  on 'cat'       (the subject -- tells me a verb is due)
      0.10  on 'hungry'
      0.05  on 'the'

  Breadth AND precision. Not one flat number for everyone.

  We already built the machinery to do that. Look back at version 3: scores ->
  mask -> softmax -> weighted sum. The only reason every word gets equal
  weight is that we filled the score table with zeros.

  So stop filling it with zeros. Let the model work out the scores.

    Next:  python3 05_self_attention.py
""")
