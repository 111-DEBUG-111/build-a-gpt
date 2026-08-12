"""
STEP 3  --  Our first neural network, and our first training loop

    Run me:  python3 03_neural_bigram.py   (a few seconds)

In step 0 we built a next-word guesser by counting. Now we build one that
LEARNS instead of counting -- and it will arrive at almost exactly the same
answer. That sounds like a waste of effort. It is the opposite: it's the
proof that this strange machinery does something sensible, checked against an
answer we already know is right.

Counting only works when you look one word back. Learning keeps working when
we look at everything. But we should verify the method on a problem we can
check before we trust it on one we can't.

WHAT YOU'LL MEET IN THIS FILE
-----------------------------
  logits         the model's raw opinion scores, one per word in the vocabulary
  softmax        turns those scores into percentages that add to 100%
  cross-entropy  how surprised the model was by the true answer -- the "loss"
  gradient       which way to nudge each number to be less wrong
  optimizer      the thing that applies the nudges

Four of these five have intimidating names and simple meanings. Take them one
at a time and none of them is hard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

from lib import grammar
from lib.display import title, section, bar

torch.manual_seed(1337)

title(3, "Our first neural network: learning instead of counting")

# ---------------------------------------------------------------------------
# THE DATA
#
# A "training example" is a question and its correct answer:
#     question: the word 'cat'      answer: the word that actually came next
# We get thousands of these free, just by sliding along our text. Nobody had
# to label anything. This is why language models can be trained on the whole
# internet -- the text labels itself.
# ---------------------------------------------------------------------------
section("Building training examples")

corpus = grammar.make_corpus(n_sentences=20000)
word_to_id = {w: i for i, w in enumerate(grammar.VOCAB)}
id_to_word = {i: w for i, w in enumerate(grammar.VOCAB)}

data = torch.tensor([word_to_id[w] for w in corpus])
inputs = data[:-1]        # This channel is debug explains
targets = data[1:]        # channel is debug explains .

print(f"  {len(inputs):,} training examples, free of charge.")
print(f"  For example:  input '{id_to_word[int(inputs[0])]}'"
      f"  ->  correct answer '{id_to_word[int(targets[0])]}'")

# ---------------------------------------------------------------------------
# THE MODEL
# ---------------------------------------------------------------------------
section("The model: two layers, that's all")

D_MODEL = 32


class NeuralBigram(nn.Module):
    """
    Word in -> guess for the next word out. Two steps:

        1. EMBED     look up the 32 numbers that describe this word (step 2)
        2. PREDICT   turn those 32 numbers into 500 scores, one per vocabulary
                     word, saying how likely each is to come next

    That second step is a "linear layer", which is just: multiply by a matrix
    and add a bias. Every weight in both steps starts as random noise and gets
    improved by training.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(grammar.VOCAB_SIZE, D_MODEL) 
        self.predict = nn.Linear(D_MODEL, grammar.VOCAB_SIZE)

    def forward(self, word_ids):
        vectors = self.embed(word_ids)        # (batch, 32) 
        logits = self.predict(vectors)        # (batch, 500)
        return logits


model = NeuralBigram()
n_params = sum(p.numel() for p in model.parameters())
print(f"  {n_params:,} parameters -- {n_params} individual numbers to be tuned.")

# ---------------------------------------------------------------------------
# LOGITS AND SOFTMAX
# ---------------------------------------------------------------------------
section("Logits -> probabilities (softmax)")

test_word = torch.tensor([word_to_id["cat"]])
logits = model(test_word)

print(f"  Feeding in 'cat' gives {logits.shape[1]} raw scores called LOGITS.")
print(f"  The first eight: {[round(x, 2) for x in logits[0, :8].tolist()]}")
print("""
  Logits are unbounded opinions: big means 'likely', negative means 'unlikely',
  but they don't add up to anything meaningful. SOFTMAX fixes that. It does
  two things:
      1. e^x on every score, which makes them all positive and exaggerates
         the gaps (a strong opinion becomes a much stronger one)
      2. divide by the total, so everything adds to 100%

  Think of a race: logits are how fast each runner looks; softmax converts
  that into each runner's chance of winning.""")

probs = F.softmax(logits, dim=-1)
print(f"\n  After softmax, first eight: {[round(x, 4) for x in probs[0, :8].tolist()]}")
print(f"  They all sum to: {probs.sum().item():.4f}")

# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------
section("Loss: how wrong were we?")

print("""  We score a guess by how much probability it gave to the RIGHT answer:

      loss = -log(probability the model assigned to the true next word)

  Gave the right word 100%? loss = -log(1.0)  = 0.00   perfect
  Gave it 50%?              loss = -log(0.5)  = 0.69   decent
  Gave it 1%?               loss = -log(0.01) = 4.61   bad
  Gave it almost 0?         loss shoots to infinity     catastrophic

  This is "cross-entropy loss". It's a surprise meter: a low number means the
  true answer was no surprise. Training = making the model less surprised.

  SANITY CHECK -- do this every time you build a model. An untrained model
  spreads its guesses evenly over all 500 words, so it should give the true
  word about 1/500, and the loss should be about -log(1/500):""")

with torch.no_grad():
    starting_loss = F.cross_entropy(model(inputs[:5000]), targets[:5000])
import math
print(f"\n      expected at random : {math.log(grammar.VOCAB_SIZE):.3f}")
print(f"      our model actually : {starting_loss.item():.3f}   <- matches, nothing is broken")

# ---------------------------------------------------------------------------
# THE TRAINING LOOP
# ---------------------------------------------------------------------------
section("Training")

print("""  Four lines, repeated a few thousand times:

      logits = model(x)                 guess
      loss   = cross_entropy(logits, y) measure how wrong
      loss.backward()                   work out which way each number should move
      optimizer.step()                  move them all a tiny bit

  loss.backward() is the only genuinely clever line. It uses the chain rule
  from calculus to compute, for all 32,000 numbers at once, the direction that
  would reduce the loss. PyTorch does it automatically -- for today, treat it
  as a black box that answers "which way is downhill?".

  The blindfolded-hiker picture: you're on a foggy hillside trying to reach
  the valley. You can't see anything, but you can feel which way the ground
  slopes under your feet. So you take a small step downhill, and repeat. The
  step size is the LEARNING RATE. Too big and you leap over the valley; too
  small and you're there all week.
""")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
BATCH_SIZE = 512
STEPS = 3000

print("  step      loss")
for step in range(STEPS + 1):
    # Grab a random handful of examples rather than all 163,000 -- far faster,
    # and the noise it introduces actually helps training. This is a "batch".
    pick = torch.randint(0, len(inputs), (BATCH_SIZE,))
    x, y = inputs[pick], targets[pick]

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()      # clear last round's gradients
    loss.backward()            # which way is downhill?
    optimizer.step()           # take the step

    if step % 500 == 0:
        print(f"  {step:>5}   {loss.item():7.4f}   {bar(1 - loss.item() / 6.5, 30)}")

# ---------------------------------------------------------------------------
# DID IT REDISCOVER COUNTING?
# ---------------------------------------------------------------------------
section("The check: did learning find the same answer as counting?")

counts = defaultdict(Counter)
for a, b in zip(corpus, corpus[1:]):
    counts[a][b] += 1

with torch.no_grad():
    learned = F.softmax(model(torch.tensor([word_to_id["cat"]])), dim=-1)[0]

cat_total = sum(counts["cat"].values())
print("  What follows 'cat'?\n")
print("    word           by counting    by learning")
for w, n in counts["cat"].most_common(8):
    print(f"    {w:<12}    {n / cat_total:8.1%}      {learned[word_to_id[w]].item():8.1%}")

print("""
  Two completely different methods, the same answer. Gradient descent found
  the statistics of the language on its own, without ever being told to count.
""")

# ---------------------------------------------------------------------------
# THE FREE GIFT: the embeddings organised themselves
# ---------------------------------------------------------------------------
section("A bonus nobody asked for: the embeddings sorted themselves out")

with torch.no_grad():
    E = model.embed.weight
    E_normalised = E / E.norm(dim=1, keepdim=True)
    sims = E_normalised @ E_normalised.T          # every word vs every word

for probe in ["cat", "chased", "the", "quickly"]:
    i = word_to_id[probe]
    best = sims[i].topk(6).indices.tolist()[1:]   # skip the word itself
    neighbours = ", ".join(f"{id_to_word[j]}" for j in best)
    print(f"  nearest to '{probe}' ({grammar.PART_OF_SPEECH[probe]}): {neighbours}")

print("""
  Look at the parts of speech. Nouns landed near nouns, verbs near verbs.

  We never told the model what a noun is. We never mentioned grammar. All it
  ever did was try to predict the next word -- and the most efficient way to
  do that is to notice that certain words behave alike, and file them together.
  Structure emerged from prediction. That one sentence is most of why this
  whole field works.
""")

# ---------------------------------------------------------------------------
# AND YET
# ---------------------------------------------------------------------------
section("The wall we've hit")


@torch.no_grad()
def generate(max_length=20):
    word_id = torch.tensor([word_to_id["."]])
    out = []
    for _ in range(max_length):
        probs = F.softmax(model(word_id), dim=-1)
        word_id = torch.multinomial(probs, num_samples=1)[0]
        w = id_to_word[int(word_id)]
        if w == ".":
            break
        out.append(w)
    return out


samples = [generate() for _ in range(1000)]
score = grammar.score_sentences(samples)
for s in samples[:6]:
    print("  " + " ".join(s))
print(f"\n  Grammatical: {score:6.1%}  {bar(score)}")

print("""
  Roughly the same 22% we got by counting in step 0. Of course it is -- we
  built a fancier machine to solve the identical, still-too-small problem.

  We are still looking at exactly ONE word before guessing. The model is
  perfectly trained and structurally blind.

  So: stop improving how we look at one word. Start looking at more words.

    Next:  python3 04_context_by_averaging.py
""")
