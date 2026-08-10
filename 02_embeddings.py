"""
STEP 2  --  Embeddings: giving words a place in space

    Run me:  python3 02_embeddings.py

Step 1 left us with a problem: word IDs are meaningless labels, but neural
networks can only do arithmetic. The fix is to describe every word not with
one number but with a LIST of numbers -- a vector.

THE ANALOGY THAT WORKS BEST
---------------------------
Think about how you'd describe a colour to a computer. You don't say "colour
number 47". You say (Red=255, Green=0, Blue=0). Three numbers, each meaning
something. And now arithmetic suddenly makes sense: orange really does sit
between red and yellow, and you can measure how far apart two colours are.

An embedding does the same thing for words. Instead of "cat = 63", we say
cat = [0.9, -0.2, 0.4, ...] -- a point in space. Words with similar meanings
end up near each other, and "near" becomes something you can actually compute.

The twist, and the reason this is powerful rather than tedious: nobody decides
what the numbers mean. The model works them out for itself while learning.
"""

import torch

import grammar
from display import title, section, matrix

torch.manual_seed(1337)

title(2, "Embeddings: giving words a place in space")

# ---------------------------------------------------------------------------
# PART A -- a hand-made embedding, so you can see what one IS
#
# In a real model these numbers are learned and not human-readable. Here we
# choose them by hand purely to make the idea concrete.
# ---------------------------------------------------------------------------
section("A: a hand-made 3-dimensional embedding")

#                     alive?  action?   big?
hand_made = {
    "cat":         [  0.9,    0.0,     -0.5],
    "dog":         [  0.9,    0.1,     -0.3],
    "mouse":       [  0.8,    0.0,     -0.9],
    "castle":      [ -0.9,    0.0,      0.9],
    "cottage":     [ -0.9,    0.0,      0.2],
    "chased":      [  0.0,    0.9,      0.0],
    "watched":     [  0.0,    0.8,      0.0],
}

print("  word        alive?  action?    big?")
for word, vec in hand_made.items():
    print(f"  {word:<10} " + "".join(f"{v:8.2f}" for v in vec))

print("""
Three numbers per word. We picked the meanings of the three slots ourselves:
"how alive is it", "is it an action", "how big is it". Every word is now a
point in a 3-D space, and geometry starts to mean something.
""")

# ---------------------------------------------------------------------------
# MEASURING SIMILARITY
#
# Cosine similarity = "do these two arrows point the same way?"
#   +1 = same direction (similar), 0 = unrelated, -1 = opposite.
# ---------------------------------------------------------------------------
section("Distance now means something")


def similarity(a, b):
    a, b = torch.tensor(a), torch.tensor(b)
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


pairs = [("cat", "dog"), ("cat", "mouse"), ("cat", "castle"),
         ("cat", "chased"), ("chased", "watched"), ("castle", "cottage")]
for a, b in pairs:
    s = similarity(hand_made[a], hand_made[b])
    verdict = "very similar" if s > 0.8 else "related" if s > 0.3 else \
              "unrelated" if s > -0.3 else "opposites"
    print(f"  {a:<8} vs {b:<8}  {s:+.2f}   {verdict}")

print("""
'cat' and 'dog' come out nearly identical. 'cat' and 'chased' come out
unrelated. We never wrote a rule saying so -- it falls out of the numbers.
THAT is what an embedding buys us.
""")

# ---------------------------------------------------------------------------
# PART B -- why a lookup table is secretly a matrix multiply
#
# This is the bit worth slowing down for. Beginners often see nn.Embedding as
# a magic box. It is not: it is one matrix multiplication, and understanding
# that makes the whole network feel like one uniform kind of thing.
# ---------------------------------------------------------------------------
section("B: a lookup table IS a matrix multiplication")

tiny_vocab = ["cat", "dog", "mouse", "castle", "chased"]
E = torch.tensor([hand_made[w] for w in tiny_vocab])       # 5 words x 3 numbers

print("Our embedding table E (5 words down, 3 features across):")
matrix(E, row_labels=tiny_vocab, col_labels=["alive", "act", "big"])

print("""
Now, the 'one-hot' trick. Write a word as a row of zeros with a single 1 in
the slot for that word. 'mouse' is word number 2, so:""")

one_hot_mouse = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
print(f"\n  one-hot for 'mouse' = {one_hot_mouse.tolist()[0]}")

result = one_hot_mouse @ E                       # @ means matrix multiply
print(f"  one_hot @ E         = {[round(x, 2) for x in result.tolist()[0]]}")
print(f"  E[2] (just row 2)   = {[round(x, 2) for x in E[2].tolist()]}")

print("""
Identical. Multiplying by a one-hot vector picks out one row -- the zeros
delete every other row and the single 1 keeps the one you want.

Why does this matter? Two reasons:

  1. Looking up a word is not a special operation bolted onto the network.
     It is the same matrix multiply as every other layer. The whole model is
     one consistent kind of machine.

  2. Because it's a matrix multiply, gradient descent can train it. The
     embedding table is not fixed data -- it is a set of PARAMETERS that get
     nudged and improved, exactly like every other weight in the model.

(In practice libraries skip the multiplication and just grab the row directly,
because multiplying by hundreds of zeros is a waste of electricity. Same
result, less work.)
""")

# ---------------------------------------------------------------------------
# PART C -- the real thing, as we'll actually use it
# ---------------------------------------------------------------------------
section("C: the real embedding table for our 500 words")

D_MODEL = 16          # how many numbers describe each word
embedding = torch.nn.Embedding(grammar.VOCAB_SIZE, D_MODEL)

print(f"  vocabulary size : {grammar.VOCAB_SIZE} words")
print(f"  d_model         : {D_MODEL} numbers per word")
print(f"  table shape     : {tuple(embedding.weight.shape)}")
print(f"  parameters      : {embedding.weight.numel():,} numbers to be learned")

word_to_id = {w: i for i, w in enumerate(grammar.VOCAB)}
cat_id = torch.tensor([word_to_id["cat"]])
print(f"\n  'cat' before any training:")
print(f"    {[round(x, 2) for x in embedding(cat_id)[0].tolist()]}")

print(f"""
Random noise. Right now 'cat' and 'dog' are as unrelated as 'cat' and 'the'.
Nothing has been learned, because nothing has been trained.

  d_model is the name the paper gives to that 16. In "Attention Is All You
  Need" (section 3.2.2) they use d_model = 512. GPT-3 uses 12,288. It is
  simply how much room each word gets to describe itself -- more room, more
  nuance, more compute.

WHERE WE ARE
------------
We can turn words into numbers we're allowed to do maths on. But the numbers
are currently garbage. Next we build a training loop that improves them --
and we'll watch 'cat' and 'dog' drift together on their own.

    Next:  python3 03_neural_bigram.py
""")
