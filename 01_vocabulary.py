"""
STEP 1  --  Turning words into numbers

    Run me:  python3 01_vocabulary.py

A neural network is a pile of multiplication and addition. It cannot multiply
the word "cat". So before anything else, we need to turn text into numbers.

This step is short and easy. But it ends with a trap that most beginners walk
straight into, and spotting that trap is what makes step 2 feel necessary
rather than arbitrary.
"""

import grammar
from display import title, section

title(1, "Turning words into numbers")

# ---------------------------------------------------------------------------
# THE VOCABULARY: a numbered list of every word the model can know
# ---------------------------------------------------------------------------
section("The vocabulary")

vocab = grammar.VOCAB                     # already sorted, so it never changes
print(f"Our model knows exactly {len(vocab)} words.")
print(f"  first five: {vocab[:5]}")
print(f"  last five:  {vocab[-5:]}")

print("""
This is a hard boundary. A word that is not on this list simply does not exist
as far as the model is concerned -- it cannot read it and cannot say it.
Real models have the same limit, just with a bigger list: GPT-4's vocabulary is
about 100,000 entries. Ours is 500 so that everything stays inspectable.
""")

# ---------------------------------------------------------------------------
# THE TWO DICTIONARIES
# ---------------------------------------------------------------------------
section("Two lookup tables")

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for i, word in enumerate(vocab)}

for w in ["cat", "dog", "chased", "the", "."]:
    print(f"  '{w}'  is word number {word_to_id[w]}")


def encode(words):
    """Words in, numbers out. ['the','cat'] -> [432, 63]"""
    return [word_to_id[w] for w in words]


def decode(ids):
    """Numbers in, words out. [432, 63] -> ['the','cat']"""
    return [id_to_word[i] for i in ids]


section("Encoding and decoding")

sentence = ["the", "hungry", "cat", "quietly", "chased", "a", "small", "mouse", "."]
ids = encode(sentence)

print(f"  text:    {' '.join(sentence)}")
print(f"  numbers: {ids}")
print(f"  back:    {' '.join(decode(ids))}")
print("\n  Nothing is lost -- encode and decode are perfect mirrors of each other.")

print("""
This is all "tokenization" means. Real models chop text into pieces smaller
than words ("unbelievable" -> "un" + "believ" + "able") so that a fixed-size
list can cover any text at all, including words it has never seen. That's a
practical trick, not a conceptual one, so we skip it: for us, one word is one
token. Everything you learn here transfers directly.
""")

# ---------------------------------------------------------------------------
# THE TRAP
#
# This is the most valuable 20 seconds in the whole step. Ask the audience the
# question and let them sit with it before revealing the answer.
# ---------------------------------------------------------------------------
section("The trap: these numbers are labels, not quantities")

print(f"  'cat' is {word_to_id['cat']}, 'dog' is {word_to_id['dog']}, "
      f"'castle' is {word_to_id['castle']}")

print(f"""
Ask your audience: is 'dog' ({word_to_id['dog']}) bigger than 'cat' ({word_to_id['cat']})?
Is 'cat' plus 'dog' equal to {word_to_id['cat']} + {word_to_id['dog']} = {word_to_id['cat'] + word_to_id['dog']}, which is '{id_to_word[(word_to_id['cat'] + word_to_id['dog']) % len(vocab)]}'?

Obviously not. These IDs are jersey numbers. Player 10 is not twice as good as
player 5. The numbers came from alphabetical order and carry no meaning at all:
'cat' and 'castle' sit next to each other only because of how they're spelled,
while 'cat' and 'kitten' are hundreds apart despite being nearly the same idea.

But a neural network only knows how to do arithmetic. Feed it these IDs and it
WILL do arithmetic on them, and it will be arithmetic on nonsense.

So we have a real problem:

    a word must be represented by numbers you are allowed to do maths on.

The answer is to stop describing each word with one number, and start
describing it with a whole list of numbers -- where each number means
something, and is LEARNED rather than assigned.

That is the single most important idea in modern AI, and it's next.

    Next:  python3 02_embeddings.py
""")
