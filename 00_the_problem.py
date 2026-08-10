"""
STEP 0  --  What is a language model, really?

    Run me:  python3 00_the_problem.py

There is no machine learning in this file. None. We are going to build a
working text generator using nothing but counting, and it will genuinely
produce English-looking sentences. The point is to strip away the mystique
before we add any mathematics.

THE ONE IDEA
------------
A language model does exactly one thing:

    Given the words so far, guess the next word.

That's it. Everything else -- attention, embeddings, the entire Transformer --
is machinery in service of that single question. ChatGPT writing an essay is
this operation run a few thousand times in a row, each guess appended to the
text and fed back in.

TEACHING NOTE (an analogy that lands well)
------------------------------------------
It's the predictive text on your phone keyboard. You type "I'll be there in
five" and it offers "minutes". It isn't thinking about time; it has just seen
that "five" is followed by "minutes" an awful lot. Today we build that. Over
the next ten steps we make it dramatically smarter -- but never anything other
than a next-word guesser.
"""

import random
from collections import defaultdict, Counter

import grammar
from display import title, section, bar

rng = random.Random(0)

title(0, "A language model is just a next-word guesser")

# ---------------------------------------------------------------------------
# THE TRAINING TEXT
# ---------------------------------------------------------------------------
section("Our training text")

corpus = grammar.make_corpus(n_sentences=20000)
print(f"We have {len(corpus):,} words of text, written in a 500-word language.")
print(f"It begins:  {' '.join(corpus[:14])} ...")

# ---------------------------------------------------------------------------
# BASELINE: what does "no knowledge at all" look like?
#
# Always start here when teaching. If you don't show the floor, nobody can
# appreciate the ceiling.
# ---------------------------------------------------------------------------
section("Baseline: picking words completely at random")

random_sentences = [[rng.choice(grammar.VOCAB) for _ in range(6)] for _ in range(500)]
for s in random_sentences[:3]:
    print("  " + " ".join(s))
random_score = grammar.score_sentences(random_sentences)
print(f"\n  Grammatical: {random_score:6.1%}  {bar(random_score)}")

# ---------------------------------------------------------------------------
# THE COUNTING MODEL
#
# Walk through the text. Every time we see word A followed by word B, add one
# tally mark to the box labelled (A -> B). That's the entire training process.
# ---------------------------------------------------------------------------
section("Training by counting (this IS the training)")

counts = defaultdict(Counter)
for current_word, next_word in zip(corpus, corpus[1:]):
    counts[current_word][next_word] += 1

print(f"We filled in {len(counts)} boxes, one per word we've ever seen.")
print("\nHere is the box for the word 'cat' -- what tends to follow it:")
for word, n in counts["cat"].most_common(6):
    total = sum(counts["cat"].values())
    print(f"    cat -> {word:<12} seen {n:>4} times   ({n/total:5.1%})")

# ---------------------------------------------------------------------------
# GENERATING
# ---------------------------------------------------------------------------
section("Generating new sentences from the tallies")


def next_word_after(word):
    """
    Pick the next word by rolling a weighted die.

    IMPORTANT and easy to gloss over: we SAMPLE, we don't take the most common
    word. If we always took the most likely word, the model would say the exact
    same sentence every single time. Randomness is where novelty comes from.
    A model that always picks the top choice is a parrot; one that samples is
    a writer. (We revisit this in step 9 as "temperature".)
    """
    options = counts[word]
    words = list(options.keys())
    weights = list(options.values())
    return rng.choices(words, weights=weights, k=1)[0]


def generate_sentence(max_length=20):
    """Start just after a full stop, and keep guessing until we hit one."""
    word = "."                      # '.' means "a sentence is about to begin"
    sentence = []
    for _ in range(max_length):
        word = next_word_after(word)
        if word == ".":
            break
        sentence.append(word)
    return sentence


print("Ten sentences this model invented (it has never seen these before):")
for _ in range(10):
    print("  " + " ".join(generate_sentence()))

bigram_sentences = [generate_sentence() for _ in range(1000)]
bigram_score = grammar.score_sentences(bigram_sentences)
print(f"\n  Grammatical: {bigram_score:6.1%}  {bar(bigram_score)}")
print(f"  (up from {random_score:.1%} with pure randomness -- counting really does work)")

# ---------------------------------------------------------------------------
# NOW THE IMPORTANT PART: WHY IT FAILS
#
# Do not rush past this. The rest of the course is a sequence of fixes for the
# problem we are about to expose.
# ---------------------------------------------------------------------------
section("Where it breaks, and why")

print("Look closely at the failures -- they are all the same failure:\n")
shown = 0
for s in bigram_sentences:
    if not grammar.is_grammatical(s) and 3 <= len(s) <= 9:
        print(f"    {' '.join(s)}")
        shown += 1
        if shown == 5:
            break

print("""
Every sentence above is fine if you read any TWO adjacent words. The breakage
only appears when you look at the whole thing.

Here is the reason, and it is embarrassingly simple:

    our model looks at exactly one word before deciding the next one.

When it is standing on the word 'cat', it has no idea whether that 'cat' is
the subject of a sentence just getting started, or the object of a sentence
about to finish. It has forgotten. Watch:""")

cat_total = sum(counts["cat"].values())
by_pos = Counter()
for w, n in counts["cat"].items():
    by_pos[grammar.PART_OF_SPEECH.get(w, "?")] += n

print("\n  After the word 'cat', our model thinks the next word is:")
for pos, n in by_pos.most_common():
    meaning = {
        "VERB": "the cat is a subject, sentence continues",
        "ADV":  "the cat is a subject, sentence continues",
        "PREP": "the cat was an object, phrase continues",
        "END":  "the cat was an object, sentence is over",
    }.get(pos, "")
    print(f"    {n/cat_total:5.1%}  a {pos:<5}  {bar(n/cat_total, 24)}  {meaning}")

print("""
It is guessing blind between two completely different situations, because the
one clue that would settle it -- 'have we had a verb yet?' -- may be five
words back, and our model cannot see five words back.

THE FIX WE NEED
---------------
Look at ALL the previous words, not just one.

That sounds easy. It is not, and the difficulties are what produce every
component of the Transformer:

  * A word must become something we can do arithmetic on   -> step 2, embeddings
  * We need a way to learn from data rather than tally     -> step 3, training
  * We need to combine many previous words into one clue   -> step 4, averaging
  * The model must decide WHICH previous words matter      -> step 5, ATTENTION
  * 'the cat' and 'cat the' must not look identical        -> step 6, position

Attention -- the thing the paper is named after -- is the answer to a question
we have now actually earned the right to ask.

    Next:  python3 01_vocabulary.py
""")
