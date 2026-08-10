"""
grammar.py  --  our tiny artificial language

WHY WE INVENT A LANGUAGE INSTEAD OF USING A REAL BOOK
-----------------------------------------------------
We could train on Shakespeare. But with only 500 words, the model's output
would be mush, and we'd have no way to answer the one question that matters
when you're learning: "did the model actually learn anything?"

So we invent a language with EXACTLY 500 words and a handful of strict rules.
Now we can ask a precise question: what percentage of the sentences our model
invents obey the rules? That number runs from 0% (words picked at random),
through 22% (counting word pairs, step 0), to 97% (the finished Transformer,
step 9) -- and you can watch it climb. Learning becomes visible.

Better still, because we wrote the generator ourselves we can compute the
language's exact entropy: the lowest loss any model could ever achieve on it.
See entropy_per_token() at the bottom of this file. Having a known finish line
turns out to be the sharpest teaching tool in the whole course.

Think of it like teaching someone chess. You don't start them on a grandmaster
game; you start with "here's how a knight moves" so they can tell right from
wrong immediately.

THE RULES OF OUR LANGUAGE  (read "->" as "is made of")

    SENTENCE     -> NOUN_PHRASE + VERB_PHRASE.
    
    NOUN_PHRASE  -> DETERMINER + NOUN,
                  | DETERMINER +  ADJECTIVE + NOUN
    VERB_PHRASE  -> VERB + NOUN_PHRASE,
                  | ADVERB + VERB + NOUN_PHRASE,
                  | VERB + NOUN_PHRASE + PREPOSITION + NOUN_PHRASE

So these are legal:

    the cat chased a mouse .
    every hungry dog quietly watched the tall bird .
    some child found a golden coin beneath the wooden bridge .

and these are not:

    cat the chased mouse .          (determiner must come before its noun)
    the cat mouse .                 (no verb)
    the hungry hungry cat slept .   (only one adjective; verb needs an object)

Notice that these rules force LONG-RANGE thinking. When the model is choosing
the word after "the hungry", it needs a noun. But to know the sentence isn't
finished, it has to remember there hasn't been a verb yet -- which might be
six words back. A model that only looks at the previous word cannot do this.
That limitation is exactly what drives us, step by step, to the Transformer.
"""

import math
import random

# ---------------------------------------------------------------------------
# THE 500 WORDS
#
# One small stylistic rule: every adjective and noun starts with a consonant.
# That way "a cat" and "a golden coin" are always correct and we never have to
# worry about "a apple" vs "an apple". One less thing to explain.
# ---------------------------------------------------------------------------

DETERMINERS = [
    "the", "a", "every", "some", "this", "that", "one", "another",
]

PREPOSITIONS = [
    "near", "beside", "under", "behind", "beneath", "toward",
    "past", "across", "through", "around", "beyond", "without",
]

ADVERBS = [
    "quickly", "slowly", "quietly", "loudly", "gently", "fiercely",
    "calmly", "suddenly", "carefully", "boldly", "sadly", "happily",
    "bravely", "wildly", "softly", "sharply", "silently", "patiently",
    "eagerly", "nervously", "cheerfully", "clumsily", "gracefully",
    "hungrily", "lazily", "politely", "proudly", "rudely", "seriously",
    "sweetly", "tenderly", "wearily", "wisely", "angrily", "blindly",
    "briskly", "cautiously", "dearly", "deeply", "doubtfully",
]

ADJECTIVES = [
    "hungry", "clever", "small", "tall", "quiet", "loud", "brave", "gentle",
    "happy", "sad", "young", "wise", "kind", "cruel", "lazy", "quick",
    "slow", "bright", "dark", "heavy", "light", "warm", "cold", "dry",
    "wet", "rough", "smooth", "sharp", "dull", "clean", "dirty", "rich",
    "poor", "strong", "weak", "thick", "thin", "wide", "narrow", "deep",
    "shallow", "distant", "new", "fresh", "stale", "sweet", "sour", "bitter",
    "salty", "plain", "fancy", "simple", "strange", "common", "rare", "wild",
    "tame", "free", "busy", "calm", "restless", "curious", "careful",
    "careless", "cheerful", "gloomy", "friendly", "lonely", "noisy",
    "peaceful", "polite", "proud", "shy", "silly", "serious", "silent",
    "gracious", "graceful", "clumsy", "sturdy", "fragile", "golden",
    "silver", "crimson", "purple", "green", "blue", "black", "white",
    "grey", "brown", "yellow", "pink", "tiny", "giant", "huge", "massive",
    "minor", "major", "curly", "straight", "crooked", "hollow", "solid",
    "frozen", "burning", "dusty", "muddy", "misty", "foggy", "sunny",
    "cloudy", "stormy", "windy", "snowy", "hidden", "secret", "forgotten",
    "faithful", "wooden", "crowded", "cracked", "worn",
]

NOUNS = [
    # creatures
    "cat", "dog", "bird", "mouse", "horse", "fish", "fox", "wolf", "bear",
    "deer", "rabbit", "squirrel", "frog", "turtle", "snake", "lizard",
    "heron", "hawk", "crow", "sparrow", "robin", "duck", "goose", "swan",
    "chicken", "cow", "sheep", "goat", "pig", "donkey", "camel", "lion",
    "tiger", "monkey", "whale", "dolphin", "shark", "crab", "beetle",
    "spider", "bee", "moth", "kitten", "lamb", "calf", "pony", "beaver",
    "badger", "hedgehog", "seal", "raven", "finch", "trout",
    # people
    "child", "boy", "girl", "man", "woman", "farmer", "baker", "sailor",
    "soldier", "teacher", "student", "doctor", "painter", "singer",
    "dancer", "writer", "hunter", "fisher", "merchant", "traveller",
    "stranger", "neighbour", "friend", "brother", "sister", "mother",
    "father", "king", "queen", "prince", "princess", "knight", "guard",
    "thief", "beggar", "shepherd", "miller", "smith", "potter", "weaver",
    "cook", "driver", "pilot", "captain",
    # places
    "house", "barn", "tower", "bridge", "castle", "cottage", "garden",
    "field", "forest", "meadow", "hill", "mountain", "valley", "river",
    "lake", "pond", "stream", "sea", "shore", "beach", "desert", "cave",
    "path", "road", "street", "village", "city", "market", "harbour",
    # things
    "ship", "boat", "cart", "wagon", "train", "bicycle", "ladder",
    "window", "door", "roof", "wall", "fence", "gate", "chair", "table",
    "basket", "bottle", "cup", "bowl", "plate", "spoon", "knife", "candle",
    "lantern", "mirror", "clock", "book", "letter", "map", "coin", "ring",
    "crown", "sword", "shield", "hammer", "needle", "thread", "rope",
    "blanket", "pillow", "hat", "coat", "shoe", "glove", "drum", "flute",
    "bell", "kettle", "barrel", "chest", "key", "lock", "brush", "comb",
    # nature and abstractions
    "peach", "bread", "cheese", "honey", "berry", "flower", "tree", "leaf",
    "branch", "root", "stone", "feather", "shadow", "cloud", "storm",
    "wind", "rain", "snow", "moon", "star", "fire", "song", "story",
    "dream", "message", "promise", "puzzle", "riddle", "journey",
    "mushroom", "pebble", "ribbon", "button", "pocket", "curtain",
    "carpet", "saddle",
]

VERBS = [
    "chased", "watched", "found", "followed", "carried", "dropped",
    "caught", "pushed", "pulled", "lifted", "held", "wanted", "needed",
    "loved", "feared", "trusted", "guarded", "buried", "painted",
    "drew", "wrote", "sold", "bought", "traded", "gave", "took",
    "brought", "sent", "kept", "lost", "broke", "fixed", "built",
    "burned", "washed", "cleaned", "filled", "emptied", "opened",
    "closed", "locked", "counted", "measured", "weighed", "tasted",
    "smelled", "touched", "heard", "noticed", "studied", "remembered",
    "imagined", "described", "praised", "blamed", "warned", "greeted",
    "thanked", "invited", "visited", "joined", "passed", "crossed",
    "entered", "circled", "approached", "avoided", "ignored", "answered",
    "questioned", "taught", "showed", "offered", "refused", "accepted",
    "borrowed", "returned", "shared", "tied", "wrapped", "folded",
    "sliced", "stirred", "mixed", "poured", "planted", "picked",
    "gathered", "sorted", "stacked", "moved", "dragged", "tossed",
    "rescued", "hunted", "traced", "sketched", "polished", "repaired",
]

# The full stop is a real word as far as the model is concerned. It is the
# token that means "this sentence is over" -- and learning WHEN to produce it
# is one of the harder things the model has to figure out.
END = "."

# Every word our model will ever know, in one fixed, sorted order.
# Sorting matters only because it makes the word<->number mapping stable:
# rerun the code tomorrow and "cat" still gets the same ID.
VOCAB = sorted(
    set(DETERMINERS + PREPOSITIONS + ADVERBS + ADJECTIVES + NOUNS + VERBS)
    | {END}
)

VOCAB_SIZE = len(VOCAB)

# A promise we make to the reader in the very first lesson. If someone edits
# the word lists above, this line tells them immediately.
assert VOCAB_SIZE == 500, f"vocabulary should be 500 words, got {VOCAB_SIZE}"

# No word may appear in two categories -- otherwise "is this sentence legal?"
# would have no single answer.
_all_listed = DETERMINERS + PREPOSITIONS + ADVERBS + ADJECTIVES + NOUNS + VERBS
assert len(_all_listed) == len(set(_all_listed)), "a word appears in two categories"

# Handy lookup: word -> its part of speech. Used by the grammar checker.
PART_OF_SPEECH = {}
for _group_name, _group in [
    ("DET", DETERMINERS), ("PREP", PREPOSITIONS), ("ADV", ADVERBS),
    ("ADJ", ADJECTIVES), ("NOUN", NOUNS), ("VERB", VERBS),
]:
    for _w in _group:
        PART_OF_SPEECH[_w] = _group_name
PART_OF_SPEECH[END] = "END"


# ---------------------------------------------------------------------------
# MAKING SENTENCES
# ---------------------------------------------------------------------------

def _noun_phrase(rng):
    """DETERMINER [ADJECTIVE] NOUN  --  e.g. 'the cat' or 'a hungry cat'."""
    words = [rng.choice(DETERMINERS)]
    if rng.random() < 0.5:                      # half our noun phrases get an adjective
        words.append(rng.choice(ADJECTIVES))
    words.append(rng.choice(NOUNS))
    return words


def _verb_phrase(rng):
    """One of three shapes, picked at random."""
    roll = rng.random()
    if roll < 0.45:
        # 'chased the mouse'
        return [rng.choice(VERBS)] + _noun_phrase(rng)
    elif roll < 0.75:
        # 'quietly chased the mouse'
        return [rng.choice(ADVERBS), rng.choice(VERBS)] + _noun_phrase(rng)
    else:
        # 'chased the mouse across a field'
        return ([rng.choice(VERBS)] + _noun_phrase(rng)
                + [rng.choice(PREPOSITIONS)] + _noun_phrase(rng))


def make_sentence(rng=random):
    """Build one legal sentence, as a list of words ending with '.'."""
    return _noun_phrase(rng) + _verb_phrase(rng) + [END]


def make_corpus(n_sentences=20000, seed=1337):
    """
    Build our training text: one long list of words, sentences run together.

    We do NOT keep sentences separate. We glue them into a single stream,
    exactly the way real language models are trained. The '.' token is what
    tells the model where one sentence ends and the next begins -- and it has
    to learn that from the data, we never tell it.
    """
    rng = random.Random(seed)
    words = []
    for _ in range(n_sentences):
        words.extend(make_sentence(rng))
    return words


# ---------------------------------------------------------------------------
# THE GRAMMAR CHECKER  --  our scoreboard
#
# This is what lets us put a number on "is the model learning?". It walks a
# sentence left to right and checks it against the rules at the top of this
# file. It is an ordinary hand-written parser -- no machine learning here.
# ---------------------------------------------------------------------------

def _match_noun_phrase(words, i):
    """If words[i:] starts with a noun phrase, return the index just past it."""
    if i >= len(words) or PART_OF_SPEECH.get(words[i]) != "DET":
        return None
    i += 1
    if i < len(words) and PART_OF_SPEECH.get(words[i]) == "ADJ":
        i += 1
    if i >= len(words) or PART_OF_SPEECH.get(words[i]) != "NOUN":
        return None
    return i + 1


def is_grammatical(words):
    """
    True if this exact list of words is a legal sentence of our language.
    (The trailing '.' is optional here so we can also check partial output.)
    """
    words = list(words)
    if words and words[-1] == END:
        words = words[:-1]
    if not words:
        return False

    i = _match_noun_phrase(words, 0)             # the subject
    if i is None:
        return False

    if i < len(words) and PART_OF_SPEECH.get(words[i]) == "ADV":
        i += 1                                   # optional adverb
    if i >= len(words) or PART_OF_SPEECH.get(words[i]) != "VERB":
        return False
    i += 1

    i = _match_noun_phrase(words, i)             # the object
    if i is None:
        return False
    if i == len(words):
        return True                              # 'the cat chased a mouse'

    if PART_OF_SPEECH.get(words[i]) != "PREP":   # optional trailing phrase
        return False
    i = _match_noun_phrase(words, i + 1)
    return i == len(words)                       # 'chased a mouse past the barn'


def score_sentences(sentences):
    """What fraction of these sentences are grammatical? Returns 0.0 to 1.0."""
    if not sentences:
        return 0.0
    return sum(is_grammatical(s) for s in sentences) / len(sentences)


# ---------------------------------------------------------------------------
# HOW GOOD COULD ANY MODEL POSSIBLY GET?
#
# Worth its own section, because it answers a question beginners always ask
# and courses rarely address: "the loss stopped falling -- is the model broken,
# or is it finished?"
#
# Our language is genuinely random. After "the hungry", ANY of our 217 nouns
# is legal and equally likely, so even a flawless model must spread its guess
# across all 217 and will score -log(1/217) on that word. That irreducible
# surprise is the language's ENTROPY, and it is a hard floor: no model, of any
# size, trained for any length of time, can score below it.
#
# Because we invented this language, we can compute the floor exactly -- which
# almost never happens with real text, and is a luxury worth exploiting.
# ---------------------------------------------------------------------------

# The branch probabilities used by _verb_phrase and _noun_phrase above.
P_ADJECTIVE = 0.5           # chance a noun phrase carries an adjective
P_PLAIN_VP = 0.45           # VERB NP
P_ADVERB_VP = 0.30          # ADV VERB NP
P_PREP_VP = 0.25            # VERB NP PREP NP


def _noun_phrase_surprise():
    """Expected -log(probability) and expected length of one noun phrase."""
    surprise = (
        math.log(len(DETERMINERS))                     # which determiner
        + math.log(2)                                  # adjective: yes or no
        + P_ADJECTIVE * math.log(len(ADJECTIVES))      # which adjective, if any
        + math.log(len(NOUNS))                         # which noun
    )
    length = 2 + P_ADJECTIVE                           # det + noun (+ maybe adj)
    return surprise, length


def entropy_per_token():
    """
    The information-theoretic floor: the best average cross-entropy loss that
    any model could ever achieve on our language.

    We add up the expected surprise of a whole sentence and divide by its
    expected length in tokens, because that is exactly what cross-entropy loss
    averages over.
    """
    np_surprise, np_length = _noun_phrase_surprise()
    verb = math.log(len(VERBS))

    branches = [
        # (probability, surprise of this branch, tokens in this branch)
        (P_PLAIN_VP,
         verb + np_surprise,
         1 + np_length),
        (P_ADVERB_VP,
         math.log(len(ADVERBS)) + verb + np_surprise,
         2 + np_length),
        (P_PREP_VP,
         verb + np_surprise + math.log(len(PREPOSITIONS)) + np_surprise,
         2 + 2 * np_length),
    ]

    # Choosing which branch to take is itself information the model must guess.
    vp_surprise = sum(p * (math.log(1 / p) + s) for p, s, _ in branches)
    vp_length = sum(p * n for p, _, n in branches)

    # The final '.' costs nothing: once the sentence is complete it is certain.
    total_surprise = np_surprise + vp_surprise
    total_length = np_length + vp_length + 1
    return total_surprise / total_length


# ---------------------------------------------------------------------------
# Run this file directly to look at the language: python3 grammar.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE} words")
    print(f"  {len(DETERMINERS):>3} determiners   {DETERMINERS[:4]} ...")
    print(f"  {len(ADJECTIVES):>3} adjectives    {ADJECTIVES[:4]} ...")
    print(f"  {len(NOUNS):>3} nouns         {NOUNS[:4]} ...")
    print(f"  {len(VERBS):>3} verbs         {VERBS[:4]} ...")
    print(f"  {len(ADVERBS):>3} adverbs       {ADVERBS[:4]} ...")
    print(f"  {len(PREPOSITIONS):>3} prepositions  {PREPOSITIONS[:4]} ...")
    print(f"    1 full stop   ['.']")

    print("\nTen sentences from our language:")
    rng = random.Random(0)
    for _ in range(10):
        print("  " + " ".join(make_sentence(rng)))

    print("\nThe grammar checker in action:")
    tests = [
        ["the", "cat", "chased", "a", "mouse", "."],
        ["every", "hungry", "dog", "quietly", "watched", "the", "tall", "bird", "."],
        ["some", "child", "found", "a", "coin", "beneath", "the", "bridge", "."],
        ["cat", "the", "chased", "mouse", "."],
        ["the", "cat", "mouse", "."],
        ["the", "hungry", "hungry", "cat", "chased", "a", "mouse", "."],
    ]
    for t in tests:
        mark = "legal  " if is_grammatical(t) else "ILLEGAL"
        print(f"  {mark}  {' '.join(t)}")

    corpus = make_corpus(1000)
    print(f"\nA 1000-sentence corpus is {len(corpus)} words long.")
    print(f"First 20 words: {' '.join(corpus[:20])}")

    print(f"\nEntropy of this language: {entropy_per_token():.4f} nats per word.")
    print("That is the lowest loss any model could ever reach on it.")
