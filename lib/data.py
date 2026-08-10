"""
data.py  --  turning our text into batches the model can train on

Introduced in step 4 and reused by every step after it. It is short; read it
once and you'll understand the shape of every tensor for the rest of the course.

THE THREE LETTERS YOU'LL SEE EVERYWHERE
---------------------------------------
Almost every tensor from here on has the shape (B, T, C):

    B = batch     how many separate chunks of text we process at once
    T = time      how many words in each chunk (also called the context length)
    C = channels  how many numbers describe each word (d_model)

The names come from PyTorch convention. "Time" is a slightly odd word for
"position in the sentence", but text is a sequence, so position IS time: word
3 happens after word 2. When you see (4, 8, 32), read it as "4 chunks of text,
8 words each, 32 numbers describing every word".
"""

import torch

import grammar

# The word <-> number dictionaries from step 1.
WORD_TO_ID = {w: i for i, w in enumerate(grammar.VOCAB)}
ID_TO_WORD = {i: w for i, w in enumerate(grammar.VOCAB)}

# How many words back the model is allowed to look. Our longest sentence is
# 12 words, so 16 is enough to always see a whole sentence plus a bit of the
# one before it. In the paper this is the sequence length; in GPT-4 it is the
# "context window", and it's in the hundreds of thousands.
BLOCK_SIZE = 16


def encode(words):
    """['the','cat'] -> tensor([432, 74])"""
    return torch.tensor([WORD_TO_ID[w] for w in words], dtype=torch.long)


def decode(ids):
    """tensor([432, 74]) -> ['the','cat']"""
    return [ID_TO_WORD[int(i)] for i in ids]


def load(n_sentences=20000, seed=1337, train_fraction=0.9):
    """
    Build the corpus and split it into training and validation halves.

    WHY SPLIT? We hold back the last 10% and never train on it. Then we can
    ask: does the model do as well on text it has never seen? If training loss
    keeps falling while validation loss climbs, the model has stopped learning
    the language and started memorising the examples. That's overfitting, and
    the held-out split is the only way to catch it.
    """
    corpus = grammar.make_corpus(n_sentences=n_sentences, seed=seed)
    data = encode(corpus)
    split = int(train_fraction * len(data))
    return data[:split], data[split:]


def get_batch(data, batch_size=32, block_size=BLOCK_SIZE):
    """
    Grab `batch_size` random chunks of `block_size` words.

    Returns x and y, both shaped (B, T):

        x = the words we show the model
        y = the correct next word at every position -- x shifted along by one

    Concretely, if a chunk is "the cat chased a mouse":

        x = the   cat    chased   a       mouse
        y = cat   chased a        mouse   .

    Read that column by column. Each column is a complete training example:
    "given everything up to and including this word, what comes next?" So a
    16-word chunk isn't one example, it's SIXTEEN, all learned from in
    parallel. That efficiency is a real part of why Transformers won.
    """
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y
