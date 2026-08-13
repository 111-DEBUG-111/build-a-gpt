# Building a GPT from first principles

A teaching course. Eleven runnable Python files that start with counting word
pairs and end with a working decoder-only Transformer — the architecture from
*Attention Is All You Need* — writing new sentences in a 500-word language.

Written to be **taught**, not just read. Every file prints its own narrated
lesson, every component arrives because the previous model failed in a
specific, visible way, and every claim is backed by a number on screen.

```bash
pip install torch matplotlib      # numpy comes with torch
python3 00_the_problem.py         # start here
```

No GPU anywhere. The whole course runs on a laptop CPU.

---

## The through-line

The course is one argument, not ten topics. Each step exists because the step
before it broke:

| Step | The model | Grammatical | Why it fails → what's next |
|---|---|---|---|
| 0 | Counting word pairs | 22% | Forgets everything but the last word |
| 3 | Neural bigram | 23% | Same blindness, fancier machine |
| 4 | Average all past words | **2%** | An average is a blur — precision destroyed |
| 5 | Self-attention | 22% | Can't tell *where* any word is |
| 6 | + positional encoding | **92%** | One head must compromise between jobs |
| 7 | + multi-head | 94% | Gathers information, never thinks about it |
| 8 | + feed-forward, residuals, norm | 95% | Memorises instead of learning |
| 9 | Full GPT, 10× data | **97%** | — |

Two of those rows are failures, and they are the two most valuable lessons in
the course. Step 4 gets *worse* than the model before it; that failure is what
makes attention feel necessary instead of arbitrary. Step 5 builds the
mechanism the paper is named after and barely beats a lookup table; that
failure is what makes positional encoding land.

Don't skip the failures. They are the argument.

## The files

**Run in order.** Each one takes seconds to a few minutes.

| File | Teaches | Time |
|---|---|---|
| `00_the_problem.py` | A language model is a next-word guesser. Built by counting, no ML at all. | <1s |
| `01_vocabulary.py` | Text → numbers, and why those numbers are dangerous | <1s |
| `02_embeddings.py` | Words as vectors; a lookup table *is* a matrix multiply | 1s |
| `03_neural_bigram.py` | First training loop: softmax, cross-entropy, gradient descent | 3s |
| `04_context_by_averaging.py` | Causal masking, the triangular-matrix trick, and an instructive failure | 6s |
| `05_self_attention.py` | **Query, key, value.** Why `√d_k`. The paper's Equation 1 | 9s |
| `05a_why_sqrt_dk.py` | A short aside: what the scaling factor actually prevents | 2s |
| `06_position.py` | Proof that attention is order-blind, then the fix. Biggest jump in the course | 15s |
| `07_multi_head.py` | Several heads in parallel, and what they specialise in | 20s |
| `08_the_block.py` | Feed-forward, residuals, layer norm — with ablations for each | 2m |
| `09_gpt.py` | The whole model. Overfitting, temperature, top-k, prompting | 2m |
| `10_see_attention.py` | Attention heatmaps and measured head specialisation | 5s |

(Timings from an M-series MacBook CPU. Steps 8 and 9 are the only slow ones —
they each train several models from scratch.)

## The trick that makes this teachable

We invent the language instead of using real text. `lib/grammar.py` defines 500
words and a handful of rules:

```
SENTENCE    -> NOUN_PHRASE VERB_PHRASE .
NOUN_PHRASE -> DETERMINER [ADJECTIVE] NOUN
VERB_PHRASE -> [ADVERB] VERB NOUN_PHRASE [PREPOSITION NOUN_PHRASE]
```

so `the hungry cat quietly chased a small mouse .` is legal and
`cat the chased mouse .` is not.

This buys three things you cannot get from Shakespeare:

1. **A scoreboard.** A parser checks whether generated sentences are legal, so
   "is it learning?" has a number, not a vibe.
2. **A finish line.** Because we know the generator, we can compute the
   language's exact entropy — **3.8100 nats/word** — the lowest loss any model
   could ever achieve. Our final model reaches 3.86. You can say "this is
   essentially done" and prove it.
3. **Speed.** Everything trains on a CPU in minutes.

That entropy floor turns out to be the sharpest teaching tool in the course.
In step 9 the training loss drops *below* it — which is impossible — and that
impossibility is how overfitting gets diagnosed on screen.

## Supporting files, not lessons

`lib/` — the four modules every lesson shares:

- **`gpt_model.py`** — the finished architecture in one file, ~120 lines. The
  takeaway. Read it end to end after step 9.
- `grammar.py` — the 500-word language, its rules, the grammar checker, and
  its exact entropy
- `data.py` — batching, and the train/validation split
- `display.py` — terminal printing helpers, no lesson content

`outputs/` — what the lessons write: the trained model from step 9, and the two
pictures (step 6 saves `positional_encoding.png`, step 10 saves
`attention_maps.png`; both are slide-ready).

[`PAPER_MAP.md`](PAPER_MAP.md) — every equation and section of *Attention Is All
You Need* mapped to this code, including what we skip and why. Read it after
step 10.

## Teaching notes

**Analogies used, in order.** Predictive text on a phone (step 0) · jersey
numbers (step 1) · RGB colour codes (step 2) · a blindfolded hiker walking
downhill in fog (step 3) · an exam with the answers stapled to it, for causal
masking (step 4) · **a library: your question, the spine label, the contents —
query, key, value** (step 5) · a bag of words tipped onto a table, and clock
hands for sinusoids (step 6) · four specialists reading one contract (step 7)
· scribbling across three pages then summarising, margin notes on a manuscript,
grading on a curve (step 8).

**If you are short on time**, the irreducible core is steps 0, 4, 5, 6. Step 4
into step 5 is the conceptual hinge of the whole thing — attention is a
weighted average where the model picks the weights — and step 6 is the
emotional payoff.

**Expect the numbers to move.** Every script fixes a random seed, but results
shift with hardware and PyTorch version. The *pattern* is stable; the third
decimal place is not. Step 6 says so explicitly, which is a good moment to
talk about reading benchmark tables sceptically.

## What this is not

Our model is architecturally complete — the same design as GPT-2 and GPT-3,
differing only in scale. It is not an assistant. Missing: the encoder and
cross-attention from the original paper (see [`PAPER_MAP.md`](PAPER_MAP.md) for
why GPT drops them), sub-word tokenisation, scale, and the fine-tuning and RLHF
that turn a text-continuer into something that answers questions.

Every one of those is a real gap. None of them is a gap in your understanding
of the architecture once you finish step 10.
