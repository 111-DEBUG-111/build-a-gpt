"""
STEP 9  --  The whole thing: a working GPT

    Run me:  python3 09_gpt.py   (about 2 minutes; saves the model to gpt.pt)

Everything we've built, assembled into one model, trained properly, and used
to write sentences. Nothing new is introduced in the architecture -- if you
followed steps 0 to 8 you have already seen every line of it. That is worth
saying to an audience out loud when this file opens: the scary diagram in
Figure 1 of the paper is now just a summary of things you built yourself.

What IS new is the sampling: temperature and top-k, the two dials that decide
whether your model writes safely or adventurously. They aren't in the 2017
paper, but they're how every model you've ever used actually generates text.

THE MAP OF THE MODEL

    word IDs
       |
       +-- word embedding      "what is this word?"        (step 2)
       +-- position embedding  "where is this word?"       (step 6)
       |
     [ Block ] x N                                         (step 8)
       |   |
       |   +-- LayerNorm -> multi-head attention -> add    (steps 5, 7)
       |   +-- LayerNorm -> feed-forward         -> add    (step 8)
       |
       +-- LayerNorm
       +-- Linear to 500 logits                            (step 3)
       |
    a probability for every word in the vocabulary
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import grammar, data, OUTPUTS
from lib.display import title, section, bar

torch.manual_seed(1337)

title(9, "The whole thing: a working GPT")

# ---------------------------------------------------------------------------
# THE MODEL LIVES IN gpt_model.py
#
# Every class in that file was built up in an earlier step, so rather than
# reprint 120 lines here we import it. Open gpt_model.py alongside this one --
# it is the single most useful file in the course to read end to end, because
# it is the whole architecture with none of the teaching scaffolding.
#
#     AttentionHead        step 5
#     MultiHeadAttention   step 7
#     FeedForward          step 8
#     Block                step 8
#     GPT                  steps 2, 6 and 3 for the outer layers
#     GPT.generate         the sampling loop, new in this step
# ---------------------------------------------------------------------------
from lib.gpt_model import (GPT, D_MODEL, N_HEADS, N_BLOCKS, BLOCK_SIZE, DROPOUT)

# Training knobs, which are ours rather than the architecture's.
LEARNING_RATE = 3e-3
BATCH_SIZE = 64
TRAIN_STEPS = 3000

FLOOR = grammar.entropy_per_token()


# ---------------------------------------------------------------------------
section("The model")

model = GPT()
n_params = sum(p.numel() for p in model.parameters())
print(f"  d_model {D_MODEL}   heads {N_HEADS}   blocks {N_BLOCKS}   "
      f"context {BLOCK_SIZE}   dropout {DROPOUT}")
print(f"  {n_params:,} parameters")
print(f"""
  For scale: GPT-3 has 175,000,000,000. Ours is about {175e9 / n_params:,.0f} times smaller,
  and structurally identical. Everything that separates this file from a
  frontier model is size, data and engineering -- not ideas.

  A word on DROPOUT, since it appears here for the first time. During training
  it randomly switches off {DROPOUT:.0%} of the activations. It sounds like sabotage;
  it's insurance. The model can't lean too heavily on any single pathway if
  that pathway might vanish, so it's pushed to spread its knowledge around and
  memorise less. The paper uses the same 0.1 (section 5.4). It switches itself
  off automatically at generation time -- that's what model.eval() does.""")

# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(net, dataset, batches=30):
    net.eval()
    losses = [net(*data.get_batch(dataset, BATCH_SIZE))[1].item()
              for _ in range(batches)]
    net.train()
    return sum(losses) / len(losses)


def train(net, train_data, steps, lr=LEARNING_RATE, report_every=500):
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    print(f"  {'step':>6}  {'train':>7}  {'val':>7}   {'gap':>6}")
    for step in range(steps + 1):
        x, y = data.get_batch(train_data, BATCH_SIZE)
        _, loss = net(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % report_every == 0:
            tr = estimate_loss(net, train_data)
            va = estimate_loss(net, val_data)
            flag = "  <-- train is BELOW the floor" if tr < FLOOR else ""
            print(f"  {step:>6}  {tr:7.4f}  {va:7.4f}   {va - tr:+6.3f}{flag}")
    return net


# ---------------------------------------------------------------------------
section("First attempt -- and a trap worth falling into")

print(f"""  Let's train on the same 20,000-sentence corpus we've used all course, and
  watch both losses. Remember: the floor is {FLOOR:.4f}, and NO model can
  legitimately score below it.
""")

small_train, val_data = data.load(n_sentences=20000)
start = time.time()
train(GPT(dropout=0.0), small_train, steps=3000)

print(f"""
  Something has gone wrong, and the floor is what makes it obvious.

  The training loss has dropped BELOW {FLOOR:.4f}. That is supposed to be
  impossible -- it's the irreducible randomness of the language itself. You
  cannot predict a coin flip better than 50%.

  Unless you've seen this particular coin flip before. Which is exactly what
  happened: our corpus is only {len(small_train):,} words, we showed it to the model
  dozens of times, and the model simply MEMORISED it. On text it has memorised
  it does the impossible; on the held-out text it has never seen, it's getting
  worse. That growing gap between the two columns is the symptom.

  This is OVERFITTING, and it is the single most common way a machine learning
  project quietly fails. Notice we could only diagnose it this crisply because
  we knew the floor. In real projects you don't, which is why the held-out
  split from data.py is non-negotiable -- the gap between the columns is the
  only warning you get.

  THE FIX
  Dropout helps a little. Far more effective: get more data. We happen to have
  an infinite supply, so let's use ten times as much.""")

# ---------------------------------------------------------------------------
section("Second attempt: ten times the data")

train_data, val_data = data.load(n_sentences=200000)
print(f"  Training corpus is now {len(train_data):,} words.\n")

start = time.time()
train(model, train_data, steps=TRAIN_STEPS)

final_val = estimate_loss(model, val_data)
print(f"""
  Trained in {time.time() - start:.0f} seconds on a CPU.

  The two columns now sit on top of each other, and both settle just above the
  floor rather than crashing through it. The model is learning the LANGUAGE
  instead of memorising our sentences.

      theoretical floor : {FLOOR:.4f}
      our model         : {final_val:.4f}   ({final_val - FLOOR:+.4f})

  That gap is small enough to call this done. There is not much left for any
  architecture to find here.""")

# ---------------------------------------------------------------------------
section("Writing sentences")

samples = []
for _ in range(400):
    out = model.generate(data.encode(["."]).unsqueeze(0), max_new_tokens=20)
    samples.append(data.decode(out[0][1:-1]))       # drop the leading and trailing '.'

print("  Twelve sentences written by our GPT:\n")
for s in samples[:12]:
    print("    " + " ".join(s))
print(f"\n  Grammatical: {grammar.score_sentences(samples):.1%}")

# ---------------------------------------------------------------------------
section("Temperature: the creativity dial")

print("  Same model, same prompt, different temperature.\n")
for temp in [0.4, 0.8, 1.0, 1.4, 2.0]:
    batch = []
    for _ in range(200):
        out = model.generate(data.encode(["."]).unsqueeze(0),
                             max_new_tokens=20, temperature=temp)
        batch.append(data.decode(out[0][1:-1]))
    example = " ".join(batch[0]) if batch[0] else "(empty)"
    print(f"  T = {temp:<4} grammatical {grammar.score_sentences(batch):6.1%}   "
          f"{example[:44]}")

print("""
  Low temperature plays it safe and scores well but writes the same few
  sentences over and over. High temperature explores, and eventually falls
  apart. There is no correct setting -- it's a dial between reliability and
  variety, and it's the same dial behind the "creativity" slider in every
  writing tool you've used.""")

# ---------------------------------------------------------------------------
section("Top-k: refusing to consider nonsense")

for k in [1, 5, 50, None]:
    batch = []
    for _ in range(200):
        out = model.generate(data.encode(["."]).unsqueeze(0),
                             max_new_tokens=20, temperature=1.0, top_k=k)
        batch.append(data.decode(out[0][1:-1]))
    unique = len({" ".join(s) for s in batch})
    label = "off" if k is None else str(k)
    print(f"  top_k = {label:<5} grammatical {grammar.score_sentences(batch):6.1%}"
          f"   distinct sentences: {unique}/200")

print("""
  top_k = 1 always takes the single best word. Notice what that costs: the
  grammar is flawless and the model says almost the same thing every time.
  That's the trade in one line -- filtering harder buys correctness and sells
  variety.""")

# ---------------------------------------------------------------------------
section("Prompting: finishing someone else's sentence")

print("""  Give it a beginning and let it continue.

  Note the '.' at the front of every prompt. Our model was trained on one
  unbroken stream of sentences, so it has no concept of "the text starts
  here" -- the full stop is the only signal it has that a fresh sentence is
  beginning. Leave it out and the model assumes it has walked in halfway
  through something, and answers accordingly. Step 10 shows that failure in
  detail; it's the reason real models have special start tokens.
""")
for prompt in [[".", "the", "hungry", "cat"],
               [".", "a", "brave", "knight", "quietly"],
               [".", "every", "small", "bird", "found", "a"]]:
    out = model.generate(data.encode(prompt).unsqueeze(0),
                         max_new_tokens=14, temperature=0.8, top_k=10)
    words = data.decode(out[0])
    print(f"    given: {' '.join(prompt[1:])}")
    print(f"    model continues: {' '.join(words[len(prompt):])}\n")

print("""  This is exactly what "prompting" means, with nothing else going on. There is
  no instruction-following and no understanding of a request -- the model is
  continuing a sequence, and everything a chat assistant appears to do sits on
  top of this one behaviour.""")

torch.save(model.state_dict(), OUTPUTS / "gpt.pt")
print("\n  Saved the trained model to outputs/gpt.pt (step 10 will load it).")

# ---------------------------------------------------------------------------
section("What we left out, and what it would take to get to ChatGPT")

print(f"""  Our model is architecturally complete. Honest list of what it is not:

  IN THE PAPER, BUT NOT IN OUR CODE
    * The ENCODER, and encoder-decoder cross-attention. The 2017 paper was
      about translation, so it has two stacks: an encoder that reads the
      French, and a decoder that writes the English while attending to the
      encoder's output. GPT keeps only the decoder, because writing text needs
      no separate thing to read. See PAPER_MAP.md.
    * Label smoothing, learning-rate warm-up with the inverse-square-root
      schedule, and beam search. All useful, none conceptual.

  NOT IN THE PAPER, AND NEEDED FOR A REAL MODEL
    * Sub-word tokenisation, so any text at all can be encoded.
    * Scale: more data, more parameters, more compute. GPT-3 is this file
      with d_model 12288, 96 blocks, 96 heads and 300 billion words.
    * The post-training that turns a text-continuer into an assistant:
      supervised fine-tuning on demonstrations, then reinforcement learning
      from human feedback. Our model will happily continue any sentence; it
      has no notion of being helpful, because nothing ever asked it to be.

  WHERE WE LANDED
    theoretical floor for our language : {FLOOR:.4f}
    our model                          : {final_val:.4f}

  Close enough that the remaining gap is mostly our small size and short
  training run rather than anything missing from the architecture.

    Next:  python3 10_see_attention.py
""")
