"""
STEP 5a  --  Why divide by the square root of d_k?

    Run me:  python3 05a_why_sqrt_dk.py   (about 30 seconds)

A side-quest off step 5. Step 5 told you that

    Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V
                                        ^^^^^^^^^

keeps softmax gentle, and moved on. That is true but it is not an explanation.
This file derives the sqrt(d_k) from scratch and then tries to break it. By the
end you will be able to answer all five of these:

    1. Where does sqrt(d_k) come from? (It is not a tuning constant. It is a
       standard deviation, and we will compute it with pen and paper.)
    2. Why does the SCALE of the scores matter at all, when softmax turns
       everything into percentages anyway?
    3. What actually goes wrong without it -- and why "attention gets too
       peaky" is the less important half of the answer.
    4. Why sqrt(d_k) rather than d_k, or log(d_k), or a learned parameter?
    5. When does it stop mattering? (It does. Knowing when is the difference
       between understanding the formula and reciting it.)

Nothing here is new machinery. It is the same nine-line head from step 5,
examined under a microscope.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import grammar, data
from lib.display import title, section, bar, matrix

torch.manual_seed(1337)

title("5a", "The square root, derived rather than asserted")

# ===========================================================================
# PART 1  --  WHAT IS A DOT PRODUCT OF TWO RANDOM VECTORS?
# ===========================================================================
section("Part 1: a dot product is a random walk")

print("""  Strip attention away entirely. The question underneath it is pure
  probability, and has nothing to do with language:

      I have two vectors of length d_k. Every entry is an independent random
      number with mean 0 and variance 1. How big is their dot product?

  That is the whole problem. At the start of training, that is EXACTLY what a
  query and a key are: two vectors of random numbers, because the matrices
  that produced them were initialised randomly.

  Write the dot product out:

      q . k  =  q1*k1 + q2*k2 + ... + q_dk * k_dk

  It is a sum of d_k separate little products. Now, two facts from probability,
  both of which you can check by hand:

  MEAN.  Each term q_i*k_i has expected value E[q_i]*E[k_i] = 0*0 = 0, because
         q_i and k_i are independent. Sum of d_k things each averaging 0
         averages 0. So the dot product is centred at zero. No surprise.

  VARIANCE.  This is the interesting one. For independent things, VARIANCES
         ADD -- not standard deviations, variances. And the variance of one
         term is

             Var(q_i*k_i) = E[(q_i*k_i)^2] - (E[q_i*k_i])^2
                          = E[q_i^2] * E[k_i^2] - 0
                          = 1 * 1
                          = 1

         So each term contributes exactly 1 unit of variance, and

             Var(q . k) = d_k          ->  std(q . k) = sqrt(d_k)

  There it is. sqrt(d_k) is not a fudge factor anyone tuned. It is the standard
  deviation of the thing we are about to divide -- computed exactly, on paper,
  before we ever ran anything.

  THE PICTURE: a drunkard's walk. Take d_k random steps of typical size 1.
  You do not end up d_k away from home, because the steps partly cancel. You
  end up about sqrt(d_k) away. A dot product is that walk. 64 coin-flip steps
  leave you about 8 paces from where you started, not 64.

  Let's check the algebra against a computer:""")

print(f"\n    {'d_k':>6}  {'predicted std':>14}  {'measured std':>13}  {'after /sqrt(d_k)':>17}")
for d in [1, 4, 16, 64, 256, 1024]:
    a, b = torch.randn(200_000, d), torch.randn(200_000, d)
    raw = (a * b).sum(dim=1)
    print(f"    {d:>6}  {math.sqrt(d):>14.3f}  {raw.std():>13.3f}"
          f"  {(raw / math.sqrt(d)).std():>17.3f}")

print("""
  Column 2 is the pen-and-paper prediction, column 3 is reality. They agree to
  three digits. Column 4 is what dividing by sqrt(d_k) buys: whatever d_k you
  chose, the scores now have standard deviation 1.

  So the honest one-line description of the scaling is:

      "divide the scores by their own standard deviation"

  which is the most ordinary operation in statistics. It is the same move as
  converting to a z-score. The only reason it looks exotic in the paper is
  that the standard deviation happens to have a closed form, sqrt(d_k), so
  nobody has to measure it at runtime.""")

# ===========================================================================
# PART 2  --  WHY DOES SCALE MATTER TO SOFTMAX AT ALL?
# ===========================================================================
section("Part 2: softmax ignores shifts but not stretches")

print("""  Reasonable objection: who cares how big the scores are? Softmax converts
  them to percentages that sum to 1 regardless. Isn't the scale washed out?

  No -- and the sharpest way to see it is to notice what softmax DOES ignore.

  Add 1000 to every score and nothing changes at all:""")

z = torch.tensor([2.0, 1.0, 0.0, -1.0])
print(f"\n    softmax({z.tolist()})")
print(f"      = {[round(p, 4) for p in F.softmax(z, -1).tolist()]}")
print(f"    softmax({(z + 1000).tolist()})")
print(f"      = {[round(p, 4) for p in F.softmax(z + 1000, -1).tolist()]}   <- identical")

print("""
  That is because softmax is built from e^z, and e^(z+c) = e^z * e^c -- the
  e^c is a common factor top and bottom and cancels. Softmax cares only about
  the GAPS between scores, never their absolute level.

  But now multiply by 4 instead of adding to it:""")

for mult in [0.25, 1, 4, 16]:
    p = F.softmax(z * mult, -1)
    print(f"    all scores x {mult:<5}  ->  {[round(x, 4) for x in p.tolist()]}")

print("""
  Completely different answers. Stretching the scores stretches the gaps, and
  gaps are the one thing softmax is sensitive to. So "how spread out are the
  scores" is precisely the quantity that decides how sharp attention is -- and
  Part 1 showed that d_k controls exactly that spread.

  The chain is now complete, and every link is forced:

      bigger d_k  ->  wider spread of scores  ->  wider gaps
                  ->  softmax closer to one-hot  ->  attention picks
                      one word instead of blending

  Watch it happen. Below, the SAME five words are scored by random q and k
  vectors of increasing width. Nothing about the words changes -- only d_k:""")

print(f"\n    {'d_k':>6}  {'largest weight':>15}  {'entropy':>8}  {'attention spread'}")
torch.manual_seed(0)
for d in [4, 16, 64, 256, 1024]:
    q_, k_ = torch.randn(2000, 1, d), torch.randn(2000, 5, d)
    s = (q_ @ k_.transpose(-2, -1)).squeeze(1)      # (2000, 5) unscaled scores
    p = F.softmax(s, dim=-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    print(f"    {d:>6}  {p.max(-1).values.mean():>15.1%}  {ent:>8.3f}  "
          f"{bar(ent / math.log(5), 24)}")

print(f"""
  A perfectly even blend over 5 words would have entropy log(5) = {math.log(5):.3f}
  (the full bar) and a largest weight of 20%. A one-hot pick has entropy 0.

  At d_k = 4 attention is still blending. By d_k = 1024 it has collapsed: one
  word gets ~97% and the other four share the crumbs -- and remember these are
  RANDOM vectors, so it collapsed onto a word chosen by pure noise. The model
  has not learned anything yet; it has just committed to an arbitrary opinion
  with enormous confidence.

  Now the same table WITH the scaling applied:""")

print(f"\n    {'d_k':>6}  {'largest weight':>15}  {'entropy':>8}  {'attention spread'}")
torch.manual_seed(0)
for d in [4, 16, 64, 256, 1024]:
    q_, k_ = torch.randn(2000, 1, d), torch.randn(2000, 5, d)
    s = (q_ @ k_.transpose(-2, -1)).squeeze(1) / math.sqrt(d)
    p = F.softmax(s, dim=-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    print(f"    {d:>6}  {p.max(-1).values.mean():>15.1%}  {ent:>8.3f}  "
          f"{bar(ent / math.log(5), 24)}")

print("""
  Flat. Every row behaves the same. That is the real goal of the scaling: make
  the model's starting behaviour INDEPENDENT of a size you chose for unrelated
  reasons. Without it, changing head_size from 16 to 64 would silently change
  how the model learns, and you would have to re-tune everything else to
  compensate. With it, d_k becomes a free architectural knob.""")

# ===========================================================================
# PART 3  --  THE REAL DAMAGE: GRADIENTS DIE
# ===========================================================================
section("Part 3: the half of the answer that actually matters")

print("""  "Attention becomes too peaky" sounds survivable. Learning would fix it,
  surely -- the model gets a bad answer, the loss is high, gradient descent
  nudges the weights, we move on.

  It would not. Here is why, and this is the part step 5 waved at in one
  sentence.

  Differentiate softmax. For p = softmax(z), the derivative of output i with
  respect to input j is

      dp_i/dz_j = p_i * (delta_ij - p_j)

  which as a matrix (the Jacobian) is

      J = diag(p) - p p^T

  Do not take my word for what that means -- just try p values.

  If p is spread out, say p = [0.25, 0.25, 0.25, 0.25]:
      diagonal entries are 0.25*(1 - 0.25) = 0.1875     healthy, non-zero

  If p has collapsed, say p = [0.999, 0.0003, 0.0003, 0.0004]:
      the winner's entry   is 0.999*(1 - 0.999) = 0.000999
      the losers' entries  are 0.0003*(1-0.0003) = 0.0003
                                                         everything is ~0

  Every route from the loss back to the query and key weights passes through
  this matrix. When it is full of zeros, the message "your scores were wrong"
  cannot get back to W_q and W_k. They receive nothing and stop moving.

  This is a trap that locks behind you. The scores are large, so softmax is
  saturated; softmax is saturated, so no gradient reaches the weights that
  produce the scores; no gradient reaches them, so the scores stay large. A
  saturated softmax is not a bad starting point you train out of. It is a
  starting point you cannot leave.

  Measured, not asserted -- total size of that Jacobian at each scale:""")

torch.manual_seed(0)
print(f"\n    {'d_k':>6}  {'unscaled |J|':>13}  {'scaled |J|':>11}  {'gradient strength lost'}")
for d in [4, 16, 64, 256, 1024]:
    q_, k_ = torch.randn(500, 1, d), torch.randn(500, 5, d)
    raw = (q_ @ k_.transpose(-2, -1)).squeeze(1)

    def jac_norm(scores):
        p = F.softmax(scores, dim=-1)
        J = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
        return J.flatten(1).norm(dim=1).mean().item()

    bad, good = jac_norm(raw), jac_norm(raw / math.sqrt(d))
    print(f"    {d:>6}  {bad:>13.4f}  {good:>11.4f}  "
          f"{bar(1 - bad / good, 24)} {1 - bad / good:.0%}")

print("""
  At d_k = 1024 the unscaled head has lost about 91% of the size of that
  Jacobian before training has taken a single step. The scaled column, mean-
  while, barely moves at all across a 256x change in d_k.

  A WARNING ABOUT MEASURING THIS, because it caught me while writing the file
  and it will catch you. The obvious next move is "measure |grad W_q| both ways
  and show the unscaled one is smaller". Do that and you get the OPPOSITE of
  what you expect -- the unscaled gradient is often bigger. The reason is that
  the gradient into W_q also carries a factor of k, and unscaled keys push
  larger numbers through the chain. Raw gradient norm mixes up "how much signal
  is there" with "what units are we measuring in", and tells you nothing.

  So we test the claim the way it should be tested: not by measuring gradients,
  but by asking whether the head can still LEARN. We give one attention head a
  job so simple there is no excuse for failing --

      "every word should attend to word 0"

  -- and train only W_q and W_k to do it, 300 steps, 8 seeds each. A run counts
  as FAILED if it never gets the loss below 0.5. For reference, a head that has
  learned nothing at all and blends every past word evenly scores 1.33, so
  failing this test means ending up worse than knowing nothing:""")


def learn_pattern(d_k, scaled, opt_name, steps=300, T=8, seed=0):
    """Train one head to attend to position 0. Returns (starting loss, final loss)."""
    torch.manual_seed(seed)
    x = torch.randn(1, T, 32)
    Wq = nn.Linear(32, d_k, bias=False)
    Wk = nn.Linear(32, d_k, bias=False)
    target = torch.zeros(T, T)
    target[:, 0] = 1.0                       # the job: all attention on word 0

    params = list(Wq.parameters()) + list(Wk.parameters())
    opt = (torch.optim.SGD(params, lr=0.1) if opt_name == "sgd"
           else torch.optim.AdamW(params, lr=1e-2))
    tril = torch.tril(torch.ones(T, T))

    first = None
    for i in range(steps):
        scores = Wq(x) @ Wk(x).transpose(-2, -1)
        if scaled:
            scores = scores / math.sqrt(d_k)
        scores = scores.masked_fill(tril == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        loss = -(target * w.clamp_min(1e-12).log()).sum() / T
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    return first, loss.item()


SEEDS = 8
for opt_name in ["sgd", "adamw"]:
    print(f"\n    optimiser = {opt_name}")
    print(f"    {'d_k':>6}  {'scaled: start':>14} {'failed':>8}   "
          f"{'unscaled: start':>16} {'failed':>8}")
    for d in [64, 256, 512, 1024, 2048]:
        good = [learn_pattern(d, True, opt_name, seed=s) for s in range(SEEDS)]
        bad = [learn_pattern(d, False, opt_name, seed=s) for s in range(SEEDS)]
        fg = sum(1 for _, e in good if e > 0.5)
        fb = sum(1 for _, e in bad if e > 0.5)
        avg = lambda v: sum(x for x, _ in v) / len(v)
        print(f"    {d:>6}  {avg(good):>14.2f} {f'{fg}/{SEEDS}':>8}   "
              f"{avg(bad):>16.2f} {f'{fb}/{SEEDS}':>8}")

print("""
  Two things in that table, and the second one surprised me.

  FIRST, the starting-loss columns. Scaled starts at 1.33 no matter how wide
  the head gets -- and 1.33 is exactly the "blend everything evenly" score. The
  scaled head begins in a state of honest ignorance, which is the best possible
  place to begin. The unscaled head starts at 2.81 and climbs to 12.90: it
  begins with a loud, confident, completely random opinion, and the wider the
  head the louder it shouts.

  SECOND -- and this is the part worth internalising -- the failure is a CLIFF,
  not a slope. At d_k = 64 the unscaled head is perfectly fine; it starts worse
  but trains to the answer every time. It is not until d_k passes a few hundred
  that runs start dying, and then it goes quickly: 2/8, 3/8, 4/8, 7/8.

  Nothing gradually degrades. The head either escapes the saturated region or
  it does not, and once it does not, it is stuck at loss ~6.9 forever, which is
  a probability of 0.1% on the word it was supposed to be looking at. That is
  the trap from the paragraph above, caught in the act.

  Compare the two optimisers and you get the last piece. Near the edge of the
  cliff AdamW genuinely helps the unscaled head: at d_k = 256 it fails 0/8
  where plain SGD fails 2/8. Well past the edge it stops helping entirely --
  by d_k = 2048 both fail 7/8, and at 1024 Adam is actually the worse of the
  two. Adam rescales each parameter's step by its own recent gradient history,
  so it can amplify a small gradient back to a useful size; but amplifying
  something requires it to be non-zero. Past a certain saturation there is
  nothing left to amplify, and no optimiser saves you.""")

# ===========================================================================
# PART 4  --  WHY sqrt(d_k) AND NOT SOMETHING ELSE?
# ===========================================================================
section("Part 4: three tempting alternatives, and why each is worse")

print("""  Part 1 justified sqrt(d_k) by deriving it. But three other ideas look
  just as plausible until you try them, and trying them is how you know the
  formula is right rather than merely traditional.

  ALTERNATIVE 1 -- divide by d_k instead. Bigger correction, surely safer?

  No: it overcorrects, and it overcorrects by a factor that GROWS with d_k.
  Scores end up with standard deviation sqrt(d_k)/d_k = 1/sqrt(d_k), which
  shrinks toward zero. Near-equal scores mean near-equal weights, and you have
  reinvented the flat average from step 4 -- the one that scored 2%.""")

torch.manual_seed(0)
print(f"\n    {'d_k':>6}  {'/1':>10}  {'/sqrt(d_k)':>12}  {'/d_k':>10}   (largest attention weight)")
for d in [16, 64, 256, 1024]:
    q_, k_ = torch.randn(3000, 1, d), torch.randn(3000, 5, d)
    s = (q_ @ k_.transpose(-2, -1)).squeeze(1)
    row = [F.softmax(s / den, -1).max(-1).values.mean().item()
           for den in [1.0, math.sqrt(d), float(d)]]
    print(f"    {d:>6}  {row[0]:>10.1%}  {row[1]:>12.1%}  {row[2]:>10.1%}")

print("""
  Read the last column downward: 20% is a perfectly flat blend over 5 words.
  Dividing by d_k walks straight into the blur. sqrt(d_k) is the unique power
  of d_k that lands between the two failure modes -- and it is not a lucky
  compromise, it is the exact standard deviation we computed in Part 1.

  ALTERNATIVE 2 -- normalise q and k to unit length, then no d_k appears.

  This genuinely works and is genuinely used (it is "cosine attention"; the
  QK-LayerNorm in several modern large models is a cousin of it). But it costs
  something real. Once q and k have fixed length, the model can no longer make
  attention sharper by growing their magnitude -- and sharpness is something a
  trained model NEEDS to control per head. Some heads should be nearly one-hot
  ("attend to the previous word"); others should blend widely. Dividing by a
  constant leaves that dial in the model's hands. Normalising takes it away.

  Which brings us to the most important point in this file:

  ALTERNATIVE 3 -- let the model learn the scale. It has W_q and W_k; it can
  multiply them by anything it likes. Why hard-code a constant at all?

  It CAN learn it. That is not the issue. The issue is that it has to learn it
  from wherever it starts -- and Part 3 showed that the unscaled starting point
  is the one place in the space where gradients are too small to learn from.
  You are asking the model to climb out of a hole using a rope that only exists
  outside the hole.

  So sqrt(d_k) is not a constraint on the trained model. It is a choice of
  STARTING POINT. The model is free to scale W_q and W_k up during training and
  become as peaky as it wants -- and it does. We are only insisting that it
  begins somewhere it can still learn from.""")

# ===========================================================================
# PART 5  --  DOES IT CHANGE THE FINAL MODEL?
# ===========================================================================
section("Part 5: the honest test -- train it both ways")

print("""  Everything so far has been about initialisation. The question that
  actually matters is whether it survives contact with training.

  Same model as step 5, same data, same steps. Only the division changes. We
  run it at the course's head size (32) and at a wider one (256), because the
  whole argument predicts the damage should scale with d_k.
""")


class Head(nn.Module):
    """Step 5's attention head, with the scaling made optional."""

    def __init__(self, d_model, head_size, block_size, scaled=True):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.scale = math.sqrt(head_size) if scaled else 1.0
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        T = x.shape[1]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / self.scale
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        self.last_weights = F.softmax(scores, dim=-1)
        return self.last_weights @ v


class Model(nn.Module):
    def __init__(self, head_size, scaled):
        super().__init__()
        d_model = 32
        self.embed = nn.Embedding(grammar.VOCAB_SIZE, d_model)
        self.attention = Head(d_model, head_size, data.BLOCK_SIZE, scaled)
        self.predict = nn.Linear(head_size, grammar.VOCAB_SIZE)

    def forward(self, idx):
        return self.predict(self.attention(self.embed(idx)))


@torch.no_grad()
def generate(model, max_new=18):
    idx = data.encode(["."]).unsqueeze(0)
    out = []
    for _ in range(max_new):
        logits = model(idx[:, -data.BLOCK_SIZE:])
        nxt = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        w = data.ID_TO_WORD[int(nxt)]
        if w == ".":
            break
        out.append(w)
    return out


train_data, val_data = data.load()


def train(head_size, scaled, steps=2000, seed=1337):
    torch.manual_seed(seed)
    model = Model(head_size, scaled)

    # How saturated is attention BEFORE any training? This is the number the
    # first four parts have been predicting.
    with torch.no_grad():
        xb, _ = data.get_batch(train_data, batch_size=64)
        model(xb)
        w = model.attention.last_weights
        start_peak = w.max(-1).values.mean().item()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(steps):
        xb, yb = data.get_batch(train_data, batch_size=64)
        loss = F.cross_entropy(model(xb).view(-1, grammar.VOCAB_SIZE), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        xb, yb = data.get_batch(val_data, batch_size=256)
        val = F.cross_entropy(model(xb).view(-1, grammar.VOCAB_SIZE),
                              yb.reshape(-1)).item()
    score = grammar.score_sentences([generate(model) for _ in range(600)])
    return start_peak, val, score


print(f"  {'head_size':>10}  {'scaling':>12}  {'peak attn at init':>18}"
      f"  {'val loss':>9}  {'gramm.':>7}")
for hs in [32, 256]:
    for scaled in [True, False]:
        peak, val, score = train(hs, scaled)
        tag = "/sqrt(d_k)" if scaled else "none"
        print(f"  {hs:>10}  {tag:>12}  {peak:>18.1%}  {val:>9.4f}  {score:>7.1%}")

print("""
  Read the "peak attn at init" column first. That is Parts 1 to 3 showing
  up in a real model exactly as predicted: the scaled head starts near an even
  blend regardless of width, the unscaled one starts far more committed, and
  gets worse as the head gets wider.

  Now read the two result columns, and let me be straight with you rather than
  sell you the formula: AT THIS SCALE IT MAKES ALMOST NO DIFFERENCE. Val loss
  and grammar score come out within noise of each other, at both head sizes.
  If you were hoping this file would end with the unscaled model in flames,
  it doesn't, and I am not going to pretend otherwise.

  That is not a hole in the argument -- it is the argument's fine print, and
  knowing it is what separates understanding the formula from reciting it. Our
  d_k is 32, so sqrt(d_k) is 5.7, and Part 3's cliff did not begin until a few
  hundred. We are simply nowhere near the edge, and one attention layer trained
  by AdamW on 16-word sequences is a forgiving place to be.

  A caution I have to give you, though, having just shown you a table with a
  cliff in it: do NOT walk away with "the danger starts around d_k = 256". That
  number belongs to Part 3's toy -- one layer, eight positions, one particular
  job, SGD. The original paper used d_k = 64 per head, which is comfortably
  safe by my table's standard, and its authors still hit this problem hard
  enough to write a sentence about it in the paper. So the cliff clearly moves,
  and it moves with things I have not isolated here: depth, how large the
  activations grow during training, what the head is trying to learn.

  What transfers is the MECHANISM, not the number. Big scores saturate softmax,
  saturated softmax has no gradient, and no gradient means no learning -- and
  d_k is one of the dials that pushes you toward it. Where exactly your model
  falls off depends on your model.

  Which makes the honest summary of sqrt(d_k) not "it makes models better" but
  INSURANCE: one division, invisible when you do not need it, and Part 3 is
  what it protects you from. A great deal of what looks like arbitrary ritual
  in deep learning turns out to be insurance written by someone who met the
  failure at a scale you have not reached yet.""")

# ===========================================================================
# WHAT TO REMEMBER
# ===========================================================================
section("The whole thing in seven lines")

print("""
    1. At init, q and k are random vectors, so q.k is a sum of d_k random
       products: mean 0, variance d_k, standard deviation sqrt(d_k).
    2. Dividing by sqrt(d_k) is therefore just "divide by the standard
       deviation" -- a z-score, not a fudge factor.
    3. Softmax ignores shifts but is exquisitely sensitive to stretches, so
       that standard deviation directly controls how peaky attention is.
    4. Peaky softmax has a near-zero Jacobian, so no gradient reaches W_q and
       W_k, so the model cannot learn its way out. That is the real damage --
       and it arrives as a cliff, not a slope.
    5. Dividing by d_k instead overshoots into step 4's useless blur.
    6. It fixes the STARTING POINT only. A trained model is still free to make
       attention as sharp as it likes -- and does.
    7. At this course's d_k = 32 you could delete it and barely notice. It is
       insurance against a failure that shows up at real scale.

  Back to the main line:  python3 06_position.py
""")
