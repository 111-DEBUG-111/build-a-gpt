"""
STEP 10  --  Looking inside: what did our model actually learn?

    Run me:  python3 10_see_attention.py     (needs gpt.pt -- run 09 first)

A trained Transformer is not a black box. Attention weights are readable
numbers, and reading them tells you what each head decided its job was.

This step is the one to end a talk on, because it converts belief into
evidence. Up to now the audience has taken your word that attention "learns
what to look at". Here they see it: a head that tracks the subject noun, a
head that only ever looks one word back, a head that hunts for the verb.

Nobody designed any of that. It is what falling down the loss surface built.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

import grammar
import data
from gpt_model import GPT
from display import title, section, bar

torch.manual_seed(1337)

title(10, "Looking inside: reading the attention maps")

# ---------------------------------------------------------------------------
section("Loading the trained model")

model = GPT()
try:
    model.load_state_dict(torch.load("gpt.pt"))
    print("  Loaded gpt.pt")
except FileNotFoundError:
    print("  gpt.pt not found -- run 'python3 09_gpt.py' first to train and save it.")
    raise SystemExit(1)
model.eval()

n_blocks = len(model.blocks)
n_heads = len(model.blocks[0].attention.heads)
print(f"  {n_blocks} blocks x {n_heads} heads = {n_blocks * n_heads} attention "
      f"patterns to inspect")

# ---------------------------------------------------------------------------
section("A single sentence, head by head")

sentence = ["the", "hungry", "cat", "quietly", "chased", "a", "small", "mouse"]
ids = data.encode(sentence).unsqueeze(0)
with torch.no_grad():
    maps = model.attention_maps(ids)          # [block][head] -> (1, T, T)

SHADES = " .:-=+*#%@"


def heatmap(weights, words):
    """Print an attention matrix as a shaded grid. Darker = more attention."""
    width = max(len(w) for w in words) + 1
    print(" " * (width + 2) + "".join(f"{w[:5]:>6}" for w in words))
    for i, row_word in enumerate(words):
        cells = ""
        for j in range(len(words)):
            w = float(weights[i, j])
            if j > i:
                cells += "     ."                      # masked: the future
            else:
                shade = SHADES[min(int(w * len(SHADES)), len(SHADES) - 1)]
                cells += f"  {shade * 3} "
        print(f"  {row_word:>{width}}{cells}")


print("""  Row = the word doing the looking. Column = the word being looked at.
  Darker means more attention. The blank upper triangle is the causal mask --
  no word may look at the future.
""")

for b in range(n_blocks):
    for h in range(n_heads):
        print(f"\n  BLOCK {b}, HEAD {h}")
        heatmap(maps[b][h][0], sentence)

# ---------------------------------------------------------------------------
# QUANTITATIVE: don't eyeball it, measure it
# ---------------------------------------------------------------------------
section("Measuring what each head does, over 300 sentences")

print("""  Reading one sentence's heatmap risks seeing patterns that aren't there.
  So let's average over hundreds of sentences and ask two precise questions
  about every head:

    1. HOW FAR BACK does it look?   (distance 0 = itself, 1 = the word behind)
    2. WHAT KIND of word does it prefer?  (its part of speech)

  Question 2 has a trap in it, and it's a trap worth showing your audience.
  If you simply ask "which part of speech gets the most attention?", every
  single head answers DET -- because our sentences contain two or three
  determiners each, so 'the' and 'a' win on sheer frequency no matter what
  the head is doing. That measures our language, not the model.

  The fix is to compare against chance. For each head we work out how much
  attention a part of speech GOT, versus how much it would get if attention
  were spread evenly over everything visible. A score of 2.0 means "this head
  looks at verbs twice as often as it would by luck".
""")

distance_profile = defaultdict(lambda: defaultdict(float))
pos_profile = defaultdict(lambda: defaultdict(float))
totals = defaultdict(float)
available = defaultdict(float)      # how often each POS was even available
available_total = 0.0

rng_sentences = [grammar.make_sentence() for _ in range(300)]
for words in rng_sentences:
    words = words[:data.BLOCK_SIZE]
    with torch.no_grad():
        m = model.attention_maps(data.encode(words).unsqueeze(0))
    T = len(words)

    # The baseline: every (looker, looked-at) pair that the mask permits.
    for i in range(T):
        for j in range(i + 1):
            available[grammar.PART_OF_SPEECH[words[j]]] += 1
            available_total += 1

    for b in range(n_blocks):
        for h in range(n_heads):
            w = m[b][h][0]
            for i in range(T):
                for j in range(i + 1):
                    weight = float(w[i, j])
                    distance_profile[(b, h)][i - j] += weight
                    pos_profile[(b, h)][grammar.PART_OF_SPEECH[words[j]]] += weight
                    totals[(b, h)] += weight

chance = {pos: n / available_total for pos, n in available.items()}

print(f"  {'head':<12} {'itself':>8} {'1 back':>8} {'2 back':>8} {'3+ back':>8}"
      f"    prefers (vs chance)")
print("  " + "-" * 76)

for b in range(n_blocks):
    for h in range(n_heads):
        key = (b, h)
        total = totals[key]
        d = distance_profile[key]
        far = sum(v for k, v in d.items() if k >= 3)

        # Enrichment: attention share divided by how common that POS is.
        enrichment = {pos: (mass / total) / chance[pos]
                      for pos, mass in pos_profile[key].items()
                      if chance.get(pos, 0) > 0.01}
        best, ratio = max(enrichment.items(), key=lambda kv: kv[1])

        print(f"  block {b} head {h} {d[0] / total:7.0%} {d[1] / total:8.0%} "
              f"{d[2] / total:8.0%} {far / total:8.0%}    {best:<5} {ratio:.1f}x")

print("""
  Now the specialities are measurable rather than suggestive. Read the table
  for these signatures:

    * a high "itself" number -- the head is largely passing its own word
      forward rather than gathering, effectively opting out of communication
      for most positions
    * a high "1 back" number -- a PREVIOUS-TOKEN head. Real GPT models grow
      these too, and they are a known building block for copying patterns
      across long distances
    * a preference score well above 1.0 -- the head is hunting for a specific
      kind of word rather than taking whatever is nearby

  Which head does what will shift if you retrain with a different seed, and
  that is worth saying out loud: the JOBS are forced by the language, but who
  takes which job is an accident of initialisation.

  Step back and notice what these heads add up to. In step 0 we diagnosed our
  entire problem as: "standing on the word 'cat', the model cannot remember
  whether a verb has happened yet." Solving that needs someone tracking the
  previous word and someone tracking words further back by category -- which
  is what the table shows. Nobody assigned those roles. Gradient descent
  discovered the division of labour our diagnosis called for.""")

# ---------------------------------------------------------------------------
section("Watching a prediction being made")

def predict(context):
    with torch.no_grad():
        logits, _ = model(data.encode(context).unsqueeze(0))
        probs = F.softmax(logits[0, -1], dim=-1)
    by_pos = defaultdict(float)
    for i, p in enumerate(probs):
        by_pos[grammar.PART_OF_SPEECH[data.ID_TO_WORD[i]]] += float(p)
    return probs, by_pos


def show(context):
    probs, by_pos = predict(context)
    print(f"\n  Context: '{' '.join(context)}'")
    for pos, p in sorted(by_pos.items(), key=lambda kv: -kv[1])[:4]:
        print(f"    {pos:<6} {p:6.1%}  {bar(p, 30)}")
    return by_pos


print("""  Let's ask the model what follows "the hungry cat". You would expect a verb.
  Watch what actually happens.""")

show(["the", "hungry", "cat"])

print("""
  A full stop?! That looks broken. It isn't -- we asked a badly-formed
  question, and this is one of the most useful mistakes in the course.

  Our model was trained on an unbroken STREAM of run-together sentences. It
  has no notion of "this is where the text begins". So when we hand it "the
  hungry cat" with nothing before it, the honest reading is: this is a noun
  phrase somewhere in the middle of a stream, and I cannot see what came
  before. In our language, "DET ADJ NOUN" is more often an OBJECT than a
  subject -- every sentence has one subject but one or two objects -- and
  after an object, a sentence usually ends. The model is answering correctly;
  we asked the wrong thing.

  Fix it by telling the model a sentence is starting. The '.' token is the
  only marker it has for that:""")

show([".", "the", "hungry", "cat"])

print("""
  There it is. Verb or adverb, nothing else -- exactly what our grammar
  demands after a subject. And now watch what happens when we supply that
  adverb, closing off the last remaining option:""")

show([".", "the", "hungry", "cat", "quietly"])

probs, _ = predict([".", "the", "hungry", "cat", "quietly"])
top = probs.topk(5)
print("\n  Most likely individual words there:")
for p, i in zip(top.values, top.indices):
    word = data.ID_TO_WORD[int(i)]
    print(f"    {word:<12} {float(p):6.2%}  {grammar.PART_OF_SPEECH[word]}")

print("""
  Near-total confidence about the CATEGORY, and spread thinly across the
  individual words within it. That is precisely right: our grammar says a verb
  must come next, and picks which verb uniformly at random. A model confident
  that the verb would be 'chased' specifically would be a model that had
  memorised our corpus rather than learned our language.

  Two things worth landing with an audience here:

    * Compare with step 0, where 'cat' was followed by a full stop 48.8% of
      the time because the counting model had no idea whether it was looking
      at a subject or an object. Our Transformer distinguishes them -- when
      you give it enough context to do so.

    * That first, broken-looking answer is exactly why real models are fussy
      about prompt format. Special tokens marking where text begins, and chat
      templates wrapping your message in a particular shape, exist because a
      model's answer depends entirely on what it believes it is continuing.
      Change the frame and you change the answer, and the model is not being
      unreliable when that happens.""")

# ---------------------------------------------------------------------------
section("Saving the picture")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_blocks, n_heads,
                             figsize=(3.1 * n_heads, 3.0 * n_blocks))
    axes = axes.reshape(n_blocks, n_heads)
    for b in range(n_blocks):
        for h in range(n_heads):
            ax = axes[b][h]
            ax.imshow(maps[b][h][0], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"block {b}, head {h}", fontsize=9)
            ax.set_xticks(range(len(sentence)))
            ax.set_yticks(range(len(sentence)))
            ax.set_xticklabels(sentence, rotation=90, fontsize=7)
            ax.set_yticklabels(sentence, fontsize=7)
    fig.suptitle('"' + " ".join(sentence) + '"    (row attends to column)')
    plt.tight_layout()
    plt.savefig("attention_maps.png", dpi=110)
    print("  Saved attention_maps.png -- one panel per head.")
except Exception as e:
    print(f"  (skipped the plot: {e})")

print("""
================================================================================
  THAT'S THE WHOLE THING.

  You started with a table of word-pair tallies that could not remember what
  it read two words ago. You finished with a decoder-only Transformer whose
  attention heads have measurably divided the work of parsing a sentence
  between them, and which writes new grammatical text within a few hundredths
  of the information-theoretic limit for the language.

  Every component arrived because the previous model failed in a specific,
  visible way:

     counting forgets everything but the last word      -> look at more words
     a flat average blurs them all together             -> ATTENTION
     attention can't tell where anything is             -> POSITION
     one head has to compromise between jobs            -> MULTI-HEAD
     gathering with no time to think                    -> FEED-FORWARD
     deep stacks stop training                          -> RESIDUALS + NORM
     memorising instead of learning                     -> MORE DATA

  That is "Attention Is All You Need", and none of it is magic.

  For where each piece lives in the paper, see PAPER_MAP.md.
================================================================================
""")
