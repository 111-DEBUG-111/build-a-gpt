"""
lib  --  the four modules every lesson shares

Nothing here teaches anything on its own; the lessons do that. These are the
pieces they all lean on, kept in one place so a change to the language or the
printing shows up everywhere at once.

    grammar.py    the 500-word language, its rules, and the grammar checker
    data.py       batching, and the train/validation split
    display.py    terminal printing helpers, no lesson content
    gpt_model.py  the finished architecture in one file, ~120 lines

A lesson picks up what it needs with:

    from lib import grammar, data
    from lib.display import title, section, bar

OUTPUTS is where the three lessons that write files put them -- the trained
model from step 9 and the two pictures from steps 6 and 10. It is an absolute
path, so those lessons land in the same place whichever directory you run
them from.
"""

import pathlib

OUTPUTS = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS.mkdir(exist_ok=True)
