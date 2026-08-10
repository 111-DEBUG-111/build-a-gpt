"""
display.py  --  printing helpers only. No lesson content here.

These just make the output of each step readable in a terminal. Feel free to
ignore this file entirely; nothing about Transformers lives in it.
"""

WIDTH = 74


def title(step, text):
    """The banner at the top of each step."""
    print()
    print("=" * WIDTH)
    print(f"  STEP {step}:  {text}")
    print("=" * WIDTH)


def section(text):
    """A sub-heading inside a step."""
    print()
    print(f"--- {text} " + "-" * max(0, WIDTH - len(text) - 5))


def bar(fraction, width=40):
    """A little text bar chart: bar(0.25) -> '##########..............'"""
    filled = int(round(float(fraction) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


def matrix(m, row_labels=None, col_labels=None, fmt="{:>6.2f}", indent="  "):
    """Print a small 2-D array of numbers with optional labels."""
    rows = [[float(x) for x in row] for row in m]
    label_w = max((len(str(r)) for r in row_labels), default=0) if row_labels else 0

    # Column headers must be exactly as wide as the numbers underneath them,
    # so work that width out from the number format rather than guessing.
    cell_w = len(fmt.format(0.0))

    if col_labels is not None:
        head = " " * (label_w + len(indent))
        for c in col_labels:
            head += f"{str(c)[:cell_w - 1]:>{cell_w}}"
        print(head)

    for i, row in enumerate(rows):
        line = indent
        if row_labels is not None:
            line += f"{str(row_labels[i]):>{label_w}}"
        for x in row:
            line += fmt.format(x)
        print(line)
