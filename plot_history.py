"""
plot_history.py - graph a training run from history.csv.

Two panels that share the generation axis:
  top    : laps - training best, training mean, validation (best car)
  bottom : % of validation tracks completed

Why two panels instead of one chart with two y-axes? Laps (0-2) and
percent (0-100) are different units; a second y-axis makes it easy to
mislead (where two lines cross depends only on how the axes were scaled).

Install:  python3 -m pip install matplotlib
Run:      python3 plot_history.py                       # ./history.csv
          python3 plot_history.py runs/seed1/history.csv runs/seed2/history.csv
Each CSV gets a PNG saved next to it (history.png).
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")                     # save to file, no window needed
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run train.py first "
                         "(or pass the path, e.g. runs/seed1/history.csv)")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def label_ends(ax, x, items, min_gap):
    """Direct labels at the right end of each line, so identity isn't color
    alone. items: (last_value, text, color). Labels closer than min_gap are
    pushed apart vertically so they never overlap."""
    items = sorted(items)
    placed = []
    for value, text, color in items:
        y = value if not placed else max(value, placed[-1] + min_gap)
        placed.append(y)
        ax.plot(x[-1], value, "o", color=color, markersize=5, zorder=4)
        ax.annotate(text, (x[-1], value), xytext=(x[-1] + 1.2, y),
                    textcoords="data", va="center", fontsize=9, color=INK)


def plot(path):
    h = load(path)
    g = h["generation"]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 2]))
    fig.patch.set_facecolor(SURFACE)

    series = [("train_best", "Training best", BLUE),
              ("train_mean", "Training mean", ORANGE),
              ("val_laps", "Validation (best car)", AQUA)]
    # dashed validation line so it stays visible where it overlaps training best
    for key, name, color in series:
        top.plot(g, h[key], color=color, linewidth=2, label=name,
                 linestyle="--" if key == "val_laps" else "-")
    label_ends(top, g, [(h[k][-1], f"{n}: {h[k][-1]:.2f}", c) for k, n, c in series],
               min_gap=0.13)
    top.set_ylim(0, 2.3)
    top.set_xlim(0, g[-1] * 1.32)          # room on the right for labels
    top.set_ylabel("laps (capped at 2.0)", color=MUTED)
    top.set_title(f"EvoDrive training - {path}", loc="left", color=INK, fontsize=12)
    top.legend(loc="lower right", frameon=False, fontsize=9)
    style(top)

    pct = [v * 100 for v in h["val_success"]]
    bot.plot(g, pct, color=AQUA, linewidth=2)
    label_ends(bot, g, [(pct[-1], f"{pct[-1]:.0f}%", AQUA)], min_gap=0)
    bot.set_ylim(0, 105)
    bot.set_ylabel("validation tracks\ncompleted (%)", color=MUTED)
    bot.set_xlabel("generation", color=MUTED)
    style(bot)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(path) or ".", "history.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", default=["history.csv"])
    for f in p.parse_args().files:
        plot(f)