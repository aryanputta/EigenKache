"""Result figure for the EigenKache paper: the two-regime fidelity frontier.

One claim: landmark compression beats eviction only when the cold region is
redundant. Left (i.i.d. random) -> eviction wins; right (structured, low-rank)
-> landmark wins. Values are the committed benchmark results (Tables 1-2).

Rendered in the paper's serif font via paper-kit/paperstyle so it does not look
like a default matplotlib plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "paper-kit"))
import paperstyle  # noqa: E402

paperstyle.use()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = paperstyle.PALETTE

RETAINED = [25, 33, 50]
RANDOM = {
    "window":   [0.558, 0.653, 0.858],
    "h2o_like": [0.657, 0.804, 0.872],
    "landmark": [0.598, 0.683, 0.857],
}
STRUCTURED = {
    "window":   [0.867, 0.930, 0.965],
    "h2o_like": [0.692, 0.736, 0.857],
    "landmark": [0.902, 0.944, 0.977],
}
STYLE = {  # same identity in both panels
    "window":   (P["grey"], "s", 1.2, "window (evict)"),
    "h2o_like": (P["rust"], "^", 1.2, "H2O-like (evict)"),
    "landmark": (P["blue"], "o", 2.4, "landmark (compress)"),
}


def panel(ax, data, title, winner):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.5)
    for key in ("window", "h2o_like", "landmark"):
        c, m, lw, _ = STYLE[key]
        z = 3 if key == "landmark" else 2
        ax.plot(RETAINED, data[key], marker=m, color=c, linewidth=lw,
                markersize=4 if key != "landmark" else 4.5,
                markeredgecolor="white", markeredgewidth=0.4, zorder=z)
    ax.set_title(title)
    ax.set_xlabel("retained KV (%)")
    ax.set_xticks(RETAINED)
    ax.set_xlim(22, 53)
    ax.tick_params(length=3)


def main():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    panel(axl, RANDOM, "i.i.d. random cold region", "h2o_like")
    panel(axr, STRUCTURED, "structured (low-rank) cold region", "landmark")
    axl.set_ylabel("attention cosine vs. exact")
    axl.set_ylim(0.5, 1.0)
    axl.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # takeaway labels, placed near the winning line, no arrows
    axl.text(33, 0.83, "eviction wins", color=P["rust"], fontsize=8,
             ha="center", style="italic")
    axr.text(37, 0.995, "landmark wins", color=P["blue"], fontsize=8,
             ha="center", style="italic")

    # one shared legend below, clean
    handles = [plt.Line2D([0], [0], color=STYLE[k][0], marker=STYLE[k][1],
                          linewidth=STYLE[k][2], markersize=4, label=STYLE[k][3])
               for k in ("window", "h2o_like", "landmark")]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8,
               columnspacing=1.4, handletextpad=0.4)
    fig.subplots_adjust(bottom=0.30, wspace=0.08)
    paperstyle.save(fig, ROOT / "docs" / "paper" / "figs" / "regimes.pdf")


if __name__ == "__main__":
    main()
