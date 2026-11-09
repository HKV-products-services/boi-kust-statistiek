"""Figures (optional; the R scripts make these interactively, they are not part of the outputs)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from kuststatistiek.gw import assigncat

# cols in the R scripts: black, blue, magenta3, cyan3, darkorange2, darkseagreen, mediumpurple1, skyblue1, sienna4
COLS = [
    "black",
    "blue",
    "#CD00CD",
    "#00CDCD",
    "#EE7600",
    "#8FBC8F",
    "#AB82FF",
    "#87CEFF",
    "#8B4726",
]


def plot_returnlevels(
    Qfreq0, cats0, freq, title, ylabel, file=None, ind=(7, 5, 3, 0), Q0sd=None, pn=0.95
):
    """Directional return values for a few return periods (as the ggplot in the R scripts).

    ind:  rows of Qfreq0 (default: return periods 1, 100, 10^4 and 10^7 yr, as R: ind <- c(1, 4, 6, 8))
    Q0sd: optional bootstrap standard deviations (same shape as Qfreq0), e.g. Q0sd from
          <station>_stats_wind_boot1.pkl (without model error) or _boot2.pkl (with model error);
          then a band Qfreq0 +- qnorm((1 + pn)/2) * Q0sd is drawn
    """
    cats0 = np.asarray(cats0) % 360
    order = np.argsort(cats0)
    qn = norm.ppf((1 + pn) / 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    for c, i in zip(COLS, ind):
        q = Qfreq0[i, order]
        ax.plot(cats0[order], q, "o-", ms=3, color=c, label=f"{1 / freq[i]:.0f} yr")
        if Q0sd is not None:
            ax.fill_between(
                cats0[order],
                q - qn * Q0sd[i, order],
                q + qn * Q0sd[i, order],
                color=c,
                alpha=0.2,
                lw=0,
            )
    ax.set(title=title, xlabel="direction", ylabel=ylabel)
    ax.legend(title="Return Period [yr]")
    ax.set_box_aspect(1)
    ax.grid(True)
    if file:
        fig.savefig(file, dpi=300)
        plt.close(fig)
    return fig


def plot_frequencies(x, mu, Mu, cats, freq, title, xlabel, file=None):
    """Omni (mu, black) and directional (Mu, one column per 45-degree sector) exceedance frequencies
    (as the plot after the GW fit in the R scripts)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, mu, color=COLS[0], lw=2, label="omni")
    for i in range(len(cats) - 1):  # last column of Mu is omni
        ax.plot(x, Mu[:, i], color=COLS[1 + i % 8], lw=2, label=f"{cats[i]:g}")
    ax.set(
        yscale="log",
        ylim=(min(freq) * 0.2, max(freq) * 2),
        title=title,
        xlabel=xlabel,
        ylabel="frequency [1/yr]",
    )
    ax.set_box_aspect(1)
    ax.grid(color="lightgray", linestyle=":")
    ax.legend(loc="lower left", fontsize=7)
    if file:
        fig.savefig(file, dpi=300)
        plt.close(fig)
    return fig


def covariate_graph(
    values, dirs, lbin, ubin, cats, N=None, title="", ax=None, color="black"
):
    """CovariateGraph0.R: conditional probability of each direction bin given exceedance of a threshold."""
    values = np.asarray(values, dtype=float)
    N = len(values) if N is None else N
    cat = assigncat(dirs, lbin, ubin)
    order = np.argsort(-values, kind="stable")
    cat = cat[order]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="fraction exceeded",
        ylabel="cond. prob. of category",
        title=title,
    )
    for i in range(cat.shape[1]):
        rank = np.flatnonzero(cat[:, i]) + 1
        if len(rank):
            ax.plot(rank / N, np.arange(1, len(rank) + 1) / rank, color=color, lw=0.8)
            ax.text(rank[-1] / N, len(rank) / rank[-1], f"{cats[i]:g}", fontsize=7)
    return ax
