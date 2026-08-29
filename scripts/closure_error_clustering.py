"""
Closure error against network clustering: a Watts-Strogatz sweep
================================================================
Companion experiment to closure_error.py. That script sweeps the DEGREE
DISTRIBUTION (scale-free, Erdos-Renyi) while clustering sits at essentially
zero throughout, because configuration-model graphs are locally tree-like at
N = 10,000. This script sweeps the other axis: it holds the degree sequence
fixed and varies the number of short cycles.

Why this axis is the one that matters
-------------------------------------
The macroscopic logistic equation closes only if neighbouring node states are
uncorrelated. On a locally tree-like graph two neighbours reach the rest of
the network by disjoint paths, so nothing drives them in common and they
decorrelate. Inside a short cycle they share drivers, their states co-move,
and the closure term survives. Wuyts and Sieber (Phys. Rev. E 106, 054312,
arXiv:2111.07643) find exactly this: "the presence of cycles beyond closure
distance is the dominant cause for mean-field model biases", with Erdos-Renyi
graphs showing considerably less bias than lattices DESPITE their degree
heterogeneity, and the bias vanishing above dimension four where random walks
stop self-intersecting.

So the existing closure figure varies the axis that the moment-closure
literature says is not the important one, and holds the important one pinned
at its most favourable value. This script supplies the missing control.

The Watts-Strogatz construction
-------------------------------
Start from a ring lattice: N nodes on a circle, each joined to its K/2
nearest neighbours on either side. This is regular (every node has degree K)
and dense in short cycles. Then rewire each edge with probability p to a
uniformly random endpoint. p = 0 is the pure lattice; p = 1 is essentially a
random graph; p ~ 0.01-0.1 is the small-world regime, still clustered but
with short paths. Sweeping p therefore moves clustering over orders of
magnitude with the mean degree held fixed.

What each model should do, and what is actually being tested
------------------------------------------------------------
At p = 0 the ring lattice is REGULAR, so the row-normalised W is doubly
stochastic and exact Gompertz closure follows from the theorem in the text.
The Gompertz curve at p = 0 is therefore a check that the code implements the
theorem, not a discovery -- the linearity of the log-domain dynamics means
neighbour correlations never enter the aggregate however many cycles exist.

The genuinely open region is the middle of the sweep. Once p > 0 the rewiring
makes degrees heterogeneous, W stops being doubly stochastic, and the closure
discrepancy eps_G ~ beta |Cov(c, delta)| of the text switches on while
clustering is still high. Both error sources are active at once there. The
sweep reports the column-sum deviation max_j |c_j - 1| alongside the errors so
the two can be told apart.

The transmission logistic has no such protection at any p and should be worst
where the cycles are.

Outputs
-------
    ../figures/closure_error_clustering.pdf,.png

Run with:  cd scripts && uv run python closure_error_clustering.py
Then copy the pdf+png into arxiv2/ if the figure is used in the paper.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")

from sim_params import BETA, DT, IC_MEAN_Z_CLOSURE, IC_STD, N, SEED, T_MAX  # noqa: F401

# The three simulators are imported rather than copied so this experiment and
# closure_error.py can never drift apart in what they measure.
from closure_error import simulate_gompertz, simulate_logistic, simulate_theta1
from closure_error import trajectory_mean_error

# Mean degree of the ring lattice. Matched to the Poisson(6) mean of the
# Erdos-Renyi case in closure_error.py so the two figures are comparable.
K = 6

# Rewiring probabilities. p = 0 is the pure lattice; p = 1 is essentially a
# random graph and should reproduce the existing figure's regime.
P_VALUES = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]

# Smaller than closure_error.py's N_ENSEMBLE because this sweep runs seven
# network types rather than two; the total simulation count is comparable.
N_ENSEMBLE_SWEEP = 8

COLOR_GOMPERTZ = "#e5a400"
COLOR_LOGISTIC = "#008000"
COLOR_THETA1 = "#2a78d6"


def ring_lattice_edges(n_nodes, k):
    """Undirected edge list of a ring lattice: each node joined to k/2 on each side."""
    half = k // 2
    u = np.repeat(np.arange(n_nodes), half)
    v = np.concatenate([(np.arange(n_nodes) + off) % n_nodes for off in range(1, half + 1)])
    v = v.reshape(half, n_nodes).T.ravel()
    return u, v


def watts_strogatz_W(n_nodes, k, p, rng):
    """Row-normalised coupling matrix of a Watts-Strogatz graph.

    Rewires each edge's second endpoint with probability p, rejecting self-loops
    and duplicate edges. Returns (W, degrees). At p = 0 the graph is regular, so
    W is doubly stochastic; for p > 0 the degrees spread and it is not.
    """
    u, v = ring_lattice_edges(n_nodes, k)
    u, v = u.copy(), v.copy()

    if p > 0:
        rewire = rng.random(len(u)) < p
        idx = np.flatnonzero(rewire)
        if len(idx):
            existing = set(zip(*(np.minimum(u, v), np.maximum(u, v))))
            for e in idx:
                a = u[e]
                for _ in range(20):  # a few tries, then leave the edge alone
                    b = int(rng.integers(n_nodes))
                    key = (min(a, b), max(a, b))
                    if b != a and key not in existing:
                        existing.discard((min(u[e], v[e]), max(u[e], v[e])))
                        v[e] = b
                        existing.add(key)
                        break

    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0  # simple graph
    deg = np.asarray(A.sum(axis=1)).ravel()
    isolated = deg == 0
    W = sp.diags(1.0 / np.where(isolated, 1.0, deg)) @ A
    if isolated.any():
        W = W + sp.diags(isolated.astype(float))
    return W.tocsr(), deg, A


def clustering_coefficient(A):
    """Mean local clustering: triangles through a node over pairs of its neighbours."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    triangles = np.asarray(A.multiply(A @ A).sum(axis=1)).ravel() / 2.0
    pairs = deg * (deg - 1) / 2.0
    ok = pairs > 0
    return float(np.mean(triangles[ok] / pairs[ok]))


def column_sum_deviation(W):
    """max_j |c_j - 1|: zero exactly when W is doubly stochastic."""
    c = np.asarray(W.sum(axis=0)).ravel()
    return float(np.max(np.abs(c - 1.0)))


CACHE = "closure_error_clustering_results.npy"


def main(recompute=False):
    if not recompute and os.path.exists(CACHE):
        rows = list(np.load(CACHE, allow_pickle=True))
        print(f"loaded cached sweep from {CACHE} "
              f"(delete it, or pass recompute=True, to re-run)\n")
        for r in rows:
            print(f"  p = {r['p']:<6g} clustering = {r['clustering']:.4f}  "
                  f"max|c_j-1| = {r['colsum']:.4f}  |  "
                  f"Gompertz {np.median(r['g']):.3e}   logistic {np.median(r['l']):.3e}   "
                  f"theta=1 {np.median(r['t1']):.3e}   "
                  f"L/G = {np.median(r['l'])/max(np.median(r['g']),1e-18):.1f}")
        return _plot(rows)

    print(f"Watts-Strogatz closure sweep: N = {N:,}, K = {K}, "
          f"{N_ENSEMBLE_SWEEP} realizations per p\n")
    rows = []
    for p in P_VALUES:
        eg, el, et1, clust, colsum = [], [], [], [], []
        for s in range(N_ENSEMBLE_SWEEP):
            seed = SEED + 100 * s
            W, deg, A = watts_strogatz_W(N, K, p, np.random.default_rng(seed + 1))
            clust.append(clustering_coefficient(A))
            colsum.append(column_sum_deviation(W))

            M_g, rel_g = simulate_gompertz(W, np.random.default_rng(seed + 2))
            M_l, rel_l = simulate_logistic(W, np.random.default_rng(seed + 2))
            M_t1, rel_t1 = simulate_theta1(W, np.random.default_rng(seed + 2))
            eg.append(trajectory_mean_error(M_g, rel_g))
            el.append(trajectory_mean_error(M_l, rel_l))
            et1.append(trajectory_mean_error(M_t1, rel_t1))

        rows.append(dict(p=p, clustering=np.mean(clust), colsum=np.mean(colsum),
                         g=np.array(eg), l=np.array(el), t1=np.array(et1)))
        print(f"  p = {p:<6g} clustering = {np.mean(clust):.4f}  "
              f"max|c_j-1| = {np.mean(colsum):.4f}  |  "
              f"Gompertz {np.median(eg):.3e}   logistic {np.median(el):.3e}   "
              f"theta=1 {np.median(et1):.3e}   L/G = {np.median(el)/max(np.median(eg),1e-18):.1f}")

    np.save(CACHE, np.array(rows, dtype=object))
    print(f"\ncached sweep to {CACHE}")
    return _plot(rows)


def _plot(rows):
    # ── figure ───────────────────────────────────────────────────────────────
    # Errors are computed as fractions. closure_error.py plots the same quantity
    # scaled to percent, so we scale identically here and use its axis label.
    PCT = 100.0
    # At p = 0 the Gompertz error is zero to machine precision. Plotting it
    # literally would stretch the log axis over fourteen decades and flatten the
    # structure that matters, so it is drawn at a floor with an explicit label.
    FLOOR = 1e-6   # in percent
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.8, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.6, 1.2]})
    p_nonzero = [p for p in P_VALUES if p > 0]
    p_zero_pos = min(p_nonzero) / 10.0
    xs = np.array([p_zero_pos if r["p"] == 0 else r["p"] for r in rows])

    for key, color, label in [("g", COLOR_GOMPERTZ, "Gompertz ($\\theta\\to0$)"),
                              ("l", COLOR_LOGISTIC, "Transmission logistic"),
                              ("t1", COLOR_THETA1, "$\\theta=1$ endpoint")]:
        med = PCT * np.array([np.median(r[key]) for r in rows])
        lo = PCT * np.array([np.percentile(r[key], 25) for r in rows])
        hi = PCT * np.array([np.percentile(r[key], 75) for r in rows])
        clipped = med < FLOOR
        ax.plot(xs[~clipped], med[~clipped], "o-", color=color, label=label, zorder=3)
        ax.fill_between(xs[~clipped], lo[~clipped], hi[~clipped], color=color, alpha=0.18, lw=0)
        if clipped.any():   # draw the exact-closure points on the floor, hollow
            ax.plot(xs[clipped], np.full(clipped.sum(), FLOOR), "o",
                    mfc="white", mec=color, mew=1.8, zorder=4)
            ax.plot(xs[:2], [FLOOR, med[1]], "-", color=color, alpha=0.35, zorder=2)
            ax.annotate("exact: $0$ to machine\nprecision ($<10^{-14}\\,\\%$)",
                        xy=(xs[0], FLOOR), xytext=(xs[1] * 1.4, FLOOR * 2.2),
                        fontsize=9, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2.5, 0.4)
    ax.set_ylabel("Relative closure error (%)")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(f"Closure error against clustering ($N={N:,}$, $K={K}$, "
                 f"{N_ENSEMBLE_SWEEP} realizations)", fontsize=11)

    # Linear scale: both diagnostics contain an exact zero, which a log axis cannot show.
    ax2.plot(xs, [r["clustering"] for r in rows], "s-", color="#52514e",
             label="clustering coefficient (short cycles)")
    ax2.plot(xs, [r["colsum"] for r in rows], "^--", color="#e34948",
             label=r"$\max_j |c_j - 1|$   ($=0$ iff doubly stochastic)")
    ax2.set_xscale("log")
    ax2.set_ylim(-0.08, 1.9)
    ax2.set_xlabel("Rewiring probability $p$     "
                   "(left-most point is the pure ring lattice, $p=0$)")
    ax2.set_ylabel("Network\ndiagnostic")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax2.grid(True, which="both", alpha=0.25)
    ticks = [p_zero_pos] + p_nonzero
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(["0"] + [f"{p:g}" for p in p_nonzero])
    ax2.minorticks_off()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"../figures/closure_error_clustering.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else None)
        print(f"\nwrote {out}")

    # Which network property does the Gompertz closure error actually track?
    from scipy.stats import spearmanr
    g = np.array([np.median(r["g"]) for r in rows])
    cl = np.array([r["clustering"] for r in rows])
    cs = np.array([r["colsum"] for r in rows])
    print(f"\nGompertz closure error vs clustering:        Spearman {spearmanr(g, cl).statistic:+.2f}")
    print(f"Gompertz closure error vs max|c_j-1|:        Spearman {spearmanr(g, cs).statistic:+.2f}")
    l = np.array([np.median(r["l"]) for r in rows])
    print(f"Transmission logistic error vs clustering:   Spearman {spearmanr(l, cl).statistic:+.2f}")


if __name__ == "__main__":
    main()
