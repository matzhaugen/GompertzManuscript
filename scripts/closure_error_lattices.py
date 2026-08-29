"""
Closure error on regular lattices: clustering varied, regularity held fixed
==========================================================================
The controlled companion to closure_error_clustering.py.

That script rewires a ring, which lowers the clustering and raises the degree
heterogeneity at the same time, so the two candidate causes of closure error are
confounded along the sweep and can only be separated at its p = 0 endpoint. This
script removes the confound entirely: every network here is REGULAR, so the
row-normalised coupling is doubly stochastic throughout and max_j |c_j - 1| = 0 in
every case, while the clustering coefficient ranges from 0.00 to 0.60.

The four networks are chosen in degree-matched pairs, so that clustering is the
only quantity that differs within a pair:

    degree 4:  ring lattice K=4     C = 0.50    vs   periodic square lattice   C = 0.00
    degree 6:  ring lattice K=6     C = 0.60    vs   periodic triangular lat.  C = 0.40

The square lattice is the important case. It is built entirely from four-cycles and
contains no triangles at all, so its clustering coefficient is exactly zero even
though it is as locally cyclic as any network here. It therefore distinguishes
"triangles" from "short cycles", which the clustering coefficient alone cannot do.

What the two models are predicted to do:

  * The Gompertz aggregate coupling term is (1/N) sum_j (c_j - 1) z_j, which
    vanishes identically on any doubly stochastic coupling regardless of the z
    values. All four networks are regular, so the prediction is exact closure in
    all four, whatever the clustering.
  * The transmission logistic has no such protection. Its discrepancy is the
    network-weighted covariance Cov_W(x, x), generated whenever neighbours share
    drivers -- which happens inside cycles of any length, not only triangles.

Outputs
-------
    prints a summary table and the corresponding LaTeX rows for the manuscript.

Run with:  cd scripts && uv run python closure_error_lattices.py
"""

import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from sim_params import BETA, DT, N, SEED, T_MAX  # noqa: F401

# The simulators and the error measure are imported, never copied, so this
# experiment and closure_error.py can never drift apart in what they measure.
from closure_error import (simulate_gompertz, simulate_logistic, simulate_theta1,
                           trajectory_mean_error)
from closure_error_clustering import clustering_coefficient, column_sum_deviation
from closure_error_clustering import ring_lattice_edges

# Side length of the periodic lattices; L*L is matched to N in sim_params.
L = int(round(np.sqrt(N)))


def _W_from_edges(edges, n_nodes):
    """Row-normalised coupling matrix from an undirected edge list."""
    u = np.array([e[0] for e in edges])
    v = np.array([e[1] for e in edges])
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0                      # simple graph: needed for the triangle count
    deg = np.asarray(A.sum(axis=1)).ravel()
    W = sp.diags(1.0 / deg) @ A
    return W.tocsr(), deg, A


def square_lattice(side):
    """Periodic square lattice: degree 4, built from four-cycles, no triangles."""
    idx = lambda r, c: (r % side) * side + (c % side)  # noqa: E731
    edges = []
    for r in range(side):
        for c in range(side):
            edges += [(idx(r, c), idx(r, c + 1)), (idx(r, c), idx(r + 1, c))]
    return _W_from_edges(edges, side * side)


def triangular_lattice(side):
    """Periodic triangular lattice: degree 6, dense in triangles."""
    idx = lambda r, c: (r % side) * side + (c % side)  # noqa: E731
    edges = []
    for r in range(side):
        for c in range(side):
            edges += [(idx(r, c), idx(r, c + 1)),
                      (idx(r, c), idx(r + 1, c)),
                      (idx(r, c), idx(r + 1, c - 1))]
    return _W_from_edges(edges, side * side)


def ring_lattice(n_nodes, k):
    """Periodic ring lattice: degree k, each node joined to k/2 on each side."""
    u, v = ring_lattice_edges(n_nodes, k)
    return _W_from_edges(list(zip(u, v)), n_nodes)


def main():
    n_nodes = L * L
    cases = [
        ("Ring lattice, $K=4$",   ring_lattice(n_nodes, 4)),
        ("Square lattice",        square_lattice(L)),
        ("Ring lattice, $K=6$",   ring_lattice(n_nodes, 6)),
        ("Triangular lattice",    triangular_lattice(L)),
    ]

    print(f"Closure error on regular lattices: N = {n_nodes:,}, deterministic dynamics\n")
    header = (f"{'network':22s} {'deg':>4s} {'C':>7s} {'max|c-1|':>9s} "
              f"{'Gompertz %':>11s} {'logistic %':>11s} {'theta=1 %':>10s}")
    print(header)
    rows = []
    for name, (W, deg, A) in cases:
        clustering = clustering_coefficient(A)
        colsum = column_sum_deviation(W)
        seed = SEED + 2
        err_g = trajectory_mean_error(*simulate_gompertz(W, np.random.default_rng(seed)))
        err_l = trajectory_mean_error(*simulate_logistic(W, np.random.default_rng(seed)))
        err_t = trajectory_mean_error(*simulate_theta1(W, np.random.default_rng(seed)))
        rows.append((name, deg.mean(), clustering, colsum, err_g, err_l, err_t))
        plain = name.replace("$", "").replace("\\", "")
        print(f"{plain:22s} {deg.mean():4.0f} {clustering:7.4f} {colsum:9.4f} "
              f"{100*err_g:11.2e} {100*err_l:11.4f} {100*err_t:10.4f}")

    print("\n% LaTeX rows for the manuscript table:")
    for name, deg, clustering, colsum, err_g, err_l, err_t in rows:
        print(f"{name:22s} & {deg:.0f} & {clustering:.2f} & {colsum:.0f} & "
              f"${100*err_g:.1f}\\times10^{{-15}}$ & {100*err_l:.4f} \\\\"
              .replace(f"${100*err_g:.1f}\\times10^{{-15}}$",
                       f"${100*err_g/1e-15:.1f}\\times 10^{{-15}}$"))


if __name__ == "__main__":
    main()
