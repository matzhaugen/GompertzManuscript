"""
Microscopic Gompertz-to-Logistic via Power-Mean Coupling
=========================================================
EXPLICIT NETWORK VERSION using scipy sparse matrix for fast
neighbor averaging. No mean-field approximation.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

# ─── Parameters ───────────────────────────────────────────────────────────────

N = 10_000
BETA = 0.25
SIGMA = 0.02
DT = 0.02
T_MAX = 40.0
SEED = 42

# ─── Network Construction ─────────────────────────────────────────────────────

def degree_log_normal(N, rng):
    z = rng.standard_normal(N)
    return np.maximum(2, np.round(np.exp(1.5 + 0.8 * z)).astype(int))

def degree_scale_free(N, rng):
    u = rng.random(N)
    return np.maximum(2, np.floor(2 * (1 - u) ** (-0.5)).astype(int))

def build_sparse_row_normalized(degrees, rng):
    """
    Build a configuration-model graph and return the row-normalized
    weight matrix as a scipy CSR sparse matrix.
    """
    degrees = degrees.copy()
    if degrees.sum() % 2 == 1:
        degrees[rng.integers(len(degrees))] += 1

    stubs = []
    for i, k in enumerate(degrees):
        stubs.extend([i] * k)
    stubs = np.array(stubs)
    rng.shuffle(stubs)

    rows, cols = [], []
    for idx in range(0, len(stubs) - 1, 2):
        u, v = stubs[idx], stubs[idx + 1]
        if u != v:
            rows.extend([u, v])
            cols.extend([v, u])

    # Build adjacency as sparse matrix
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    # Eliminate duplicates by summing
    A.sum_duplicates()

    # Row-normalize: w_ij = a_ij / k_i
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolated nodes
    D_inv = sp.diags(1.0 / row_sums)
    W = D_inv @ A

    return W


# ─── Simulation: explicit network via sparse matrix ──────────────────────────

def simulate_theta_explicit(theta, W_sparse, rng_seed, t_max=T_MAX):
    r = np.random.default_rng(rng_seed)
    z = r.normal(loc=-13.8, scale=0.03, size=N)
    steps = int(t_max / DT)
    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.exp(np.mean(z))
    noise_scale = np.sqrt(SIGMA * DT)
    eps = 1e-8

    for t_idx in range(steps):
        if theta < eps:
            # Gompertz: W @ z gives each node's neighbor average
            z_bar_local = W_sparse @ z
            drift = -BETA * z
            coupling = BETA * (z_bar_local - z)
        else:
            # Richards: power-mean via sparse matrix
            eth = np.exp(np.clip(theta * z, -100, 2))
            eth_bar_local = W_sparse @ eth  # neighbor average of e^{θz}
            eth_bar_local = np.clip(eth_bar_local, 1e-30, None)
            z_bar_local = np.log(eth_bar_local) / theta

            drift = (BETA / theta) * (1.0 - eth)
            coupling = BETA * (z_bar_local - z)

        tendency = drift + coupling
        dz = tendency * DT + noise_scale * r.standard_normal(N)
        z += dz
        M_traj[t_idx + 1] = np.exp(np.mean(z))

    return M_traj

# ─── Theoretical curves ──────────────────────────────────────────────────────

def gompertz_gamma(M):
    return -np.log(np.clip(M, 1e-15, None))

def logistic_gamma(M):
    return 1 - M

def richards_gamma(M, theta):
    return (1 - np.clip(M, 1e-15, None)**theta) / theta

# ─── Build network ───────────────────────────────────────────────────────────

print(f"Building explicit scale-free network with N = {N:,} nodes...")
rng_net = np.random.default_rng(SEED)
# degrees = degree_log_normal(N, rng_net)
degrees = degree_scale_free(N, rng_net)
import pdb; pdb.set_trace()
print(f"  Degrees: mean={degrees.mean():.1f}, max={degrees.max()}, min={degrees.min()}")
W_sparse = build_sparse_row_normalized(degrees, np.random.default_rng(SEED + 1))
print(f"  Sparse W: {W_sparse.nnz} nonzero entries ({W_sparse.nnz/N:.1f} avg per row)")

# ─── Run simulations ─────────────────────────────────────────────────────────

thetas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
labels = {
    0.0:  r"$\theta \to 0$ (Gompertz)",
    0.1:  r"$\theta = 0.1$",
    0.3:  r"$\theta = 0.3$",
    0.5:  r"$\theta = 0.5$",
    0.7:  r"$\theta = 0.7$",
    1.0:  r"$\theta = 1$ (Logistic)",
}

print(f"\nRunning explicit network simulations...")
results = {}
t_maxes = {}
for th in thetas:
    t_max = T_MAX * max(1.0, 1 + 1.1 * th)
    t_maxes[th] = t_max
    print(f"  θ = {th}, T_max = {t_max:.0f}...")
    X = simulate_theta_explicit(th, W_sparse, SEED + 2, t_max=t_max)
    results[th] = X
    print(f"    X(0)={X[0]:.4e}, X(end)={X[-1]:.4f}")

# ─── Plotting ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

M_th = np.linspace(0.005, 0.995, 500)

# ── Left panel ──

for th_ref in [0.1, 0.3, 0.5, 0.7]:
    ax1.plot(M_th, richards_gamma(M_th, th_ref),
             color="0.55", lw=1.1, ls="-", zorder=5)

ax1.plot(M_th, gompertz_gamma(M_th), color="black", lw=2.2, ls=":",
         label="Gompertz (theory)", zorder=10)
ax1.plot(M_th, logistic_gamma(M_th), color="black", lw=2.2, ls="--",
         label="Logistic (theory)", zorder=10)

annot_cfg = [
    (0.1, 0.08, (-55, -5)),
    (0.3, 0.08, (-55, -5)),
    (0.5, 0.08, (-55, -5)),
    (0.7, 0.08, (-55, -5)),
]
for th_ref, x_arrow, xytext in annot_cfg:
    y_arrow = richards_gamma(x_arrow, th_ref)
    ax1.annotate(rf"$\theta={th_ref}$", xy=(x_arrow, y_arrow),
                 xytext=xytext, textcoords="offset points",
                 fontsize=8.5, fontweight="bold", color="0.3",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5, alpha=0.9),
                 arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0,
                                 connectionstyle="arc3,rad=0.0"),
                 zorder=12)

cmap = plt.cm.RdYlBu_r
colors = [cmap(i / (len(thetas) - 1)) for i in range(len(thetas))]
markers_list = ["o", "s", "^", "D", "v", "p"]

for idx, th in enumerate(thetas):
    X = results[th]
    X_norm = (X - X.min()) / (X.max() - X.min())
    X_norm = np.clip(X_norm, 1e-6, 1 - 1e-6)

    dX = np.gradient(X_norm, DT)
    gamma = dX / X_norm
    gamma_norm = gamma / BETA

    step = max(1, len(X_norm) // 40)
    X_sub = X_norm[::step]
    g_sub = gamma_norm[::step]

    mask = (X_sub > 0.01) & (X_sub < 0.98) & (g_sub > -0.5) & (g_sub < 8)
    ax1.plot(X_sub[mask], g_sub[mask],
             color=colors[idx], marker=markers_list[idx], markersize=4.5,
             lw=1.4, alpha=0.9, markeredgewidth=0.5, markeredgecolor="white",
             label=labels[th], zorder=7)

ax1.set_xlabel("Abundance $X(t)$ [normalized]")
ax1.set_ylabel(r"Relative Growth Rate $\dot{\bar{z}} / \beta$ [normalized]")
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.3, 5.5)
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.tick_params(which="both", direction="in", top=True, right=True)
ax1.legend(loc="upper right", ncol=1, fontsize=7.5)
ax1.set_title("(a) Relative growth rate", fontsize=11, pad=8)

# ── Right panel ──

t_theory = np.linspace(0, T_MAX, 300)
X_gomp = results[0.0]
X_gomp_norm = (X_gomp - X_gomp.min()) / (X_gomp.max() - X_gomp.min())
X_gomp_norm = np.clip(X_gomp_norm, 1e-6, 1 - 1e-6)
with np.errstate(invalid="ignore", divide="ignore"):
    dlog_gomp = np.log(-np.log(X_gomp_norm))
valid_g = np.isfinite(dlog_gomp)
t_gomp = np.linspace(0, T_MAX, len(X_gomp))
from numpy.polynomial.polynomial import polyfit
t_valid_g = t_gomp[valid_g]
dlog_valid_g = dlog_gomp[valid_g]
mid = (t_valid_g > 2) & (t_valid_g < T_MAX - 2)
if mid.sum() > 10:
    c0, c1 = polyfit(t_valid_g[mid], dlog_valid_g[mid], 1)
    ax2.plot(t_theory, c0 + c1 * t_theory, color="black", lw=2.2, ls=":",
             label=f"Gompertz fit (slope={c1:.3f})", zorder=10)

for idx, th in enumerate(thetas):
    X = results[th]
    t_max = t_maxes[th]
    t_arr = np.linspace(0, t_max, len(X))

    X_norm = (X - X.min()) / (X.max() - X.min())
    X_norm = np.clip(X_norm, 1e-6, 1 - 1e-6)

    with np.errstate(invalid="ignore", divide="ignore"):
        dlog = np.log(-np.log(X_norm))

    valid = np.isfinite(dlog)
    t_valid = t_arr[valid]
    dlog_valid = dlog[valid]

    step = max(1, len(t_valid) // 60)
    ax2.plot(t_valid[::step], dlog_valid[::step],
             color=colors[idx], marker=markers_list[idx], markersize=4,
             lw=1.2, alpha=0.85, markeredgewidth=0.4, markeredgecolor="white",
             label=labels[th], zorder=7 - idx)

ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\ln\!\left(\ln(1/X(t))\right)$")
ax2.set_xlim(-1, max(t_maxes.values()) + 1)
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.tick_params(which="both", direction="in", top=True, right=True)
ax2.legend(loc="upper right", ncol=1, fontsize=7.5)
ax2.set_title("(b) Double-log transform (straight line = Gompertz)", fontsize=11, pad=8)

fig.suptitle(
    r"Microscopic Richards interpolation: $\theta$-power-mean coupling"
    f",  Explicit Scale-Free network,  $N = ${N:,},  "
    r"$\beta=$" + f"{BETA},  " + r"$\sigma=$" + f"{SIGMA}",
    fontsize=11, y=1.02
)

plt.tight_layout()
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
plt.savefig(dir_path + "/../figures/gompertz_richards_explicit.png")
plt.savefig(dir_path + "/../figures/gompertz_richards_explicit.pdf")
print("\nPlots saved.")
