"""
Explicit Network Gompertz Simulation (No Mean-Field)
=====================================================
Each node interacts with its ACTUAL neighbors via a sparse adjacency
structure. No mean-field approximation — the neighbor average for
node i is computed from its specific neighbors only.

Comparison: mean-field vs explicit network for multiple topologies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

# ─── Parameters ───────────────────────────────────────────────────────────────

N = 10_000
BETA = 0.25
SIGMA = 0.001
DT = 0.05
T_MAX = 40.0
SEED = 42

# ─── Network Construction ─────────────────────────────────────────────────────

def build_configuration_model(degrees, rng):
    """
    Build a random graph from a degree sequence using the configuration model.
    Returns a list of neighbor lists (sparse representation).
    Ensures no self-loops; allows multi-edges (negligible for large N).
    """
    # Make sure sum of degrees is even
    degrees = degrees.copy()
    if degrees.sum() % 2 == 1:
        degrees[rng.integers(len(degrees))] += 1

    # Create stub list
    stubs = []
    for i, k in enumerate(degrees):
        stubs.extend([i] * k)
    stubs = np.array(stubs)
    rng.shuffle(stubs)

    # Pair stubs
    neighbors = [[] for _ in range(N)]
    for idx in range(0, len(stubs) - 1, 2):
        u, v = stubs[idx], stubs[idx + 1]
        if u != v:  # no self-loops
            neighbors[u].append(v)
            neighbors[v].append(u)

    return neighbors


def build_row_normalized_weights(neighbors):
    """
    Build row-normalized weight matrix in sparse form.
    w_ij = 1/k_i for each neighbor j of i.
    Returns list of (neighbor_indices, weights) per node.
    """
    sparse_W = []
    for i in range(len(neighbors)):
        nbrs = np.array(neighbors[i], dtype=np.int32)
        if len(nbrs) == 0:
            sparse_W.append((np.array([], dtype=np.int32), np.array([])))
        else:
            # Remove duplicates (multi-edges) by averaging
            unique_nbrs, counts = np.unique(nbrs, return_counts=True)
            weights = counts.astype(np.float64)
            weights /= weights.sum()  # row-normalize
            sparse_W.append((unique_nbrs, weights))
    return sparse_W


# ─── Degree Generators ────────────────────────────────────────────────────────

def degree_scale_free(N, rng):
    u = rng.random(N)
    return np.maximum(2, np.floor(2 * (1 - u) ** (-0.5)).astype(int))

def degree_log_normal(N, rng):
    z = rng.standard_normal(N)
    return np.maximum(2, np.round(np.exp(1.5 + 0.8 * z)).astype(int))

def degree_erdos_renyi(N, rng):
    return np.maximum(2, rng.poisson(6, N))

def degree_power_law(N, rng):
    u = rng.random(N)
    return np.maximum(2, np.floor(2 * (1 - u) ** (-1/1.2)).astype(int))

NETWORKS = {
    "Scale-Free": degree_scale_free,
    "Log-Normal": degree_log_normal,
    r"Erdős-Rényi": degree_erdos_renyi,
    "Power Law": degree_power_law,
}

# ─── Simulation: Explicit Network ─────────────────────────────────────────────

def simulate_explicit(sparse_W, rng, theta=0.0):
    """Simulate with actual neighbor interactions (no mean-field)."""
    z = rng.normal(loc=-4.6, scale=0.3, size=N)
    steps = int(T_MAX / DT)
    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.exp(np.mean(z))
    noise_scale = np.sqrt(SIGMA * DT)
    eps = 1e-8

    for t_idx in range(steps):
        # Compute neighbor averages explicitly for each node
        if theta < eps:
            # Gompertz: neighbor average in z-domain
            z_bar_local = np.zeros(N)
            for i in range(N):
                nbrs, wts = sparse_W[i]
                if len(nbrs) > 0:
                    z_bar_local[i] = np.dot(wts, z[nbrs])
                else:
                    z_bar_local[i] = z[i]

            drift = -BETA * z
            coupling = BETA * (z_bar_local - z)
        else:
            # Richards: power-mean coupling
            z_bar_local = np.zeros(N)
            for i in range(N):
                nbrs, wts = sparse_W[i]
                if len(nbrs) > 0:
                    eth_nbrs = np.exp(np.clip(theta * z[nbrs], -100, 2))
                    eth_bar = np.dot(wts, eth_nbrs)
                    z_bar_local[i] = np.log(max(eth_bar, 1e-30)) / theta
                else:
                    z_bar_local[i] = z[i]

            eth = np.exp(np.clip(theta * z, -100, 2))
            drift = (BETA / theta) * (1.0 - eth)
            coupling = BETA * (z_bar_local - z)

        dz = (drift + coupling) * DT + noise_scale * rng.standard_normal(N)
        z += dz
        M_traj[t_idx + 1] = np.exp(np.mean(z))

    return M_traj


def simulate_mean_field(degrees, rng, theta=0.0):
    """Simulate with mean-field approximation (for comparison)."""
    k = degrees.astype(np.float64)
    weights = k / k.sum()

    z = rng.normal(loc=-4.6, scale=0.3, size=N)
    steps = int(T_MAX / DT)
    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.exp(np.mean(z))
    noise_scale = np.sqrt(SIGMA * DT)
    eps = 1e-8

    for t_idx in range(steps):
        if theta < eps:
            drift = -BETA * z
            z_bar = np.sum(weights * z)
            coupling = BETA * (z_bar - z)
        else:
            eth = np.exp(np.clip(theta * z, -100, 2))
            drift = (BETA / theta) * (1.0 - eth)
            eth_bar = np.sum(weights * eth)
            pm_log = np.log(max(eth_bar, 1e-30)) / theta
            coupling = BETA * (pm_log - z)

        dz = (drift + coupling) * DT + noise_scale * rng.standard_normal(N)
        z += dz
        M_traj[t_idx + 1] = np.exp(np.mean(z))

    return M_traj

# ─── Theoretical curves ──────────────────────────────────────────────────────

def gompertz_gamma(M):
    return -np.log(np.clip(M, 1e-15, None))

def logistic_gamma(M):
    return 1 - M

# ─── Run simulations ─────────────────────────────────────────────────────────

print(f"Running explicit network simulations with N = {N:,} nodes...")
print("(This involves per-node neighbor lookups — slower than mean-field)\n")

results_explicit = {}
results_mf = {}

for name, deg_fn in NETWORKS.items():
    print(f"  {name}:")

    # Build network
    rng1 = np.random.default_rng(SEED)
    degrees = deg_fn(N, rng1)
    print(f"    Degrees: mean={degrees.mean():.1f}, max={degrees.max()}, min={degrees.min()}")

    neighbors = build_configuration_model(degrees, np.random.default_rng(SEED + 1))
    sparse_W = build_row_normalized_weights(neighbors)

    # Explicit network simulation
    rng2 = np.random.default_rng(SEED + 2)
    print(f"    Running explicit network...")
    M_exp = simulate_explicit(sparse_W, rng2, theta=0.0)
    results_explicit[name] = M_exp
    print(f"      M(0)={M_exp[0]:.4e}, M(end)={M_exp[-1]:.4f}")

    # Mean-field simulation (same initial RNG for fair comparison)
    rng3 = np.random.default_rng(SEED + 2)
    print(f"    Running mean-field...")
    M_mf = simulate_mean_field(degrees, rng3, theta=0.0)
    results_mf[name] = M_mf
    print(f"      M(0)={M_mf[0]:.4e}, M(end)={M_mf[-1]:.4f}")

# ─── Plotting ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "legend.fontsize": 7.5,
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

# ── Left panel: Relative growth rate ──

ax1.plot(M_th, gompertz_gamma(M_th), color="black", lw=2.2, ls=":",
         label="Gompertz (theory)", zorder=10)
ax1.plot(M_th, logistic_gamma(M_th), color="0.45", lw=1.6, ls="--",
         label="Logistic (theory)", zorder=9)

colors_exp = ["#c0392b", "#2980b9", "#27ae60", "#f39c12"]
colors_mf = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]
markers_exp = ["o", "s", "^", "D"]
markers_mf = ["x", "+", "1", "2"]

for idx, name in enumerate(NETWORKS.keys()):
    for results, col, mk, suffix, lw, ms in [
        (results_explicit, colors_exp[idx], markers_exp[idx], " (explicit)", 1.4, 4.5),
        (results_mf, colors_mf[idx], markers_mf[idx], " (mean-field)", 0.9, 6),
    ]:
        M = results[name]
        M_norm = (M - M.min()) / (M.max() - M.min())
        M_norm = np.clip(M_norm, 1e-6, 1 - 1e-6)
        dM = np.gradient(M_norm, DT)
        gamma = dM / M_norm / BETA
        step = max(1, len(M_norm) // 35)
        M_sub = M_norm[::step]
        g_sub = gamma[::step]
        mask = (M_sub > 0.01) & (M_sub < 0.98) & (g_sub > -0.5) & (g_sub < 8)
        ax1.plot(M_sub[mask], g_sub[mask],
                 color=col, marker=mk, markersize=ms,
                 lw=lw, alpha=0.85, markeredgewidth=0.5,
                 markeredgecolor="white" if mk in ["o","s","^","D"] else col,
                 label=f"{name}{suffix}", zorder=7)

ax1.set_xlabel("Abundance $M(t)$ [normalized]")
ax1.set_ylabel(r"Relative Growth Rate $\gamma / \beta$ [normalized]")
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.3, 5.5)
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.tick_params(which="both", direction="in", top=True, right=True)
ax1.legend(loc="upper right", ncol=2, fontsize=6.5)
ax1.set_title("(a) Relative growth rate", fontsize=11, pad=8)

# ── Right panel: Double-log ──

t_arr = np.linspace(0, T_MAX, int(T_MAX / DT) + 1)
t_theory = np.linspace(0, T_MAX, 300)

# Theoretical line from first explicit result
M_ref = list(results_explicit.values())[0]
M_ref_norm = (M_ref - M_ref.min()) / (M_ref.max() - M_ref.min())
M_ref_norm = np.clip(M_ref_norm, 1e-6, 1 - 1e-6)
with np.errstate(invalid="ignore", divide="ignore"):
    dlog_ref = np.log(-np.log(M_ref_norm))
valid = np.isfinite(dlog_ref)
from numpy.polynomial.polynomial import polyfit
t_v = t_arr[valid]
d_v = dlog_ref[valid]
mid = (t_v > 2) & (t_v < T_MAX - 5)
if mid.sum() > 10:
    c0, c1 = polyfit(t_v[mid], d_v[mid], 1)
    ax2.plot(t_theory, c0 + c1 * t_theory, color="black", lw=2.2, ls=":",
             label=f"Gompertz fit (slope={c1:.3f})", zorder=10)

for idx, name in enumerate(NETWORKS.keys()):
    for results, col, mk, suffix, lw, ms in [
        (results_explicit, colors_exp[idx], markers_exp[idx], " (explicit)", 1.2, 4),
        (results_mf, colors_mf[idx], markers_mf[idx], " (mean-field)", 0.8, 5),
    ]:
        M = results[name]
        M_norm = (M - M.min()) / (M.max() - M.min())
        M_norm = np.clip(M_norm, 1e-6, 1 - 1e-6)
        with np.errstate(invalid="ignore", divide="ignore"):
            dlog = np.log(-np.log(M_norm))
        valid = np.isfinite(dlog)
        t_valid = t_arr[valid]
        dlog_valid = dlog[valid]
        step = max(1, len(t_valid) // 35)
        ax2.plot(t_valid[::step], dlog_valid[::step],
                 color=col, marker=mk, markersize=ms,
                 lw=lw, alpha=0.85, markeredgewidth=0.4,
                 markeredgecolor="white" if mk in ["o","s","^","D"] else col,
                 label=f"{name}{suffix}", zorder=7)

ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\ln(\ln(1/M(t)))$")
ax2.set_xlim(-1, T_MAX + 1)
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.tick_params(which="both", direction="in", top=True, right=True)
ax2.legend(loc="upper right", ncol=2, fontsize=6.5)
ax2.set_title("(b) Double-log transform", fontsize=11, pad=8)

fig.suptitle(
    "Explicit network vs mean-field: Gompertz growth ($\\theta=0$)"
    f",  $N = ${N:,},  "
    r"$\beta=$" + f"{BETA},  " + r"$\sigma=$" + f"{SIGMA}",
    fontsize=11, y=1.02
)

plt.tight_layout()
plt.savefig("../figures/gompertz_explicit_vs_meanfield.png")
plt.savefig("../figures/gompertz_explicit_vs_meanfield.pdf")
print("\nPlots saved.")
