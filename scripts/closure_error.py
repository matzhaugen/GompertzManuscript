"""
Closure Error Comparison: Gompertz vs Logistic
================================================
Measures the RELATIVE macroscopic closure error for both models
on explicit (non-mean-field) networks.

Relative Gompertz closure error:  |ε_G / (β z̄)|
Relative Logistic closure error:  |ε_L / (β x̄(1-x̄))|

This normalizes by the theoretical growth rate, giving a dimensionless
measure of how much the actual macroscopic dynamics deviate from
the closed-form equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

# ─── Parameters ───────────────────────────────────────────────────────────────

N = 10_000
BETA = 0.25
SIGMA = 0.0005
DT = 0.05
T_MAX = 50.0
SEED = 42

# ─── Network Construction ─────────────────────────────────────────────────────

def build_configuration_model(degrees, rng):
    degrees = degrees.copy()
    if degrees.sum() % 2 == 1:
        degrees[rng.integers(len(degrees))] += 1
    stubs = []
    for i, k in enumerate(degrees):
        stubs.extend([i] * k)
    stubs = np.array(stubs)
    rng.shuffle(stubs)
    neighbors = [[] for _ in range(N)]
    for idx in range(0, len(stubs) - 1, 2):
        u, v = stubs[idx], stubs[idx + 1]
        if u != v:
            neighbors[u].append(v)
            neighbors[v].append(u)
    return neighbors


def build_row_normalized_weights(neighbors):
    sparse_W = []
    for i in range(len(neighbors)):
        nbrs = np.array(neighbors[i], dtype=np.int32)
        if len(nbrs) == 0:
            sparse_W.append((np.array([], dtype=np.int32), np.array([])))
        else:
            unique_nbrs, counts = np.unique(nbrs, return_counts=True)
            weights = counts.astype(np.float64)
            weights /= weights.sum()
            sparse_W.append((unique_nbrs, weights))
    return sparse_W


# ─── Degree Generators ────────────────────────────────────────────────────────

def degree_scale_free(N, rng):
    u = rng.random(N)
    return np.maximum(2, np.floor(2 * (1 - u) ** (-0.5)).astype(int))

def degree_erdos_renyi(N, rng):
    return np.maximum(2, rng.poisson(6, N))

# ─── Gompertz simulation ─────────────────────────────────────────────────────

def simulate_gompertz(sparse_W, rng):
    z = rng.normal(loc=-2.3, scale=0.5, size=N)  # start at x ~ 0.1
    steps = int(T_MAX / DT)
    noise_scale = np.sqrt(SIGMA * DT)

    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.exp(np.mean(z))
    abs_error = np.zeros(steps)
    rel_error = np.zeros(steps)

    for t_idx in range(steps):
        z_bar_local = np.zeros(N)
        for i in range(N):
            nbrs, wts = sparse_W[i]
            if len(nbrs) > 0:
                z_bar_local[i] = np.dot(wts, z[nbrs])
            else:
                z_bar_local[i] = z[i]

        drift = -BETA * z
        coupling = BETA * (z_bar_local - z)
        tendency = drift + coupling

        z_bar = np.mean(z)
        dz_bar_dt = np.mean(tendency)
        theory = -BETA * z_bar

        abs_error[t_idx] = abs(dz_bar_dt - theory)
        if abs(theory) > 1e-10:
            rel_error[t_idx] = abs(abs_error[t_idx] / theory)
        else:
            rel_error[t_idx] = 0.0

        dz = tendency * DT + noise_scale * rng.standard_normal(N)
        z += dz
        M_traj[t_idx + 1] = np.exp(np.mean(z))

    return M_traj, abs_error, rel_error


# ─── Logistic simulation ─────────────────────────────────────────────────────

def simulate_logistic(sparse_W, rng):
    z0 = rng.normal(loc=-2.3, scale=0.5, size=N)
    x = np.exp(z0)
    x = np.clip(x, 1e-10, 1 - 1e-10)

    steps = int(T_MAX / DT)
    noise_scale = np.sqrt(SIGMA * DT)

    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.mean(x)
    abs_error = np.zeros(steps)
    rel_error = np.zeros(steps)

    for t_idx in range(steps):
        x_bar_local = np.zeros(N)
        for i in range(N):
            nbrs, wts = sparse_W[i]
            if len(nbrs) > 0:
                x_bar_local[i] = np.dot(wts, x[nbrs])
            else:
                x_bar_local[i] = x[i]

        tendency = BETA * (1.0 - x) * x_bar_local

        x_bar = np.mean(x)
        dx_bar_dt = np.mean(tendency)
        theory = BETA * x_bar * (1.0 - x_bar)

        abs_error[t_idx] = abs(dx_bar_dt - theory)
        if abs(theory) > 1e-10:
            rel_error[t_idx] = abs(abs_error[t_idx] / theory)
        else:
            rel_error[t_idx] = 0.0

        dx = tendency * DT + noise_scale * rng.standard_normal(N)
        x += dx
        x = np.clip(x, 1e-10, 2.0)
        M_traj[t_idx + 1] = np.mean(x)

    return M_traj, abs_error, rel_error


# ─── Run ──────────────────────────────────────────────────────────────────────

networks = {
    "Scale-Free": degree_scale_free,
    r"Erdős-Rényi": degree_erdos_renyi,
}

print(f"Running closure error analysis with N = {N:,} nodes...\n")

results = {}
for name, deg_fn in networks.items():
    print(f"  {name}:")
    rng1 = np.random.default_rng(SEED)
    degrees = deg_fn(N, rng1)
    print(f"    Degrees: mean={degrees.mean():.1f}, max={degrees.max()}")

    neighbors = build_configuration_model(degrees, np.random.default_rng(SEED + 1))
    sparse_W = build_row_normalized_weights(neighbors)

    rng_g = np.random.default_rng(SEED + 2)
    print(f"    Gompertz...")
    M_g, abs_g, rel_g = simulate_gompertz(sparse_W, rng_g)

    rng_l = np.random.default_rng(SEED + 2)
    print(f"    Logistic...")
    M_l, abs_l, rel_l = simulate_logistic(sparse_W, rng_l)

    results[name] = {
        "M_gompertz": M_g, "abs_gompertz": abs_g, "rel_gompertz": rel_g,
        "M_logistic": M_l, "abs_logistic": abs_l, "rel_logistic": rel_l,
    }

    # Report peak relative error during active growth (M in [0.1, 0.9])
    M_g_norm = (M_g - M_g.min()) / (M_g.max() - M_g.min())
    M_l_norm = (M_l - M_l.min()) / (M_l.max() - M_l.min())
    active_g = (M_g_norm[:-1] > 0.1) & (M_g_norm[:-1] < 0.9)
    active_l = (M_l_norm[:-1] > 0.1) & (M_l_norm[:-1] < 0.9)
    peak_g = np.max(rel_g[active_g]) if active_g.any() else 0
    peak_l = np.max(rel_l[active_l]) if active_l.any() else 0
    print(f"    Peak relative error (active phase):")
    print(f"      Gompertz: {peak_g:.4%}")
    print(f"      Logistic: {peak_l:.4%}")
    print(f"      Ratio (L/G): {peak_l/max(peak_g, 1e-15):.1f}x\n")

# ─── Plotting ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

t_arr = np.linspace(0, T_MAX, int(T_MAX / DT))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for col_idx, (name, res) in enumerate(results.items()):
    ax = axes[col_idx]
    # ax = axes[0,col_idx]
    # ax2 = axes[1, col_idx]

    # Normalize abundance for both models
    M_g = res["M_gompertz"]
    M_l = res["M_logistic"]
    M_g_norm = (M_g - M_g.min()) / (M_g.max() - M_g.min())
    M_l_norm = (M_l - M_l.min()) / (M_l.max() - M_l.min())
    M_g_plot = np.clip(M_g_norm[:-1], 1e-6, 1 - 1e-6)
    M_l_plot = np.clip(M_l_norm[:-1], 1e-6, 1 - 1e-6)

    rel_g = res["rel_gompertz"]
    rel_l = res["rel_logistic"]
    abs_g = res["abs_gompertz"]
    abs_l = res["abs_logistic"]

    # Smooth for clarity
    def smooth(arr, w=25):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    rel_g_smooth = smooth(rel_g)
    rel_l_smooth = smooth(rel_l)
    abs_g_smooth = smooth(abs_g)
    abs_l_smooth = smooth(abs_l)

    step = max(1, len(M_g_plot) // 100)

    ax.plot(M_g_plot[::step], rel_g_smooth[::step] * 100,
            color="#2980b9", marker="o", markersize=3, lw=1.3, alpha=0.85,
            markeredgewidth=0.3, markeredgecolor="white",
            label=r"Gompertz: $|\epsilon_G / \dot{\bar{z}}_{theory}|$")
    ax.plot(M_l_plot[::step], rel_l_smooth[::step] * 100,
            color="#c0392b", marker="s", markersize=3, lw=1.3, alpha=0.85,
            markeredgewidth=0.3, markeredgecolor="white",
            label=r"Logistic: $|\epsilon_L / \dot{\bar{x}}_{theory}|$")

    # ax2.plot(M_g_plot[::step], abs_g_smooth[::step] * 100,
    #         color="#2980b9", marker="o", markersize=3, lw=1.3, alpha=0.85,
    #         markeredgewidth=0.3, markeredgecolor="white",
    #         label=r"Gompertz: $|\epsilon_G|$")
    # ax2.plot(M_l_plot[::step], abs_l_smooth[::step] * 100,
    #         color="#c0392b", marker="s", markersize=3, lw=1.3, alpha=0.85,
    #         markeredgewidth=0.3, markeredgecolor="white",
    #         label=r"Logistic: $|\epsilon_L|$")
    ax.set_xlabel("Abundance $M(t)$ [normalized]")
    ax.set_ylabel("Relative closure error (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper right")
    label = "a" if col_idx == 0 else "b"
    ax.set_title(f"({label}) {name} network", fontsize=11, pad=8)
    # ax2.set_xlabel("Abundance $M(t)$ [normalized]")
    # ax2.set_ylabel("Absolute closure error (%)")
    # ax2.set_xlim(-0.02, 1.02)
    # ax2.set_ylim(1e-5, 1)
    # ax2.set_yscale("log")
    # ax2.tick_params(which="both", direction="in", top=True, right=True)
    # ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax2.legend(loc="upper right")

fig.suptitle(
    "Relative and absolute macroscopic closure error on explicit networks (no mean-field)"
    f"\n$N = ${N:,}, "
    r"$\beta=$" + f"{BETA}, " + r"$\sigma=$" + f"{SIGMA}",
    fontsize=11, y=1.04
)
plt.tight_layout()
plt.savefig("../figures/closure_error_comparison.png")
plt.savefig("../figures/closure_error_comparison.pdf")
print("Plots saved.")
