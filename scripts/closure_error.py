"""
Closure Error Comparison: Gompertz vs Logistic (+ theta=1 control)
==================================================================
Measures the RELATIVE macroscopic closure error for three models
on explicit (non-mean-field) networks, each under its OWN natural
aggregation operator:

  Gompertz (theta -> 0 of Eq. newMicro): geometric mean, X = exp(mean z)
      Relative error:  |eps_G / (beta z_bar)|
  Transmission logistic (Eq. microLogisticEpidemic): arithmetic mean, X = mean x
      Relative error:  |eps_L / (beta x_bar (1 - x_bar))|
  theta=1 endpoint of Eq. newMicro: odds/logit mean, X = sigmoid(mean logit x)
      Relative error:  |d/dt mean(logit x) - beta| / beta

This normalizes by the theoretical growth rate, giving a dimensionless
measure of how much the actual macroscopic dynamics deviate from the
closed-form equation.

Why the theta=1 endpoint is a distinct, fair control (reviewer point 2)
-----------------------------------------------------------------------
The transmission logistic (Eq. microLogisticEpidemic) is NOT the theta=1
endpoint of the Richards-family model (Eq. newMicro); the paper says so
explicitly. The transmission logistic is purely bilinear (growth driven
solely by neighbor transmission, no intrinsic self-term), which is the
worst case for closure. Pitting Gompertz only against that model risks
stacking the deck for a "within-family uniqueness" claim.

To make the uniqueness argument honestly we therefore ALSO measure the
theta=1 endpoint of Eq. newMicro,

    dz_i = [ beta (1 - e^{z_i}) + beta (zbar_{1,i} - z_i) ] dt,
    zbar_{1,i} = ln( sum_j w_ij e^{z_j} )  (order-1 power mean of neighbors),

which combines an intrinsic logistic self-term with a multiplicative
synchronizing coupling. Its natural aggregation operator is the odds/logit
mean: logistic growth is exactly linear in logit(x) (d/dt logit(x) = beta),
just as Gompertz growth is exactly linear in ln(x). Measuring closure under
this natural mean is the like-for-like analog of measuring Gompertz closure
under the geometric mean, so any residual error is due purely to the
network coupling term (self-correcting as nodes synchronize), NOT to using
a mismatched aggregator.

DETERMINISTIC measurement (sigma = 0)
-------------------------------------
Closure under aggregation is a property of the DETERMINISTIC vector field:
it asks whether d<z>/dt equals the closed-form macroscopic RHS. Measuring it
with stochastic forcing would conflate closure with noise fluctuations, and
near saturation (where the theoretical rate -> 0) the relative error is
dominated by whichever model's noise structure injects more dispersion. That
is precisely the additive-vs-multiplicative ambiguity a reviewer raised.

We therefore measure the closure error on the noise-free dynamics (sigma = 0),
which (a) is exactly what the paper's closure/coherence theory is about, and
(b) removes any dependence on the noise structure, so it is automatically
consistent with the microscopic model Eq. (newMicro) whose noise is additive
in the log-domain (multiplicative in abundance).

Uncertainty quantification still comes from an ENSEMBLE of N_ENSEMBLE
independent network + initial-condition realizations (both models share the
SAME initial condition within each realization); the figure shows the median
relative error with an interquartile (25-75th percentile) band, and the trajectory-averaged ratio is
reported as mean +/- std across realizations.

Set SIGMA_CLOSURE > 0 below only to reproduce the noise-sensitivity check
discussed in the text (where the stochastic relative error is noise-dominated
and the simple ratio is not meaningful).
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

# ─── Parameters (shared across all figure scripts, see sim_params.py) ─────────

from sim_params import (N, BETA, DT, T_MAX, SEED, IC_STD,
                        IC_MEAN_Z_CLOSURE, N_ENSEMBLE)

# Range of normalized abundance over which the closure error is shown and
# averaged. Because the measurement is deterministic (see below), the error is
# well behaved over the whole growth trajectory, so we use the full range
# 0 < X < 1 with no restriction to a special "active" window.
X_LO, X_HI = 0.0, 1.0

# Closure is a deterministic property -> primary result uses sigma = 0.
# (Set > 0, together with MULTIPLICATIVE_LOGISTIC, only for the noise check.)
SIGMA_CLOSURE = 0.0
MULTIPLICATIVE_LOGISTIC = True   # Eq. (newMicro)-consistent noise IF SIGMA_CLOSURE > 0

# ─── Network Construction (vectorized, scipy.sparse) ──────────────────────────

def build_row_normalized_W(degrees, rng):
    """Configuration-model random graph -> row-normalized CSR coupling matrix.

    w_ij = (multiplicity of edge i-j) / k_i, so rows sum to 1. Self-loops are
    discarded; isolated nodes get w_ii = 1 (zero coupling). Identical
    construction to explicit_network.py."""
    degrees = degrees.copy()
    if degrees.sum() % 2 == 1:
        degrees[rng.integers(len(degrees))] += 1
    stubs = np.repeat(np.arange(N), degrees)
    rng.shuffle(stubs)
    m = len(stubs) // 2
    u = stubs[0:2 * m:2]
    v = stubs[1:2 * m:2]
    keep = u != v                       # drop self-loops
    u, v = u[keep], v[keep]
    rows = np.concatenate([u, v])       # symmetrize
    cols = np.concatenate([v, u])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    A.sum_duplicates()                  # multi-edges -> integer weights
    deg = np.asarray(A.sum(axis=1)).ravel()
    isolated = deg == 0
    deg_safe = np.where(isolated, 1.0, deg)
    W = sp.diags(1.0 / deg_safe) @ A
    if isolated.any():                  # self-loop for isolated nodes
        W = W + sp.diags(isolated.astype(float))
    return W.tocsr()


# ─── Degree Generators ────────────────────────────────────────────────────────

def degree_scale_free(N, rng):
    u = rng.random(N)
    return np.maximum(2, np.floor(2 * (1 - u) ** (-0.5)).astype(int))

def degree_erdos_renyi(N, rng):
    return np.maximum(2, rng.poisson(6, N))

# ─── Gompertz simulation (additive noise in log-domain) ───────────────────────

def simulate_gompertz(W, rng):
    z = rng.normal(loc=IC_MEAN_Z_CLOSURE, scale=IC_STD, size=N)  # shared IC
    steps = int(T_MAX / DT)
    noise_scale = SIGMA_CLOSURE * np.sqrt(DT)   # 0 for the deterministic diagnostic

    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.exp(np.mean(z))
    rel_error = np.zeros(steps)

    for t_idx in range(steps):
        z_bar_local = W @ z
        tendency = -BETA * z + BETA * (z_bar_local - z)

        z_bar = np.mean(z)
        dz_bar_dt = np.mean(tendency)
        theory = -BETA * z_bar

        abs_error = abs(dz_bar_dt - theory)
        rel_error[t_idx] = abs_error / abs(theory) if abs(theory) > 1e-10 else 0.0

        z = z + tendency * DT
        if noise_scale > 0:                     # additive sigma dB in the log-domain
            z = z + noise_scale * rng.standard_normal(N)
        M_traj[t_idx + 1] = np.exp(np.mean(z))

    return M_traj, rel_error


# ─── Logistic simulation (multiplicative noise sigma x dB) ────────────────────

def simulate_logistic(W, rng):
    z0 = rng.normal(loc=IC_MEAN_Z_CLOSURE, scale=IC_STD, size=N)  # SAME IC as Gompertz
    x = np.clip(np.exp(z0), 1e-10, 1 - 1e-10)

    steps = int(T_MAX / DT)
    noise_scale = SIGMA_CLOSURE * np.sqrt(DT)
    sqrt_dt = np.sqrt(DT)

    M_traj = np.zeros(steps + 1)
    M_traj[0] = np.mean(x)
    rel_error = np.zeros(steps)

    for t_idx in range(steps):
        x_bar_local = W @ x
        tendency = BETA * (1.0 - x) * x_bar_local

        x_bar = np.mean(x)
        dx_bar_dt = np.mean(tendency)
        theory = BETA * x_bar * (1.0 - x_bar)

        abs_error = abs(dx_bar_dt - theory)
        rel_error[t_idx] = abs_error / abs(theory) if abs(theory) > 1e-10 else 0.0

        x = x + tendency * DT
        if noise_scale > 0:
            # Eq. (newMicro)-consistent noise is multiplicative in abundance
            # (additive in the log-domain); additive-in-x kept only as a
            # non-consistent alternative for the sensitivity discussion.
            if MULTIPLICATIVE_LOGISTIC:
                x = x + SIGMA_CLOSURE * x * sqrt_dt * rng.standard_normal(N)
            else:
                x = x + noise_scale * rng.standard_normal(N)
        x = np.clip(x, 1e-10, 2.0)
        M_traj[t_idx + 1] = np.mean(x)

    return M_traj, rel_error


# ─── theta=1 endpoint of Eq. newMicro (odds/logit natural mean) ───────────────

def _odds_macro(z):
    """Macroscopic abundance under the odds/logit mean: aggregate logit(x) then
    map back. This is the natural aggregator for logistic-family growth, the
    exact analog of the geometric mean for Gompertz."""
    x = np.clip(np.exp(z), 1e-12, 1.0 - 1e-12)
    lbar = np.mean(np.log(x / (1.0 - x)))
    return 1.0 / (1.0 + np.exp(-lbar))

def simulate_theta1(W, rng):
    """theta=1 endpoint of Eq. newMicro, integrated in the log-domain and
    measured under its own natural (odds/logit) mean.

    Dynamics (deterministic, sigma = 0 like the other two models):
        dz_i = [ beta (1 - e^{z_i}) + beta (zbar_{1,i} - z_i) ] dt,
        zbar_{1,i} = ln( sum_j w_ij e^{z_j} ).

    Closure test: logistic growth is exactly linear in the logit, so the closed
    macroscopic rate is the CONSTANT beta, d/dt mean(logit x) = beta. The
    relative closure error is |d/dt mean(logit x) - beta| / beta. Per node,
        d logit(x_i)/dt = (dz_i/dt) / (1 - x_i),
    so the whole error comes from the network coupling term (it vanishes as
    nodes synchronize, exactly like the Gompertz column-sum error)."""
    z = rng.normal(loc=IC_MEAN_Z_CLOSURE, scale=IC_STD, size=N)  # SAME IC family
    steps = int(T_MAX / DT)

    M_traj = np.zeros(steps + 1)
    M_traj[0] = _odds_macro(z)
    rel_error = np.zeros(steps)

    for t_idx in range(steps):
        x = np.exp(z)
        zbar_1 = np.log(W @ x)                       # order-1 power mean of neighbors
        tendency = BETA * (1.0 - x) + BETA * (zbar_1 - z)

        one_minus_x = np.clip(1.0 - x, 1e-12, None)  # guard occasional overshoot
        dlbar_dt = np.mean(tendency / one_minus_x)   # d/dt mean(logit x)
        theory = BETA                                # closed logit rate is constant beta
        rel_error[t_idx] = abs(dlbar_dt - theory) / abs(theory)

        z = z + tendency * DT
        M_traj[t_idx + 1] = _odds_macro(z)

    return M_traj, rel_error


# ─── Ensemble helpers ─────────────────────────────────────────────────────────

def smooth(arr, w=25):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")

def trajectory_mean_error(M, rel):
    """Mean relative closure error over the growth trajectory (normalized
    abundance X in [X_LO, X_HI]). This is the per-model summary error; the
    reported L/G ratio is the ratio of these two means."""
    M_norm = (M - M.min()) / (M.max() - M.min())
    window = (M_norm[:-1] > X_LO) & (M_norm[:-1] < X_HI)
    return np.mean(rel[window]) if window.any() else 0.0

def rel_vs_abundance(M, rel, grid):
    """Interpolate a single realization's relative error onto a common
    normalized-abundance grid (abundance increases monotonically on average;
    a running max enforces monotonicity for the interpolation)."""
    M_norm = (M[:-1] - M.min()) / (M.max() - M.min())
    order = np.argsort(M_norm)
    xp = np.maximum.accumulate(M_norm[order])
    fp = smooth(rel)[order]
    return np.interp(grid, xp, fp, left=np.nan, right=np.nan)

# ─── Run ensemble ─────────────────────────────────────────────────────────────

networks = {
    "Scale-Free": degree_scale_free,
    r"Erdős-Rényi": degree_erdos_renyi,
}

grid = np.linspace(X_LO, X_HI, 150)

print(f"Running closure-error ensemble: N = {N:,} nodes, "
      f"{N_ENSEMBLE} realizations per network...\n")

results = {}
for name, deg_fn in networks.items():
    print(f"  {name}:")
    curves_g, curves_l, curves_t1 = [], [], []
    err_g, err_l, err_t1, ratios, ratios_t1 = [], [], [], [], []
    for s in range(N_ENSEMBLE):
        seed = SEED + 100 * s
        degrees = deg_fn(N, np.random.default_rng(seed))
        W = build_row_normalized_W(degrees, np.random.default_rng(seed + 1))

        M_g, rel_g = simulate_gompertz(W, np.random.default_rng(seed + 2))
        M_l, rel_l = simulate_logistic(W, np.random.default_rng(seed + 2))
        M_t1, rel_t1 = simulate_theta1(W, np.random.default_rng(seed + 2))

        curves_g.append(rel_vs_abundance(M_g, rel_g, grid))
        curves_l.append(rel_vs_abundance(M_l, rel_l, grid))
        curves_t1.append(rel_vs_abundance(M_t1, rel_t1, grid))

        # per-model summary = mean relative closure error over the trajectory
        eg, el = trajectory_mean_error(M_g, rel_g), trajectory_mean_error(M_l, rel_l)
        et1 = trajectory_mean_error(M_t1, rel_t1)
        err_g.append(eg); err_l.append(el); err_t1.append(et1)
        ratios.append(el / max(eg, 1e-15))
        ratios_t1.append(et1 / max(eg, 1e-15))

    curves_g = np.array(curves_g); curves_l = np.array(curves_l)
    curves_t1 = np.array(curves_t1)
    ratios = np.array(ratios); ratios_t1 = np.array(ratios_t1)
    results[name] = {
        # median line with interquartile (25-75th percentile) band (robust on a log axis)
        "g_med": np.nanmedian(curves_g, axis=0),
        "g_lo": np.nanpercentile(curves_g, 25, axis=0),
        "g_hi": np.nanpercentile(curves_g, 75, axis=0),
        "l_med": np.nanmedian(curves_l, axis=0),
        "l_lo": np.nanpercentile(curves_l, 25, axis=0),
        "l_hi": np.nanpercentile(curves_l, 75, axis=0),
        "t1_med": np.nanmedian(curves_t1, axis=0),
        "t1_lo": np.nanpercentile(curves_t1, 25, axis=0),
        "t1_hi": np.nanpercentile(curves_t1, 75, axis=0),
        "ratio_median": np.median(ratios),
        "ratio_q25": np.percentile(ratios, 25),
        "ratio_q75": np.percentile(ratios, 75),
        "ratio_frac_gt1": np.mean(ratios > 1.0),
        "ratio_t1_median": np.median(ratios_t1),
        "ratio_t1_q25": np.percentile(ratios_t1, 25),
        "ratio_t1_q75": np.percentile(ratios_t1, 75),
    }
    print(f"    Gompertz     trajectory mean rel error: {np.mean(err_g):.4%} "
          f"+/- {np.std(err_g):.4%}")
    print(f"    Logistic     trajectory mean rel error: {np.mean(err_l):.4%} "
          f"+/- {np.std(err_l):.4%}")
    print(f"    theta=1 (odds) trajectory mean rel error: {np.mean(err_t1):.4%} "
          f"+/- {np.std(err_t1):.4%}")
    print(f"    Ratio L/G      of trajectory mean errors: median {np.median(ratios):.1f} "
          f"[IQR {np.percentile(ratios,25):.1f}-{np.percentile(ratios,75):.1f}], "
          f"range {ratios.min():.1f}-{ratios.max():.1f}")
    print(f"    Ratio (t1)/G   of trajectory mean errors: median {np.median(ratios_t1):.1f} "
          f"[IQR {np.percentile(ratios_t1,25):.1f}-{np.percentile(ratios_t1,75):.1f}], "
          f"range {ratios_t1.min():.1f}-{ratios_t1.max():.1f}\n")

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

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for col_idx, (name, res) in enumerate(results.items()):
    ax = axes[col_idx]

    ax.plot(grid, res["g_med"] * 100, color="#2980b9", lw=1.6,
            label=r"Gompertz: $|\epsilon_G / \dot{\bar{z}}_{theory}|$")
    ax.fill_between(grid, res["g_lo"] * 100, res["g_hi"] * 100,
                    color="#2980b9", alpha=0.20, linewidth=0)
    ax.plot(grid, res["l_med"] * 100, color="#c0392b", lw=1.6,
            label=r"Transmission logistic: $|\epsilon_L / \dot{\bar{x}}_{theory}|$")
    ax.fill_between(grid, res["l_lo"] * 100, res["l_hi"] * 100,
                    color="#c0392b", alpha=0.20, linewidth=0)
    # theta=1 endpoint control curve omitted for now (may re-enable later):
    # ax.plot(grid, res["t1_med"] * 100, color="#27ae60", lw=1.6, ls="--",
    #         label=r"$\theta{=}1$ endpoint (odds mean): $|\dot{\bar{\ell}}/\beta - 1|$")
    # ax.fill_between(grid, res["t1_lo"] * 100, res["t1_hi"] * 100,
    #                 color="#27ae60", alpha=0.18, linewidth=0)

    ax.text(0.03, 0.04,
            rf"median $\bar{{\epsilon}}_L/\bar{{\epsilon}}_G = {res['ratio_median']:.1f}$"
            + f"\n[IQR {res['ratio_q25']:.1f}-{res['ratio_q75']:.1f}], $n={N_ENSEMBLE}$"
            # theta=1 control annotation omitted for now (may re-enable later):
            # + rf"$\;\;$ median $\bar{{\epsilon}}_{{\theta=1}}/\bar{{\epsilon}}_G = {res['ratio_t1_median']:.1f}$"
            # + f"\n[IQR {res['ratio_t1_q25']:.1f}-{res['ratio_t1_q75']:.1f}]"
            ,
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

    ax.set_xlabel("Abundance $X(t)$ [normalized]")
    ax.set_ylabel("Relative closure error (%)")
    ax.set_xlim(X_LO - 0.02, X_HI + 0.02)
    ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper right")
    label = "a" if col_idx == 0 else "b"
    ax.set_title(f"({label}) {name} network", fontsize=11, pad=8)

noise_lbl = (f"$\\sigma={SIGMA_CLOSURE}$ (stochastic check)" if SIGMA_CLOSURE > 0
             else "deterministic ($\\sigma=0$)")
fig.suptitle(
    "Ensemble macroscopic closure error on explicit networks (no mean-field)"
    f"\n$N = ${N:,}, "
    r"$\beta=$" + f"{BETA}, "
    + f"{N_ENSEMBLE} realizations (median, IQR band), {noise_lbl}",
    fontsize=11, y=1.04
)
plt.tight_layout()
plt.savefig("../figures/closure_error_comparison.png")
plt.savefig("../figures/closure_error_comparison.pdf")
print("Plots saved.")
