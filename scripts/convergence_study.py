"""
Time-step (Δt) convergence study for the microscopic Richards model
====================================================================
Reviewer request: show that the high-θ near-saturation deviations in Fig. 1
survive Δt → 0, separating the genuine structural effect (mean-variance
coupling, Eq. meanVarianceCoupling) from Euler–Maruyama discretization error.

Method
------
We measure the deviation of the simulated macroscopic trajectory from the
CLOSED-FORM Richards solution, as a function of Δt, for several θ and two
noise levels (σ = 0 deterministic, and σ = 0.01 the paper's value).

The closed form is exact: with y = X^θ the θ-logistic dX/dt = (β/θ)X(1−X^θ)
becomes the logistic ẏ = βy(1−y), so
    X_theory(t) = [ y0 / (y0 + (1−y0) e^{−βt}) ]^{1/θ},   y0 = X(0)^θ,
and X_theory(t) = exp(ln X0 · e^{−βt}) for θ → 0 (Gompertz).

Deviation metric (in the double-log domain the reviewer refers to):
    g(X) = ln(ln(1/X)),   D = RMS_t[ g(X_sim) − g(X_theory) ]
over the window where X_theory ∈ [0.1, 0.99] (active + near-saturation).

Interpretation
--------------
* θ = 0 is a CONTROL: Gompertz is exactly closed under aggregation, so its
  deviation must converge to ~0 as Δt → 0 — this validates that the scheme
  converges and that any residual at θ > 0 is not a generic bug.
* If D(θ>0) converges to a NONZERO floor as Δt → 0, that floor is the genuine
  structural mean-variance coupling; the part that shrinks with Δt is the
  Euler–Maruyama error.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sim_params import N, BETA, SIGMA, IC_STD, IC_MEAN_Z_SWEEP

# ─── Model (mean-field Richards, variable Δt and σ) ───────────────────────────

def degree_log_normal(n, rng):
    z = rng.standard_normal(n)
    return np.maximum(2, np.round(np.exp(1.5 + 0.8 * z)).astype(int))

def compute_tendency(z, theta, weights, eps=1e-8):
    if theta < eps:
        z_bar = np.sum(weights * z)
        return -BETA * z + BETA * (z_bar - z)
    eth = np.exp(np.clip(theta * z, -100, 2))
    drift = (BETA / theta) * (1.0 - eth)
    eth_bar = np.sum(weights * eth)
    pm_log = np.log(np.clip(eth_bar, 1e-30, None)) / theta
    return drift + BETA * (pm_log - z)

def simulate(theta, dt, sigma, seed, t_max):
    r = np.random.default_rng(seed)
    degrees = degree_log_normal(N, r)
    weights = degrees.astype(np.float64) / degrees.sum()
    z = r.normal(IC_MEAN_Z_SWEEP, IC_STD, N)
    steps = int(t_max / dt)
    noise_scale = sigma * np.sqrt(dt)
    t = np.arange(steps + 1) * dt
    X = np.empty(steps + 1)
    X[0] = np.exp(z.mean())
    for i in range(steps):
        z = z + compute_tendency(z, theta, weights) * dt
        if sigma > 0:
            z = z + noise_scale * r.standard_normal(N)
        X[i + 1] = np.exp(z.mean())
    return t, X

def richards_theory(X0, theta, t, eps=1e-8):
    """Closed-form macroscopic Richards / Gompertz trajectory."""
    if theta < eps:
        return np.exp(np.log(X0) * np.exp(-BETA * t))
    y0 = X0 ** theta
    y = y0 / (y0 + (1.0 - y0) * np.exp(-BETA * t))
    return y ** (1.0 / theta)

def g(X):
    return np.log(-np.log(np.clip(X, 1e-12, 1 - 1e-12)))

def deviation(theta, dt, sigma, seed, t_max):
    t, X = simulate(theta, dt, sigma, seed, t_max)
    Xth = richards_theory(X[0], theta, t)
    window = (Xth >= 0.1) & (Xth <= 0.99)
    if window.sum() < 5:
        return np.nan
    return np.sqrt(np.mean((g(X[window]) - g(Xth[window])) ** 2))

# ─── Run the study ────────────────────────────────────────────────────────────

DTS = [0.02, 0.01, 0.005, 0.0025, 0.00125]
THETAS = [0.0, 0.5, 1.0]
T_MAX = {0.0: 50.0, 0.5: 80.0, 1.0: 110.0}
SEEDS_DET = 3     # IC variability for the deterministic case
SEEDS_STO = 6     # ensemble for the stochastic case
BASE = 42

CACHE = os.path.join(os.path.dirname(__file__), "convergence_results.npy")


def compute():
    results = {}   # (theta, sigma) -> list of (mean D, std D) over DTS
    for sigma, nseed in [(0.0, SEEDS_DET), (SIGMA, SEEDS_STO)]:
        for theta in THETAS:
            row = []
            for dt in DTS:
                ds = [deviation(theta, dt, sigma, BASE + 100 * s, T_MAX[theta])
                      for s in range(nseed)]
                row.append((np.nanmean(ds), np.nanstd(ds)))
            results[(theta, sigma)] = row
            line = "  ".join(f"dt={dt}: {m:.4f}±{s:.4f}"
                             for dt, (m, s) in zip(DTS, row))
            print(f"theta={theta}, sigma={sigma}:  {line}")
    np.save(CACHE, results, allow_pickle=True)
    return results


def plot(results):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
        "axes.labelsize": 11, "legend.fontsize": 8.5, "legend.frameon": True,
        "legend.framealpha": 0.92, "legend.edgecolor": "0.8",
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    dts = np.array(DTS)
    colors = {0.0: "#2980b9", 0.5: "#f39c12", 1.0: "#c0392b"}
    markers = {0.0: "o", 0.5: "s", 1.0: "^"}
    labels = {0.0: r"$\theta \to 0$ (Gompertz)", 0.5: r"$\theta = 0.5$",
              1.0: r"$\theta = 1$ (Logistic)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, sigma, title in [(ax1, 0.0, r"(a) Deterministic ($\sigma = 0$)"),
                             (ax2, SIGMA, rf"(b) Stochastic ($\sigma = {SIGMA}$)")]:
        for theta in THETAS:
            m = np.array([v[0] for v in results[(theta, sigma)]])
            s = np.array([v[1] for v in results[(theta, sigma)]])
            ax.errorbar(dts, m, yerr=s, color=colors[theta], marker=markers[theta],
                        markersize=5, lw=1.4, capsize=2.5, alpha=0.9,
                        markeredgewidth=0.4, markeredgecolor="white",
                        label=labels[theta], zorder=5)
        if sigma == 0.0:
            # first-order Euler reference (slope 1) anchored to the Gompertz point
            m0 = results[(0.0, 0.0)][0][0]
            ax.plot(dts, m0 * (dts / dts[0]), color="0.5", ls="--", lw=1.1,
                    zorder=2, label=r"$\propto \Delta t$ (Euler)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Time step $\Delta t$")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(loc="lower right")
        ax.set_title(title, fontsize=11, pad=8)
    ax1.set_ylabel(r"Deviation from Richards theory  $D$")
    fig.suptitle(
        r"Time-step convergence of the macroscopic deviation from theory"
        f"\n$N = ${N:,}, " + r"$\beta = $" + f"{BETA}; "
        + "Gompertz ($\\theta\\to0$) converges as $\\Delta t$; "
        + "$\\theta>0$ retains a structural floor",
        fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig("../figures/convergence_deltat.png")
    plt.savefig("../figures/convergence_deltat.pdf")
    print("Plot saved.")


if __name__ == "__main__":
    if os.path.exists(CACHE):
        print(f"Loading cached results from {CACHE}")
        results = np.load(CACHE, allow_pickle=True).item()
    else:
        results = compute()
    plot(results)
