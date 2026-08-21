"""
Gumbel (Gompertz) vs Log-Normal: how close are they really?
============================================================

The normalized Gompertz curve is EXACTLY the Gumbel CDF:

    F(t) = exp(-exp(-(t - T)/s))

Sartwell's observation is that latent periods are log-normal.
This script asks: over what range of the log-normal shape parameter
sigma are the two actually close, and where do they part?

Comparisons made:
  (a) CDFs, location-scale matched (equal mean and variance)
  (b) The Gompertz double-log transform g(F) = ln(-ln F) vs t
  (c) Skewness as a function of log-normal sigma, vs the fixed
      Gumbel skewness (12*sqrt(6)*zeta(3)/pi^3 ~ 1.1395)
  (d) Maximum CDF discrepancy (Kolmogorov distance) vs sigma
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats, optimize, special
import warnings
warnings.filterwarnings("ignore")

# ─── Gumbel constants ────────────────────────────────────────────────────────

EULER_GAMMA = 0.5772156649015329
# Gumbel skewness = 12*sqrt(6)*zeta(3)/pi^3
GUMBEL_SKEW = 12 * np.sqrt(6) * special.zeta(3, 1) / np.pi**3

print(f"Gumbel skewness (fixed, no shape parameter): {GUMBEL_SKEW:.4f}\n")

# ─── Distribution helpers ────────────────────────────────────────────────────

def lognormal_moments(mu, s):
    """Mean, variance, skewness of log-normal with log-params (mu, s)."""
    mean = np.exp(mu + s**2 / 2)
    var = (np.exp(s**2) - 1) * np.exp(2 * mu + s**2)
    skew = (np.exp(s**2) + 2) * np.sqrt(np.exp(s**2) - 1)
    return mean, var, skew


def gumbel_moments(loc, scale):
    """Mean, variance, skewness of Gumbel(loc, scale)."""
    mean = loc + EULER_GAMMA * scale
    var = (np.pi**2 / 6) * scale**2
    return mean, var, GUMBEL_SKEW


def match_gumbel_to_lognormal(mu, s):
    """
    Find Gumbel (loc, scale) with the same mean and variance as the
    log-normal with log-params (mu, s). This is the fairest comparison:
    location and scale are matched, so any remaining difference is SHAPE.
    """
    ln_mean, ln_var, _ = lognormal_moments(mu, s)
    scale = np.sqrt(6 * ln_var) / np.pi
    loc = ln_mean - EULER_GAMMA * scale
    return loc, scale


def skewness_match_sigma():
    """Find the log-normal sigma whose skewness equals the Gumbel skewness."""
    f = lambda s: (np.exp(s**2) + 2) * np.sqrt(np.exp(s**2) - 1) - GUMBEL_SKEW
    return optimize.brentq(f, 1e-4, 2.0)


S_STAR = skewness_match_sigma()
print(f"Log-normal sigma matching Gumbel skewness: s* = {S_STAR:.4f}\n")

# ─── Kolmogorov distance between matched distributions ───────────────────────

def kolmogorov_distance(mu, s, n_grid=20000):
    """
    Max |F_lognormal(t) - F_gumbel(t)| for location/scale-matched pair.
    """
    loc, scale = match_gumbel_to_lognormal(mu, s)
    ln_mean, ln_var, _ = lognormal_moments(mu, s)
    lo = max(1e-9, ln_mean - 6 * np.sqrt(ln_var))
    hi = ln_mean + 10 * np.sqrt(ln_var)
    t = np.linspace(lo, hi, n_grid)
    F_ln = stats.lognorm.cdf(t, s, scale=np.exp(mu))
    F_gu = stats.gumbel_r.cdf(t, loc=loc, scale=scale)
    return np.max(np.abs(F_ln - F_gu))

# ─── Plot setup ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sigmas_show = [0.2, S_STAR, 0.8]
colors = ["#2980b9", "#27ae60", "#c0392b"]
labels_s = [rf"$s={sigmas_show[0]}$",
            rf"$s={S_STAR:.2f}$ (skew-matched)",
            rf"$s={sigmas_show[2]}$"]

MU = 0.0  # log-scale location; only sets units

# ── (a) CDF comparison ──
ax = axes[0, 0]
for s, col, lab in zip(sigmas_show, colors, labels_s):
    loc, scale = match_gumbel_to_lognormal(MU, s)
    ln_mean, ln_var, _ = lognormal_moments(MU, s)
    t = np.linspace(max(1e-6, ln_mean - 4*np.sqrt(ln_var)),
                    ln_mean + 6*np.sqrt(ln_var), 2000)
    F_ln = stats.lognorm.cdf(t, s, scale=np.exp(MU))
    F_gu = stats.gumbel_r.cdf(t, loc=loc, scale=scale)
    # Normalize t by lognormal mean so curves are comparable across s
    ax.plot(t / ln_mean, F_ln, color=col, lw=1.6, ls="-", label=f"LN, {lab}")
    ax.plot(t / ln_mean, F_gu, color=col, lw=1.4, ls="--", alpha=0.75)

ax.plot([], [], color="0.3", lw=1.6, ls="-", label="Log-normal (solid)")
ax.plot([], [], color="0.3", lw=1.4, ls="--", label="Gumbel/Gompertz (dashed)")
ax.set_xlabel(r"$t$ / mean")
ax.set_ylabel(r"$F(t)$")
ax.set_xlim(0, 3)
ax.set_ylim(-0.02, 1.02)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc="lower right", fontsize=7)
ax.set_title("(a) CDFs, mean- and variance-matched", fontsize=10.5, pad=8)

# ── (b) Gompertz double-log transform ──
ax = axes[0, 1]
for s, col, lab in zip(sigmas_show, colors, labels_s):
    loc, scale = match_gumbel_to_lognormal(MU, s)
    ln_mean, ln_var, _ = lognormal_moments(MU, s)
    t = np.linspace(max(1e-6, ln_mean - 3*np.sqrt(ln_var)),
                    ln_mean + 6*np.sqrt(ln_var), 2000)
    F_ln = np.clip(stats.lognorm.cdf(t, s, scale=np.exp(MU)), 1e-12, 1-1e-12)
    F_gu = np.clip(stats.gumbel_r.cdf(t, loc=loc, scale=scale), 1e-12, 1-1e-12)
    g_ln = np.log(-np.log(F_ln))
    g_gu = np.log(-np.log(F_gu))
    ax.plot(t / ln_mean, g_ln, color=col, lw=1.6, ls="-", label=f"LN, {lab}")
    ax.plot(t / ln_mean, g_gu, color=col, lw=1.4, ls="--", alpha=0.75)

ax.set_xlabel(r"$t$ / mean")
ax.set_ylabel(r"$g(F) = \ln(-\ln F)$")
ax.set_xlim(0, 3)
ax.set_ylim(-6, 3)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc="upper right", fontsize=7)
ax.set_title("(b) Double-log transform (Gumbel = exactly straight)",
             fontsize=10.5, pad=8)

# ── (c) Skewness vs sigma ──
ax = axes[1, 0]
s_range = np.linspace(0.01, 1.5, 500)
skew_ln = (np.exp(s_range**2) + 2) * np.sqrt(np.exp(s_range**2) - 1)
ax.plot(s_range, skew_ln, color="#2c3e50", lw=1.8, label="Log-normal skewness")
ax.axhline(GUMBEL_SKEW, color="#c0392b", lw=1.8, ls="--",
           label=f"Gumbel skewness = {GUMBEL_SKEW:.3f} (fixed)")
ax.axvline(S_STAR, color="0.5", lw=1.0, ls=":")
ax.plot([S_STAR], [GUMBEL_SKEW], "o", color="#27ae60", markersize=7,
        markeredgecolor="white", markeredgewidth=0.8, zorder=10,
        label=rf"match at $s^*={S_STAR:.3f}$")
ax.set_xlabel(r"log-normal shape parameter $s$")
ax.set_ylabel("skewness")
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 8)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc="upper left", fontsize=8)
ax.set_title("(c) Shape mismatch: skewness", fontsize=10.5, pad=8)

# ── (d) Kolmogorov distance vs sigma ──
ax = axes[1, 1]
s_grid = np.linspace(0.05, 1.5, 120)
ks = np.array([kolmogorov_distance(MU, s) for s in s_grid])
ax.plot(s_grid, ks, color="#2c3e50", lw=1.8)
ax.axvline(S_STAR, color="0.5", lw=1.0, ls=":")
i_min = np.argmin(ks)
ax.plot([s_grid[i_min]], [ks[i_min]], "o", color="#27ae60", markersize=7,
        markeredgecolor="white", markeredgewidth=0.8, zorder=10,
        label=f"min KS = {ks[i_min]:.4f} at s = {s_grid[i_min]:.3f}")

# Reference: KS detectable at n=100, 1000 (approx critical values, alpha=0.05)
for n, style in [(100, "--"), (1000, "-.")]:
    crit = 1.36 / np.sqrt(n)
    ax.axhline(crit, color="#c0392b", lw=1.0, ls=style, alpha=0.8,
               label=f"KS detectable, n={n} ({crit:.3f})")

ax.set_xlabel(r"log-normal shape parameter $s$")
ax.set_ylabel("max |CDF difference| (Kolmogorov distance)")
ax.set_xlim(0, 1.5)
ax.set_yscale("log")
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc="lower right", fontsize=7.5)
ax.set_title("(d) How different are they? (location/scale matched)",
             fontsize=10.5, pad=8)

fig.suptitle(
    "Gumbel (= Gompertz) vs Log-Normal: closeness is confined to a narrow "
    r"band of shape parameter $s$",
    fontsize=11.5, y=1.01
)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/gumbel_vs_lognormal.png")
plt.savefig("/mnt/user-data/outputs/gumbel_vs_lognormal.pdf")

# ─── Numerical summary ───────────────────────────────────────────────────────

print("=" * 68)
print("Kolmogorov distance (max |CDF difference|), mean/variance matched")
print("=" * 68)
print(f"{'sigma':>8} {'KS dist':>12} {'LN skew':>12} {'detectable n':>16}")
print("-" * 68)
for s in [0.1, 0.2, 0.3, S_STAR, 0.5, 0.6, 0.8, 1.0, 1.2]:
    ksd = kolmogorov_distance(MU, s)
    _, _, sk = lognormal_moments(MU, s)
    # sample size at which this difference becomes detectable (alpha=0.05)
    n_det = (1.36 / ksd) ** 2 if ksd > 0 else np.inf
    tag = "  <-- skew-matched" if abs(s - S_STAR) < 1e-6 else ""
    print(f"{s:>8.3f} {ksd:>12.5f} {sk:>12.3f} {n_det:>16.0f}{tag}")
print("-" * 68)
print("\nInterpretation: 'detectable n' is the approximate sample size at")
print("which a KS test at alpha=0.05 could distinguish the two distributions.")
print("Small n means easy to tell apart; large n means practically identical.")
print("\nPlots saved.")
