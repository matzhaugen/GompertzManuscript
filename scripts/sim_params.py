"""
Shared simulation parameters
=============================
Single source of truth for every figure-producing script in this directory
(micro_gompertz.py, micro_richards.py, explicit_network.py, closure_error.py).

Importing these constants everywhere guarantees that the network simulations
behind the four figures use identical numerics (N, beta, sigma, dt, seed,
initial-condition spread, integration scheme), which is what the reviewer's
"numerical reproducibility" comment asks for.

Any per-experiment deviation from these canonical values is defined locally in
the individual script with an explicit comment explaining why (currently only
two justified exceptions exist, see below).
"""

# ─── Canonical parameters (shared by ALL experiments) ─────────────────────────

N = 10_000          # number of nodes
BETA = 0.25         # growth-rate constant
SIGMA = 0.1         # noise AMPLITUDE (identical across every stochastic figure).
                    # Convention matches Eq. (newMicro), dz_i = [...]dt + SIGMA dB_i,
                    # so the Euler-Maruyama increment is SIGMA * sqrt(DT) * xi.
                    # NOTE: earlier versions used an increment sqrt(SIGMA * DT) with
                    # SIGMA = 0.01, i.e. an effective amplitude of sqrt(0.01) = 0.1.
                    # SIGMA = 0.1 reproduces that same noise level under the correct
                    # (amplitude) convention, so the figures are unchanged.
DT = 0.02           # Euler-Maruyama time step
T_MAX = 50.0        # integration horizon
SEED = 42           # base RNG seed
IC_STD = 0.3        # std of the initial log-abundance z_i across ALL experiments

# ─── Ensemble size for uncertainty quantification ─────────────────────────────

N_ENSEMBLE = 20     # independent network+IC realizations for the closure figure

# ─── Plot trimming ────────────────────────────────────────────────────────────
# The double-log time panels are only informative up to saturation; beyond that
# the trajectory sits on a flat plateau and the ln(ln(1/X)) transform amplifies
# tiny residual fluctuations into a long noisy tail. We stop each curve once the
# abundance is within SAT_THRESH of its saturation level (i.e. 99.99% saturated).

SAT_THRESH = 0.9999

def saturation_index(M):
    """First time index at which the (running-max) abundance reaches
    SAT_THRESH of its plateau, estimated robustly as the median of the final
    20% of the trajectory. Returns the last index if saturation is never
    reached. Used to trim the double-log time panels."""
    import numpy as np
    M = np.asarray(M, dtype=float)
    plateau = np.median(M[int(0.8 * len(M)):])
    reached = np.maximum.accumulate(M) >= SAT_THRESH * plateau
    return int(np.argmax(reached)) if reached.any() else len(M) - 1

# ─── Documented, deliberate exception (initial abundance only) ────────────────
#
# closure_error.py uses a higher initial abundance (IC_MEAN_Z_CLOSURE) than the
# growth-sweep experiments. The closure metric is measured during the active
# growth phase X in [0.1, 0.9]; the logistic model also needs a non-negligible
# seed x_i(0) to grow at all (dx ~ x). All growth-sweep figures share
# IC_MEAN_Z_SWEEP; only the closure comparison differs, and within it the
# Gompertz and logistic models share the exact same initial condition, so the
# comparison remains like-for-like. beta, sigma, dt, N, and the IC spread are
# identical everywhere.
IC_MEAN_Z_SWEEP = -13.8      # x_i(0) ~ 1e-6, used by the three growth-sweep figures
IC_MEAN_Z_CLOSURE = -2.3     # x_i(0) ~ 0.1, used by the closure comparison
