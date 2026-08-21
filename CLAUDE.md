# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A scientific manuscript ("Gompertz growth as a signature of coherent, non-local dynamics in biological systems") plus the Python scripts that generate its figures. It is a paper repo, not an application — there is no app to run and no test suite. `main.py` is an unused `uv` stub.

The **active paper is `arxiv2/new_draft.tex`**. The many sibling directories (`elsarticle/`, `PNAS/`, `BulletinOfMathBio/`, `JournalOfMathBio/`, `NonlinearDynamics/`, `ScientificReports/`, `letterToEditor*/`) and the older `arxiv/` are journal-template / submission-format variants — do not edit these unless explicitly asked; they are not the source of truth.

## Building the paper

```bash
cd arxiv2 && latexmk -pdf new_draft.tex     # -> new_draft.pdf
```

Bibliography is `references2.bib` (present in both `arxiv2/` and repo root). Figures are pulled in via `\insertPdfFig{name}{caption}` (defined in the preamble), which includes `../figures/name.pdf` and auto-labels it `fig:name`. So a figure's `\ref{}` key is its filename. A new figure needs a second `latexmk` pass before its reference resolves.

## Running the figure scripts

Python is managed by `uv` (see `uv.lock`, `pyproject.toml`, `.python-version` = 3.11). A bare `python`/`python3` will not have numpy/scipy. Always:

```bash
cd scripts && uv run python micro_gompertz.py
```

**Run from `scripts/`** — most scripts write to the literal relative path `../figures/*.pdf,png`, so the working directory must be `scripts/` (exception: `richards_explicit.py` resolves relative to its own location).

`.pre-commit-config.yaml` strips output from committed `.ipynb` notebooks.

## Architecture

**`scripts/sim_params.py` is the single source of truth** for the numerics (`N`, `BETA`, `SIGMA`, `DT`, `T_MAX`, `SEED`, `IC_STD`, `IC_MEAN_Z_SWEEP`, `IC_MEAN_Z_CLOSURE`, `N_ENSEMBLE`, `SAT_THRESH`, and the `saturation_index()` plot-trimming helper). Every figure script imports from it, so changing a value there keeps all figures consistent. Any deliberate per-experiment deviation lives locally in the script with a comment explaining why. When editing a script's parameters or a figure caption, keep the values in sync across `sim_params.py`, the script, and the corresponding caption/Methods text in `new_draft.tex`.

**Each figure script maps to one figure and one `\insertPdfFig` in `new_draft.tex`:**

| Script | Figure file | What it shows |
|---|---|---|
| `micro_gompertz.py` | `gompertz_networks` | Mean-field Gompertz across network topologies |
| `micro_richards.py` | `gompertz_richards_interpolation` | θ-interpolation between Gompertz (θ→0) and logistic (θ=1) |
| `convergence_study.py` | `convergence_deltat` | Δt convergence separating structural vs Euler–Maruyama error |
| `explicit_network.py` | `gompertz_explicit_vs_meanfield` | Explicit (scipy.sparse) network vs mean-field |
| `closure_error.py` | `closure_error_comparison` | Gompertz vs logistic macroscopic closure error |
| `richards_explicit.py` | `gompertz_richards_explicit` | Richards on explicit networks |

**The model.** Dynamics live in the log-domain `z = ln x`; the macroscopic observable is the geometric mean `X = exp(mean(z))`. "Mean-field" replaces each node's neighbor average with a degree-weighted global average (a weight vector); "explicit" uses actual neighbors via a row-normalized `scipy.sparse` coupling matrix `W` (so neighbor averages are one `W @ z`). The θ parameter interpolates Gompertz↔Richards↔logistic via a power-mean coupling.

**Noise convention (matters, easy to get wrong).** The microscopic model (Eq. `newMicro`, i.e. equation 12) adds noise in the log-domain, `σ dB` on `dz` — which is *multiplicative in abundance*. The closure-error comparison (`closure_error.py`) is measured on the **deterministic** dynamics (`SIGMA_CLOSURE = 0`) because closure under aggregation is a property of the deterministic vector field; a stochastic comparison is noise-structure-dependent and can invert the result.

**Regeneration workflow.** Scripts write to `figures/`, but `arxiv2/` also holds tracked copies of each figure (for self-contained arXiv packaging). After regenerating, copy the updated `pdf`+`png` from `figures/` into `arxiv2/` and recompile, e.g.:

```bash
cp figures/closure_error_comparison.{pdf,png} arxiv2/
```

`convergence_study.py` caches its ~6-minute computation to `scripts/convergence_results.npy`; it loads the cache if present, otherwise recomputes.
