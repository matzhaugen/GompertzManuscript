# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A scientific manuscript ("Gompertz growth as a signature of coherent, non-local dynamics in biological systems") plus the Python scripts that generate its figures. It is a paper repo, not an application — there is no app to run and no test suite. `main.py` is an unused `uv` stub.

The **active paper is `arxiv2/new_draft.tex`**. The many sibling directories (`elsarticle/`, `PNAS/`, `BulletinOfMathBio/`, `JournalOfMathBio/`, `NonlinearDynamics/`, `ScientificReports/`, `letterToEditor*/`) and the older `arxiv/` are journal-template / submission-format variants — do not edit these unless explicitly asked; they are not the source of truth.

## Building the paper

```bash
cd arxiv2 && latexmk -pdf new_draft.tex     # -> new_draft.pdf
```

Bibliography is `references2.bib` (present in both `arxiv2/` and repo root). Figures are written out as plain `figure` environments that include `name.pdf` **from `arxiv2/` itself** and are labelled `fig:name`, so a figure's `\ref{}` key is still its filename. The old `\insertPdfFig{name}{caption}` macro was removed on purpose: an `\includegraphics` inside a `\newcommand` hides the filename from arXiv's file scanner, which then reports the macro parameter as a missing file. Do not reintroduce it. A new figure needs a second `latexmk` pass before its reference resolves.

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
| `closure_error_clustering.py` | `closure_error_clustering` | Closure error vs clustering: Watts-Strogatz rewiring sweep |
| `closure_error_lattices.py` | *(table, no figure)* | Closure error on four regular lattices; prints the LaTeX rows |
| `gompertz_vs_sird_lnln.py` | `Gompertz_vs_SIRD_lnln_infty` | Fig. 1 of `arxiv/main.tex`, days counted from the WHO declaration |

**The model.** Dynamics live in the log-domain `z = ln x`; the macroscopic observable is the geometric mean `X = exp(mean(z))`. "Mean-field" replaces each node's neighbor average with a degree-weighted global average (a weight vector); "explicit" uses actual neighbors via a row-normalized `scipy.sparse` coupling matrix `W` (so neighbor averages are one `W @ z`). The θ parameter interpolates Gompertz↔Richards↔logistic via a power-mean coupling.

**Noise convention (matters, easy to get wrong).** The microscopic model (Eq. `newMicro`, i.e. equation 12) adds noise in the log-domain, `σ dB` on `dz` — which is *multiplicative in abundance*. The closure-error comparison (`closure_error.py`) is measured on the **deterministic** dynamics (`SIGMA_CLOSURE = 0`) because closure under aggregation is a property of the deterministic vector field; a stochastic comparison is noise-structure-dependent and can invert the result. `SIGMA` in `sim_params.py` is the noise **amplitude**, matching the `σ dB` of Eq. `newMicro`, so every Euler-Maruyama step uses `SIGMA * np.sqrt(DT)` — *not* `np.sqrt(SIGMA * DT)`, which an earlier version used and which makes the effective amplitude `√σ`.

**Regeneration workflow.** Scripts write to `figures/`, and `arxiv2/` holds the copies the paper actually compiles against — so copying is **required**, not just for packaging: a stale copy in `arxiv2/` means the PDF silently shows an old figure. After regenerating, copy the updated `pdf`+`png` from `figures/` into `arxiv2/` and recompile, e.g.:

```bash
cp figures/closure_error_comparison.{pdf,png} arxiv2/
```

`arxiv/main.tex` has the same trap in a worse form: its `\graphicspath` used `~`, which TeX does
**not** expand, so it silently resolved to nothing and figures were read from `arxiv/` itself. The
line has been removed; keep figure copies in `arxiv/` beside the `.tex`. `main.fls` tells you which
file was actually read (`grep INPUT main.fls`) — the only reliable check.

**The three closure experiments share one implementation.** `closure_error_clustering.py` and
`closure_error_lattices.py` **import** `simulate_gompertz` / `simulate_logistic` / `simulate_theta1`
and `trajectory_mean_error` from `closure_error.py` rather than copying them, so the three can never
drift apart in what they measure. This is why `closure_error.py`'s ensemble now sits under
`if __name__ == "__main__":` — importing it used to run the whole 2-network ensemble as a side
effect. Its printed statistics are unchanged by that refactor (L/G median 11.8, IQR 9.4-15.0;
θ=1 median 35.7, IQR 27.2-42.4), which is the regression check if you touch it.
`closure_error_clustering.py` caches its ~9-minute sweep to
`scripts/closure_error_clustering_results.npy` and replots from it; delete the file or pass
`recompute=True` to re-run.

`convergence_study.py` caches its ~6-minute computation to `scripts/convergence_results.npy`; it loads the cache if present, otherwise recomputes.

## Deferred: the θ=1 endpoint control in the closure figure

`closure_error.py` computes a third model on every run — `simulate_theta1`, the θ=1
endpoint of Eq. `newMicro` measured under its own natural (odds/logit) aggregator — and
prints its error ratio, but the curve and annotation are **deliberately commented out** of
the figure (see the `theta=1 endpoint control curve omitted for now` comments around the
`ax.plot` and `ax.text` calls). Do not silently re-enable them.

Rationale and current numbers (deterministic, 20 realizations, full 0<X<1 window):

| ratio vs Gompertz | Scale-Free | Erdős-Rényi |
|---|---|---|
| transmission logistic (in the paper) | median 6.0, IQR 5.1–13.7 | median 11.8, IQR 9.4–15.0 |
| θ=1 endpoint (omitted) | median 8.6, IQR 3.6–14.5 | median 35.7, IQR 27.2–42.4 |

Showing the θ=1 curve would *strengthen* the paper — it answers the likely referee
objection that the headline comparison is against a model outside the interpolating family
(Eq. `microLogisticEpidemic`), and on Erdős-Rényi the family's own endpoint closes worse
than the transmission logistic. It is held back pending referee feedback; revisit then.

Related: the Results note that the θ=1 limit "would close in the aggregate under a harmonic
mean coordinate transformation" refers to a **modified** coupling that is linear in the
harmonic coordinate u=1/x (which then closes exactly under double stochasticity, the same
mechanism by which Gompertz closes in the log coordinate) — *not* to the θ=1 endpoint of
Eq. `newMicro` as written, whose coupling is a power mean in the log domain and which does
not close under the harmonic mean (verified numerically: harmonic-mean closure error is
~5x worse than the logit mean).

## arXiv submission package

`arxiv2/arxiv_submission.zip` is a **flat, verbatim copy** of eight files from `arxiv2/`:
`new_draft.tex`, `new_draft.bbl`, `arxiv.sty`, and the five figure PDFs. No path rewriting or
macro expansion happens at packaging time — that is why the figure macros were expanded in the
source and why the paper reads figures from `arxiv2/`.

Two things to preserve when regenerating it:

- **Ship `new_draft.bbl`, never `references2.bib`.** arXiv does not run BibTeX. The committed
  `.bbl` is in *citation order*, but `\bibliographystyle{abbrvnat}` regenerates it
  *alphabetically*; including the `.bib` locally triggers that rebuild and renumbers every
  citation in the paper. (Unresolved: the `.bbl` and the declared style disagree.)
- **No `#` may appear in the shipped `.tex`** — see the figure-macro note above.

Verify with a clean-room build: unzip into an empty directory, `latexmk -pdf`, and check the
page count and that `new_draft.fls` lists no `INPUT ../` or `INPUT /Users` lines.

Note: `latexmk` repeatedly reports success while leaving a **corrupt or missing** PDF — twice
`Output written ... (20 pages)` with a null top-level pages object, and once exit code 12 with no
`new_draft.pdf` at all after a figure was swapped underneath it. **`rm -f new_draft.pdf` before
building is NOT sufficient** (an earlier version of this note said it was). The reliable fix is a
full clean first:

```bash
latexmk -C -cd arxiv2/new_draft.tex && rm -f arxiv2/new_draft.pdf
latexmk -pdf -cd -interaction=nonstopmode arxiv2/new_draft.tex
```

Always verify afterwards — latexmk's exit code and success message are both worthless here:

```bash
pdfinfo arxiv2/new_draft.pdf | grep -i pages        # must print a page count, not an error
grep -cE '^! ' arxiv2/new_draft.log                 # real errors (0 expected)
grep -c 'undefined' arxiv2/new_draft.log            # unresolved refs/citations (0 expected)
```

Note that `grep -c Overfull` and hyperref's "destination with the same identifier" lines are
warnings, not errors; a build can be clean with those present.
