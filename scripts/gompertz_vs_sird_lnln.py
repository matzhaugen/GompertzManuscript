"""
Figure `Gompertz_vs_SIRD_lnln_infty` -- Figure 1 of arxiv/main.tex.

Cumulative COVID-19 deaths under the Gompertz-linearising transform
g(Y) = ln ln(Y_inf / Y(t)), with an SIRD fit and a Gompertz fit overlaid, for the
sixteen countries of Carletti et al. (2020).

This is the standalone form of cell 7 of gompertz_vs_sird.ipynb, with two changes:

  * TIME IS COUNTED FROM THE WHO PANDEMIC DECLARATION, 11 March 2020, which is
    marked on every panel, rather than from the country-specific date at which
    Y(t)/Y_max first exceeds TAKEOFF. The take-off rule still selects which
    observations enter the fit -- only the axis origin has moved. A country whose
    take-off precedes the declaration (Italy, 5 March) therefore starts at a
    negative day count, which is the intended reading: the common origin is what
    makes the panels comparable, since the old per-country origin silently aligned
    sixteen different calendar dates at zero.
  * The Gompertz model is fitted in its 2-parameter form (Eq. GompertzODE2param),
    matching the redrafted text. This is a reparameterisation, not a different
    model, so the plotted curve is unchanged.

The SIRD fit and the reported R_0 are untouched: R_0 comes from the unnormalised
fit, as in cell 6, and its interval from the parametric bootstrap of the Methods.

No pandas -- the manuscript venv does not have it -- so the JHU file is parsed with
the csv module.

Run with:  cd scripts && uv run python gompertz_vs_sird_lnln.py
Then copy ../figures/Gompertz_vs_SIRD_lnln_infty.pdf into the manuscript directory
that main.tex reads from (see \graphicspath in main.tex).
"""

import csv
import datetime
from pathlib import Path

import matplotlib
import numpy as np
import scipy.optimize as spo
from scipy import integrate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA = Path(__file__).resolve().parent / "time_series_covid19_deaths_global.csv"

START_DATE = datetime.date(2020, 1, 22)
END_DATE = datetime.date(2020, 5, 16)
WINDOW = 6  # days, centred rolling mean applied to daily deaths
TAKEOFF = 0.005  # fit starts one step before Y(t)/Y_max first exceeds this

# WHO declared COVID-19 a pandemic on this date; it is the origin of the time axis.
DECLARATION = datetime.date(2020, 3, 11)
DECLARATION_LABEL = "WHO pandemic declaration"

COUNTRIES = ["Belgium", "Portugal", "Austria", "Switzerland",
             "Sweden", "Italy", "France", "Spain",
             "Germany", "Denmark", "United Kingdom", "Netherlands",
             "Turkey", "Iran", "US", "China"]

COLOR_GOMPERTZ = "#e5a400"
COLOR_SIRD = "#008000"
COLOR_DATA = "#0000ff"
COLOR_RULE = "#52514e"

SIRD_MIN = (10, 1e-6, 0.001)  # gamma, xi, beta
SIRD_MAX = (10000, 1e-2, 1)
GOMP_MIN = (1e-6, 1.0 + 1e-9)  # beta, Y_inf (normalised, so Y_inf > 1)
GOMP_MAX = (10.0, 50.0)

BOOTSTRAP_DRAWS = 10000
RNG = np.random.default_rng(0)


def sird(D, gamma, xi, beta):
    return gamma * (1 - np.exp(-xi * D)) - beta * D


def sird_jac(D, gamma, xi, beta):
    jac = np.empty((D.size, 3))
    jac[:, 0] = 1 - np.exp(-xi * D)
    jac[:, 1] = gamma * D * np.exp(-xi * D)
    jac[:, 2] = -D
    return jac


def sird_time(t, D0, gamma, xi, beta):
    obj = integrate.solve_ivp(lambda _, y: sird(y, gamma, xi, beta), (0, np.max(t)),
                              (D0,), t_eval=t)
    return obj["y"].flatten()


def gompertz(Y, beta, Y_inf):
    """dY/dt for Eq. GompertzODE2param."""
    return -beta * Y * np.log(Y / Y_inf)


def gompertz_jac(Y, beta, Y_inf):
    jac = np.empty((Y.size, 2))
    jac[:, 0] = -Y * np.log(Y / Y_inf)
    jac[:, 1] = beta * Y / Y_inf
    return jac


def gompertz_time(t, Y0, beta, Y_inf):
    return Y_inf * np.power(Y0 / Y_inf, np.exp(-beta * t))


def gompertz_seed(Y):
    """Levitt's closed form, used as the optimiser start point."""
    log_y, dlog_y = np.log(Y)[:-1], np.diff(np.log(Y))
    intercept, beta = np.linalg.lstsq(
        np.column_stack([np.ones(len(log_y)), -log_y]), dlog_y, rcond=None)[0]
    return beta, np.exp(intercept / beta)


def safe_lnln(x, k=1.0):
    z = np.array(x, dtype=float) / k
    eps = 5e-3
    z[z <= 0] = eps
    z[z >= 1.0] = 1 - eps
    return np.log(-np.log(z))


def reproduction_number(gamma, xi, beta, cov):
    """R_0 = xi*gamma/beta with a parametric-bootstrap interval, as in the Methods."""
    draws = RNG.multivariate_normal(mean=np.array([gamma, xi, beta]), cov=cov,
                                    size=BOOTSTRAP_DRAWS)
    r0 = draws[:, 0] * draws[:, 1] / draws[:, 2]
    r0 = r0[np.isfinite(r0) & (r0 > 0)]  # negative rates are unphysical
    return gamma * xi / beta, np.quantile(r0, 0.975) - np.quantile(r0, 0.025)


def load():
    """Daily and cumulative deaths per country over the fitting window."""
    with open(DATA, newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    dates = [datetime.datetime.strptime(c, "%m/%d/%y").date() for c in header[4:]]

    out = {}
    for country in COUNTRIES:
        cum = np.zeros(len(dates))
        for row in body:
            if row[1] == country:
                cum += np.array([float(v or 0) for v in row[4:]])
        daily = np.diff(cum)
        day = dates[1:]
        if country == "China":
            keep = daily <= 1000  # the notebook drops China's reporting spike
            daily, day = daily[keep], [d for d, k in zip(day, keep) if k]
        # Centred rolling mean; the window is even, so it lags by half a day. This
        # is the convention of the published fits and is kept deliberately -- see
        # the sensitivity columns of Table 1 in main.tex.
        pad = WINDOW // 2
        roll = np.full(len(daily), np.nan)
        for i in range(pad, len(daily) - pad + 1):
            roll[i] = daily[i - pad:i - pad + WINDOW].mean()
        cumulative = np.cumsum(daily)
        inside = np.array([START_DATE <= d <= END_DATE for d in day]) & np.isfinite(roll)
        out[country] = (np.array(day)[inside], cumulative[inside] + 1.0, roll[inside])
    return out


def fit(country, day, cumulative, per_day):
    scale = cumulative.max()
    y, dy = cumulative / scale, per_day / scale

    takeoff = max(int(np.argmax(y / y.max() > TAKEOFF)) - 1, 0)
    y, dy, day = y[takeoff:], dy[takeoff:], day[takeoff:]

    # R_0 from the unnormalised fit, as in the published tables.
    popt_raw, pcov_raw = spo.curve_fit(sird, cumulative, per_day, bounds=(SIRD_MIN, SIRD_MAX),
                                       jac=sird_jac, method="trf")
    r0, r0_width = reproduction_number(*popt_raw, pcov_raw)

    popt_sird, _ = spo.curve_fit(sird, y, dy, jac=sird_jac)
    seed = gompertz_seed(y)
    popt_gomp, _ = spo.curve_fit(gompertz, y, dy, p0=(seed[0], max(seed[1], 1.001)),
                                 bounds=(GOMP_MIN, GOMP_MAX), jac=gompertz_jac)

    t = np.arange(len(y) - 1)
    days = np.array([(day[i] - DECLARATION).days for i in t], dtype=float)
    return dict(
        days=days, y_inf=popt_gomp[1], r0=r0, r0_width=r0_width,
        observed=safe_lnln(y[t], popt_gomp[1]),
        sird=safe_lnln(sird_time(t, y[0], *popt_sird), popt_gomp[1]),
        gompertz=safe_lnln(gompertz_time(t, y[0], *popt_gomp), popt_gomp[1]),
    )


def main():
    plt.rcParams["text.usetex"] = True
    data = load()
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))

    for n, country in enumerate(COUNTRIES):
        ax = axs[n // 4, n % 4]
        f = fit(country, *data[country])

        ax.axvline(0, color=COLOR_RULE, linestyle=":", linewidth=1, zorder=0,
                   label=DECLARATION_LABEL if n == 0 else None)
        ax.plot(f["days"], f["observed"], ".", color=COLOR_DATA, alpha=0.3)
        ax.plot(f["days"], f["sird"], label="SIRD", color=COLOR_SIRD)
        ax.plot(f["days"], f["gompertz"], label="Gompertz", color=COLOR_GOMPERTZ)
        ax.set_ylim((np.min(f["observed"]), 2.5))

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        yscale = ymax - ymin
        x_alpha, y_alpha = (0.40, 0.70) if n == 15 else (0.90, 0.90)
        if n == 15:
            xloc = xmin * (1 - x_alpha) + xmax * x_alpha
            yloc = ymin * (1 - y_alpha) + ymax * y_alpha
        else:
            xloc = xmin * x_alpha + xmax * (1 - x_alpha)
            yloc = ymin * y_alpha + ymax * (1 - y_alpha)
        ax.text(xloc, yloc + yscale * 0.12, country, fontsize=11)
        ax.text(xloc, yloc, r"$\mathcal{R}_0$: " + f"{f['r0']:.2f} ({f['r0_width']:.2f})",
                fontsize=11)

        if n % 4 == 0:
            ax.set_ylabel(r"$\ln{(\ln{(Y_{\infty}/Y(t))})}$")
        if n // 4 == 3:
            ax.set_xlabel("Days from declaration")
        if n == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

    fig.tight_layout()
    out = FIG_DIR / "Gompertz_vs_SIRD_lnln_infty.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
