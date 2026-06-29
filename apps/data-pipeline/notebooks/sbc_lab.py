"""Support code for the simulation-based-calibration (SBC) walkthrough.

SBC (Talts, Betancourt, Simpson, Vehtari & Gelman 2018, arXiv 1804.06788) answers one
narrow but load-bearing question: *is my posterior sampler actually returning the
posterior?* It does not ask whether the model is right for the world — only whether the
computation is faithful to the model and prior you wrote down. That makes it the natural
acceptance test for an inference engine.

The whole thing rests on one self-consistency identity. Average the posterior over
datasets drawn from the prior predictive and you get the prior back:

    ∫∫ p(θ | y) p(y | θ') p(θ') dy dθ' = p(θ).

So if you (1) draw a parameter from the prior, (2) simulate a dataset from it, and
(3) draw L samples from the *correct* posterior, the prior draw is exchangeable with the
L posterior draws — they are L+1 iid draws from the same distribution. Its **rank**
among them (how many of the L fall below it) is therefore Uniform{0, 1, ..., L}. Repeat
S times, histogram the ranks: a correct sampler gives a flat histogram; the *way* it
departs from flat names the defect.

This module builds that experiment on a toy where the posterior is known in closed form,
so we can be the *correct* sampler (uniform ranks, by construction) and then deliberately
be a *wrong* one — too narrow, too wide, off-centre, under-thinned — and watch each
failure print its signature.

The toy: a conjugate Gaussian with unknown mean and known variance.

    prior        θ ~ Normal(MU0, TAU0²)
    likelihood   yᵢ ~ Normal(θ, SIGMA²),  i = 1..N
    posterior    θ | y ~ Normal(m, v)      (closed form, below)

Only the sample mean ȳ matters (it is sufficient), and ȳ | θ ~ Normal(θ, SIGMA²/N), so a
whole SBC run vectorises to a few array ops — thousands of simulations are instant, which
is what makes the sliders in the walkthrough live.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from continuous_time_lab import C_CONT, C_DISC, C_OBS, C_PROBE, C_SET, C_TRUTH
from plotly.subplots import make_subplots

# ── The toy model ──────────────────────────────────────────────────────────────
MU0 = 0.0  # prior mean
TAU0 = 3.0  # prior standard deviation
SIGMA = 1.0  # known observation standard deviation
N_DATA = 8  # observations per simulated dataset

# ── SBC defaults ───────────────────────────────────────────────────────────────
S_DEFAULT = 2000  # number of SBC simulations
L_DEFAULT = 99  # posterior draws per simulation -> ranks live in {0, ..., L}
BINS_DEFAULT = 20  # histogram bins; L+1 divisible by BINS avoids aliasing teeth
ALPHA = 0.05  # 1 - simultaneous coverage of the null bands
NSIM_BAND = 2000  # Monte-Carlo replicates used to calibrate a simultaneous band
SEED = 0

# ── Semantic palette (built on the shared continuous-time colours) ─────────────
C_UNIFORM = C_TRUTH  # the uniform target / reference
C_PASS = C_SET  # calibrated — green
C_OVER = C_DISC  # overconfident: posterior too narrow — red
C_UNDER = C_CONT  # underconfident: posterior too wide — blue
C_BIAS = C_PROBE  # biased location — orange
C_TRUE = C_PROBE  # the prior draw θ̃ (the "truth" of a run)
C_BAND = "rgba(148,163,184,0.30)"  # simultaneous null band fill (slate)


def _layout(fig: go.Figure, title: str, height: int = 460, width: int = 940) -> go.Figure:
    fig.update_layout(
        title=f"<b>{title}</b>",
        height=height,
        width=width,
        margin=dict(t=66, b=58, l=66, r=26),
        template="plotly_white",
        legend=dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.78)"
        ),
        bargap=0.04,
    )
    return fig


# ── The conjugate posterior and one SBC run ────────────────────────────────────
def posterior(ybar: np.ndarray | float, n: int) -> tuple[np.ndarray | float, float]:
    """Closed-form Normal posterior mean and variance from the sufficient statistic ȳ."""
    v = 1.0 / (1.0 / TAU0**2 + n / SIGMA**2)
    m = v * (MU0 / TAU0**2 + n * ybar / SIGMA**2)
    return m, v


def sbc_ranks(
    scale: float = 1.0,
    bias: float = 0.0,
    rho: float = 0.0,
    *,
    S: int = S_DEFAULT,
    L: int = L_DEFAULT,
    n: int = N_DATA,
    seed: int = SEED,
    thin: int = 1,
) -> np.ndarray:
    """Run S SBC simulations against a (possibly deformed) candidate posterior.

    Each simulation draws a truth θ̃ from the prior, a dataset's sufficient statistic ȳ
    from the likelihood, then L draws from the candidate posterior, and records the rank
    of θ̃ among those L draws. The candidate is the EXACT conjugate posterior, knobbed:

        scale   multiplies the posterior sd   (scale < 1 overconfident, > 1 underconfident)
        bias    shifts the posterior mean by  bias · posterior_sd  (location error)
        rho     AR(1) autocorrelation of the L draws (an under-thinned MCMC chain),
                optionally thinned by keeping every `thin`-th draw

    scale = 1, bias = 0, rho = 0 is the exact sampler, whose ranks are Uniform{0,...,L}.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(MU0, TAU0, S)  # the truth for each run
    ybar = theta + rng.normal(0.0, SIGMA / np.sqrt(n), S)  # ȳ | θ̃, the sufficient statistic
    m, v = posterior(ybar, n)
    sd = np.sqrt(v)
    m_c = m + bias * sd  # candidate mean
    sd_c = scale * sd  # candidate sd
    if rho == 0.0:
        z = rng.standard_normal((S, L))
    else:
        k = L * thin + thin  # run a longer chain, then thin down to L draws
        z = np.empty((S, k))
        z[:, 0] = rng.standard_normal(S)
        root = np.sqrt(1.0 - rho**2)  # keeps the stationary marginal exactly Normal(0,1)
        for j in range(1, k):
            z[:, j] = rho * z[:, j - 1] + root * rng.standard_normal(S)
        z = z[:, thin - 1 :: thin][:, :L]
    draws = m_c[:, None] + sd_c * z
    return (draws < theta[:, None]).sum(axis=1).astype(int)


def ess_fraction(rho: float, thin: int = 1) -> float:
    """Effective fraction of L kept after AR(1) autocorrelation at the given thinning."""
    r = rho**thin
    return float((1.0 - r) / (1.0 + r))


# ── Simultaneous null bands (Säilynoja, Bürkner & Vehtari 2022, arXiv 2103.10522) ──
def _simultaneous_envelope(
    stats_sim: np.ndarray, alpha: float = ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise bounds tuned to a single γ so their *joint* coverage is 1 - alpha.

    `stats_sim` is [n_sim, n_point], one row per simulated null dataset. Per-point γ/2 and
    1-γ/2 empirical quantiles shrink as γ grows, so joint coverage falls monotonically;
    bisect γ until the band contains a whole null dataset exactly 1 - alpha of the time.
    This is the binning-free SBC band: it accounts automatically for the dependence
    between points (e.g. an ECDF is monotone) that a naive pointwise interval ignores.
    """
    n_sim = stats_sim.shape[0]
    col = np.sort(stats_sim, axis=0)

    def at(gamma: float) -> tuple[np.ndarray, np.ndarray]:
        lo = max(int(np.floor(gamma / 2 * n_sim)) - 1, 0)
        hi = min(int(np.ceil((1 - gamma / 2) * n_sim)) - 1, n_sim - 1)
        return col[lo], col[hi]

    def coverage(gamma: float) -> float:
        lo, hi = at(gamma)
        return float(np.mean(np.all((stats_sim >= lo) & (stats_sim <= hi), axis=1)))

    glo, ghi = 1e-4, alpha
    for _ in range(40):
        gm = 0.5 * (glo + ghi)
        if coverage(gm) > 1 - alpha:
            glo = gm
        else:
            ghi = gm
    return at(0.5 * (glo + ghi))


_BAND_CACHE: dict[tuple, tuple] = {}


def _null_ranks(S: int, L: int, seed: int) -> np.ndarray:
    """NSIM_BAND replicate datasets of S exact-uniform ranks in {0, ..., L}."""
    return np.random.default_rng(seed).integers(0, L + 1, size=(NSIM_BAND, S))


def hist_band(S: int, L: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simultaneous band for the per-bin counts of a uniform rank histogram."""
    key = ("hist", S, L, bins)
    if key not in _BAND_CACHE:
        null = _null_ranks(S, L, 20240601)
        idx = np.minimum((null * bins) // (L + 1), bins - 1)
        counts = np.zeros((NSIM_BAND, bins), dtype=int)
        rows = np.repeat(np.arange(NSIM_BAND), S)
        np.add.at(counts, (rows, idx.ravel()), 1)
        lo, hi = _simultaneous_envelope(counts)
        edges = np.linspace(-0.5, L + 0.5, bins + 1)
        _BAND_CACHE[key] = (lo, hi, edges)
    return _BAND_CACHE[key]


def ecdf_band(S: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Simultaneous band for the ECDF of normalised ranks, evaluated on the support."""
    key = ("ecdf", S, L)
    if key not in _BAND_CACHE:
        null = _null_ranks(S, L, 20240602)
        counts = np.zeros((NSIM_BAND, L + 1), dtype=int)
        rows = np.repeat(np.arange(NSIM_BAND), S)
        np.add.at(counts, (rows, null.ravel()), 1)
        cdf = np.cumsum(counts, axis=1) / S  # ECDF at support points k/L
        _BAND_CACHE[key] = _simultaneous_envelope(cdf)
    return _BAND_CACHE[key]


def ecdf_diff(
    ranks: np.ndarray, S: int, L: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ECDF-minus-diagonal of the ranks, with the simultaneous null band (also centred)."""
    grid = np.arange(L + 1) / L
    cdf = np.cumsum(np.bincount(ranks, minlength=L + 1)) / len(ranks)
    lo, hi = ecdf_band(S, L)
    return grid, cdf - grid, lo - grid, hi - grid


# ── Verdict heuristic for the interactive dial ─────────────────────────────────
def diagnose(ranks: np.ndarray, S: int, L: int, bins: int) -> tuple[str, str]:
    """A plain-language read of a rank histogram and the colour to render it in."""
    lo, hi, edges = hist_band(S, L, bins)
    counts, _ = np.histogram(ranks, bins=edges)
    n_out = int(np.sum((counts < lo) | (counts > hi)))
    if n_out == 0:
        return "consistent with uniform — calibrated ✓", C_PASS
    x = np.linspace(-1.0, 1.0, bins)
    trend = float(np.polyfit(x, counts, 1)[0]) * bins  # net rise across the histogram
    half = bins // 2
    curv = float(counts[0] + counts[-1] - counts[half - 1] - counts[half])  # ends minus middle
    if abs(curv) >= abs(trend):
        if curv > 0:
            return "∪ valley — overconfident / under-thinned: too little effective spread", C_OVER
        return "∩ dome — underconfident: posterior too wide", C_UNDER
    if trend < 0:
        return "ramp ↘ — biased high: candidate sits above the truth", C_BIAS
    return "ramp ↗ — biased low: candidate sits below the truth", C_BIAS


# ── Plot helper: a rank histogram with its simultaneous band ───────────────────
def _add_rank_hist(
    fig: go.Figure,
    ranks: np.ndarray,
    S: int,
    L: int,
    bins: int,
    color: str,
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = False,
) -> None:
    lo, hi, edges = hist_band(S, L, bins)
    counts, _ = np.histogram(ranks, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    where = dict(row=row, col=col) if row is not None else {}
    expected = S / bins
    # the simultaneous band as a filled ribbon, drawn first so the bars sit on top
    fig.add_trace(
        go.Scatter(
            x=centers, y=lo, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
        ),
        **where,
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C_BAND,
            name="95% simultaneous band",
            legendgroup="band",
            showlegend=show_legend,
            hoverinfo="skip",
        ),
        **where,
    )
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=(edges[1] - edges[0]) * 0.92,
            marker=dict(color=color, line=dict(width=0)),
            name="rank counts",
            legendgroup="bars",
            showlegend=show_legend,
            hoverinfo="skip",
        ),
        **where,
    )
    fig.add_hline(y=expected, line=dict(color=C_UNIFORM, width=1.4, dash="dot"), **(where or {}))


# ── Figure 1: one SBC run, unpacked ────────────────────────────────────────────
def one_run(seed: int, n: int) -> dict:
    """A single simulation, materialised with real data points (not just ȳ)."""
    rng = np.random.default_rng(1000 + seed)
    theta = float(rng.normal(MU0, TAU0))
    y = rng.normal(theta, SIGMA, n)
    m, v = posterior(float(y.mean()), n)
    draws = rng.normal(m, np.sqrt(v), L_DEFAULT)
    rank = int((draws < theta).sum())
    return dict(theta=theta, y=y, m=float(m), v=float(v), draws=draws, rank=rank)


def fig_one_run(seed: int = 0, n: int = N_DATA) -> go.Figure:
    """Prior, simulated data and posterior on top; the L posterior draws and the rank below.

    The square-edged story of one SBC simulation: a truth θ̃ falls out of the prior (orange
    line), a dataset is drawn from it (orange ticks), the posterior contracts around the
    data (blue), and we sample it L times. The rank is just how many of those draws land to
    the *left* of θ̃ — coloured green below, grey above, counted in the title.
    """
    r = one_run(seed, n)
    theta, m, v, draws, rank = r["theta"], r["m"], r["v"], r["draws"], r["rank"]
    grid = np.linspace(MU0 - 3.2 * TAU0, MU0 + 3.2 * TAU0, 400)
    prior = np.exp(-0.5 * (grid - MU0) ** 2 / TAU0**2) / np.sqrt(2 * np.pi * TAU0**2)
    post = np.exp(-0.5 * (grid - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.74, 0.26],
        vertical_spacing=0.12,
        subplot_titles=(
            "prior → data → posterior",
            f"rank of θ̃ among the {L_DEFAULT} posterior draws  =  {rank} / {L_DEFAULT}",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=prior,
            mode="lines",
            line=dict(color=C_OBS, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(107,114,128,0.10)",
            name="prior p(θ)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=post,
            mode="lines",
            line=dict(color=C_UNDER, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.10)",
            name="posterior p(θ|y)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=r["y"],
            y=np.zeros_like(r["y"]),
            mode="markers",
            marker=dict(color=C_OBS, size=8, symbol="line-ns-open"),
            name="data y",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=theta, line=dict(color=C_TRUE, width=2.2, dash="dash"), row=1, col=1)
    fig.add_annotation(
        x=theta,
        y=max(prior.max(), post.max()),
        text="θ̃ (truth)",
        showarrow=False,
        yshift=10,
        font=dict(color=C_TRUE, size=12),
        row=1,
        col=1,
    )
    below = draws < theta
    fig.add_trace(
        go.Scatter(
            x=draws[below],
            y=np.zeros(int(below.sum())),
            mode="markers",
            marker=dict(color=C_PASS, size=7, symbol="line-ns-open"),
            name=f"draws below θ̃ ({rank})",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=draws[~below],
            y=np.zeros(int((~below).sum())),
            mode="markers",
            marker=dict(color=C_OBS, size=7, symbol="line-ns-open"),
            name=f"draws above θ̃ ({L_DEFAULT - rank})",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_vline(x=theta, line=dict(color=C_TRUE, width=2.2, dash="dash"), row=2, col=1)
    fig.update_xaxes(title_text="θ", row=2, col=1)
    fig.update_yaxes(title_text="density", row=1, col=1)
    fig.update_yaxes(showticklabels=False, range=[-1, 1], row=2, col=1)
    fig.update_xaxes(matches="x", row=2, col=1)
    return _layout(
        fig, "One SBC simulation: where does the truth fall in its own posterior?", height=520
    )


# ── Figure 2: the calibrated baseline ──────────────────────────────────────────
def fig_calibrated(S: int = S_DEFAULT, bins: int = BINS_DEFAULT) -> go.Figure:
    """Ranks from the exact posterior: flat, inside the band — what 'calibrated' looks like."""
    ranks = sbc_ranks(S=S)
    fig = go.Figure()
    _add_rank_hist(fig, ranks, S, L_DEFAULT, bins, C_PASS, show_legend=True)
    fig.update_xaxes(title_text=f"rank of the truth  (0 … {L_DEFAULT})")
    fig.update_yaxes(title_text="count")
    return _layout(
        fig,
        f"Exact sampler, {S} simulations — uniform ranks, every bar inside the band",
        height=440,
        width=860,
    )


# ── Figure 3: the dictionary of shapes (hero) ──────────────────────────────────
_GALLERY = (
    ("calibrated  (exact)", dict(), C_PASS),
    ("overconfident  (σ × 0.5)", dict(scale=0.5), C_OVER),
    ("underconfident  (σ × 2)", dict(scale=2.0), C_UNDER),
    ("biased high  (+0.7σ)", dict(bias=0.7), C_BIAS),
    ("biased low  (−0.7σ)", dict(bias=-0.7), C_BIAS),
    ("under-thinned  (ρ = 0.9)", dict(rho=0.9), C_OVER),
)


def fig_gallery(S: int = S_DEFAULT, bins: int = BINS_DEFAULT) -> go.Figure:
    """Six candidate posteriors, six rank histograms — the shapes you learn to read."""
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[g[0] for g in _GALLERY],
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )
    for i, (_label, kw, color) in enumerate(_GALLERY):
        row, col = divmod(i, 3)
        ranks = sbc_ranks(**kw, S=S, seed=SEED + i)
        _add_rank_hist(fig, ranks, S, L_DEFAULT, bins, color, row=row + 1, col=col + 1)
    fig.update_yaxes(title_text="count", col=1)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="rank", row=2, col=c)
    fig.update_annotations(font=dict(size=12))
    return _layout(
        fig,
        "The SBC dictionary: every miscalibration writes a different histogram",
        height=560,
        width=980,
    )


# ── Figure 4: the interactive miscalibration dial ──────────────────────────────
def fig_dial(
    scale: float = 1.0, bias: float = 0.0, S: int = S_DEFAULT, bins: int = BINS_DEFAULT
) -> go.Figure:
    """Histogram (left) and ECDF-difference (right) for any (scale, bias), with a verdict."""
    ranks = sbc_ranks(scale=scale, bias=bias, S=S)
    verdict, color = diagnose(ranks, S, L_DEFAULT, bins)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.1,
        subplot_titles=("rank histogram", "ECDF − uniform, with simultaneous band"),
    )
    _add_rank_hist(fig, ranks, S, L_DEFAULT, bins, color, row=1, col=1)
    grid, diff, lo, hi = ecdf_diff(ranks, S, L_DEFAULT)
    fig.add_trace(
        go.Scatter(
            x=grid, y=lo, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C_BAND,
            name="band",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=diff,
            mode="lines",
            line=dict(color=color, width=2.4),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.0, line=dict(color=C_UNIFORM, width=1.2, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="rank", row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_xaxes(title_text="normalised rank", row=1, col=2)
    fig.update_yaxes(title_text="ECDF − uniform", row=1, col=2)
    return _layout(fig, f"Verdict: {verdict}", height=460, width=980)


# ── Figure 5: the ECDF-difference signatures ───────────────────────────────────
def fig_ecdf_gallery(S: int = S_DEFAULT) -> go.Figure:
    """The same defects as the histogram gallery, read off binning-free ECDF curves."""
    cases = (
        ("calibrated", dict(), C_PASS),
        ("overconfident", dict(scale=0.5), C_OVER),
        ("underconfident", dict(scale=2.0), C_UNDER),
        ("biased high", dict(bias=0.7), C_BIAS),
        ("biased low", dict(bias=-0.7), C_PROBE),
    )
    fig = go.Figure()
    _, _, lo, hi = ecdf_diff(sbc_ranks(S=S), S, L_DEFAULT)
    grid = np.arange(L_DEFAULT + 1) / L_DEFAULT
    fig.add_trace(
        go.Scatter(
            x=grid, y=lo, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C_BAND,
            name="95% band (calibrated)",
            hoverinfo="skip",
        )
    )
    for i, (label, kw, color) in enumerate(cases):
        _, diff, _, _ = ecdf_diff(sbc_ranks(**kw, S=S, seed=SEED + i), S, L_DEFAULT)
        dash = "solid" if label == "calibrated" else "dot" if "low" in label else "dash"
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=diff,
                mode="lines",
                line=dict(color=color, width=2.2, dash=dash),
                name=label,
                hoverinfo="skip",
            )
        )
    fig.add_hline(y=0.0, line=dict(color=C_UNIFORM, width=1.2, dash="dot"))
    fig.update_xaxes(title_text="normalised rank (≈ posterior CDF at the truth)")
    fig.update_yaxes(title_text="ECDF − uniform")
    return _layout(
        fig, "Binning-free view: each defect bends the ECDF its own way", height=460, width=900
    )


# ── Figure 6: binning sensitivity ──────────────────────────────────────────────
def fig_binning(S: int = S_DEFAULT) -> go.Figure:
    """One overconfident run, three bin counts: the choice changes how the ∪ reads."""
    ranks = sbc_ranks(scale=0.6, S=S)
    bin_choices = (10, 20, 50)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{b} bins" for b in bin_choices],
        horizontal_spacing=0.06,
    )
    for c, b in enumerate(bin_choices):
        _add_rank_hist(fig, ranks, S, L_DEFAULT, b, C_OVER, row=1, col=c + 1)
        fig.update_xaxes(title_text="rank", row=1, col=c + 1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    return _layout(
        fig,
        "Same overconfident ranks — coarse blurs the valley, fine adds noise (so the ECDF wins)",
        height=400,
        width=980,
    )


# ── Figure 7: how many simulations? (power) ────────────────────────────────────
def fig_sample_size(scale: float = 0.8, bins: int = BINS_DEFAULT) -> go.Figure:
    """A mild defect at four budgets: invisible when S is small, unmistakable when large."""
    budgets = (100, 500, 2000, 8000)
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[f"S = {s}" for s in budgets],
        horizontal_spacing=0.045,
        shared_yaxes=False,
    )
    for c, s in enumerate(budgets):
        ranks = sbc_ranks(scale=scale, S=s)
        _add_rank_hist(fig, ranks, s, L_DEFAULT, bins, C_OVER, row=1, col=c + 1)
        fig.update_xaxes(title_text="rank", row=1, col=c + 1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    return _layout(
        fig,
        f"A mild defect (σ × {scale}) — the band shrinks with √S until the ∪ pokes out",
        height=400,
        width=1000,
    )


# ── Figure 8: the autocorrelation gotcha and its fix ───────────────────────────
def fig_autocorrelation(
    rho: float = 0.9, thin: int = 1, S: int = S_DEFAULT, bins: int = BINS_DEFAULT
) -> go.Figure:
    """An exact-but-correlated chain vs. the same chain thinned — uniformity, restored."""
    raw = sbc_ranks(rho=rho, thin=1, S=S)
    thinned = sbc_ranks(rho=rho, thin=thin, S=S)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"every draw kept · ESS ≈ {ess_fraction(rho):.0%} of L",
            f"thinned ×{thin} · ESS ≈ {ess_fraction(rho, thin):.0%} of L",
        ),
        horizontal_spacing=0.08,
    )
    _add_rank_hist(fig, raw, S, L_DEFAULT, bins, C_OVER, row=1, col=1)
    _add_rank_hist(fig, thinned, S, L_DEFAULT, bins, C_PASS if thin > 1 else C_OVER, row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(title_text="rank", row=1, col=c)
    fig.update_yaxes(title_text="count", row=1, col=1)
    return _layout(
        fig,
        f"Correlated draws (ρ = {rho}) fake overconfidence — thinning is the cure, not effort",
        height=420,
        width=960,
    )
