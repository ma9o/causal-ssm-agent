"""Support code for the inference-geometry walkthrough notebook.

This is the companion to ``confounder_marginalization_walkthrough.py``. That
notebook asks *where the confounder goes* when you marginalize it; this one asks
*why marginalizing helps the sampler*, and answers it with two pictures of
likelihood geometry:

1. **The funnel** — the explicit-latent parameterization (loading ``lambda`` times
   latent scale ``tau``) has a redundant direction: only the product ``lambda*tau``
   reaches the data, so the likelihood is a flat trench. Marginalizing switches to
   the single identified product and the trench becomes one clean peak. This happens
   *whether or not the latent confounds anything* — it is the unconditional win.

2. **The ridge** — once the latent is marginalized into a residual covariance ``c``
   between two children, the causal slope ``beta`` and that handshake ``c`` trade off
   along a flat ridge whenever they land in the same observable cell (the back-door
   case). Marginalizing does not remove this ridge; it *reveals* it. Identification is
   whichever constraint ("knife") cuts across the ridge — the graph itself (``c = 0``
   off-path) or an observable anchor on the latent (on-path).

We reuse the simulators and palette from ``confounder_lab`` so every number matches
the companion notebook; only the geometry helpers and their plots are new.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from confounder_lab import (
    A_UX,
    B_UY,
    BETA_TRUE,
    C_CLASS,
    C_ID,
    C_NAIVE,
    C_OBS,
    C_TRUTH,
    SX,
    SY,
    simulate_confounded,
)
from plotly.subplots import make_subplots


def _layout(fig: go.Figure, title: str, height: int = 420, width: int = 820) -> go.Figure:
    fig.update_layout(
        title=f"<b>{title}</b>",
        height=height,
        width=width,
        margin=dict(t=64, b=48, l=64, r=24),
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
    )
    return fig


# ── Act 1: wiggling the latent is just redrawing one covariance ───────────────
def fig_wiggle_invariance(logk: float, *, n_scatter: int = 500) -> go.Figure:
    """The loading-scale redundancy, made interactive.

    Rescale the latent by ``k = exp(logk)`` and compensate the loadings the other
    way (``a -> a/k``, ``b -> b/k``, ``tau -> k``). Every product ``a*tau``, ``b*tau``
    is invariant, so the population covariance of (X, Y) — and therefore the data
    cloud the sampler fits — does not move at all while the latent's own scale breathes.
    That frozen direction *is* the funnel: a whole curve of latent parameters, one
    dataset. Left panel: the latent U the sampler resamples (scale ``tau`` changing).
    Right panel: the data (X, Y), frozen.
    """
    k = float(np.exp(logk))
    a, b, tau = A_UX / k, B_UY / k, k

    var_x = A_UX**2 + SX**2
    cov_xy = BETA_TRUE * var_x + A_UX * B_UY
    var_y = BETA_TRUE**2 * var_x + 2.0 * BETA_TRUE * A_UX * B_UY + B_UY**2 + SY**2
    sigma = np.array([[var_x, cov_xy], [cov_xy, var_y]])

    rng = np.random.default_rng(0)  # fixed: the scatter never depends on k
    data = simulate_confounded(rng, n=n_scatter)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=(
            "the latent U the sampler resamples — its scale τ is free",
            "the data (X, Y) the likelihood sees — frozen",
        ),
    )

    grid = np.linspace(-6.0, 6.0, 400)
    dens = np.exp(-0.5 * (grid / tau) ** 2) / (tau * np.sqrt(2.0 * np.pi))
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=dens,
            mode="lines",
            line=dict(color=C_OBS, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(154,160,166,0.25)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data["X"],
            y=data["Y"],
            mode="markers",
            marker=dict(color=C_CLASS, size=4, opacity=0.30),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    th = np.linspace(0.0, 2.0 * np.pi, 240)
    ell = np.linalg.cholesky(sigma) @ np.vstack([2.0 * np.cos(th), 2.0 * np.sin(th)])
    fig.add_trace(
        go.Scatter(
            x=ell[0], y=ell[1], mode="lines", line=dict(color=C_NAIVE, width=3), showlegend=False
        ),
        row=1,
        col=2,
    )

    fig.add_annotation(
        x=0.04,
        y=0.96,
        xref="x domain",
        yref="y domain",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        text=(
            f"loadings a = {a:.2f}, b = {b:.2f}<br>latent scale τ = {tau:.2f}<br>"
            f"<b>identified:</b> aτ = {a * tau:.2f}, bτ = {b * tau:.2f} (fixed)"
        ),
        bgcolor="rgba(255,255,255,0.75)",
        font=dict(size=12),
    )

    fig.update_xaxes(title_text="latent value U", range=[-6, 6], row=1, col=1)
    fig.update_yaxes(title_text="density", range=[0.0, 1.2], row=1, col=1)
    fig.update_xaxes(title_text="treatment X", range=[-4, 4], row=1, col=2)
    fig.update_yaxes(title_text="outcome Y", range=[-5, 5], row=1, col=2)
    return _layout(
        fig,
        "Slide the latent's scale: U breathes, the data never moves",
        height=420,
        width=900,
    )


# ── Act 2: the funnel marginalization always kills ────────────────────────────
def fig_funnel(*, n: int = 120, sigma2: float = 0.5, v0: float = 1.0, grid: int = 160) -> go.Figure:
    """The loading-scale funnel as a likelihood surface, and its collapse.

    One latent child ``W = lambda * U + noise`` with ``U ~ N(0, tau^2)``. The data give
    one number, ``Var(W) = lambda^2 * tau^2 + sigma^2``, so the likelihood over
    ``(lambda, tau)`` depends only on the product ``v = lambda^2 * tau^2``: a flat trench
    along ``lambda * tau = sqrt(v0)`` (a straight diagonal in log-log). Reparameterize
    by the single identified ``v`` and the trench becomes one clean peak. The private
    noise ``sigma^2`` is held at its known value so the picture isolates the
    loading-scale redundancy.
    """
    s_var = v0 + sigma2
    log_l = np.linspace(-1.0, 1.0, grid)
    log_t = np.linspace(-1.0, 1.0, grid)
    mesh_l, mesh_t = np.meshgrid(log_l, log_t, indexing="xy")
    m = (10.0**mesh_l * 10.0**mesh_t) ** 2 + sigma2
    ll = -0.5 * n * (np.log(2.0 * np.pi) + np.log(m) + s_var / m)
    ll -= ll.max()

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=(
            "explicit (λ, τ): a flat trench the sampler crawls",
            "marginalized to v = λ²τ²: one clean peak",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=ll,
            x=log_l,
            y=log_t,
            coloraxis="coloraxis",
            hovertemplate="log₁₀λ=%{x:.2f}, log₁₀τ=%{y:.2f}<br>Δloglik=%{z:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    ridge_t = 0.5 * np.log10(v0) - log_l
    inside = (ridge_t >= log_t.min()) & (ridge_t <= log_t.max())
    fig.add_trace(
        go.Scatter(
            x=log_l[inside],
            y=ridge_t[inside],
            mode="lines",
            line=dict(color="white", width=2.5, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=0.0,
        y=0.5 * np.log10(v0),
        xref="x",
        yref="y",
        text="λτ = const → identical fit",
        showarrow=False,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.30)",
    )

    v = np.linspace(0.05, 4.0 * v0, 400)
    llv = -0.5 * n * (np.log(2.0 * np.pi) + np.log(v + sigma2) + s_var / (v + sigma2))
    llv -= llv.max()
    fig.add_trace(
        go.Scatter(x=v, y=llv, mode="lines", line=dict(color=C_ID, width=3), showlegend=False),
        row=1,
        col=2,
    )
    fig.add_vline(x=v0, line=dict(color=C_TRUTH, width=2, dash="dot"), row=1, col=2)
    fig.add_annotation(
        x=v0,
        y=0.0,
        xref="x2",
        yref="y2",
        text="identified product v₀",
        showarrow=False,
        yshift=12,
        font=dict(color=C_TRUTH, size=12),
    )

    fig.update_xaxes(title_text="log₁₀ loading λ", row=1, col=1)
    fig.update_yaxes(title_text="log₁₀ latent scale τ", row=1, col=1)
    fig.update_xaxes(title_text="induced variance  v = λ²τ²", row=1, col=2)
    fig.update_yaxes(title_text="profile log-likelihood (centred at max)", row=1, col=2)
    fig.update_layout(
        coloraxis=dict(
            colorscale="Viridis",
            cmin=-12.0,
            cmax=0.0,
            colorbar=dict(title="Δ log-lik", thickness=12, len=0.85),
        )
    )
    return _layout(
        fig,
        "The funnel marginalization deletes: a flat trench becomes a single peak",
        height=430,
        width=940,
    )


# ── Act 3: the ridge marginalization reveals ──────────────────────────────────
def _marginalized_ll_grid(s2: np.ndarray, betas: np.ndarray, cs: np.ndarray, n: int) -> np.ndarray:
    """Profile log-lik of the marginalized (X, Y) model over (beta, c).

    The model is X, Y jointly Gaussian with Var(X) matched to the data, off-diagonal
    ``Sigma_XY = beta * Var(X) + c``, and Var(Y) profiled to its conditional optimum.
    Because the model only feels (beta, c) through ``Sigma_XY``, the surface is a flat
    ridge along ``beta * S_XX + c = S_XY`` (the saturated, exactly-fitting set) and
    falls away off it. Returned with the max subtracted; invalid (non-PD) cells are NaN.
    """
    sxx, sxy, syy = s2[0, 0], s2[0, 1], s2[1, 1]
    mesh_b, mesh_c = np.meshgrid(betas, cs, indexing="xy")
    q = mesh_b * sxx + mesh_c
    syy_star = syy + 2.0 * q * (q - sxy) / sxx
    det = sxx * syy_star - q**2
    nt = sxx * syy_star + sxx * syy - 2.0 * sxy * q
    with np.errstate(invalid="ignore", divide="ignore"):
        ll = -0.5 * n * (2.0 * np.log(2.0 * np.pi) + np.log(det) + nt / det)
    ll = np.where((det > 0.0) & (syy_star > 0.0), ll, np.nan)
    return ll - np.nanmax(ll)


def fig_ridge_and_cuts(
    s2_on: np.ndarray, s2_off: np.ndarray, *, n: int = 150, grid: int = 220
) -> go.Figure:
    """The (beta, c) likelihood for the on-path and off-path worlds, side by side.

    Both are the *same* flat ridge — (X, Y) alone can never split beta from the
    handshake c. What differs is the knife: off-path the graph supplies ``c = 0`` for
    free (no U->X edge, so X and Y residuals are uncorrelated); on-path nothing
    observable does, so you must measure U to learn c. The knife cuts the ridge at the
    true beta; without one, the whole ridge is admissible (drop).
    """
    betas = np.linspace(BETA_TRUE - 2.3, BETA_TRUE + 2.3, grid)
    cs = np.linspace(-1.1, 2.6, grid)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "§1  on-path: U → X exists ⇒ c is free",
            "§7  off-path: no U → X ⇒ graph pins c = 0",
        ),
    )

    notes = {
        1: "no structural knife —<br>you must <b>measure U</b> to learn c.<br>without it: the whole ridge (drop)",
        2: "X ⟂ U ⇒ residuals uncorrelated<br>⇒ <b>c ≡ 0</b> for free<br>⇒ β identified",
    }
    for col, s2, knife_c in (
        (1, s2_on, float(s2_on[0, 1] - BETA_TRUE * s2_on[0, 0])),
        (2, s2_off, 0.0),
    ):
        ll = _marginalized_ll_grid(s2, betas, cs, n)
        fig.add_trace(
            go.Heatmap(
                z=ll,
                x=betas,
                y=cs,
                coloraxis="coloraxis",
                hovertemplate="β=%{x:.2f}, c=%{y:.2f}<br>Δloglik=%{z:.1f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        sxx, sxy = s2[0, 0], s2[0, 1]
        ridge_c = sxy - betas * sxx
        inside = (ridge_c >= cs.min()) & (ridge_c <= cs.max())
        fig.add_trace(
            go.Scatter(
                x=betas[inside],
                y=ridge_c[inside],
                mode="lines",
                line=dict(color=C_CLASS, width=6),
                opacity=0.55,
                name="ridge: (β, c) the data can't separate",
                legendgroup="ridge",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[betas.min(), betas.max()],
                y=[knife_c, knife_c],
                mode="lines",
                line=dict(color=C_ID, width=2.5, dash="dash"),
                name="knife: a constraint on c",
                legendgroup="knife",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[(sxy - knife_c) / sxx],
                y=[knife_c],
                mode="markers",
                marker=dict(
                    color=C_ID, size=15, symbol="diamond", line=dict(color="white", width=1)
                ),
                name="identified β",
                legendgroup="id",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=BETA_TRUE, line=dict(color=C_TRUTH, width=2, dash="dot"), row=1, col=col)
        ax = "" if col == 1 else str(col)
        fig.add_annotation(
            x=0.04,
            y=0.04,
            xref=f"x{ax} domain",
            yref=f"y{ax} domain",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            align="left",
            text=notes[col],
            bgcolor="rgba(255,255,255,0.78)",
            font=dict(size=11),
        )
        fig.update_xaxes(title_text="causal effect β", row=1, col=col)

    fig.update_yaxes(title_text="handshake covariance c", row=1, col=1)
    fig.update_layout(
        coloraxis=dict(
            colorscale="Viridis",
            cmin=-25.0,
            cmax=0.0,
            colorbar=dict(title="Δ log-lik", thickness=12, len=0.85),
        )
    )
    fig = _layout(
        fig,
        "The same flat ridge — identification is whichever knife cuts across it",
        height=480,
        width=980,
    )
    fig.update_layout(
        margin=dict(b=104),
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", yanchor="top"),
    )
    return fig
