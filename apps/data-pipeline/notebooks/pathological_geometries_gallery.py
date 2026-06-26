import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pathological geometries gallery, with sampler overlays

    A visual tour of every primitive in `synthetic_posteriors`, one 3D surface per pathology. Each panel overlays a sampler run (dots coloured by stage — purple → yellow = early → late) so you can see directly where the sampler mixes well and where it gets stuck or dies.

    Each figure also shows a low-opacity **MAP + Laplace** approximation (orange surface / ellipses) as a baseline — the local Gaussian around the mode that an optimizer-backed `MAP + IEKS` run would collapse to. Toggle it via `SHOW_LAPLACE` in the imports cell, or click the legend entry on any individual figure.

    **Swap sampler** by flipping the two lines in the imports cell (`SAMPLER_FN` and `SAMPLER_CFG`). The rest of the notebook is sampler-agnostic — per-pathology cells pass a dict of `cfg_overrides` (e.g. initial position for NUTS, prior scale for SMC), and only the keys that exist on the active config are applied.
    """)
    return


@app.cell
def _():
    from dataclasses import fields, replace
    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    import plotly.graph_objects as go
    import scipy.optimize as spo
    from plotly.subplots import make_subplots
    from samplers import (
        MultiPathfinderConfig,
        PathfinderConfig,
        run_multipath_pathfinder,
        run_pathfinder,
        run_pathfinder_sampler,
    )
    from synthetic_posteriors import (
        Bend,
        Cauchy,
        Chain,
        Funnel,
        Gaussian,
        Identity,
        Logit,
        Mirror,
        Mixture,
        Shear,
        Shift,
        Softplus,
        StudentT,
        TransformedTarget,
        invariance,
    )

    # ── swap sampler + default config here ──────────────────────────────────
    # Available sampler entry points in `samplers` (each returns a SamplerTrace):
    #   (run_pathfinder_sampler, PathfinderConfig())
    #   (run_multipath_pathfinder_sampler, MultiPathfinderConfig())  # add to the import above
    # The appendix at the bottom of the notebook shows Pathfinder's L-BFGS path +
    # ELBO curve and the multi-path PSIR internals via the richer run_pathfinder /
    # run_multipath_pathfinder entry points.
    SAMPLER_FN = run_pathfinder_sampler
    SAMPLER_CFG = PathfinderConfig()

    # ── MAP + Laplace baseline overlay (low-opacity orange) ─────────────────
    # Flip this to False to skip the MAP fit + Gaussian approximation entirely.
    # When True, each figure also exposes the overlay in its legend so you can
    # click-toggle it per-panel.
    SHOW_LAPLACE = True
    return (
        Bend,
        Cauchy,
        Chain,
        Funnel,
        Gaussian,
        Identity,
        Logit,
        Mirror,
        Mixture,
        MultiPathfinderConfig,
        PathfinderConfig,
        SAMPLER_CFG,
        SAMPLER_FN,
        SHOW_LAPLACE,
        Shear,
        Shift,
        Softplus,
        StudentT,
        TransformedTarget,
        fields,
        go,
        invariance,
        jax,
        jnp,
        make_subplots,
        np,
        partial,
        replace,
        run_multipath_pathfinder,
        run_pathfinder,
        spo,
    )


@app.cell
def _(
    SAMPLER_CFG,
    SAMPLER_FN,
    SHOW_LAPLACE,
    fields,
    go,
    jax,
    jnp,
    make_subplots,
    np,
    replace,
    spo,
):
    # ── MAP + Laplace approximation helpers ──────────────────────────────────
    def fit_laplace(target, init):
        """Find the MAP by L-BFGS on ``-log_prob`` and return (mode, covariance).

        Covariance is the inverse of the Hessian of ``-log_prob`` at the mode —
        the same Gaussian approximation that a MAP + IEKS / KFAS-style fit uses
        for its parameter posterior. Returns ``None`` when the optimizer fails
        or the Hessian is not positive-definite (e.g. ring-shaped posteriors
        with a flat tangent direction).
        """
        neg_lp = jax.jit(lambda x: -target.log_prob(x))
        neg_lp_grad = jax.jit(jax.grad(lambda x: -target.log_prob(x)))
        neg_lp_hess = jax.jit(jax.hessian(lambda x: -target.log_prob(x)))
        x0 = np.asarray(init, dtype=np.float64)
        res = spo.minimize(
            lambda x: float(neg_lp(jnp.asarray(x))),
            x0,
            jac=lambda x: np.asarray(neg_lp_grad(jnp.asarray(x)), dtype=np.float64),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if not (res.success and np.all(np.isfinite(res.x))):
            return None
        mode = np.asarray(res.x, dtype=np.float64)
        H = np.asarray(neg_lp_hess(jnp.asarray(mode)), dtype=np.float64)
        H = 0.5 * (H + H.T) + 1e-08 * np.eye(H.shape[0])
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        cov = 0.5 * (cov + cov.T)
        eigs = np.linalg.eigvalsh(cov)
        if not np.all(np.isfinite(eigs)) or np.any(eigs <= 0):
            return None
        return (mode, cov)

    def laplace_surface_z(mode, cov, a, b, *, log_scale, clip):
        """Gaussian density normalised to peak 1.0 at the mode, on the (a, b) grid."""
        A, B = np.meshgrid(a, b, indexing="xy")
        pts = np.stack([A, B], axis=-1).reshape(-1, 2)
        diff = pts - mode[None, :]
        cov_inv = np.linalg.inv(cov)
        mahal = np.einsum("ni,ij,nj->n", diff, cov_inv, diff).reshape(A.shape)
        log_ratio = np.clip(-0.5 * mahal, -clip, 0.0)
        return log_ratio if log_scale else np.exp(log_ratio)

    def gaussian_ellipse(mode, cov, scale, *, n=120):
        """Iso-Mahalanobis ellipse at distance ``scale`` from the Gaussian mode."""
        theta = np.linspace(0.0, 2.0 * np.pi, n)
        L = np.linalg.cholesky(cov)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
        pts = mode[:, None] + scale * (L @ circle)
        return (pts[0], pts[1])

    def density_figure(
        title,
        target,
        *,
        a_range=(-4.0, 4.0),
        b_range=(-4.0, 4.0),
        n=140,
        log_scale=False,
        clip=18.0,
        trace=None,
        laplace=None,
    ):
        a = np.linspace(a_range[0], a_range[1], n)
        b = np.linspace(b_range[0], b_range[1], n)
        A, B = np.meshgrid(a, b, indexing="xy")
        grid = jnp.stack([jnp.asarray(A), jnp.asarray(B)], axis=-1)
        log_p = np.asarray(target.log_prob(grid))
        log_p_max = float(np.nanmax(log_p))
        log_p_shift = log_p - log_p_max
        if log_scale:
            z = np.clip(log_p_shift, -clip, 0.0)
            z_label = "log p − max"
            z_offset = 0.3
        else:
            z = np.exp(log_p_shift)
            z_label = "p (unnorm.)"
            z_offset = 0.015
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "surface"}, {"type": "xy"}]],
            subplot_titles=("3D surface + sampler", "top-down + sampler"),
            column_widths=[0.58, 0.42],
            horizontal_spacing=0.04,
        )
        fig.add_trace(
            go.Surface(
                x=a,
                y=b,
                z=z,
                colorscale="Viridis",
                showscale=False,
                opacity=0.9,
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(x=a, y=b, z=z, colorscale="Viridis", showscale=False), row=1, col=2
        )
        if trace is not None:
            samples = np.asarray(trace.positions)
            stage = np.asarray(trace.stage)
            killed = np.asarray(trace.killed).astype(bool)
            lp_samples = np.asarray(target.log_prob(jnp.asarray(samples))) - log_p_max
            lp_samples = np.where(np.isfinite(lp_samples), lp_samples, -clip)
            if log_scale:
                z_samples = np.clip(lp_samples, -clip, 0.0) + z_offset
            else:
                z_samples = np.exp(lp_samples) + z_offset
            gradient = dict(color=stage, colorscale="Plasma", showscale=False)
            mode = "markers+lines" if trace.connect else "markers"
            line3d = (
                dict(color="rgba(220,220,220,0.35)", width=1.2) if trace.connect else dict(width=0)
            )
            line2d = (
                dict(color="rgba(150,150,150,0.45)", width=0.6) if trace.connect else dict(width=0)
            )
            fig.add_trace(
                go.Scatter3d(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    z=z_samples,
                    mode=mode,
                    marker=dict(size=2.4, opacity=0.85, **gradient),
                    line=line3d,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    mode=mode,
                    marker=dict(size=3.8, opacity=0.75, **gradient),
                    line=line2d,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            if killed.any():
                kxy = samples[killed]
                kz = z_samples[killed]
                fig.add_trace(
                    go.Scatter3d(
                        x=kxy[:, 0],
                        y=kxy[:, 1],
                        z=kz,
                        mode="markers",
                        marker=dict(size=5, color="black", symbol="x"),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=kxy[:, 0],
                        y=kxy[:, 1],
                        mode="markers",
                        marker=dict(
                            size=8, color="black", symbol="x", line=dict(color="white", width=1)
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
        if laplace is not None:
            lap_mode, lap_cov = laplace
            z_lap = laplace_surface_z(lap_mode, lap_cov, a, b, log_scale=log_scale, clip=clip)
            lap_colorscale = [[0.0, "rgba(255, 170, 60, 0.0)"], [1.0, "rgba(255, 120, 0, 1.0)"]]
            fig.add_trace(
                go.Surface(
                    x=a,
                    y=b,
                    z=z_lap,
                    colorscale=lap_colorscale,
                    cmin=float(z_lap.min()),
                    cmax=float(z_lap.max()),
                    showscale=False,
                    opacity=0.32,
                    contours=dict(z=dict(show=False)),
                    name="MAP + Laplace",
                    legendgroup="laplace",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            for scale, dash in ((1.0, "solid"), (2.0, "dot")):
                ex, ey = gaussian_ellipse(lap_mode, lap_cov, scale)
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color="rgba(255, 120, 0, 0.9)", width=2, dash=dash),
                        name="MAP + Laplace",
                        legendgroup="laplace",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            fig.add_trace(
                go.Scatter(
                    x=[lap_mode[0]],
                    y=[lap_mode[1]],
                    mode="markers",
                    marker=dict(
                        color="orange", symbol="cross", size=11, line=dict(color="black", width=1)
                    ),
                    name="MAP mode",
                    legendgroup="laplace",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.update_scenes(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.05)),
            row=1,
            col=1,
        )
        fig.update_xaxes(title="a", row=1, col=2)
        fig.update_yaxes(title="b", scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_layout(
            title=f"<b>{title}</b>",
            height=480,
            width=1150,
            margin=dict(t=70, b=40, l=20, r=20),
            legend=dict(
                x=1.0, y=1.0, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.7)"
            ),
        )
        return fig

    # z on surface for each sample (clip for out-of-support / -inf)
    def _stationary_samples(trace):
        """Live samples from the stationary distribution.

        Chain samplers (connect=True): all post-warmup iterations count.
        Particle samplers (connect=False): if ``stage`` partitions samples into
        cohorts, the final cohort is the target-distributed one; use it. When
        ``stage`` is a per-draw dummy index (Pathfinder draws are all iid from the
        approximate target), the final "cohort" is a single point, so use every
        live draw instead.
        """
        xs = np.asarray(trace.positions)
        alive = ~np.asarray(trace.killed).astype(bool)
        if not trace.connect:
            stage = np.asarray(trace.stage)
            final_mask = alive & (stage == stage.max())
            if final_mask.sum() > 1:
                return xs[final_mask]
        return xs[alive]

    def metrics_point(target, trace, *, ci=0.95, n_truth=20000, seed=42, robust=False):
        """Point-valued truth (via target.sample). Uses median instead of mean when robust=True."""
        key = jax.random.PRNGKey(seed)
        truth_samples = np.asarray(target.sample(key, n_truth))
        xs = _stationary_samples(trace)
        n = xs.shape[0]
        if n == 0:
            return "MAE=n/a · CI95 width=n/a · truth-in-CI=n/a (no live samples)"
        if robust:
            truth = np.median(truth_samples, axis=0)
            est = np.median(xs, axis=0)
            label = "median"
        else:
            truth = truth_samples.mean(axis=0)
            est = xs.mean(axis=0)
            label = "mean"
        mae = float(np.mean(np.abs(est - truth)))
        alpha = (1 - ci) / 2
        lo = np.quantile(xs, alpha, axis=0)
        hi = np.quantile(xs, 1 - alpha, axis=0)
        ci_width = float(np.mean(hi - lo))
        covered = float(np.mean((lo <= truth) & (truth <= hi)))
        return f"MAE({label})={mae:.3f} · CI95 width={ci_width:.2f} · truth-in-CI={covered:.0%} (truth≈[{truth[0]:.2f}, {truth[1]:.2f}], n={n})"

    def metrics_manifold(trace, phi, target_value, tol):
        """Manifold truth (φ(x) = target_value). MAE of |φ−τ|, 95% CI width, fraction within tol."""
        xs = _stationary_samples(trace)
        n = xs.shape[0]
        if n == 0:
            return "MAE=n/a · CI95 width=n/a · within-tol=n/a (no live samples)"
        residuals = np.asarray(phi(jnp.asarray(xs))) - target_value
        mae = float(np.mean(np.abs(residuals)))
        lo, hi = np.quantile(residuals, [0.025, 0.975])
        ci_width = float(hi - lo)
        within = float(np.mean(np.abs(residuals) <= tol))
        return f"MAE|φ−τ|={mae:.3f} · CI95 width={ci_width:.2f} · within-tol({tol:.2g})={within:.0%} (n={n})"

    def metrics_invariance(phi, target_value, tol):
        """Wrap metrics_manifold into a (target, trace) -> str recovery callable."""

        def _fn(_target, trace):
            return metrics_manifold(trace, phi, target_value, tol)

        return _fn

    def plot_with(
        title, target, cfg_overrides=None, *, recovery=None, sampler_fn=None, cfg=None, **fig_kwargs
    ):
        """Build the figure and return it; recovery metrics print to the console."""
        fn = sampler_fn or SAMPLER_FN
        active_cfg = cfg or SAMPLER_CFG
        if cfg_overrides:
            valid = {f.name for f in fields(active_cfg)}
            filtered = {k: v for k, v in cfg_overrides.items() if k in valid}
            if filtered:
                active_cfg = replace(active_cfg, **filtered)
        trace = fn(target, active_cfg)
        laplace_fit = None
        if SHOW_LAPLACE:
            lap_init = (cfg_overrides or {}).get("initial_position", (0.0, 0.0))
            laplace_fit = fit_laplace(target, lap_init)
        fig = density_figure(title, target, trace=trace, laplace=laplace_fit, **fig_kwargs)
        footer = f"{trace.summary}  ·  colour = {trace.stage_label} (Plasma)"  # 2D: 1σ (solid) + 2σ (dot) Mahalanobis ellipses + mode marker
        if SHOW_LAPLACE:
            footer += "  ·  Laplace: " + (
                "mode=({:+.2f}, {:+.2f})".format(*laplace_fit[0])
                if laplace_fit is not None
                else "fit failed"
            )
        fig.add_annotation(
            text=footer,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.05,
            showarrow=False,
            font=dict(size=11, color="#555"),
        )
        if recovery is not None:
            # recovery metrics (MAE · CI95 width · coverage) print to console
            print(recovery(target, trace))
        return fig

    return (
        fit_laplace,
        gaussian_ellipse,
        laplace_surface_z,
        metrics_invariance,
        metrics_point,
        plot_with,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Baseline — isotropic Gaussian
    """)
    return


@app.cell
def _(Gaussian, Identity, TransformedTarget, metrics_point, plot_with):
    _target = TransformedTarget(Gaussian(dim=2), Identity())
    plot_with("Gaussian (baseline)", _target, recovery=metrics_point)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Heavy tails — Student-t (ν = 2)
    """)
    return


@app.cell
def _(
    Identity,
    StudentT,
    TransformedTarget,
    metrics_point,
    partial,
    plot_with,
):
    _target = TransformedTarget(StudentT(dim=2, df=2.0), Identity())
    plot_with(
        "Student-t (heavy tails, log-scale)",
        _target,
        a_range=(-6, 6),
        b_range=(-6, 6),
        log_scale=True,
        recovery=partial(metrics_point, robust=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cauchy — pathological tails
    """)
    return


@app.cell
def _(Cauchy, Identity, TransformedTarget, metrics_point, partial, plot_with):
    _target = TransformedTarget(Cauchy(dim=2, scale=0.5), Identity())
    plot_with(
        "Cauchy (log-scale)",
        _target,
        a_range=(-4, 4),
        b_range=(-4, 4),
        log_scale=True,
        recovery=partial(metrics_point, robust=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Shear — linear correlation / elongated ridge
    """)
    return


@app.cell
def _(Gaussian, Shear, TransformedTarget, metrics_point, plot_with):
    _target = TransformedTarget(Gaussian(dim=2), Shear(theta=0.7, scale=(2.2, 0.4)))
    plot_with(
        "Shear — correlated ridge",
        _target,
        a_range=(-5, 5),
        b_range=(-5, 5),
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Bend — banana (Rosenbrock)
    """)
    return


@app.cell
def _(Bend, Gaussian, TransformedTarget, metrics_point, plot_with):
    _target = TransformedTarget(Gaussian(dim=2), Bend(f=lambda x: 0.5 * x**2 - 1.0))
    plot_with(
        "Bend — banana curvature",
        _target,
        a_range=(-3.5, 3.5),
        b_range=(-2.5, 5.0),
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Funnel — Neal's hierarchical scale
    """)
    return


@app.cell
def _(Funnel, Gaussian, TransformedTarget, metrics_point, plot_with):
    _target = TransformedTarget(Gaussian(dim=2, scale=1.2), Funnel(g=lambda x: 0.6 * x))
    plot_with(
        "Funnel (log-scale)",
        _target,
        a_range=(-3.5, 3.5),
        b_range=(-8, 8),
        log_scale=True,
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Softplus — positive-orthant support

    NUTS needs an in-support init; SMC needs a narrower prior or it wastes ~¾ of particles in the negative quadrants.
    """)
    return


@app.cell
def _(Gaussian, Softplus, TransformedTarget, metrics_point, plot_with):
    _target = TransformedTarget(Gaussian(dim=2, loc=0.0, scale=1.0), Softplus(axes=(0, 1)))
    plot_with(
        "Softplus — positive orthant",
        _target,
        cfg_overrides={"initial_position": (1.0, 1.0), "prior_scale": 1.5},
        a_range=(0.01, 4.0),
        b_range=(0.01, 4.0),
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Logit — bounded support (unit square)
    """)
    return


@app.cell
def _(Gaussian, Logit, TransformedTarget, metrics_point, plot_with):
    _target = TransformedTarget(Gaussian(dim=2, scale=1.5), Logit(axes=(0, 1)))
    plot_with(
        "Logit — bounded (0, 1)^2",
        _target,
        cfg_overrides={"initial_position": (0.5, 0.5), "prior_scale": 0.5},
        a_range=(0.01, 0.99),
        b_range=(0.01, 0.99),
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Mirror — bimodality
    """)
    return


@app.cell
def _(Gaussian, Mirror, Shift, TransformedTarget, metrics_point, plot_with):
    _shifted = TransformedTarget(Gaussian(dim=2, scale=0.4), Shift(offset=(1.4, 0.0)))
    _target = Mirror(_shifted, flip_axes=(0,))
    plot_with(
        "Mirror — two modes",
        _target,
        cfg_overrides={"initial_position": (1.4, 0.0)},
        a_range=(-3.5, 3.5),
        b_range=(-2.5, 2.5),
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Mixture — three unequal modes
    """)
    return


@app.cell
def _(Gaussian, Mixture, Shift, TransformedTarget, metrics_point, plot_with):
    def _blob(dx, dy, scale=0.4):
        return TransformedTarget(Gaussian(dim=2, scale=scale), Shift(offset=(dx, dy)))

    _target = Mixture(
        components=(_blob(1.6, 0.0), _blob(-0.8, 1.4), _blob(-0.8, -1.4)), weights=(0.5, 0.3, 0.2)
    )
    plot_with(
        "Mixture — 3 unequal modes",
        _target,
        cfg_overrides={"initial_position": (1.6, 0.0)},
        a_range=(-3.0, 3.0),
        b_range=(-3.0, 3.0),
        recovery=metrics_point,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Invariance — additive ridge (`φ = a + b`)
    """)
    return


@app.cell
def _(
    Gaussian,
    Identity,
    TransformedTarget,
    invariance,
    metrics_invariance,
    plot_with,
):
    _base = TransformedTarget(Gaussian(dim=2, scale=3.0), Identity())
    _phi = lambda x: x[..., 0] + x[..., 1]
    _target = invariance(_base, phi=_phi, target_value=0.0, tol=0.2)
    plot_with(
        "Invariance — additive (φ = a + b, log-scale)",
        _target,
        a_range=(-4, 4),
        b_range=(-4, 4),
        log_scale=True,
        recovery=metrics_invariance(_phi, 0.0, 0.2),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Invariance — hyperbolic ridge (`φ = a · b`)
    """)
    return


@app.cell
def _(
    Gaussian,
    Identity,
    TransformedTarget,
    invariance,
    metrics_invariance,
    plot_with,
):
    _base = TransformedTarget(Gaussian(dim=2, scale=3.0), Identity())
    _phi = lambda x: x[..., 0] * x[..., 1]
    _target = invariance(_base, phi=_phi, target_value=1.0, tol=0.15)
    plot_with(
        "Invariance — hyperbolic (φ = a·b = 1, log-scale)",
        _target,
        cfg_overrides={"initial_position": (1.0, 1.0)},
        a_range=(-3.5, 3.5),
        b_range=(-3.5, 3.5),
        log_scale=True,
        recovery=metrics_invariance(_phi, 1.0, 0.15),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Invariance — rotational (ring)
    """)
    return


@app.cell
def _(
    Gaussian,
    Identity,
    TransformedTarget,
    invariance,
    metrics_invariance,
    plot_with,
):
    _base = TransformedTarget(Gaussian(dim=2, scale=3.0), Identity())
    _phi = lambda x: x[..., 0] ** 2 + x[..., 1] ** 2
    _target = invariance(_base, phi=_phi, target_value=4.0, tol=0.5)
    plot_with(
        "Invariance — rotational (ring, log-scale)",
        _target,
        cfg_overrides={"initial_position": (2.0, 0.0)},
        a_range=(-3.5, 3.5),
        b_range=(-3.5, 3.5),
        log_scale=True,
        recovery=metrics_invariance(_phi, 4.0, 0.5),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Invariance — angular (`φ = atan2(b, a)`)
    """)
    return


@app.cell
def _(
    Gaussian,
    Identity,
    TransformedTarget,
    invariance,
    jnp,
    metrics_invariance,
    np,
    plot_with,
):
    _base = TransformedTarget(Gaussian(dim=2, scale=3.0), Identity())
    _phi = lambda x: jnp.arctan2(x[..., 1], x[..., 0])
    _target = invariance(_base, phi=_phi, target_value=float(np.pi / 4), tol=0.1)
    plot_with(
        "Invariance — angular (φ = atan2(b, a) = π/4, log-scale)",
        _target,
        cfg_overrides={"initial_position": (1.5, 1.5)},
        a_range=(-4, 4),
        b_range=(-4, 4),
        log_scale=True,
        recovery=metrics_invariance(_phi, float(np.pi / 4), 0.1),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Kitchen sink — stack them all
    """)
    return


@app.cell
def _(
    Bend,
    Chain,
    Funnel,
    Mirror,
    Shear,
    Shift,
    StudentT,
    TransformedTarget,
    invariance,
    metrics_invariance,
    plot_with,
):
    warped = TransformedTarget(
        StudentT(dim=2, df=4.0),
        Chain(
            (
                Bend(f=lambda x: 0.3 * x**2 - 0.5),
                Funnel(g=lambda x: 0.25 * x),
                Shear(theta=0.25, scale=(1.2, 0.9)),
                Shift(offset=(0.6, 0.0)),
            )
        ),
    )
    bimodal = Mirror(warped, flip_axes=(0,))
    _phi = lambda x: x[..., 0] * x[..., 1]
    _target = invariance(bimodal, phi=_phi, target_value=0.4, tol=0.6)
    plot_with(
        "Kitchen sink: heavy tails + banana + funnel + shear + mirror + hyperbolic invariance",
        _target,
        cfg_overrides={"initial_position": (0.8, 0.5)},
        a_range=(-3.5, 3.5),
        b_range=(-4, 4),
        log_scale=True,
        recovery=metrics_invariance(_phi, 0.4, 0.6),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Appendix: Pathfinder internals

    Pathfinder fits the sampler abstraction above — set `SAMPLER_FN = run_pathfinder_sampler` in the imports cell to swap it into the same 15 gallery panels. That view only shows the final draws, though; what makes Pathfinder interesting is *how it produces those draws*:

    1. Run L-BFGS. Each iterate carries a local Gaussian built from the inverse-Hessian estimate.
    2. Score every iterate by ELBO; pick the argmax.
    3. Draw importance-resampled samples from the chosen Gaussian.

    The cells below use the richer `run_pathfinder` entry point (returns a `PathfinderTrace`) so we can plot the L-BFGS trajectory, flag `ELBO = −∞` iterates, mark the chosen iterate, and show the ELBO curve alongside the target density.
    """)
    return


@app.cell
def _(
    PathfinderConfig,
    SHOW_LAPLACE,
    fields,
    fit_laplace,
    gaussian_ellipse,
    go,
    jax,
    jnp,
    laplace_surface_z,
    make_subplots,
    np,
    replace,
    run_pathfinder,
):
    def pathfinder_figure(
        title,
        target,
        trace,
        *,
        a_range=(-4.0, 4.0),
        b_range=(-4.0, 4.0),
        n=140,
        log_scale=False,
        clip=18.0,
        laplace=None,
    ):
        a = np.linspace(a_range[0], a_range[1], n)
        b = np.linspace(b_range[0], b_range[1], n)
        A, B = np.meshgrid(a, b, indexing="xy")
        grid = jnp.stack([jnp.asarray(A), jnp.asarray(B)], axis=-1)
        log_p = np.asarray(target.log_prob(grid))
        log_p_max = float(np.nanmax(log_p))
        log_p_shift = log_p - log_p_max
        if log_scale:
            z = np.clip(log_p_shift, -clip, 0.0)
            z_label = "log p − max"
            z_offset = 0.3
        else:
            z = np.exp(log_p_shift)
            z_label = "p (unnorm.)"
            z_offset = 0.015
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "surface"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=("3D surface + path", "top-down + draws", "ELBO vs L-BFGS iter"),
            column_widths=[0.42, 0.34, 0.24],
            horizontal_spacing=0.04,
        )
        fig.add_trace(
            go.Surface(
                x=a,
                y=b,
                z=z,
                colorscale="Viridis",
                showscale=False,
                opacity=0.9,
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(x=a, y=b, z=z, colorscale="Viridis", showscale=False), row=1, col=2
        )
        path = np.asarray(trace.path_positions)
        elbo = np.asarray(trace.path_elbo)
        finite = np.isfinite(elbo)
        stage = np.arange(path.shape[0])
        lp_path = np.asarray(target.log_prob(jnp.asarray(path))) - log_p_max
        lp_path = np.where(np.isfinite(lp_path), lp_path, -clip)
        z_path = (np.clip(lp_path, -clip, 0.0) if log_scale else np.exp(lp_path)) + z_offset
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=z_path,
                mode="markers+lines",
                marker=dict(
                    size=3.2, opacity=0.95, color=stage, colorscale="Plasma", showscale=False
                ),
                line=dict(color="rgba(220,220,220,0.55)", width=2),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="markers+lines",
                marker=dict(size=5, opacity=0.9, color=stage, colorscale="Plasma", showscale=False),
                line=dict(color="rgba(140,140,140,0.6)", width=1.0),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        if (~finite).any():
            bad = path[~finite]
            fig.add_trace(
                go.Scatter(
                    x=bad[:, 0],
                    y=bad[:, 1],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color="rgba(90,90,90,0.9)",
                        symbol="x",
                        line=dict(color="white", width=1),
                    ),
                    name="ELBO = −∞",
                    showlegend=True,
                ),
                row=1,
                col=2,
            )
        draws = np.asarray(trace.draws)
        lp_draws = np.asarray(target.log_prob(jnp.asarray(draws))) - log_p_max
        lp_draws = np.where(np.isfinite(lp_draws), lp_draws, -clip)
        z_draws = (np.clip(lp_draws, -clip, 0.0) if log_scale else np.exp(lp_draws)) + z_offset
        fig.add_trace(
            go.Scatter3d(
                x=draws[:, 0],
                y=draws[:, 1],
                z=z_draws,
                mode="markers",
                marker=dict(size=2.2, opacity=0.55, color="rgba(80,180,230,0.9)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=draws[:, 0],
                y=draws[:, 1],
                mode="markers",
                marker=dict(size=3.2, opacity=0.55, color="rgba(80,180,230,0.9)"),
                name="Pathfinder draws",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        best = int(trace.best_iter)
        fig.add_trace(
            go.Scatter3d(
                x=[path[best, 0]],
                y=[path[best, 1]],
                z=[z_path[best] + z_offset],
                mode="markers",
                marker=dict(
                    size=7, color="limegreen", symbol="diamond", line=dict(color="black", width=1)
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[path[best, 0]],
                y=[path[best, 1]],
                mode="markers",
                marker=dict(
                    size=14, color="limegreen", symbol="star", line=dict(color="black", width=1)
                ),
                name=f"chosen iter {best}",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        if laplace is not None:
            lap_mode, lap_cov = laplace
            z_lap = laplace_surface_z(lap_mode, lap_cov, a, b, log_scale=log_scale, clip=clip)
            lap_cs = [[0.0, "rgba(255, 170, 60, 0.0)"], [1.0, "rgba(255, 120, 0, 1.0)"]]
            fig.add_trace(
                go.Surface(
                    x=a,
                    y=b,
                    z=z_lap,
                    colorscale=lap_cs,
                    cmin=float(z_lap.min()),
                    cmax=float(z_lap.max()),
                    showscale=False,
                    opacity=0.28,
                    contours=dict(z=dict(show=False)),
                    name="MAP + Laplace",
                    legendgroup="laplace",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            for scale, dash in ((1.0, "solid"), (2.0, "dot")):
                ex, ey = gaussian_ellipse(lap_mode, lap_cov, scale)
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color="rgba(255, 120, 0, 0.9)", width=2, dash=dash),
                        legendgroup="laplace",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
        elbo_display = np.where(finite, elbo, np.nan)
        fig.add_trace(
            go.Scatter(
                x=stage,
                y=elbo_display,
                mode="markers+lines",
                marker=dict(size=6, color=stage, colorscale="Plasma", showscale=False),
                line=dict(color="rgba(140,140,140,0.6)", width=1),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        if finite[best]:
            fig.add_trace(
                go.Scatter(
                    x=[best],
                    y=[float(elbo[best])],
                    mode="markers",
                    marker=dict(
                        size=12, color="limegreen", symbol="star", line=dict(color="black", width=1)
                    ),
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
        fig.update_scenes(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.05)),
            row=1,
            col=1,
        )
        fig.update_xaxes(title="a", row=1, col=2)
        fig.update_yaxes(title="b", scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_xaxes(title="L-BFGS iter", row=1, col=3)
        fig.update_yaxes(title="ELBO", row=1, col=3)
        fig.update_layout(
            title=f"<b>{title}</b>",
            height=480,
            width=1300,
            margin=dict(t=70, b=40, l=20, r=20),
            legend=dict(
                x=1.0, y=1.0, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.7)"
            ),
        )
        return fig

    def metrics_on_draws(target, draws, *, ci=0.95, n_truth=20000, seed=42, robust=False):
        key = jax.random.PRNGKey(seed)
        truth_samples = np.asarray(target.sample(key, n_truth))
        xs = np.asarray(draws)
        n = xs.shape[0]
        if robust:
            truth = np.median(truth_samples, axis=0)
            est = np.median(xs, axis=0)
            lab = "median"
        else:
            truth = truth_samples.mean(axis=0)
            est = xs.mean(axis=0)
            lab = "mean"
        mae = float(np.mean(np.abs(est - truth)))
        alpha = (1 - ci) / 2
        lo = np.quantile(xs, alpha, axis=0)
        hi = np.quantile(xs, 1 - alpha, axis=0)
        ci_width = float(np.mean(hi - lo))
        covered = float(np.mean((lo <= truth) & (truth <= hi)))
        return f"MAE({lab})={mae:.3f} · CI95 width={ci_width:.2f} · truth-in-CI={covered:.0%} (truth≈[{truth[0]:.2f}, {truth[1]:.2f}], n={n})"

    PATHFINDER_DEFAULT = PathfinderConfig()

    def plot_pathfinder(title, target, cfg_overrides=None, *, robust=False, cfg=None, **fig_kwargs):
        active_cfg = cfg or PATHFINDER_DEFAULT
        if cfg_overrides:
            valid = {f.name for f in fields(active_cfg)}
            filtered = {k: v for k, v in cfg_overrides.items() if k in valid}
            if filtered:
                active_cfg = replace(active_cfg, **filtered)
        trace = run_pathfinder(target, active_cfg)
        laplace_fit = None
        if SHOW_LAPLACE:
            lap_init = (cfg_overrides or {}).get("initial_position", (0.0, 0.0))
            laplace_fit = fit_laplace(target, lap_init)
        fig = pathfinder_figure(title, target, trace, laplace=laplace_fit, **fig_kwargs)
        footer = f"{trace.summary}  ·  path colour = L-BFGS iter (Plasma)"
        fig.add_annotation(
            text=footer,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.05,
            showarrow=False,
            font=dict(size=11, color="#555"),
        )
        print(metrics_on_draws(target, trace.draws, robust=robust))
        return fig

    return metrics_on_draws, plot_pathfinder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A. Bend (banana) — one Gaussian can't track curvature

    Watch L-BFGS descend into the trough then oscillate between iterates whose ELBO cannot improve. The chosen ★ sits near the mode but its draws ignore the banana's tails.
    """)
    return


@app.cell
def _(Bend, Gaussian, TransformedTarget, plot_pathfinder):
    _target = TransformedTarget(Gaussian(dim=2), Bend(f=lambda x: 0.5 * x**2 - 1.0))
    plot_pathfinder(
        "Bend — banana curvature",
        _target,
        cfg_overrides={"initial_position": (2.5, 2.5)},
        a_range=(-3.5, 3.5),
        b_range=(-2.5, 5.0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B. Funnel — ELBO peaks *before* L-BFGS reaches the neck

    The chosen iterate is typically mid-trajectory, not the final converged point — Pathfinder explicitly trades off mode-finding for better Gaussian fit.
    """)
    return


@app.cell
def _(Funnel, Gaussian, TransformedTarget, plot_pathfinder):
    _target = TransformedTarget(Gaussian(dim=2, scale=1.2), Funnel(g=lambda x: 0.6 * x))
    plot_pathfinder(
        "Funnel (log-scale)", _target, a_range=(-3.5, 3.5), b_range=(-8, 8), log_scale=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Appendix: Multi-path Pathfinder

    Single-path Pathfinder collapses onto one mode. **Multi-path Pathfinder** ([Algorithm 4, Zhang et al. 2022](https://arxiv.org/abs/2108.03782)) fixes this:

    1. Run `K` L-BFGS paths from an overdispersed distribution over starts.
    2. Each path picks its best-ELBO Gaussian `q_k`.
    3. Draw `M` samples per path → candidate pool of size `K·M`.
    4. Weight candidates by `log p(x) − log q_mix(x)` with `q_mix = (1/K) Σ_k q_k`.
    5. **Pareto-smooth** the tail of importance weights (PSIS) and resample. The summary's `k̂` flags tails too heavy for IS (`k̂ > 0.7` ⇒ untrustworthy).

    `run_multipath_pathfinder_sampler` swaps into the main gallery; `run_multipath_pathfinder` returns the richer trace used below.
    """)
    return


@app.cell
def _(
    MultiPathfinderConfig,
    SHOW_LAPLACE,
    fields,
    fit_laplace,
    gaussian_ellipse,
    go,
    jnp,
    make_subplots,
    metrics_on_draws,
    np,
    replace,
    run_multipath_pathfinder,
):
    def multipath_figure(
        title,
        target,
        trace,
        *,
        a_range=(-4.0, 4.0),
        b_range=(-4.0, 4.0),
        n=140,
        log_scale=False,
        clip=18.0,
        laplace=None,
    ):
        a = np.linspace(a_range[0], a_range[1], n)
        b = np.linspace(b_range[0], b_range[1], n)
        A, B = np.meshgrid(a, b, indexing="xy")
        grid = jnp.stack([jnp.asarray(A), jnp.asarray(B)], axis=-1)
        log_p = np.asarray(target.log_prob(grid))
        log_p_max = float(np.nanmax(log_p))
        log_p_shift = log_p - log_p_max
        if log_scale:
            z = np.clip(log_p_shift, -clip, 0.0)
            z_label = "log p − max"
        else:
            z = np.exp(log_p_shift)
            z_label = "p (unnorm.)"
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=("K L-BFGS paths + per-path Gaussians", "PSIR draws over target"),
            column_widths=[0.5, 0.5],
            horizontal_spacing=0.06,
        )
        fig.add_trace(
            go.Heatmap(x=a, y=b, z=z, colorscale="Viridis", showscale=False), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(x=a, y=b, z=z, colorscale="Viridis", showscale=False), row=1, col=2
        )
        path_positions = np.asarray(trace.path_positions)
        path_elbo = np.asarray(trace.path_elbo)
        K = path_positions.shape[0]
        colors = [f"hsl({int(360 * k / K)},70%,50%)" for k in range(K)]
        for k in range(K):
            pk = path_positions[k]
            finite = np.isfinite(path_elbo[k])
            if not finite.all():
                first_bad = int(np.argmin(finite))
                pk_draw = pk[: first_bad + 1] if first_bad > 0 else pk
            else:
                pk_draw = pk
            fig.add_trace(
                go.Scatter(
                    x=pk_draw[:, 0],
                    y=pk_draw[:, 1],
                    mode="markers+lines",
                    marker=dict(size=4, color=colors[k]),
                    line=dict(color=colors[k], width=1.5),
                    legendgroup=f"p{k}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[pk[0, 0]],
                    y=[pk[0, 1]],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=colors[k],
                        symbol="circle-open",
                        line=dict(color="white", width=2),
                    ),
                    legendgroup=f"p{k}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        best_iters = np.asarray(trace.best_iter)
        for k in range(K):
            if not np.isfinite(trace.path_best_elbo[k]):
                continue
            mask = np.asarray(trace.candidate_path_id) == k
            pts = np.asarray(trace.candidate_positions)[mask]
            if pts.shape[0] < 3:
                continue
            mu = pts.mean(axis=0)
            C = np.cov(pts.T)
            try:
                ex, ey = gaussian_ellipse(mu, C, 1.0)
            except np.linalg.LinAlgError:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(color=colors[k], width=1.8, dash="dot"),
                    legendgroup=f"p{k}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            b_idx = int(best_iters[k])
            fig.add_trace(
                go.Scatter(
                    x=[path_positions[k, b_idx, 0]],
                    y=[path_positions[k, b_idx, 1]],
                    mode="markers",
                    marker=dict(
                        size=12, color=colors[k], symbol="star", line=dict(color="black", width=1)
                    ),
                    legendgroup=f"p{k}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        cands = np.asarray(trace.candidate_positions)
        fig.add_trace(
            go.Scatter(
                x=cands[:, 0],
                y=cands[:, 1],
                mode="markers",
                marker=dict(size=2.2, opacity=0.15, color="white"),
                name="candidates (pre-PSIR)",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        draws = np.asarray(trace.draws)
        fig.add_trace(
            go.Scatter(
                x=draws[:, 0],
                y=draws[:, 1],
                mode="markers",
                marker=dict(
                    size=4.0,
                    opacity=0.7,
                    color="rgba(80,180,230,1)",
                    line=dict(color="white", width=0.3),
                ),
                name="PSIR draws",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        if laplace is not None:
            lap_mode, lap_cov = laplace
            for scale, dash in ((1.0, "solid"), (2.0, "dot")):
                ex, ey = gaussian_ellipse(lap_mode, lap_cov, scale)
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color="rgba(255,120,0,0.85)", width=2, dash=dash),
                        name="MAP + Laplace",
                        legendgroup="laplace",
                        showlegend=scale == 1.0,
                    ),
                    row=1,
                    col=2,
                )
        fig.update_xaxes(title="a", row=1, col=1)
        fig.update_yaxes(title="b", scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_xaxes(title="a", row=1, col=2)
        fig.update_yaxes(title="b", scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_layout(
            title=f"<b>{title}</b>  <i>({z_label})</i>",
            height=520,
            width=1200,
            margin=dict(t=70, b=40, l=20, r=20),
            legend=dict(
                x=1.0, y=1.0, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.7)"
            ),
        )
        return fig

    MULTIPATH_DEFAULT = MultiPathfinderConfig()

    def plot_multipath(title, target, cfg_overrides=None, *, robust=False, cfg=None, **fig_kwargs):
        active_cfg = cfg or MULTIPATH_DEFAULT
        if cfg_overrides:
            valid = {f.name for f in fields(active_cfg)}
            filtered = {k: v for k, v in cfg_overrides.items() if k in valid}
            if filtered:
                active_cfg = replace(active_cfg, **filtered)
        trace = run_multipath_pathfinder(target, active_cfg)
        laplace_fit = None
        if SHOW_LAPLACE:
            lap_init = tuple(active_cfg.init_center)
            laplace_fit = fit_laplace(target, lap_init)
        fig = multipath_figure(title, target, trace, laplace=laplace_fit, **fig_kwargs)
        fig.add_annotation(
            text=trace.summary,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            showarrow=False,
            font=dict(size=11, color="#555"),
        )
        print(metrics_on_draws(target, trace.draws, robust=robust))
        return fig

    return (plot_multipath,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### C. Mirror — both modes recovered

    With `K=10` dispersed starts, ≥1 path lands near each mode. PSIR covers both lobes roughly 50/50 (the true weights).
    """)
    return


@app.cell
def _(Gaussian, Mirror, Shift, TransformedTarget, plot_multipath):
    _shifted = TransformedTarget(Gaussian(dim=2, scale=0.4), Shift(offset=(1.4, 0.0)))
    _target = Mirror(_shifted, flip_axes=(0,))
    plot_multipath(
        "Mirror — multi-path (K=10)",
        _target,
        cfg_overrides={"num_paths": 10, "init_scale": 1.5, "seed": 1},
        a_range=(-3.5, 3.5),
        b_range=(-2.5, 2.5),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### D. Mixture — three modes, coverage ~ true weights

    `K=12`, `init_scale=1.8` seeds enough starts in each basin. PSIR coverage tracks the 0.5 / 0.3 / 0.2 mixture weights.
    """)
    return


@app.cell
def _(Gaussian, Mixture, Shift, TransformedTarget, plot_multipath):
    def _blob(dx, dy, scale=0.4):
        return TransformedTarget(Gaussian(dim=2, scale=scale), Shift(offset=(dx, dy)))

    _target = Mixture(
        components=(_blob(1.6, 0.0), _blob(-0.8, 1.4), _blob(-0.8, -1.4)), weights=(0.5, 0.3, 0.2)
    )
    plot_multipath(
        "Mixture — multi-path (K=12)",
        _target,
        cfg_overrides={"num_paths": 12, "init_scale": 1.8, "seed": 3},
        a_range=(-3.0, 3.0),
        b_range=(-3.0, 3.0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E. Funnel — scale mixture captures more than a single Gaussian

    Different paths settle at different points along the funnel, giving a mixture of wide and narrow Gaussians in the candidate pool. PSIR then blends them; watch `k̂` to decide whether the fit is trustworthy.
    """)
    return


@app.cell
def _(Funnel, Gaussian, TransformedTarget, plot_multipath):
    _target = TransformedTarget(Gaussian(dim=2, scale=1.2), Funnel(g=lambda x: 0.6 * x))
    plot_multipath(
        "Funnel — multi-path (K=12)",
        _target,
        cfg_overrides={"num_paths": 12, "init_scale": 2.5, "seed": 0},
        a_range=(-3.5, 3.5),
        b_range=(-8, 8),
        log_scale=True,
    )
    return


if __name__ == "__main__":
    app.run()
