import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    from dataclasses import dataclass, replace

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    return dataclass, jax, jnp, np, plt, replace


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Gradual model building along causal arrows — a Stage 4 rebuild lab (D = 3)

    **The proposal.** Stage 4 should not be "elicit a full model, then validate it". It should
    be a *build loop*: admit one construct at a time, following the causal arrows
    (cause → effect → observation), and gate every admission with prior-predictive checks run
    on the **cumulative partial model** using **exact simulation**. A node is admitted together
    with its self-dynamics prior, its incoming-edge priors, and the likelihood for its
    indicators. A red check sends the LLM back to revise *that fragment* — not the whole model.

    **Why order by causal arrows.** In a DAG-structured drift, the marginal law of every
    already-admitted node is *invariant* to anything added downstream. So checks compose
    monotonically: a green stage stays green forever, and a red stage points at exactly one
    fragment. We verify this invariance path-by-path below.

    **Why this matters for us.** The worst failures we have hit were prior-predictive failures
    with terrible attribution: a manifest-mean prior 43σ from the data, a loading 9σ out — both
    discovered only after a failed fit. The existing gates
    (the fit-boundary preflight and the full-model prior-predictive suite) run at the *end*,
    on the *whole* model. Staging moves them to admission time and makes every failure local.

    **This lab** prototypes the workflow from scratch on a D = 3 chain
    (sleep → stress → mood, one indicator each): the check battery, the happy path, the
    invariance proof, and a gallery of four realistic LLM prior errors — each caught at its own
    stage, with the feedback the LLM would receive.
    """)
    return


@app.cell
def workflow_diagram(mo, plt):
    from matplotlib.patches import FancyBboxPatch

    _fig, _ax = plt.subplots(figsize=(11.0, 3.4))

    def _box(x, y, w, h, label, fc, ec, fs=9.0, tc="black"):
        _ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.6,
            )
        )
        _ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=tc)

    def _arrow(p0, p1, color, ls="-", rad=0.0):
        _ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.7,
                linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    _green, _gray, _red, _blue = "#4a9d5b", "#8a8a8a", "#c0504d", "#3b6ea5"
    _stages = [
        (0.0, "admit CAUSE\nself-dynamics prior\n+ emission prior"),
        (2.6, "admit EFFECT\n+ incoming edge prior\n+ emission prior"),
        (5.2, "admit next EFFECT\n+ incoming edge prior\n+ emission prior"),
    ]
    for _x, _label in _stages:
        _box(_x, 1.0, 1.9, 1.0, _label, "white", _green, tc=_green)
        _box(
            _x,
            -0.55,
            1.9,
            0.62,
            "checks C1–C6\n(exact prior predictive)",
            "white",
            _gray,
            fs=8.0,
            tc=_gray,
        )
        _arrow((_x, 0.48), (_x, -0.22), _gray)
        _arrow((_x + 0.72, -0.55), (_x + 1.02, 0.62), _red, ls="--", rad=-0.35)
    _ax.text(1.15, -1.25, "revise fragment ⟲", color=_red, fontsize=8.5, ha="center")
    _arrow((1.0, 1.0), (1.62, 1.0), _green)
    _arrow((3.6, 1.0), (4.22, 1.0), _green)
    _box(
        7.6,
        1.0,
        2.1,
        1.0,
        "full-model gate\n(existing preflight +\nprior-predictive suite)",
        "white",
        _blue,
        tc=_blue,
    )
    _arrow((6.2, 1.0), (6.52, 1.0), _green)
    _ax.text(8.9, 1.0, "→ fit()", fontsize=11, va="center", color=_blue, fontweight="bold")
    _ax.set_xlim(-1.3, 9.6)
    _ax.set_ylim(-1.6, 1.9)
    _ax.axis("off")
    _ax.set_title(
        "Stage 4 as a staged admission machine: fragment in, checks gate, red loops back to the fragment",
        fontsize=11.5,
        fontweight="bold",
    )
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def world_md(mo):
    mo.md(r"""
    ## 1. The true world and the observed data

    Following the exact-ground-truth convention: we first build the world the checks will be
    run against. Three latent constructs on a continuous-time nonlinear SSM, drift given by a
    node potential (gradient of a quadratic + quartic well — the NodePotential primitive) plus
    linear causal couplings:

    - **sleep** — OU-like well, no parents
    - **stress** — well with quartic self-limitation, driven by sleep (negative edge)
    - **mood** — well, driven by stress (negative edge)

    Each construct has one indicator, deliberately heterogeneous:

    - sleep_quality — continuous, identity link, small noise
    - stress_report — a 0–100 self-report: identity link with loading ≈ 14, intercept ≈ 55
    - mood_slider — a 0–100 slider through a **saturating sigmoid link** (the nonlinear emission)

    We observe ≈ 48 irregular time points over 60 days (daily prompts with jitter, 20 %
    missingness) — the usual ILD regime.
    """)
    return


@app.cell
def true_world(jnp):
    params_true = {
        "stiff": jnp.array([[1.0, 1.5, 1.0]]),
        "center": jnp.array([[0.0, 0.0, 0.0]]),
        "quart": jnp.array([[0.0, 0.3, 0.0]]),
        "sigma": jnp.array([[1.2, 1.5, 1.1]]),
        "x0": jnp.array([[0.3, -0.2, 0.1]]),
        "W": jnp.array([[[0.0, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, -0.9, 0.0]]]),
    }
    true_emissions = {
        "sleep_quality": ("identity", 1.0, 0.0, 0.4, 0),
        "stress_report": ("identity", 14.0, 55.0, 8.0, 1),
        "mood_slider": ("sigmoid100", 0.9, 0.2, 5.0, 2),
    }
    return params_true, true_emissions


@app.cell
def dataset(T_GRID, WORLD_KEY, jax, np, params_true, simulate_latents, true_emissions):
    _rng = np.random.default_rng(11)
    _days = np.arange(60) + 0.5 + _rng.uniform(-0.3, 0.3, size=60)
    _keep = _rng.random(60) < 0.8
    obs_times = np.sort(_days[_keep])
    obs_idx = np.round(obs_times / float(T_GRID[1] - T_GRID[0])).astype(int)

    lat_true = simulate_latents(WORLD_KEY, params_true, T_GRID)

    data = {}
    for _j, (_name, (_link, _lam, _b, _sig, _d)) in enumerate(true_emissions.items()):
        _x = lat_true[0, obs_idx, _d]
        _xi = _lam * _x + _b
        _mean = 100.0 * jax.nn.sigmoid(_xi) if _link == "sigmoid100" else _xi
        _noise = _sig * jax.random.normal(jax.random.fold_in(WORLD_KEY, 9000 + _j), _x.shape)
        data[_name] = np.asarray(_mean + _noise)
    return data, lat_true, obs_idx, obs_times


@app.cell
def data_plot(data, mo, obs_times, plt):
    _fig, _axs = plt.subplots(3, 1, figsize=(9.5, 5.4), sharex=True)
    _colors = {"sleep_quality": "#3b6ea5", "stress_report": "#c0504d", "mood_slider": "#4a9d5b"}
    for _ax, (_name, _y) in zip(_axs, data.items(), strict=True):
        _ax.scatter(obs_times, _y, s=14, color=_colors[_name])
        _ax.set_ylabel(_name, fontsize=9)
        _ax.spines[["top", "right"]].set_visible(False)
    _axs[-1].set_xlabel("day")
    _fig.suptitle(
        "The observed dataset: three indicators on wildly different scales, irregular times",
        fontsize=11.5,
        fontweight="bold",
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def engine_md(mo):
    mo.md(r"""
    ## 2. The build vocabulary and the exact engine

    A **fragment** is what one admission step contributes, and it mirrors the codebase's
    dynamics vocabulary one-to-one:

    - NodeFragment — priors over the node-potential parameters (stiffness, center, quartic
      self-limitation), the diffusion scale, and the initial state
      (↔ NodePotentialSpec + diffusion block)
    - EdgeFragment — prior over a linear drift coupling from an already-admitted parent
      (↔ LinearEdgeSpec; a Hill edge would slot in the same way)
    - EmissionFragment — link, loading, intercept, and noise priors for one indicator
      (↔ a row of the loading/means/noise blocks)

    A **BuildState** is just the tuple of admitted fragments. Simulation of the partial model
    is **Euler–Maruyama over the true nonlinear drift**, vectorized over prior draws — the
    exact engine, per the linearization-init-only policy. No linearized shortcut appears
    anywhere in the checks.

    One deliberate implementation detail: parameter draws and Wiener noise are keyed **per
    node**, so the same node sees the same randomness in every stage it participates in. That
    is what lets us verify upstream invariance exactly, below.
    """)
    return


@app.cell
def prior_helpers(jax, jnp):
    def sample_prior(key, spec, n):
        kind, a, b = spec
        if kind == "normal":
            return a + b * jax.random.normal(key, (n,))
        if kind == "lognormal":
            return jnp.exp(a + b * jax.random.normal(key, (n,)))
        if kind == "halfnormal":
            return jnp.abs(b * jax.random.normal(key, (n,)))
        if kind == "delta":
            return jnp.full((n,), a)
        raise ValueError(f"unknown prior kind: {kind}")

    return (sample_prior,)


@app.cell
def fragment_types(dataclass):
    @dataclass(frozen=True)
    class EdgeFragment:
        parent: str
        weight: tuple[str, float, float]

    @dataclass(frozen=True)
    class EmissionFragment:
        indicator: str
        link: str  # "identity" | "sigmoid100"
        loading: tuple[str, float, float]
        intercept: tuple[str, float, float]
        noise: tuple[str, float, float]

    @dataclass(frozen=True)
    class NodeFragment:
        name: str
        stiffness: tuple[str, float, float]
        center: tuple[str, float, float]
        quartic: tuple[str, float, float]
        diffusion: tuple[str, float, float]
        x0: tuple[str, float, float]
        edges_in: tuple[EdgeFragment, ...] = ()
        emission: EmissionFragment | None = None

    return EdgeFragment, EmissionFragment, NodeFragment


@app.cell
def sim_engine(jax, jnp, sample_prior):
    N_DRAWS = 200
    T_GRID = jnp.linspace(0.0, 60.0, 1201)
    ADMIT_KEY = jax.random.key(7)
    WORLD_KEY = jax.random.key(123)

    def draw_params(key, nodes, n):
        _dim = len(nodes)
        _idx = {nd.name: i for i, nd in enumerate(nodes)}

        def _stack(specs, slot):
            return jnp.stack(
                [
                    sample_prior(jax.random.fold_in(key, 10 * i + slot), spec, n)
                    for i, spec in enumerate(specs)
                ],
                axis=1,
            )

        _w = jnp.zeros((n, _dim, _dim))
        for _j, _nd in enumerate(nodes):
            for _e in _nd.edges_in:
                _i = _idx[_e.parent]
                _draw = sample_prior(jax.random.fold_in(key, 1000 + 37 * _j + _i), _e.weight, n)
                _w = _w.at[:, _j, _i].set(_draw)
        return {
            "stiff": _stack([nd.stiffness for nd in nodes], 0),
            "center": _stack([nd.center for nd in nodes], 1),
            "quart": _stack([nd.quartic for nd in nodes], 2),
            "sigma": _stack([nd.diffusion for nd in nodes], 3),
            "x0": _stack([nd.x0 for nd in nodes], 4),
            "W": _w,
        }

    def simulate_latents(key, params, t_grid):
        n, _dim = params["x0"].shape
        _dt = float(t_grid[1] - t_grid[0])
        _steps = t_grid.shape[0] - 1
        _eps = jnp.stack(
            [
                jax.random.normal(jax.random.fold_in(key, 5000 + d), (n, _steps))
                for d in range(_dim)
            ],
            axis=-1,
        )

        def _drift(x, p):
            _dev = x - p["center"]
            return -(p["stiff"] * _dev + p["quart"] * _dev**3) + p["W"] @ x

        def _rollout(p, eps_i):
            def _step(x, e):
                _xn = x + _drift(x, p) * _dt + p["sigma"] * (_dt**0.5) * e
                return _xn, _xn

            _, _xs = jax.lax.scan(_step, p["x0"], eps_i)
            return jnp.concatenate([p["x0"][None, :], _xs], axis=0)

        return jax.vmap(_rollout)(params, _eps)

    def emission_mean(em):
        if em.link == "identity":

            def _m(x, lam, b):
                return lam * x + b

        elif em.link == "sigmoid100":

            def _m(x, lam, b):
                return 100.0 * jax.nn.sigmoid(lam * x + b)

        else:
            raise ValueError(f"unknown link: {em.link}")
        return _m

    return ADMIT_KEY, N_DRAWS, T_GRID, WORLD_KEY, draw_params, emission_mean, simulate_latents


@app.cell(hide_code=True)
def checks_md(mo):
    mo.md(r"""
    ## 3. The check battery

    Every admission runs the applicable subset, all computed from exact prior-predictive
    simulation of the cumulative partial model:

    - **C1 confinement** — no non-finite values, and essentially no trajectories escaping the
      plausible region. Replaces eigenvalue-based "drift stability": for a nonlinear
      node-potential drift, linearized stability is wrong in both directions (zero stiffness
      with positive quartic is confined; the linearization calls it neutral/unstable).
    - **C2 latent scale** — stationary sd of the new node inside the standardized-latent
      convention band. Latents have no data anchor, so this check is *conventional* — which is
      precisely why every construct that has indicators should get its emission at admission
      time (the data anchor arrives with C5/C6, not at the end of the build).
    - **C3 timescale vs cadence** — the prior-implied relaxation time (autocorrelation 1/e
      crossing, measured on simulated paths) must sit between the observation cadence and the
      study length. Too fast → the process is white noise between prompts and the dynamics are
      unidentifiable; too slow → the study window sees a constant.
    - **C4 edge detectability** — with common random numbers, simulate the new node with its
      incoming edges on vs off: the share of stationary variance contributed by the parents.
      The gate is on the *upper* quantile of that share: a prior is allowed to include small
      effects, but a prior whose 90th percentile share is still negligible *forces*
      undetectability — the edge posterior can only echo the prior, and any downstream effect
      estimate is fake precision. The median guards the other side (a parent that overwhelms
      the node's own dynamics and the scale convention).
    - **C5 emission scale adequacy** — observed indicator quantiles must fall inside the pooled
      prior-predictive band (location reachability), and the prior-predictive spread must be
      within a sane ratio of the data spread (width).
    - **C6 emission sensitivity — "the Jacobian check"** — the derivative of the emission mean
      map, evaluated pointwise along exact prior-predictive latent paths:
      median |∂m/∂x| · sd(x) / σ_noise. This is the prior-predictive signal-to-noise of the
      *link*: it catches saturating links (sigmoid pushed into its flat region ⇒ derivative ≈ 0
      ⇒ observations carry no local information about the latent) that C5 is structurally
      blind to, because saturation makes the predictive band *maximally wide* and coverage
      passes.

    **Which Jacobian, and the linearization policy.** C6 differentiates the *emission map* and
    evaluates it pointwise under exact simulation — a sensitivity functional of the exact
    model. Nothing is linearized and no surrogate stands in for the model, so the
    init-only-linearization policy is untouched. What we deliberately do *not* do is gate on
    eigenvalues of a linearized drift — C1 covers stability by simulation instead.
    """)
    return


@app.cell
def checks_engine(dataclass, jax, jnp, np):
    @dataclass(frozen=True)
    class CheckResult:
        check: str
        target: str
        value: str
        band: str
        passed: bool
        note: str

    def check_confinement(name, x):
        _nonfinite = float(jnp.mean(~jnp.isfinite(x)))
        _explode = float(jnp.mean(jnp.max(jnp.abs(x), axis=1) > 10.0))
        _ok = _nonfinite == 0.0 and _explode < 0.01
        return CheckResult(
            "C1 confinement",
            name,
            f"nonfinite {_nonfinite:.1%} · P(max|x|>10) {_explode:.1%}",
            "0% · <1%",
            _ok,
            f"trajectories of {name} diverge or escape: revise stiffness/quartic/diffusion priors.",
        )

    def check_scale(name, x):
        _half = x.shape[1] // 2
        _sds = jnp.std(x[:, _half:], axis=1)
        _med = float(jnp.median(_sds))
        _q05, _q95 = np.percentile(np.asarray(_sds), [5, 95])
        return CheckResult(
            "C2 latent scale",
            name,
            f"median sd {_med:.2f} (5–95%: {_q05:.2f}–{_q95:.2f})",
            "[0.30, 3.0]",
            bool(0.3 <= _med <= 3.0),
            f"stationary scale of {name} violates the standardized-latent convention: "
            "rebalance diffusion vs stiffness priors (or the incoming edge weight).",
        )

    def check_timescale(name, x, dt, lo, hi):
        _xs = np.asarray(x)[:, x.shape[1] // 2 :]
        _xs = _xs - _xs.mean(axis=1, keepdims=True)
        _var = (_xs * _xs).mean(axis=1) + 1e-12
        _taus = np.full(_xs.shape[0], 15.0)
        _found = np.zeros(_xs.shape[0], dtype=bool)
        for _k in range(1, int(15.0 / dt)):
            _ac = (_xs[:, :-_k] * _xs[:, _k:]).mean(axis=1) / _var
            _hit = (~_found) & (_ac < np.exp(-1.0))
            _taus[_hit] = _k * dt
            _found |= _hit
        _med = float(np.median(_taus))
        return CheckResult(
            "C3 timescale",
            name,
            f"median relaxation {_med:.2f} d",
            f"[{lo:.2f}, {hi:.1f}] d",
            bool(lo <= _med <= hi),
            f"prior-implied dynamics of {name} are invisible at the observation cadence: "
            "revise the stiffness prior (relaxation time) or reconsider the sampling design.",
        )

    def check_edge_share(name, x_on, x_off):
        _half = x_on.shape[1] // 2
        _v_on = jnp.var(x_on[:, _half:], axis=1)
        _v_off = jnp.var(x_off[:, _half:], axis=1)
        _share = np.asarray(1.0 - _v_off / (_v_on + 1e-12))
        _med = float(np.median(_share))
        _q90 = float(np.percentile(_share, 90))
        return CheckResult(
            "C4 edge detectability",
            name,
            f"parent variance share: median {_med:.1%}, q90 {_q90:.1%}",
            "q90 ≥ 2% · median ≤ 90%",
            bool(_q90 >= 0.02 and _med <= 0.9),
            f"incoming edges of {name} are a priori undetectable (or overwhelming): "
            "revise the edge-weight prior — as stated, the posterior on this edge will "
            "just return the prior.",
        )

    def check_coverage(indicator, pp_y, y_obs):
        _pp = np.asarray(pp_y).ravel()
        _lo, _hi = np.percentile(_pp, [1, 99])
        _qs = np.percentile(np.asarray(y_obs), [5, 25, 50, 75, 95])
        _cov_ok = bool(np.all((_qs >= _lo) & (_qs <= _hi)))
        _q75, _q25 = np.percentile(_pp, [75, 25])
        _o75, _o25 = np.percentile(np.asarray(y_obs), [75, 25])
        _ratio = float((_q75 - _q25) / max(_o75 - _o25, 1e-9))
        _width_ok = bool(1.0 / 3.0 <= _ratio <= 50.0)
        return [
            CheckResult(
                "C5a location reach",
                indicator,
                f"obs quantiles in pp [1,99]% band [{_lo:.1f}, {_hi:.1f}]: {'yes' if _cov_ok else 'NO'}",
                "all inside",
                _cov_ok,
                f"the prior predictive cannot reach where {indicator} actually lives: revise "
                "the emission intercept/loading priors (this is the manifest-means failure "
                "the fit-boundary preflight catches — caught here at admission instead).",
            ),
            CheckResult(
                "C5b width",
                indicator,
                f"IQR ratio prior-pred/data {_ratio:.2f}",
                "[0.33, 50]",
                _width_ok,
                f"prior predictive spread for {indicator} is out of proportion to the data: "
                "tighten or widen emission priors.",
            ),
        ]

    def check_sensitivity(indicator, em, x_obs, lam, b, sigma_e, emission_mean):
        _m = emission_mean(em)
        _lam_b = jnp.broadcast_to(lam[:, None], x_obs.shape)
        _b_b = jnp.broadcast_to(b[:, None], x_obs.shape)
        _dm = jax.vmap(jax.vmap(jax.grad(_m, argnums=0)))(x_obs, _lam_b, _b_b)
        _snr = jnp.median(jnp.abs(_dm), axis=1) * jnp.std(x_obs, axis=1) / sigma_e
        _med = float(jnp.median(_snr))
        return CheckResult(
            "C6 link sensitivity",
            indicator,
            f"median |∂m/∂x|·sd(x)/σ = {_med:.2f}",
            "[0.20, 100]",
            bool(0.2 <= _med <= 100.0),
            f"the link for {indicator} is saturated or inert under the prior: observations "
            "carry almost no local information about the latent. Revise the loading prior "
            "(typical cause: identity-scale loading reused inside a logit/sigmoid link).",
        )

    return (
        CheckResult,
        check_confinement,
        check_coverage,
        check_edge_share,
        check_scale,
        check_sensitivity,
        check_timescale,
    )


@app.cell
def admit_api(
    ADMIT_KEY,
    N_DRAWS,
    T_GRID,
    check_confinement,
    check_coverage,
    check_edge_share,
    check_scale,
    check_sensitivity,
    check_timescale,
    data,
    dataclass,
    draw_params,
    emission_mean,
    jax,
    np,
    obs_idx,
    obs_times,
    sample_prior,
    simulate_latents,
):
    @dataclass(frozen=True)
    class BuildState:
        nodes: tuple = ()

    def admit(state, frag):
        _nodes = (*state.nodes, frag)
        _d = len(_nodes) - 1
        _params = draw_params(ADMIT_KEY, _nodes, N_DRAWS)
        _lat = simulate_latents(ADMIT_KEY, _params, T_GRID)
        _x = _lat[:, :, _d]
        _dt = float(T_GRID[1] - T_GRID[0])
        _dt_obs = float(np.median(np.diff(obs_times)))
        _results = [
            check_confinement(frag.name, _x),
            check_scale(frag.name, _x),
            check_timescale(frag.name, _x, _dt, 0.5 * _dt_obs, 0.5 * float(T_GRID[-1])),
        ]
        _art = {"name": frag.name, "latents": _lat}
        if frag.edges_in:
            _p_off = dict(_params)
            _p_off["W"] = _params["W"].at[:, _d, :].set(0.0)
            _x_off = simulate_latents(ADMIT_KEY, _p_off, T_GRID)[:, :, _d]
            _results.append(check_edge_share(frag.name, _x, _x_off))
        if frag.emission is not None:
            _em = frag.emission

            def _ekey(slot, d=_d):
                return jax.random.fold_in(ADMIT_KEY, 2000 + 10 * d + slot)

            _lam = sample_prior(_ekey(0), _em.loading, N_DRAWS)
            _b = sample_prior(_ekey(1), _em.intercept, N_DRAWS)
            _sig = sample_prior(_ekey(2), _em.noise, N_DRAWS)
            _x_obs = _x[:, obs_idx]
            _mean_y = emission_mean(_em)(_x_obs, _lam[:, None], _b[:, None])
            _pp_y = _mean_y + _sig[:, None] * jax.random.normal(_ekey(3), _x_obs.shape)
            _results.extend(check_coverage(_em.indicator, _pp_y, data[_em.indicator]))
            _results.append(
                check_sensitivity(_em.indicator, _em, _x_obs, _lam, _b, _sig, emission_mean)
            )
            _art.update({"pp_y": _pp_y, "indicator": _em.indicator})
        return BuildState(_nodes), _results, _art

    return BuildState, admit


@app.cell
def viz_report(mo, np):
    def report_md(title, results):
        _rows = "\n".join(
            f"| {r.check} | {r.target} | {r.value} | {r.band} | {'✅' if r.passed else '❌'} |"
            for r in results
        )
        _failed = [r for r in results if not r.passed]
        _notes = (
            "\n\n**Feedback to the proposer:**\n"
            + "\n".join(f"- **{r.check}** — {r.note}" for r in _failed)
            if _failed
            else "\n\n**All checks green — fragment admitted.**"
        )
        return mo.md(
            f"### {title}\n\n"
            "| check | target | prior-predictive value | band | verdict |\n"
            "|---|---|---|---|---|\n" + _rows + _notes
        )

    def fan(ax, t, xs, color, label=None):
        _qs = np.percentile(np.asarray(xs), [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(t, _qs[0], _qs[4], color=color, alpha=0.15, lw=0)
        ax.fill_between(t, _qs[1], _qs[3], color=color, alpha=0.3, lw=0)
        ax.plot(t, _qs[2], color=color, lw=1.5, label=label)
        ax.spines[["top", "right"]].set_visible(False)

    return fan, report_md


@app.cell(hide_code=True)
def stage_a_md(mo):
    mo.md(r"""
    ## 4. The happy path

    ### Stage A — admit the cause (sleep) with its indicator

    The LLM proposes: stiffness ~ LogNormal(ln 1.0, 0.5), center ~ Normal(0, 0.5), quartic ~
    HalfNormal(0.2), diffusion ~ LogNormal(ln 1.2, 0.4), x₀ ~ Normal(0, 1); emission
    sleep_quality with identity link, loading ~ LogNormal(ln 1.0, 0.3), intercept ~
    Normal(0, 0.5), noise ~ LogNormal(ln 0.4, 0.3). No parents, so C4 does not apply.
    """)
    return


@app.cell
def stage_a(BuildState, EmissionFragment, NodeFragment, admit, np, report_md):
    sleep_frag = NodeFragment(
        name="sleep",
        stiffness=("lognormal", 0.0, 0.5),
        center=("normal", 0.0, 0.5),
        quartic=("halfnormal", 0.0, 0.2),
        diffusion=("lognormal", float(np.log(1.2)), 0.4),
        x0=("normal", 0.0, 1.0),
        emission=EmissionFragment(
            indicator="sleep_quality",
            link="identity",
            loading=("lognormal", 0.0, 0.3),
            intercept=("normal", 0.0, 0.5),
            noise=("lognormal", float(np.log(0.4)), 0.3),
        ),
    )
    state_a, _res_a, art_a = admit(BuildState(), sleep_frag)
    report_md("Stage A — admit sleep (cause)", _res_a)
    return art_a, state_a


@app.cell
def stage_a_view(T_GRID, art_a, data, fan, mo, obs_times, plt):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10.5, 3.2))
    fan(_ax0, T_GRID, art_a["latents"][:, :, 0], "#3b6ea5")
    _ax0.set_title("latent sleep — prior predictive fan", fontsize=10)
    _ax0.set_xlabel("day")
    fan(_ax1, obs_times, art_a["pp_y"], "#3b6ea5")
    _ax1.scatter(obs_times, data["sleep_quality"], s=12, color="black", zorder=5, label="observed")
    _ax1.set_title("sleep_quality — prior predictive vs data", fontsize=10)
    _ax1.set_xlabel("day")
    _ax1.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def stage_b_md(mo):
    mo.md(r"""
    ### Stage B — admit the effect (stress): self-dynamics + the causal edge + its indicator

    The fragment now carries an incoming edge: weight ~ Normal(0, 0.7) for sleep → stress
    (sign left to the data). The emission targets the 0–100 stress_report, so the LLM proposes
    an identity link with loading ~ LogNormal(ln 12, 0.3) and intercept ~ Normal(55, 10) —
    scale-aware priors. C4 now applies: with common random numbers we simulate stress with the
    edge on and off and measure the parent's variance share.
    """)
    return


@app.cell
def stage_b(EdgeFragment, EmissionFragment, NodeFragment, admit, np, report_md, state_a):
    stress_frag = NodeFragment(
        name="stress",
        stiffness=("lognormal", 0.0, 0.4),
        center=("normal", 0.0, 0.5),
        quartic=("halfnormal", 0.0, 0.3),
        diffusion=("lognormal", float(np.log(1.5)), 0.4),
        x0=("normal", 0.0, 1.0),
        edges_in=(EdgeFragment(parent="sleep", weight=("normal", 0.0, 0.7)),),
        emission=EmissionFragment(
            indicator="stress_report",
            link="identity",
            loading=("lognormal", float(np.log(12.0)), 0.3),
            intercept=("normal", 55.0, 10.0),
            noise=("lognormal", float(np.log(8.0)), 0.3),
        ),
    )
    state_b, _res_b, art_b = admit(state_a, stress_frag)
    report_md("Stage B — admit stress (effect) + edge sleep → stress", _res_b)
    return art_b, state_b, stress_frag


@app.cell
def stage_b_view(T_GRID, art_b, data, fan, mo, obs_times, plt):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10.5, 3.2))
    fan(_ax0, T_GRID, art_b["latents"][:, :, 1], "#c0504d")
    _ax0.set_title("latent stress — prior predictive fan", fontsize=10)
    _ax0.set_xlabel("day")
    fan(_ax1, obs_times, art_b["pp_y"], "#c0504d")
    _ax1.scatter(obs_times, data["stress_report"], s=12, color="black", zorder=5, label="observed")
    _ax1.set_title("stress_report — prior predictive vs data", fontsize=10)
    _ax1.set_xlabel("day")
    _ax1.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def stage_c_md(mo):
    mo.md(r"""
    ### Stage C — admit mood: the saturating emission done right

    mood_slider is a bounded 0–100 response, so the link is a scaled sigmoid. The scale-aware
    proposal keeps the loading near 1 on the logit scale: loading ~ LogNormal(ln 1.0, 0.3),
    intercept ~ Normal(0, 0.5). C6 confirms the link is responsive where the prior predictive
    actually puts the latent.
    """)
    return


@app.cell
def stage_c(EdgeFragment, EmissionFragment, NodeFragment, admit, np, report_md, state_b):
    mood_frag = NodeFragment(
        name="mood",
        stiffness=("lognormal", 0.0, 0.4),
        center=("normal", 0.0, 0.5),
        quartic=("halfnormal", 0.0, 0.2),
        diffusion=("lognormal", float(np.log(1.1)), 0.4),
        x0=("normal", 0.0, 1.0),
        edges_in=(EdgeFragment(parent="stress", weight=("normal", 0.0, 0.7)),),
        emission=EmissionFragment(
            indicator="mood_slider",
            link="sigmoid100",
            loading=("lognormal", 0.0, 0.3),
            intercept=("normal", 0.0, 0.5),
            noise=("lognormal", float(np.log(5.0)), 0.3),
        ),
    )
    _state_c, _res_c, art_c = admit(state_b, mood_frag)
    report_md("Stage C — admit mood + edge stress → mood", _res_c)
    return art_c, mood_frag


@app.cell
def stage_c_view(T_GRID, art_c, data, fan, mo, obs_times, plt):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10.5, 3.2))
    fan(_ax0, T_GRID, art_c["latents"][:, :, 2], "#4a9d5b")
    _ax0.set_title("latent mood — prior predictive fan", fontsize=10)
    _ax0.set_xlabel("day")
    fan(_ax1, obs_times, art_c["pp_y"], "#4a9d5b")
    _ax1.scatter(obs_times, data["mood_slider"], s=12, color="black", zorder=5, label="observed")
    _ax1.set_title("mood_slider — prior predictive vs data", fontsize=10)
    _ax1.set_xlabel("day")
    _ax1.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def invariance_md(mo):
    mo.md(r"""
    ## 5. Why staging is sound: upstream marginals never move

    The whole workflow rests on one structural fact: **admitting a node cannot change the
    marginal law of anything upstream of it** — causal arrows only push influence forward. So
    a check that passed at stage A can never be invalidated by stage B or C, and re-validation
    of the prefix is unnecessary. Because parameters and noise are keyed per node, we can
    verify this path-by-path: the sleep trajectories simulated at stage A and inside the full
    stage-C model coincide draw-for-draw. The residual below sits at float32
    rounding-accumulation level (XLA re-associates the arithmetic differently for a 1-D and a
    3-D state), seven orders of magnitude under the path scale — the marginal law itself is
    unchanged by construction, since an upstream drift never references a downstream state.

    This is also the property that breaks if the drift ever gains feedback loops — then
    admission must happen per strongly-connected block, not per node.
    """)
    return


@app.cell
def invariance_check(art_a, art_c, jnp, mo):
    _gap = float(jnp.max(jnp.abs(art_a["latents"][:, :, 0] - art_c["latents"][:, :, 0])))
    mo.md(
        f"Max absolute difference between stage-A sleep paths and stage-C sleep paths, over "
        f"all {art_a['latents'].shape[0]} prior draws and every grid point: **{_gap:.1e}** "
        f"(float32 rounding accumulation; typical path scale is 1)"
    )
    return


@app.cell(hide_code=True)
def failures_md(mo):
    mo.md(r"""
    ## 6. Failure gallery — four realistic LLM errors, each caught at its own stage

    Each failure below is a plausible elicitation mistake. The point is not that a check fires
    — the full-model gate would eventually fire too — but *where* it fires and *what feedback*
    it produces: one fragment, one named prior, one actionable sentence.
    """)
    return


@app.cell(hide_code=True)
def f1_md(mo):
    mo.md(r"""
    ### F1 — stage B, C3: dynamics faster than the sampling cadence

    The LLM reasons "stress spikes and decays within hours" and sets stiffness ~
    LogNormal(ln 8, 0.3) — relaxation ≈ 3 h. Psychologically plausible; statistically
    invisible with daily prompts. Note C1 and C2 stay green (the process is well-behaved and
    the diffusion prior keeps the scale right): a histogram of the latent looks perfectly
    fine. Only the timescale check sees the problem — and the fast child also averages away
    its parent's input, so C4 degrades with it. Both notes point at the same fragment.
    """)
    return


@app.cell
def f1_fast_dynamics(admit, np, replace, report_md, state_a, stress_frag):
    _frag = replace(
        stress_frag,
        stiffness=("lognormal", float(np.log(8.0)), 0.3),
        diffusion=("lognormal", float(np.log(3.6)), 0.3),
    )
    _, _res, _ = admit(state_a, _frag)
    report_md("F1 — stress with hours-scale relaxation, observed daily", _res)
    return


@app.cell(hide_code=True)
def f2_md(mo):
    mo.md(r"""
    ### F2 — stage B, C5: the unreachable location (the 43σ bug, replayed)

    The LLM keeps the default intercept ~ Normal(0, 10) for an indicator that lives around 55.
    This is exactly the manifest-means failure that once produced a truth 43σ from the prior —
    discovered, back then, only after a failed fit. Here it is caught the moment the fragment
    is proposed, and the note names the prior to fix. The fit-boundary preflight
    (LOCATION_REACH_SIGMAS = 6) remains as the terminal backstop.
    """)
    return


@app.cell
def f2_unreachable_location(admit, replace, report_md, state_a, stress_frag):
    _frag = replace(
        stress_frag, emission=replace(stress_frag.emission, intercept=("normal", 0.0, 10.0))
    )
    _, _res, _ = admit(state_a, _frag)
    report_md("F2 — stress_report intercept prior centered at 0, data at ≈55", _res)
    return


@app.cell(hide_code=True)
def f3_md(mo):
    mo.md(r"""
    ### F3 — stage B, C4: the a-priori-undetectable edge

    An over-cautious weight prior, Normal(0, 0.03), makes the causal edge contribute ≈ 0 % of
    the child's variance. Nothing else changes — scales, timescales, and the emission all stay
    green. But the question Stage 4 exists to answer is *causal*: with this prior the edge
    posterior can only echo the prior, and any downstream effect estimate is fake precision.
    Detectability is a property of the *prior*, checkable before any data is fit.
    """)
    return


@app.cell
def f3_undetectable_edge(EdgeFragment, admit, replace, report_md, state_a, stress_frag):
    _frag = replace(
        stress_frag, edges_in=(EdgeFragment(parent="sleep", weight=("normal", 0.0, 0.03)),)
    )
    _, _res, _ = admit(state_a, _frag)
    report_md("F3 — edge sleep → stress with weight ~ Normal(0, 0.03)", _res)
    return


@app.cell(hide_code=True)
def f4_md(mo):
    mo.md(r"""
    ### F4 — stage C, C6: the saturated link that scale adequacy cannot see

    The LLM reuses the identity-link intuition from stress_report — "≈15 slider points per
    latent unit" — inside the sigmoid: loading ~ LogNormal(ln 15, 0.25). On the logit scale
    that rails the link: prior-predictive sliders sit at 0 or 100 and flip between them.

    Watch the report: **C5 passes**. Saturation makes the predictive band maximally wide, so
    the observed quantiles are comfortably covered and the width ratio is unremarkable. Only
    the derivative sees it: median |∂m/∂x| ≈ 0 along the prior-predictive paths, so the
    observations carry essentially no local information about the latent — a flat likelihood
    wearing a well-covered predictive band. This is the concrete answer to "should the checks
    include a Jacobian?": yes, and this is the Jacobian that matters.
    """)
    return


@app.cell
def f4_saturated_link(
    admit, data, fan, mo, mood_frag, np, obs_times, plt, replace, report_md, state_b
):
    _frag = replace(
        mood_frag,
        emission=replace(mood_frag.emission, loading=("lognormal", float(np.log(15.0)), 0.25)),
    )
    _, _res, _art = admit(state_b, _frag)
    _fig, _ax = plt.subplots(figsize=(9.0, 3.0))
    fan(_ax, obs_times, _art["pp_y"], "#c0504d")
    _ax.scatter(obs_times, data["mood_slider"], s=12, color="black", zorder=5, label="observed")
    _ax.set_title(
        "mood_slider prior predictive under the saturated loading — railed at 0/100, "
        "yet the band covers the data (C5 passes, C6 fails)",
        fontsize=10,
    )
    _ax.set_xlabel("day")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.vstack(
        [report_md("F4 — mood_slider loading ~ LogNormal(ln 15, 0.25)", _res), mo.as_html(_fig)]
    )
    return


@app.cell(hide_code=True)
def machine_md(mo):
    mo.md(r"""
    ## 7. What changes in the Stage 4 state machine

    The existing (currently disabled) frontier machine already has blocks, a reducer, repair
    campaigns, and checkpointing. What this proposal changes is the **unit of iteration**. The
    current plan walks *parameter kinds* horizontally across the whole model — configuration →
    indicators → model review → measurement priors → observation priors → dynamics priors →
    effect priors → correlation priors → global prior review. Validation therefore only ever
    runs against a *full assembly*, so a red prior-predictive result names no fragment, and
    repair becomes its own campaign machinery with scopes, certificates, and barriers.

    The rebuild walks *graph nodes* vertically instead:

    - **State** — the tuple of admitted fragments plus their check reports. Each fragment
      bundles what the current machine splits across four block kinds: the node's dynamics
      prior, its incoming-edge (effect) priors, and its indicators' measurement/observation
      priors. Fully serializable (the dynamics-spec layer already round-trips to JSON), so
      the build checkpoints per node instead of relying on a monolithic runtime snapshot.
    - **Transitions** — ADMIT(fragment) runs the battery on the cumulative partial model;
      green appends, red emits the per-check feedback and loops to REVISE(fragment) with a
      bounded retry budget before escalating. Upstream stages are never re-run (the invariance
      property above) — which is what makes the repair loop *local* and lets most of the
      campaign/barrier machinery collapse into a per-fragment retry.
    - **Order** — topological over the proposed DAG. Nodes with indicators get their emission
      *at admission* (the data anchor arrives with the node); explicit latent confounders have
      no emission, so they run C1–C4 only and their scale stays purely conventional — worth
      surfacing rather than hiding.
    - **Terminal gate** — the fit-boundary preflight and a full-model prior-predictive pass
      stay. Staging front-loads attribution; it does not replace the final joint check
      (marginal checks cannot see joint pathologies such as loading-vs-latent-scale trades).
    - **Placement** — per the standing decision, the checks themselves belong at the
      fit()/compiler boundary (next to the existing preflight), with the state machine as a
      thin orchestration layer above; the LLM only ever sees fragment-level feedback.
    """)
    return


@app.cell(hide_code=True)
def bridge_md(mo):
    mo.md(r"""
    ## 8. Bridge to the codebase

    Nearly everything this lab fakes already exists — what is missing is the staging and three
    of the six checks:

    - **Fragments** ↔ the dynamics component vocabulary: NodePotentialSpec (stiffness, center,
      quartic — exactly the well used here), LinearEdgeSpec, HillEdgeSpec for saturating
      couplings, all composed in DynamicsSpec.components, which is naturally *incremental*
      (a tuple you append to) and JSON-serializable.
    - **Priors** ↔ PriorRegistry with per-field defaults in DEFAULT_PRIOR_SPECS_BY_FIELD.
    - **Exact simulation** ↔ the Diffrax SDE simulator (Heun + VirtualBrownianTree) in the
      dynamics simulator module, orchestrated for prior predictive by
      sample_prior_predictive_from_runtime with family-aware emission sampling.
    - **C5a location reach** ↔ validate_observations_for_fit in models/ssm/preflight.py
      (LOCATION_REACH_SIGMAS = 6) — today a fit-time backstop; staged, it fires at admission.
    - **C1/C2-adjacent full-model gates** ↔ validate_prior_predictive in
      models/prior_predictive.py (nan/inf, constraints, extreme values, scale plausibility) —
      today invoked by Stage 4's grounding path (stage4/grounding.py → validate_assembly) only
      once a full assembly exists. Its drift-stability check is eigenvalue/Lyapunov-based —
      i.e. linearized; replacing it with simulation-based confinement (C1) would also bring
      the gates in line with the init-only-linearization policy.
    - **Orchestration** ↔ the frontier machine under flows/stages/stage4/agentic/ (reducer,
      plan blocks, repair campaigns, checkpoints). The rebuild replaces its parameter-kind
      block plan with node-fragment admission and retires most of the repair-campaign
      machinery in favor of the local revise loop.
    - **Genuinely new** — the admission ordering itself, C3 timescale-vs-cadence, C4 edge
      detectability, and C6 link sensitivity.
    """)
    return


@app.cell(hide_code=True)
def closing(mo):
    mo.md(r"""
    ## 9. Verdict and open questions

    The staged workflow holds up on the D = 3 prototype: the happy path is green end-to-end,
    the invariance that justifies never re-checking upstream is exact, and all four planted
    failures are caught at the stage that owns them, with fragment-local feedback. The two
    checks beyond plain prior-predictive coverage earn their place: C4 because edge
    detectability is the causal question in disguise, and C6 because saturation is invisible
    to coverage by construction.

    Open questions before porting into Stage 4 proper:

    - **Thresholds** — the bands here are hand-set. They should live in one config, and the
      right calibration is against the recovery fixtures (which prior pathologies actually
      break which samplers, at what check values?).
    - **Multiple indicators per construct** — run C5/C6 per indicator; a construct is only as
      anchored as its best indicator.
    - **Non-Gaussian families** — C6 generalizes by replacing σ with the family's local noise
      scale (variance function at the predictive mean); the emission registry already exposes
      variance functions.
    - **Feedback loops** — everything here assumes DAG drift. If reciprocal influence is ever
      admitted, admission must be per strongly-connected block and the invariance argument
      weakens to block granularity.
    - **Joint pathologies** — staging checks marginals; keep the terminal full-model gate, and
      consider one joint check for the loading-vs-latent-scale trade (the classic λ·x
      non-identifiability) once per build.
    """)
    return


if __name__ == "__main__":
    app.run()
