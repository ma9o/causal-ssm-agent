"""Staged model-building tooling: fragments, exact-simulation checks, severity, reports.

Extracted from gradual_model_building_lab.py (which keeps an inline pedagogical copy of
the original battery) so other notebooks can drive the incremental admission workflow on
real problems. This module is the seed for the production port of the Stage 4 rebuild:
the declarative tables (CHECK_MODES, CHECK_CONSEQUENCES, CHECK_VIZ) and the check
functions are meant to move toward models/ssm largely as-is.

The battery is scoped to **reachability + one design-observability screen** — the
recognized remit of prior-predictive checking. Practical-identifiability verdicts (is a
parameter prior-dominated / estimable from n_obs points) are deliberately NOT here: those
belong post-fit (posterior contraction, power-scaling). Concretely, versus the original
lab copy:
- C3 is a schedule-only **design-resolvability** screen (is the prior's τ inside
  [cadence/3, span/4]?), not a persistence estimator — the observed autocorrelation mixes
  self and inherited dynamics, an unidentified split left to the fit;
- C4a (edge detectability, a 2/√n_obs SNR floor) is dropped; only C4b (overwhelm, a
  degeneracy/plausibility check) remains;
- C6 (link Fisher-information SNR) is dropped as a standalone check; its structural kernel
  survives as C5c (transmission / saturated-link dead zone), a noise-free reachability test;
- emission families: "gaussian" (identity / sigmoid100 links) and "poisson" (exp link);
- everything parametrized (no module-level study constants): build an admitter with
  make_admitter(...) for a given dataset and design.

All checks run on exact Euler-Maruyama simulation of the true nonlinear drift; nothing
is linearized anywhere (init-only-linearization policy).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- priors


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


def prior_median(spec):
    kind, a, b = spec
    if kind == "normal":
        return a
    if kind == "lognormal":
        return float(jnp.exp(a))
    if kind == "halfnormal":
        return 0.6745 * b
    if kind == "delta":
        return a
    raise ValueError(f"unknown prior kind: {kind}")


# ---------------------------------------------------------------- fragments


@dataclass(frozen=True)
class EdgeFragment:
    parent: str
    weight: tuple[str, float, float]


@dataclass(frozen=True)
class EmissionFragment:
    indicator: str
    link: str  # "identity" | "sigmoid100" | "exp"
    loading: tuple[str, float, float]
    intercept: tuple[str, float, float]
    noise: tuple[str, float, float]  # unused for family="poisson"
    family: str = "gaussian"  # "gaussian" | "poisson"


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


@dataclass(frozen=True)
class BuildState:
    nodes: tuple = ()
    annotations: tuple = ()


# ---------------------------------------------------------------- simulation engine


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
        [jax.random.normal(jax.random.fold_in(key, 5000 + d), (n, _steps)) for d in range(_dim)],
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

    elif em.link == "exp":

        def _m(x, lam, b):
            return jnp.exp(jnp.clip(lam * x + b, -20.0, 20.0))

    else:
        raise ValueError(f"unknown link: {em.link}")
    return _m


def emission_inverse_link(em):
    if em.link == "identity":

        def _ginv(y):
            return y

    elif em.link == "sigmoid100":

        def _ginv(y):
            _p = np.clip(y / 100.0, 0.01, 0.99)
            return np.log(_p / (1.0 - _p))

    elif em.link == "exp":

        def _ginv(y):
            return np.log(np.maximum(y, 0.5))

    else:
        raise ValueError(f"unknown link: {em.link}")
    return _ginv


# ---------------------------------------------------------------- checks


@dataclass(frozen=True)
class CheckResult:
    check: str
    target: str
    value: str
    band: str
    passed: bool
    note: str
    diagnosis: tuple[str, ...] = ()
    evidence: dict | None = None


def check_confinement(name, x, dt):
    _bad = ~np.isfinite(np.asarray(x))
    _nonfinite = float(np.mean(_bad))
    _xa = np.abs(np.nan_to_num(np.asarray(x), nan=np.inf, posinf=np.inf, neginf=np.inf))
    _q = _xa.shape[1] // 4
    _early = np.quantile(_xa[:, _q : 2 * _q], 0.95, axis=1)
    _late = np.max(_xa[:, -_q:], axis=1)
    _growth = _late / (_early + 1e-9)
    _explode = float(np.mean(_growth > 5.0))
    _ev = {"x": np.asarray(x), "growth": _growth, "dt": dt}
    _diag_a = ()
    if _nonfinite > 0.0:
        _bad_draws = _bad.any(axis=1)
        _onset = float(np.median(np.argmax(_bad, axis=1)[_bad_draws]) * dt)
        _diag_a = (
            f"{float(np.mean(_bad_draws)):.0%} of prior draws go non-finite; median "
            f"onset t ≈ {_onset:.1f} d (integrator dt = {dt:g} d)",
            "mechanism: explicit Euler–Maruyama diverges when step × drift gradient "
            "exceeds the well width; the gradient grows with the stiffness and quartic "
            "draws, and the diffusion draw sets how far paths wander into the steep "
            "region",
        )
    _diag_b = ()
    if _explode >= 0.01:
        _diag_b = (
            f"{_explode:.1%} of draws end the window above 5× their own early amplitude "
            f"(growth-ratio q99: {float(np.percentile(_growth, 99)):.1f})",
            "mechanism: growth without settling occurs in draws where the confining "
            "terms (stiffness, quartic) are small relative to the variance input "
            "(diffusion, incoming edges)",
        )
    return [
        CheckResult(
            "C1a finiteness",
            name,
            f"nonfinite {_nonfinite:.1%}",
            "0%",
            _nonfinite == 0.0,
            f"simulation of {name} produced non-finite values — the fragment cannot be evaluated.",
            _diag_a,
            _ev,
        ),
        CheckResult(
            "C1b confinement",
            name,
            f"P(late/early amplitude > 5) {_explode:.1%}",
            "<1% (self-calibrating growth)",
            _explode < 0.01,
            f"trajectories of {name} grow without settling within the study window.",
            _diag_b,
            _ev,
        ),
    ]


def check_scale(name, x, scale_anchor, anchor_src, anchor_detail):
    _half = x.shape[1] // 2
    _sds = jnp.std(x[:, _half:], axis=1)
    _med = float(jnp.median(_sds))
    _q05, _q95 = np.percentile(np.asarray(_sds), [5, 95])
    _lo, _hi = scale_anchor / 3.0, scale_anchor * 3.0
    _ok = bool(_lo <= _med <= _hi)
    _ev = {"sds": np.asarray(_sds), "lo": _lo, "hi": _hi, "anchor": scale_anchor}
    _diag = ()
    if not _ok:
        _side = "above" if _med > _hi else "below"
        _factor = _med / _hi if _med > _hi else _lo / max(_med, 1e-9)
        _diag = (
            f"prior-predictive stationary sd: median {_med:.2f} (5–95% "
            f"{_q05:.2f}–{_q95:.2f}) vs band [{_lo:.2f}, {_hi:.2f}] — {_factor:.1f}× "
            f"{_side} the edge",
            f"band derivation: {anchor_detail} — the latent scale the indicator data "
            "implies given the emission priors",
            "dependence: the statistic rises with the diffusion prior and falls with "
            "the stiffness prior (incoming edges add parent variance); the band scales "
            "inversely with the prior-median loading — this red can equally reflect a "
            "dynamics–emission inconsistency",
            "related evidence: C6 on the same indicator also takes the loading prior as "
            "input; the joint (C2, C6) pattern localizes which side is active",
        )
    return CheckResult(
        "C2 latent scale",
        name,
        f"median sd {_med:.2f} (5–95%: {_q05:.2f}–{_q95:.2f})",
        f"[{_lo:.2f}, {_hi:.2f}] ({anchor_src})",
        _ok,
        f"stationary scale of {name} is inconsistent with the scale its indicator "
        f"implies under the emission priors ({anchor_src}).",
        _diag,
        _ev,
    )


def check_resolvability(name, tau_draws, cadence, span):
    """Design-observability screen (schedule-only): is the prior's self-relaxation timescale
    inside the window this sampling design can resolve?

    τ below ~cadence/3 is aliased — the process relaxes ≥3× between observations, so
    consecutive samples are near-independent draws of the stationary law. τ above ~span/4 is
    frozen — the window holds < ~4 relaxation times, no replication. This compares the PRIOR's
    τ to the DESIGN (median gap, span) only; it does NOT estimate τ or decompose observed
    persistence into self vs inherited (that split is unidentified pre-fit and belongs to the
    post-fit contraction gate). Reachability-flavored: it catches a prior positing dynamics the
    schedule cannot see, not a mis-centered-but-resolvable τ.
    """
    _tau = np.asarray(tau_draws, dtype=float)
    _med = float(np.median(_tau))
    _q10, _q90 = (float(_v) for _v in np.percentile(_tau, [10, 90]))
    _lo, _hi = cadence / 3.0, span / 4.0
    _frac_in = float(np.mean((_tau >= _lo) & (_tau <= _hi)))
    _ok = bool(_lo <= _med <= _hi)
    _ev = {"tau": _tau, "lo": _lo, "hi": _hi, "cadence": cadence, "span": span}
    _diag = ()
    if not _ok and _med < _lo:
        _diag = (
            f"prior self-relaxation τ: median {_med:.2f} d (10–90% {_q10:.2f}–{_q90:.2f}) "
            f"below the design floor cadence/3 = {_lo:.2f} d (median observation gap "
            f"{cadence:.2f} d); {_frac_in:.0%} of prior mass is resolvable",
            "reading: the prior posits dynamics faster than the sampling can follow — the "
            "process relaxes ≥3× between observations, so consecutive samples are "
            "near-independent draws of the stationary law and this timescale is invisible "
            "to the design regardless of the fit",
            "this is a prior/design mismatch, not an estimate: the observed autocorrelation "
            "mixes this node's own relaxation with inherited parent persistence, and that "
            "split is resolved only by the joint fit — confirm with post-fit contraction",
        )
    elif not _ok:
        _diag = (
            f"prior self-relaxation τ: median {_med:.2f} d (10–90% {_q10:.2f}–{_q90:.2f}) "
            f"above the design ceiling span/4 = {_hi:.2f} d (span {span:.0f} d); "
            f"{_frac_in:.0%} of prior mass is resolvable",
            "reading: the prior posits dynamics so slow the window holds < ~4 relaxation "
            "times — the process is near-frozen over the record, so its timescale and "
            "stationary law are not resolvable by this design",
        )
    return CheckResult(
        "C3 resolvability",
        name,
        f"prior τ median {_med:.2f} d (10–90% {_q10:.2f}–{_q90:.2f}); {_frac_in:.0%} in window",
        f"cadence/3 ≤ τ ≤ span/4 = [{_lo:.2f}, {_hi:.2f}] d",
        _ok,
        f"the timescale posited for {name} lies outside the window this sampling design can "
        "resolve; the fit cannot inform its dynamics from this schedule.",
        _diag,
        _ev,
    )


def check_edge_share(name, x_on_obs, x_off_obs):
    # C4b only (degeneracy/plausibility). C4a edge *detectability* was dropped: its 2/√n_obs
    # SNR floor is a data-quantity detectability threshold — practical identifiability of the
    # edge weight, which belongs to the post-fit gate (posterior contraction on the weight),
    # not the prior-predictive stage. Overwhelm stays because a child fully slaved to a parent
    # is a degenerate *prior*, a reachability concern.
    _a = np.asarray(x_on_obs)
    _b = np.asarray(x_off_obs)
    _disp = np.sqrt(np.mean((_a - _b) ** 2, axis=1))
    _scale = np.sqrt(np.var(_a, axis=1)) + 1e-12
    _e = _disp / _scale
    _med = float(np.median(_e))
    _i90 = int(np.argsort(_e)[int(0.9 * (_e.size - 1))])
    _ev = {"e": _e, "on": _a[_i90], "off": _b[_i90]}
    _diag_b = ()
    if _med > 0.95:
        _diag_b = (
            f"for the median prior draw the edge accounts for {_med:.0%} of the child's "
            "entire path variation: parent input, not self-dynamics, sets the path",
            "dependence: the statistic falls with the edge-weight prior scale and "
            "rises when the child's own stiffness/diffusion contribute little",
        )
    return [
        CheckResult(
            "C4b edge overwhelm",
            name,
            f"edge path displacement / child scale: median {_med:.1%}",
            "median ≤ 95%",
            bool(_med <= 0.95),
            f"parent input dominates the path variation of {name}; its self-dynamics "
            "are left uninformed.",
            _diag_b,
            _ev,
        ),
    ]


def check_coverage(indicator, pp_y, signal_y, y_obs):
    _pp = np.asarray(pp_y).ravel()
    _lo, _hi = np.percentile(_pp, [1, 99])
    _pp_med = float(np.percentile(_pp, 50))
    _qs = np.percentile(np.asarray(y_obs), [5, 25, 50, 75, 95])
    _cov_ok = bool(np.all((_qs >= _lo) & (_qs <= _hi)))
    _q75, _q25 = np.percentile(_pp, [75, 25])
    _o75, _o25 = np.percentile(np.asarray(y_obs), [75, 25])
    _obs_iqr = float(_o75 - _o25)
    _ratio = float((_q75 - _q25) / max(_obs_iqr, 1e-9))
    _width_ok = bool(1.0 / 3.0 <= _ratio <= 50.0)
    # C5c transmission (structural dead-zone, noise-free): can the link, driven by the latent's
    # prior mass, produce a *signal* comparable to the observed variation, or is it saturated /
    # flat so the observed spread would have to be almost all measurement noise? This is the
    # structural kernel salvaged from the old C6 — a reachability question (can the signal reach
    # the observed variation) with NO n_obs SNR threshold. C5b can pass on a saturated link
    # because noise widens the predictive band; this catches what C5b misses.
    _sig = np.asarray(signal_y).ravel()
    _s75, _s25 = np.percentile(_sig, [75, 25])
    _transmit = float((_s75 - _s25) / max(_obs_iqr, 1e-9))
    _trans_ok = bool(_transmit >= 0.2)
    _ev = {
        "pp": _pp[:: max(1, _pp.size // 20000)],
        "signal": _sig[:: max(1, _sig.size // 20000)],
        "y_obs": np.asarray(y_obs),
        "lo": _lo,
        "hi": _hi,
    }
    _diag_a = ()
    if not _cov_ok:
        _gap = float(_qs[2] - _pp_med)
        _diag_a = (
            f"observed quantiles (5/25/50/75/95%): "
            f"[{', '.join(f'{_q:.1f}' for _q in _qs)}]; prior-predictive [1,99]% band "
            f"[{_lo:.1f}, {_hi:.1f}], centered at {_pp_med:.1f}",
            f"the gap is in location: the data's median sits {abs(_gap):.1f} units "
            f"{'above' if _gap > 0 else 'below'} the predictive center; location is set "
            "by the intercept/manifest-means priors, while loading and noise priors "
            "enter only the spread",
            "a fit run under this prior cannot reach the data location — the "
            "fit-boundary preflight enforces the same condition at fit time "
            "(LOCATION_REACH_SIGMAS = 6)",
        )
    _diag_b = ()
    if not _width_ok:
        _diag_b = (
            f"prior-predictive IQR is {_ratio:.2f}× the data IQR (band [0.33, 50])",
            "dependence: predictive width compounds the loading spread × latent scale, "
            "the intercept spread, and the noise scale (for count families the noise "
            "is tied to the rate); the data-side IQR is fixed",
        )
    _diag_c = ()
    if not _trans_ok:
        _diag_c = (
            f"noise-free signal IQR is {_transmit:.0%} of the data IQR (floor 20%): the "
            "link, driven by the latent's prior mass, transmits almost none of the "
            "observed variation — the data spread would have to be explained by "
            "measurement noise",
            "geometry: the latent moves but the emission mean does not, i.e. the link is "
            "operating in a flat / saturated region over the prior mass (sigmoid tail, "
            "near-zero exp rate, or a near-zero loading)",
            "dependence: transmitted signal scales with the loading prior and where the "
            "latent mass lands on the link (C2); it is independent of the noise prior — "
            "C5b can pass here because noise alone widens the predictive band",
        )
    return [
        CheckResult(
            "C5a location reach",
            indicator,
            f"obs quantiles in pp [1,99]% band [{_lo:.1f}, {_hi:.1f}]: "
            f"{'yes' if _cov_ok else 'NO'}",
            "all inside",
            _cov_ok,
            f"the prior predictive cannot reach the location where {indicator} actually lives.",
            _diag_a,
            _ev,
        ),
        CheckResult(
            "C5b width",
            indicator,
            f"IQR ratio prior-pred/data {_ratio:.2f}",
            "[0.33, 50]",
            _width_ok,
            f"prior-predictive spread for {indicator} is out of proportion to the observed spread.",
            _diag_b,
            _ev,
        ),
        CheckResult(
            "C5c transmission",
            indicator,
            f"signal IQR / data IQR {_transmit:.0%}",
            "≥ 20% (link not saturated)",
            _trans_ok,
            f"the link for {indicator} transmits little of the latent's variation: the "
            "observed spread would be explained almost entirely by measurement noise.",
            _diag_c,
            _ev,
        ),
    ]


# ---------------------------------------------------------------- severity (declarative)

CHECK_MODES = {
    "C1a finiteness": "hard",
    "C1b confinement": "soft",
    "C2 latent scale": "soft",
    "C3 resolvability": "soft",
    "C4b edge overwhelm": "soft",
    "C5a location reach": "hard",
    "C5b width": "soft",
    "C5c transmission": "soft",
}

CHECK_CONSEQUENCES = {
    "C1b confinement": "{target}: excursions beyond the confinement screen accepted as "
    "intentional; extreme-value behavior is prior-set, not data-vetted",
    "C2 latent scale": "{target}: latent scale departs from its data anchor; magnitude "
    "statements about this construct are convention-bound, not data-bound",
    "C3 resolvability": "{target}: its posited timescale sits outside what this sampling "
    "design resolves; the fit cannot inform its dynamics from this schedule — treat any "
    "timescale or trajectory statement as prior-set and confirm with post-fit contraction",
    "C4b edge overwhelm": "{target}: parent-driven variation dominates; its own dynamics "
    "parameters are weakly informed",
    "C5b width": "{target}: prior-predictive width imbalance accepted; expect weak "
    "regularization or slow warmup",
    "C5c transmission": "{target}: the link passes little of the latent's variation to the "
    "indicator; the observed spread is largely measurement noise, so this construct's "
    "trajectory is weakly grounded in the data",
}


def stage_outcome(results, accepted):
    _failed = [r for r in results if not r.passed]
    _hard = [r for r in _failed if CHECK_MODES[r.check] == "hard"]
    _pending = [r for r in _failed if CHECK_MODES[r.check] == "soft" and r.check not in accepted]
    _annotations = tuple(
        CHECK_CONSEQUENCES[r.check].format(target=r.target)
        + (f" [rationale: {accepted[r.check]}]" if accepted[r.check] else "")
        for r in _failed
        if CHECK_MODES[r.check] == "soft" and r.check in accepted
    )
    if _hard:
        return "BLOCKED — hard failure: revise the fragment (no override)", _annotations
    if _pending:
        _ids = ", ".join(r.check for r in _pending)
        return (
            f"NEEDS DECISION — revise the fragment or accept the consequence ({_ids})",
            _annotations,
        )
    if _annotations:
        return "ADMITTED with accepted consequences", _annotations
    return "ADMITTED", _annotations


# ---------------------------------------------------------------- admitter


def make_admitter(*, admit_key, n_draws, t_grid, obs_times, obs_idx, data):
    """Bind a dataset and design; return admit(state, frag, accepted=None)."""
    _dt = float(t_grid[1] - t_grid[0])
    _dt_obs = float(np.median(np.diff(obs_times)))
    _span = float(np.ptp(np.asarray(obs_times)))

    def admit(state, frag, accepted=None):
        _nodes = (*state.nodes, frag)
        _d = len(_nodes) - 1
        _params = draw_params(admit_key, _nodes, n_draws)
        _lat = simulate_latents(admit_key, _params, t_grid)
        _x = _lat[:, :, _d]
        _anchor, _anchor_src = 1.0, "convention: no indicator"
        _anchor_detail = "convention anchor 1.0 — no indicator, so no possible data anchor"
        if frag.emission is not None:
            _ginv = emission_inverse_link(frag.emission)
            _q75, _q25 = np.percentile(data[frag.emission.indicator], [75, 25])
            _iqr_xi = abs(float(_ginv(_q75) - _ginv(_q25)))
            _lam_med = abs(prior_median(frag.emission.loading))
            _anchor = _iqr_xi / (1.349 * _lam_med)
            _anchor_src = f"data via {frag.emission.indicator} (inverse-link IQR)"
            _anchor_detail = (
                f"anchor {_anchor:.2f} = ({frag.emission.indicator} inverse-link IQR "
                f"{_iqr_xi:.2f} / 1.349) / |prior-median loading| {_lam_med:.2f}"
            )
        _results = [
            *check_confinement(frag.name, _x, _dt),
            check_scale(frag.name, _x, _anchor, _anchor_src, _anchor_detail),
            check_resolvability(
                frag.name, 1.0 / np.asarray(_params["stiff"][:, _d]), _dt_obs, _span
            ),
        ]
        _art = {"name": frag.name, "latents": _lat}
        if frag.edges_in:
            _p_off = dict(_params)
            _p_off["W"] = _params["W"].at[:, _d, :].set(0.0)
            _x_off = simulate_latents(admit_key, _p_off, t_grid)[:, :, _d]
            _results.extend(check_edge_share(frag.name, _x[:, obs_idx], _x_off[:, obs_idx]))
        if frag.emission is not None:
            _em = frag.emission

            def _ekey(slot, d=_d):
                return jax.random.fold_in(admit_key, 2000 + 10 * d + slot)

            _lam = sample_prior(_ekey(0), _em.loading, n_draws)
            _b = sample_prior(_ekey(1), _em.intercept, n_draws)
            _sig = sample_prior(_ekey(2), _em.noise, n_draws)
            _x_obs = _x[:, obs_idx]
            _mean_y = emission_mean(_em)(_x_obs, _lam[:, None], _b[:, None])
            if _em.family == "poisson":
                _pp_y = jax.random.poisson(_ekey(3), _mean_y).astype(jnp.float32)
            else:
                _pp_y = _mean_y + _sig[:, None] * jax.random.normal(_ekey(3), _x_obs.shape)
            _results.extend(check_coverage(_em.indicator, _pp_y, _mean_y, data[_em.indicator]))
            _art.update({"pp_y": _pp_y, "indicator": _em.indicator})
        _outcome, _annotations = stage_outcome(_results, accepted or {})
        _admitted = _outcome.startswith("ADMITTED")
        _art["outcome"] = _outcome
        _art["annotations"] = _annotations if _admitted else ()
        _next = BuildState(_nodes, (*state.annotations, *_annotations)) if _admitted else state
        return _next, _results, _art

    return admit


# ---------------------------------------------------------------- visualizations


def _viz_confinement(ev):
    _x, _growth, _dt = ev["x"], ev["growth"], ev["dt"]
    _t = np.arange(_x.shape[1]) * _dt
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10.0, 3.0))
    for _row in _x[:25]:
        _ax0.plot(_t, _row, color="#c5c5c5", lw=0.6)
    for _i in np.argsort(np.nan_to_num(_growth, nan=np.inf))[-5:]:
        _ax0.plot(_t, _x[_i], color="#c0504d", lw=1.2)
    _fin = _x[np.isfinite(_x)]
    if _fin.size:
        _lo_y, _hi_y = np.percentile(_fin, [0.1, 99.9])
        _pad = 0.25 * (_hi_y - _lo_y + 1e-9)
        _ax0.set_ylim(_lo_y - _pad, _hi_y + _pad)
    _ax0.set_title("prior draws (gray) vs the 5 highest-growth draws (red)", fontsize=9)
    _ax0.set_xlabel("day")
    _finite_g = _growth[np.isfinite(_growth)]
    _ax1.hist(np.clip(_finite_g, 0, 20), bins=40, color="#3b6ea5")
    _ax1.axvline(5.0, color="#c0504d", ls="--", label="growth gate ×5")
    _ax1.set_title("late/early amplitude ratio per draw", fontsize=9)
    _ax1.legend(frameon=False, fontsize=8)
    for _ax in (_ax0, _ax1):
        _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    return _fig


def _viz_scale(ev):
    _fig, _ax = plt.subplots(figsize=(8.0, 2.6))
    _ax.hist(ev["sds"], bins=40, color="#3b6ea5")
    _ax.axvline(ev["lo"], color="#c0504d", ls="--", label="band")
    _ax.axvline(ev["hi"], color="#c0504d", ls="--")
    _ax.axvline(ev["anchor"], color="#4a9d5b", lw=2, label="anchor")
    _ax.set_title("per-draw stationary sd vs the scale-anchor band", fontsize=9)
    _ax.set_xlabel("stationary sd")
    _ax.legend(frameon=False, fontsize=8)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    return _fig


def _viz_resolvability(ev):
    _fig, _ax = plt.subplots(figsize=(8.5, 2.8))
    _tau = ev["tau"]
    _hi_x = float(np.percentile(_tau, 99))
    _ax.hist(np.clip(_tau, 0, _hi_x), bins=50, color="#3b6ea5", label="prior τ = 1/stiffness")
    _ax.axvspan(
        ev["lo"], min(ev["hi"], _hi_x), color="#4a9d5b", alpha=0.12, label="resolvable window"
    )
    _ax.axvline(ev["lo"], color="#c0504d", ls="--", label="cadence/3 floor")
    _ax.axvline(ev["hi"], color="#7d6bb0", ls=":", label="span/4 ceiling")
    _ax.axvline(ev["cadence"], color="#333333", ls="-", lw=0.8, label="observation cadence")
    _ax.set_xlabel("self-relaxation τ (days)")
    _ax.set_title("prior timescale vs the design's resolvable window", fontsize=9)
    _ax.legend(frameon=False, fontsize=7)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    return _fig


def _viz_edge(ev):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10.0, 3.0))
    _idx = np.arange(ev["on"].size)
    _ax0.plot(_idx, ev["on"], color="#3b6ea5", lw=1.4, label="edge on")
    _ax0.plot(_idx, ev["off"], color="#e08a3c", lw=1.4, ls="--", label="edge off (same noise)")
    _ax0.set_xlabel("observation #")
    _ax0.set_title("high-displacement draw: how much the edge moves the child", fontsize=9)
    _ax0.legend(frameon=False, fontsize=8)
    _hi = float(np.percentile(ev["e"], 99))
    _ax1.hist(np.clip(ev["e"], 0, _hi), bins=40, color="#3b6ea5")
    _ax1.axvline(0.95, color="#7d6bb0", ls=":", label="overwhelm cap")
    _ax1.set_title("per-draw displacement / child scale", fontsize=9)
    _ax1.legend(frameon=False, fontsize=8)
    for _ax in (_ax0, _ax1):
        _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    return _fig


def _viz_coverage(ev):
    _fig, _ax = plt.subplots(figsize=(8.5, 2.8))
    _ax.hist(ev["pp"], bins=60, density=True, color="#c5c5c5", label="prior predictive (pooled)")
    _ax.hist(
        ev["signal"],
        bins=60,
        density=True,
        color="#4a9d5b",
        alpha=0.4,
        label="signal only (noise-free)",
    )
    _ax.hist(ev["y_obs"], bins=20, density=True, color="#3b6ea5", alpha=0.6, label="observed")
    _ax.axvline(ev["lo"], color="#c0504d", ls="--", label="pp [1,99]% band")
    _ax.axvline(ev["hi"], color="#c0504d", ls="--")
    _ax.set_title("prior predictive vs observed — location, width, transmission", fontsize=9)
    _ax.legend(frameon=False, fontsize=8)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    return _fig


CHECK_VIZ = {
    "C1a finiteness": _viz_confinement,
    "C1b confinement": _viz_confinement,
    "C2 latent scale": _viz_scale,
    "C3 resolvability": _viz_resolvability,
    "C4b edge overwhelm": _viz_edge,
    "C5a location reach": _viz_coverage,
    "C5b width": _viz_coverage,
    "C5c transmission": _viz_coverage,
}


# ---------------------------------------------------------------- report rendering

_PATTERN_HINTS = (
    (
        {"C2 latent scale", "C5c transmission"},
        "shared input — both depend on the loading prior: C2's band divides by the "
        "loading median, and C5c's transmitted signal scales with the loading and where "
        "the latent mass lands on the link. Their joint failure places the inconsistency "
        "on the emission side.",
    ),
)


def render_report(title, results, art):
    _rows = "\n".join(
        f"| {r.check} | {CHECK_MODES[r.check]} | {r.target} | {r.value} | {r.band} | "
        f"{'✅' if r.passed else '❌'} |"
        for r in results
    )
    _failed = [r for r in results if not r.passed]
    _fb = []
    for _r in _failed:
        _mode = CHECK_MODES[_r.check]
        _fb.append(f"- **{_r.check}** ({_mode}) — {_r.note}")
        _fb.extend(f"    - {_line}" for _line in _r.diagnosis)
        if _mode == "soft":
            _fb.append(
                "    - *accepting means:* " + CHECK_CONSEQUENCES[_r.check].format(target=_r.target)
            )
    _failed_ids = {_r.check for _r in _failed}
    _fb.extend(
        f"- **differential** — {_txt}" for _pat, _txt in _PATTERN_HINTS if _pat <= _failed_ids
    )
    if _failed:
        _fb.append(
            "- *diagnostics are measurements, not recommendations: the revision "
            "decision belongs to the proposer, and any revised fragment is re-verified "
            "by the exact checks*"
        )
    _notes = "\n\n**Feedback to the proposer:**\n" + "\n".join(_fb) if _failed else ""
    _ann = (
        "\n\n**Annotations attached to the build state:**\n"
        + "\n".join(f"- {a}" for a in art["annotations"])
        if art["annotations"]
        else ""
    )
    _md = mo.md(
        f"### {title}\n\n"
        "| check | mode | target | prior-predictive value | band | verdict |\n"
        "|---|---|---|---|---|---|\n" + _rows + f"\n\n**Outcome: {art['outcome']}**" + _notes + _ann
    )
    _figs = []
    _seen = set()
    for _r in _failed:
        _fn = CHECK_VIZ.get(_r.check)
        if _fn is not None and id(_fn) not in _seen and _r.evidence is not None:
            _figs.append(mo.as_html(_fn(_r.evidence)))
            _seen.add(id(_fn))
    return mo.vstack([_md, *_figs])
