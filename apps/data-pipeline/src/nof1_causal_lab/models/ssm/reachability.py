"""Prior-predictive reachability battery for gradual construct admission.

This is the production port of the checks validated in
``notebooks/gradual_build_tools.py`` (which keeps the from-scratch pedagogical
copy). The battery is scoped to **reachability + one design-observability
screen** — the recognized remit of prior-predictive checking. Practical
identifiability verdicts (is a parameter prior-dominated / estimable from
``n_obs`` points) are deliberately NOT here: those belong post-fit (posterior
contraction, power-scaling).

The checks are **pure**: each takes arrays already produced by the exact
forward engine (Euler-Maruyama over the true nonlinear drift for latents,
Diffrax for the prior predictive) and returns :class:`CheckResult`s. Nothing is
simulated or linearized here — the caller (the Stage 4 construct reducer) feeds
these from ``sample_prior_predictive_from_runtime``. Keeping them array-in makes
them engine-agnostic and trivially testable, and keeps this module free of any
plotting or notebook dependency.

Severity and consequences are declarative tables (:data:`CHECK_MODES`,
:data:`CHECK_CONSEQUENCES`); :func:`stage_outcome` derives the admit / revise /
accept verdict from them plus the proposer's accepted-consequence decisions —
there is no status enum stored on any artifact.

Checks, by family:

- ``C1a``/``C1b`` — finiteness and self-calibrating confinement of the latent path.
- ``C2`` — the construct's stationary latent scale vs the scale its indicator implies.
- ``C3`` — design-resolvability: is the prior's self-relaxation τ inside
  ``[cadence/3, span/4]``? Schedule-only; does not estimate τ (the observed
  autocorrelation mixes self and inherited dynamics, an unidentified split left
  to the fit).
- ``C4b`` — edge overwhelm (a parent slaving the child is a degenerate prior).
- ``C4c`` — Hill saturation-exercised: is the EC50 inside the parent's realized
  range, so the saturation is actually exercised (not a dead linear arm or a
  flat saturated response)?
- ``C5a``/``C5b`` — location reach and width of the prior predictive vs the data.
- ``C5c`` — transmission: can the link, driven by the latent's prior mass,
  produce a signal comparable to the observed variation, or is it saturated /
  flat (the structural kernel salvaged from the retired C6, noise-free).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class CheckResult:
    """One reachability check outcome (a measurement, not a recommendation)."""

    check: str
    target: str
    value: str
    band: str
    passed: bool
    note: str
    diagnosis: tuple[str, ...] = ()
    evidence: dict | None = None


def check_confinement(name: str, x: np.ndarray, dt: float) -> list[CheckResult]:
    """C1a finiteness + C1b confinement of a construct's latent trajectories."""
    _bad = ~np.isfinite(np.asarray(x))
    _nonfinite = float(np.mean(_bad))
    _xa = np.abs(np.nan_to_num(np.asarray(x), nan=np.inf, posinf=np.inf, neginf=np.inf))
    _q = _xa.shape[1] // 4
    _early = np.quantile(_xa[:, _q : 2 * _q], 0.95, axis=1)
    _late = np.max(_xa[:, -_q:], axis=1)
    _growth = _late / (_early + 1e-9)
    _explode = float(np.mean(_growth > 5.0))
    _ev = {"x": np.asarray(x), "growth": _growth, "dt": dt}
    _diag_a: tuple[str, ...] = ()
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
    _diag_b: tuple[str, ...] = ()
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


def check_scale(
    name: str,
    x: np.ndarray,
    scale_anchor: float,
    anchor_src: str,
    anchor_detail: str,
) -> CheckResult:
    """C2 — the construct's stationary latent scale vs its data-implied anchor."""
    _half = x.shape[1] // 2
    _sds = jnp.std(x[:, _half:], axis=1)
    _med = float(jnp.median(_sds))
    _q05, _q95 = np.percentile(np.asarray(_sds), [5, 95])
    _lo, _hi = scale_anchor / 3.0, scale_anchor * 3.0
    _ok = bool(_lo <= _med <= _hi)
    _ev = {"sds": np.asarray(_sds), "lo": _lo, "hi": _hi, "anchor": scale_anchor}
    _diag: tuple[str, ...] = ()
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
            "related evidence: C5c on the same indicator also takes the loading prior as "
            "input; the joint (C2, C5c) pattern localizes which side is active",
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


def check_resolvability(
    name: str,
    tau_draws: np.ndarray,
    cadence: float,
    span: float,
) -> CheckResult:
    """C3 — design-observability screen (schedule-only): is the prior's self-relaxation
    timescale inside the window this sampling design can resolve?

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
    _diag: tuple[str, ...] = ()
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


def check_edge_share(name: str, x_on_obs: np.ndarray, x_off_obs: np.ndarray) -> list[CheckResult]:
    """C4b edge overwhelm — is the child's path variation slaved to a parent (degenerate prior)?

    C4a edge *detectability* was dropped: its 2/√n_obs SNR floor is a data-quantity
    detectability threshold — practical identifiability of the edge weight, which belongs to
    the post-fit gate (posterior contraction on the weight), not the prior-predictive stage.
    Overwhelm stays because a child fully slaved to a parent is a degenerate *prior*.
    """
    _a = np.asarray(x_on_obs)
    _b = np.asarray(x_off_obs)
    _disp = np.sqrt(np.mean((_a - _b) ** 2, axis=1))
    _scale = np.sqrt(np.var(_a, axis=1)) + 1e-12
    _e = _disp / _scale
    _med = float(np.median(_e))
    _i90 = int(np.argsort(_e)[int(0.9 * (_e.size - 1))])
    _ev = {"e": _e, "on": _a[_i90], "off": _b[_i90]}
    _diag_b: tuple[str, ...] = ()
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


def check_saturation(
    edge_label: str,
    ec50_draws: np.ndarray,
    parent_values: np.ndarray,
) -> CheckResult:
    """C4c — is a Hill (saturating) edge's operating point actually exercised by the parent?

    A Hill edge only earns its extra parameters if its EC50 (half-saturation point) sits
    inside the range the parent's prior mass actually visits. If EC50 ≫ the parent's realized
    range the response never bends — it is an effectively-linear dead arm, and a LinearEdge
    would be the honest form. If EC50 ≪ the range the child sees a flat, fully-saturated
    response — the gradient the edge is supposed to carry is gone. This is a reachability
    question (does the nonlinearity reach the data-relevant region), schedule/noise-free, with
    no ``n_obs`` SNR threshold.
    """
    _ec50 = np.asarray(ec50_draws, dtype=float)
    _parent = np.asarray(parent_values, dtype=float)
    _ec50 = _ec50[np.isfinite(_ec50)]
    _med = float(np.median(_ec50)) if _ec50.size else float("nan")
    _p10, _p90 = (float(_v) for _v in np.percentile(_parent, [10, 90]))
    _ok = bool(_p10 <= _med <= _p90)
    _ev = {"ec50": _ec50, "p10": _p10, "p90": _p90, "parent": _parent}
    _diag: tuple[str, ...] = ()
    if not _ok:
        _where = "above" if _med > _p90 else "below"
        _reading = (
            "the response never bends over the parent's range — an effectively-linear dead "
            "arm; a linear edge is the honest form"
            if _med > _p90
            else "the parent sits on the flat saturated arm — the response carries no "
            "gradient over its realized range"
        )
        _diag = (
            f"prior EC50 median {_med:.2f} sits {_where} the parent's realized "
            f"10–90% range [{_p10:.2f}, {_p90:.2f}]",
            f"reading: {_reading}",
            "dependence: shift the EC50 prior toward the parent's realized range, or drop "
            "the Hill form for a linear edge if the bend is not exercised",
        )
    return CheckResult(
        "C4c saturation",
        edge_label,
        f"EC50 median {_med:.2f} vs parent 10–90% [{_p10:.2f}, {_p90:.2f}]",
        "EC50 inside parent range",
        _ok,
        f"the saturating edge {edge_label} is not exercised over the parent's prior range; "
        "its nonlinearity is either a dead linear arm or a flat saturated response.",
        _diag,
        _ev,
    )


def check_coverage(
    indicator: str,
    pp_y: np.ndarray,
    signal_y: np.ndarray,
    y_obs: np.ndarray,
) -> list[CheckResult]:
    """C5a location reach + C5b width + C5c transmission for one indicator."""
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
    # flat so the observed spread would have to be almost all measurement noise? C5b can pass on
    # a saturated link because noise widens the predictive band; this catches what C5b misses.
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
    _diag_a: tuple[str, ...] = ()
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
    _diag_b: tuple[str, ...] = ()
    if not _width_ok:
        _diag_b = (
            f"prior-predictive IQR is {_ratio:.2f}× the data IQR (band [0.33, 50])",
            "dependence: predictive width compounds the loading spread × latent scale, "
            "the intercept spread, and the noise scale (for count families the noise "
            "is tied to the rate); the data-side IQR is fixed",
        )
    _diag_c: tuple[str, ...] = ()
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
    "C4c saturation": "soft",
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
    "C4c saturation": "{target}: the saturating edge's bend is not exercised over the "
    "parent's prior range; treat it as effectively linear (or narrow the EC50 prior) — the "
    "extra Hill parameters are weakly informed",
    "C5b width": "{target}: prior-predictive width imbalance accepted; expect weak "
    "regularization or slow warmup",
    "C5c transmission": "{target}: the link passes little of the latent's variation to the "
    "indicator; the observed spread is largely measurement noise, so this construct's "
    "trajectory is weakly grounded in the data",
}


def stage_outcome(
    results: list[CheckResult],
    accepted: dict[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Derive the admit / revise / accept verdict + carried annotations from the tables.

    Hard failure blocks (no override). An unaccepted soft failure needs a decision (revise or
    accept the consequence). Accepted soft failures become build-state annotations. The verdict
    is a returned string over the checks + the proposer's decisions, never an enum stored on an
    artifact.
    """
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
