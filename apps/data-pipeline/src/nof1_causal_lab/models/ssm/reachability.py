"""Prior-predictive reachability battery for gradual construct admission.

This is the single, authoritative implementation of the checks (the from-scratch
notebook prototype that seeded it has been retired; the blind case-study
walkthroughs under ``notebooks/`` now drive *this* module directly). The battery
is scoped to **reachability + one design-observability screen** — the recognized
remit of prior-predictive checking. Practical
identifiability verdicts (is a parameter prior-dominated / estimable from
``n_obs`` points) are deliberately NOT here: those belong post-fit (posterior
contraction, power-scaling).

The checks are **pure**: each takes arrays already produced by the exact
forward engine (Euler-Maruyama over the true nonlinear drift for latents,
Diffrax for the prior predictive) and returns :class:`CheckResult`s. Nothing is
simulated or linearized here — the caller (the model-spec construct reducer) feeds
these from ``sample_prior_predictive_from_runtime``. Keeping them array-in makes
them engine-agnostic and trivially testable, and keeps this module free of any
plotting or notebook dependency.

Severity and consequences are declarative tables (:data:`CHECK_MODES`,
:data:`CHECK_CONSEQUENCES`); :func:`stage_outcome` derives the admit / revise /
accept verdict from them plus the proposer's accepted-consequence decisions —
there is no status enum stored on any artifact.

Checks, by family:

- ``C1a``/``C1b`` — finiteness and self-calibrating confinement of the latent path.
  C1b's calibration pair (growth ratio, tolerated explosive fraction) is a design
  choice, not part of the statistic: the defaults encode the model class's
  confinement commitment (every self-dynamics component is a restoring force) and
  are threaded from the admission design so an intrinsically trending domain can
  recalibrate them without touching the check.
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
- ``C5c`` — transmission: what fraction of prior-predictive variation is carried
  by temporal movement in the emission mean rather than conditional observation
  variance? This check applies only to time-varying constructs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


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


def _robust_scale(values: np.ndarray, *, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Normal-consistent IQR scale without epsilon-denominator semantics."""
    q75 = np.percentile(values, 75, axis=axis)
    q25 = np.percentile(values, 25, axis=axis)
    return np.asarray((q75 - q25) / 1.349)


# C1b calibration: a draw "explodes" when its late-window amplitude exceeds
# ``growth_ratio`` times its own early amplitude, and the check fails when at
# least ``max_explosive_frac`` of draws do. Window-relative and generous — only
# near-exponential within-window growth trips the defaults.
C1B_GROWTH_RATIO = 5.0
C1B_MAX_EXPLOSIVE_FRAC = 0.01

# For additive, approximately Gaussian emissions, signal share is the square of
# the retired signal/predictive scale ratio. Four percent therefore preserves
# the former 20% scale threshold while making the statistic valid for sparse
# discrete channels.
C5C_MIN_SIGNAL_FRACTION = 0.04


def check_confinement(
    name: str,
    x: np.ndarray,
    times: np.ndarray,
    *,
    growth_ratio: float = C1B_GROWTH_RATIO,
    max_explosive_frac: float = C1B_MAX_EXPLOSIVE_FRAC,
) -> list[CheckResult]:
    """C1a finiteness + C1b confinement of a construct's latent trajectories."""
    _bad = ~np.isfinite(np.asarray(x))
    _nonfinite = float(np.mean(_bad))
    _raw = np.asarray(x)
    _xa = np.abs(_raw)
    _times = np.asarray(times, dtype=float)
    if _xa.ndim != 2 or _xa.shape[1] != _times.size or _times.size < 4:
        raise ValueError("confinement requires draw-by-time paths on at least four times")
    _q = _xa.shape[1] // 4
    _finite_draws = ~_bad.any(axis=1)
    _growth = np.full(_xa.shape[0], np.inf)
    if np.any(_finite_draws):
        _early = np.quantile(_xa[_finite_draws, _q : 2 * _q], 0.95, axis=1)
        _late = np.max(_xa[_finite_draws, -_q:], axis=1)
        _growth[_finite_draws] = _late / (_early + 1e-9)
    _explode = float(np.mean(_growth > growth_ratio))
    _ev = {
        "x": np.asarray(x),
        "growth": _growth,
        "times": _times,
        "growth_ratio": float(growth_ratio),
        "max_explosive_frac": float(max_explosive_frac),
    }
    _diag_a: tuple[str, ...] = ()
    if _nonfinite > 0.0:
        _bad_draws = _bad.any(axis=1)
        _onset_indices = np.argmax(_bad, axis=1)[_bad_draws]
        _onset = float(np.median(_times[_onset_indices]))
        _diag_a = (
            f"{float(np.mean(_bad_draws)):.0%} of prior draws go non-finite; median "
            f"onset t ≈ {_onset:.1f} d on the predictive output grid",
            "the exact Diffrax Heun solve returned a non-finite path; inspect the sampled "
            "drift, diffusion, and solver diagnostics for the implicated draws",
        )
    _diag_b: tuple[str, ...] = ()
    if _explode >= max_explosive_frac:
        _finite_growth = _growth[np.isfinite(_growth)]
        _growth_q99 = (
            float(np.percentile(_finite_growth, 99)) if _finite_growth.size else float("inf")
        )
        _diag_b = (
            f"{_explode:.1%} of draws end the window above {growth_ratio:g}× their own early "
            f"amplitude (finite-draw growth-ratio q99: {_growth_q99:.1f})",
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
            f"P(late/early amplitude > {growth_ratio:g}) {_explode:.1%}",
            f"<{max_explosive_frac:.1%} (self-calibrating growth)",
            _explode < max_explosive_frac,
            f"trajectories of {name} grow without settling within the study window.",
            _diag_b,
            _ev,
        ),
    ]


def check_scale(
    name: str,
    x: np.ndarray,
    scale_anchor: float = 1.0,
    anchor_src: str = "standardized-latent convention",
    anchor_detail: str = "latent marginal scale convention",
) -> CheckResult:
    """C2 — marginal prior scale against the standardized-latent convention."""
    _half = x.shape[1] // 2
    _scales = _robust_scale(np.asarray(x)[:, _half:], axis=0)
    _med = float(np.median(_scales))
    _q05, _q95 = np.percentile(_scales, [5, 95])
    _lo, _hi = scale_anchor / 3.0, scale_anchor * 3.0
    _ok = bool(_lo <= _med <= _hi)
    _ev = {"marginal_scales": _scales, "lo": _lo, "hi": _hi, "anchor": scale_anchor}
    _diag: tuple[str, ...] = ()
    if not _ok:
        _side = "above" if _med > _hi else "below"
        _factor = _med / max(_hi, 1e-9) if _med > _hi else _lo / max(_med, 1e-9)
        _diag = (
            f"prior-predictive marginal scale: median {_med:.2f} (5–95% "
            f"{_q05:.2f}–{_q95:.2f}) vs band [{_lo:.2f}, {_hi:.2f}] — {_factor:.1f}× "
            f"{_side} the edge",
            f"band derivation: {anchor_detail}",
            "dependence: the statistic rises with the diffusion prior and falls with "
            "the stiffness prior (incoming edges add parent variance); the band scales "
            "inversely with the prior-median loading — this red can equally reflect a "
            "dynamics–emission inconsistency",
            "emission-to-data compatibility is evaluated separately by the C5 replicated-data "
            "checks",
        )
    return CheckResult(
        "C2 latent scale",
        name,
        f"median sd {_med:.2f} (5–95%: {_q05:.2f}–{_q95:.2f})",
        f"[{_lo:.2f}, {_hi:.2f}] ({anchor_src})",
        _ok,
        f"marginal prior scale of {name} is inconsistent with the standardized latent "
        f"convention ({anchor_src}).",
        _diag,
        _ev,
    )


def check_resolvability(
    name: str,
    tau_draws: np.ndarray,
    observation_times: np.ndarray,
    *,
    min_resolvable_mass: float = 0.8,
) -> CheckResult:
    """C3 — prior mass resolvable by this construct's actual irregular schedule."""
    _tau = np.asarray(tau_draws, dtype=float)
    _times = np.unique(np.asarray(observation_times, dtype=float))
    if _times.size < 2:
        return CheckResult(
            "C3 resolvability",
            name,
            f"{_times.size} distinct observation time(s)",
            ">= 2 distinct times",
            False,
            f"the schedule for {name} contains too few distinct observations to resolve dynamics.",
            ("no temporal contrast is available for this construct",),
            {"observation_times": _times},
        )
    _gaps = np.diff(_times)
    _span = float(np.ptp(_times))
    _med = float(np.median(_tau))
    _q10, _q90 = (float(_v) for _v in np.percentile(_tau, [10, 90]))
    _gap_resolved = np.mean(_gaps[None, :] <= 3.0 * _tau[:, None], axis=1) >= 0.5
    _span_resolved = _span >= 4.0 * _tau
    _resolved = _gap_resolved & _span_resolved
    _frac_in = float(np.mean(_resolved))
    _ok = bool(_frac_in >= min_resolvable_mass)
    _ev = {"tau": _tau, "gaps": _gaps, "span": _span, "resolved": _resolved}
    _diag: tuple[str, ...] = ()
    _median_gap = float(np.median(_gaps))
    if not _ok and np.mean(_gap_resolved) < np.mean(_span_resolved):
        _diag = (
            f"prior self-relaxation τ: median {_med:.2f} d (10–90% {_q10:.2f}–{_q90:.2f}) "
            f"is too fast for the construct's actual gaps (median {_median_gap:.2f} d); "
            f"{_frac_in:.0%} of prior mass is resolvable",
            "reading: the prior posits dynamics faster than the sampling can follow — the "
            "process relaxes at least three times across most adjacent observations",
            "this is a prior/design mismatch, not an estimate: the observed autocorrelation "
            "mixes this node's own relaxation with inherited parent persistence, and that "
            "split is resolved only by the joint fit — confirm with post-fit contraction",
        )
    elif not _ok:
        _diag = (
            f"prior self-relaxation τ: median {_med:.2f} d (10–90% {_q10:.2f}–{_q90:.2f}) "
            f"is too slow for four replications within span {_span:.0f} d; "
            f"{_frac_in:.0%} of prior mass is resolvable",
            "reading: the prior posits dynamics so slow the window holds < ~4 relaxation "
            "times — the process is near-frozen over the record, so its timescale and "
            "stationary law are not resolvable by this design",
        )
    return CheckResult(
        "C3 resolvability",
        name,
        f"prior τ median {_med:.2f} d (10–90% {_q10:.2f}–{_q90:.2f}); {_frac_in:.0%} resolvable",
        f">= {min_resolvable_mass:.0%} of prior mass resolved by actual gaps and span",
        _ok,
        f"the timescale posited for {name} lies outside the window this sampling design can "
        "resolve; the fit cannot inform its dynamics from this schedule.",
        _diag,
        _ev,
    )


def check_edge_share(
    edge_label: str, x_on_obs: np.ndarray, x_off_obs: np.ndarray
) -> list[CheckResult]:
    """C4b edge overwhelm — is the child's path variation slaved to a parent (degenerate prior)?

    C4a edge *detectability* was dropped: its 2/√n_obs SNR floor is a data-quantity
    detectability threshold — practical identifiability of the edge weight, which belongs to
    the post-fit gate (posterior contraction on the weight), not the prior-predictive stage.
    Overwhelm stays because a child fully slaved to a parent is a degenerate *prior*.
    """
    _a = np.asarray(x_on_obs)
    _b = np.asarray(x_off_obs)
    _delta = _a - _b
    _level = np.abs(np.median(_delta, axis=1))
    _centered_delta = _delta - np.median(_delta, axis=1, keepdims=True)
    _centered_on = _a - np.median(_a, axis=1, keepdims=True)
    _disp = _robust_scale(_centered_delta, axis=1)
    _scale = _robust_scale(_centered_on, axis=1)
    _e = np.zeros_like(_disp)
    np.divide(_disp, _scale, out=_e, where=_scale > 0)
    _e[(_scale == 0) & (_disp > 0)] = np.inf
    _med = float(np.median(_e))
    _i90 = int(np.argsort(_e)[int(0.9 * (_e.size - 1))])
    _ev = {"e": _e, "level_shift": _level, "on": _a[_i90], "off": _b[_i90]}
    _diag_b: tuple[str, ...] = ()
    if _med > 0.95:
        _diag_b = (
            f"for the median prior draw the edge changes {_med:.0%} as much temporal "
            "variation as the child path itself",
            "dependence: the statistic falls with the edge-weight prior scale and "
            "rises when the child's own stiffness/diffusion contribute little",
        )
    return [
        CheckResult(
            "C4b edge overwhelm",
            edge_label,
            f"edge path displacement / child scale: median {_med:.1%}",
            "median ≤ 95%",
            bool(_med <= 0.95),
            f"the {edge_label} input dominates the child's temporal variation; its self-dynamics "
            "are left uninformed.",
            _diag_b,
            _ev,
        ),
    ]


def check_saturation(
    edge_label: str,
    ec50_draws: np.ndarray,
    hill_n_draws: np.ndarray,
    parent_values: np.ndarray,
    *,
    min_exercised_mass: float = 0.8,
) -> CheckResult:
    """C4c — draw-paired Hill occupancy over the actual clamped input."""
    _ec50 = np.asarray(ec50_draws, dtype=float).reshape(-1)
    _n = np.asarray(hill_n_draws, dtype=float).reshape(-1)
    _parent = np.maximum(np.asarray(parent_values, dtype=float), 0.0)
    if _parent.shape[0] != _ec50.size or _n.size != _ec50.size:
        raise ValueError("Hill saturation inputs must preserve draw-wise parameter/path pairing")
    _log_parent = np.full_like(_parent, -np.inf)
    np.log(_parent, out=_log_parent, where=_parent > 0)
    _logit = _n[:, None] * (_log_parent - np.log(_ec50)[:, None])
    _occupancy = 1.0 / (1.0 + np.exp(-np.clip(_logit, -60.0, 60.0)))
    _occupancy[_parent == 0] = 0.0
    _bend_mass = np.mean((_occupancy >= 0.1) & (_occupancy <= 0.9), axis=1)
    _exercised = _bend_mass >= 0.1
    _exercised_mass = float(np.mean(_exercised))
    _dead_low = float(np.mean(np.quantile(_occupancy, 0.9, axis=1) < 0.1))
    _saturated_high = float(np.mean(np.quantile(_occupancy, 0.1, axis=1) > 0.9))
    _med = float(np.median(_ec50))
    _ok = bool(_exercised_mass >= min_exercised_mass)
    _ev = {"ec50": _ec50, "hill_n": _n, "bend_mass": _bend_mass}
    _diag: tuple[str, ...] = ()
    if not _ok:
        _diag = (
            f"only {_exercised_mass:.0%} of paired prior draws spend at least 10% of the "
            "schedule on the Hill bend",
            f"draw classification: {_dead_low:.0%} dead-low and {_saturated_high:.0%} "
            "flat-saturated",
            "dependence: shift the EC50 prior toward the parent's realized range, or drop "
            "the Hill form for a linear edge if the bend is not exercised",
        )
    return CheckResult(
        "C4c saturation",
        edge_label,
        f"EC50 median {_med:.2f}; bend exercised in {_exercised_mass:.0%} of paired draws",
        f">= {min_exercised_mass:.0%} of paired draws exercise the bend",
        _ok,
        f"the saturating edge {edge_label} is not exercised over the parent's prior range; "
        "its nonlinearity is either a dead linear arm or a flat saturated response.",
        _diag,
        _ev,
    )


def check_coverage(
    indicator: str,
    pp_y: np.ndarray,
    y_obs: np.ndarray,
    *,
    distribution: str,
    level_count: int | None = None,
) -> list[CheckResult]:
    """C5a/C5b replicated-data location and family-aware dispersion."""
    _pp_matrix = np.asarray(pp_y, dtype=float)
    _obs = np.asarray(y_obs, dtype=float).reshape(-1)
    if _pp_matrix.ndim != 2 or _pp_matrix.shape[1] != _obs.size:
        raise ValueError("coverage requires draws by observed-time replicates")

    def _band(values: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(values, [1, 99])
        return float(lo), float(hi)

    def _inside(value: float, band: tuple[float, float]) -> bool:
        return bool(band[0] <= value <= band[1])

    _categorical = distribution in {"bernoulli", "ordered_logistic", "categorical"}
    _count = distribution in {"poisson", "negative_binomial"}
    if _categorical:
        _levels = level_count or (2 if distribution == "bernoulli" else 0)
        if _levels < 2:
            raise ValueError(f"{distribution} coverage requires declared level count")

        def _frequencies(values: np.ndarray) -> np.ndarray:
            return np.stack(
                [np.mean(values == level, axis=-1) for level in range(_levels)], axis=-1
            )

        _rep_freq = _frequencies(_pp_matrix)
        _obs_freq = _frequencies(_obs[None, :])[0]
        _prior_freq = np.mean(_rep_freq, axis=0)
        _rep_location = 0.5 * np.sum(np.abs(_rep_freq - _prior_freq), axis=1)
        _obs_location = float(0.5 * np.sum(np.abs(_obs_freq - _prior_freq)))
        _location_band = (0.0, float(np.percentile(_rep_location, 99)))
        _location_ok = _inside(_obs_location, _location_band)
        _eps = 1e-12
        _rep_width = -np.sum(_rep_freq * np.log(_rep_freq + _eps), axis=1)
        _obs_width = float(-np.sum(_obs_freq * np.log(_obs_freq + _eps)))
        _width_band = _band(_rep_width)
        _width_ok = _inside(_obs_width, _width_band)
        _location_value = f"frequency TV from prior center {_obs_location:.2f}"
        _location_band_text = f"≤ {_location_band[1]:.2f} (99% replicate envelope)"
        _width_value = f"category entropy {_obs_width:.2f}"
        _width_band_text = f"[{_width_band[0]:.2f}, {_width_band[1]:.2f}] replicate envelope"
    elif _count:
        _rep_location = np.mean(_pp_matrix, axis=1)
        _obs_location = float(np.mean(_obs))
        _location_band = _band(_rep_location)
        _location_ok = _inside(_obs_location, _location_band)
        _rep_variance = np.var(_pp_matrix, axis=1)
        _obs_variance = float(np.var(_obs))
        _variance_band = _band(_rep_variance)
        _rep_zero = np.mean(_pp_matrix == 0, axis=1)
        _obs_zero = float(np.mean(_obs == 0))
        _zero_band = _band(_rep_zero)
        _width_ok = _inside(_obs_variance, _variance_band) and _inside(_obs_zero, _zero_band)
        _location_value = f"observed mean {_obs_location:.2f}"
        _location_band_text = (
            f"[{_location_band[0]:.2f}, {_location_band[1]:.2f}] replicate envelope"
        )
        _width_value = f"variance {_obs_variance:.2f}; zero fraction {_obs_zero:.0%}"
        _width_band_text = (
            f"variance [{_variance_band[0]:.2f}, {_variance_band[1]:.2f}], "
            f"zero fraction [{_zero_band[0]:.0%}, {_zero_band[1]:.0%}]"
        )
    else:
        _rep_location = np.median(_pp_matrix, axis=1)
        _obs_location = float(np.median(_obs))
        _location_band = _band(_rep_location)
        _location_ok = _inside(_obs_location, _location_band)
        _rep_width = _robust_scale(_pp_matrix, axis=1)
        _obs_width = float(_robust_scale(_obs))
        _width_band = _band(_rep_width)
        _width_ok = _inside(_obs_width, _width_band)
        _location_value = f"observed median {_obs_location:.2f}"
        _location_band_text = (
            f"[{_location_band[0]:.2f}, {_location_band[1]:.2f}] replicate envelope"
        )
        _width_value = f"robust scale {_obs_width:.2f}"
        _width_band_text = f"[{_width_band[0]:.2f}, {_width_band[1]:.2f}] replicate envelope"

    _pp = _pp_matrix.ravel()
    _ev = {
        "pp": _pp[:: max(1, _pp.size // 20000)],
        "y_obs": _obs,
        "replicate_location": _rep_location,
    }
    _diag_a: tuple[str, ...] = ()
    if not _location_ok:
        _diag_a = (
            f"the observed replicated-data location statistic ({_location_value}) lies outside "
            f"the prior replicate envelope {_location_band_text}",
            "this indicates little prior mass near the observed location; it is not a proof "
            "that a continuous-support prior makes the data impossible",
        )
    _diag_b: tuple[str, ...] = ()
    if not _width_ok:
        _diag_b = (
            f"the observed dispersion statistic ({_width_value}) lies outside the "
            f"prior replicate envelope {_width_band_text}",
            "the family-specific statistic avoids ratios against a zero empirical IQR",
        )
    return [
        CheckResult(
            "C5a location reach",
            indicator,
            _location_value,
            _location_band_text,
            _location_ok,
            f"the prior predictive puts little mass near the location of {indicator}.",
            _diag_a,
            _ev,
        ),
        CheckResult(
            "C5b width",
            indicator,
            _width_value,
            _width_band_text,
            _width_ok,
            f"prior-predictive spread for {indicator} is out of proportion to the observed spread.",
            _diag_b,
            _ev,
        ),
    ]


def check_transmission(
    indicator: str,
    signal_y: np.ndarray,
    conditional_variance_y: np.ndarray | None = None,
    *,
    min_signal_fraction: float = C5C_MIN_SIGNAL_FRACTION,
) -> CheckResult:
    """C5c — fraction of predictive variation attributable to temporal signal movement.

    Scalar emissions use the law-of-total-variance decomposition within each
    prior draw. Categorical emissions use its label-invariant one-hot analogue:
    probability-vector resolution divided by resolution plus conditional Gini
    uncertainty. The caller omits this check for time-invariant constructs.
    """
    _sig = np.asarray(signal_y, dtype=float)
    if _sig.ndim not in {2, 3}:
        raise ValueError("transmission signal requires draws by observed-time values")
    if not 0.0 <= min_signal_fraction <= 1.0:
        raise ValueError("min_signal_fraction must lie in [0, 1]")
    if np.any(~np.isfinite(_sig)):
        raise ValueError("transmission signal must be finite")

    if _sig.ndim == 3:
        if _sig.shape[2] < 2:
            raise ValueError("categorical transmission requires at least two levels")
        if np.any(_sig < 0.0) or not np.allclose(np.sum(_sig, axis=2), 1.0, atol=1e-6):
            raise ValueError("categorical transmission requires probability vectors")
        _center = np.mean(_sig, axis=1, keepdims=True)
        _signal_variance = np.mean(np.sum((_sig - _center) ** 2, axis=2), axis=1)
        _conditional_variance = np.mean(1.0 - np.sum(_sig**2, axis=2), axis=1)
    else:
        if conditional_variance_y is None:
            raise ValueError("scalar transmission requires conditional observation variance")
        _conditional = np.asarray(conditional_variance_y, dtype=float)
        if _conditional.shape != _sig.shape:
            raise ValueError("conditional variance must match the scalar signal shape")
        if np.any(np.isnan(_conditional)) or np.any(_conditional < 0.0):
            raise ValueError("conditional observation variance must be non-negative")
        _signal_variance = np.var(_sig, axis=1)
        _conditional_variance = np.mean(_conditional, axis=1)

    _total_variance = _signal_variance + _conditional_variance
    _signal_fraction = np.divide(
        _signal_variance,
        _total_variance,
        out=np.zeros_like(_signal_variance),
        where=np.isfinite(_total_variance) & (_total_variance > 0.0),
    )
    _transmit = float(np.median(_signal_fraction))
    _trans_ok = bool(_transmit >= min_signal_fraction)
    _transmit_value = f"median temporal signal share {_transmit:.1%}"
    _transmit_band = f">= {min_signal_fraction:.0%} of predictive variation"
    _diag: tuple[str, ...] = ()
    if not _trans_ok:
        _diag = (
            f"too little predictive variation comes from temporal movement in the conditional "
            f"emission mean ({_transmit_value})",
            "possible causes include a weak loading, a link operating in a flat or saturated "
            "region, or broad conditional observation variance",
            "dependence: C2 constrains the latent scale and C5b checks total predictive width; "
            "C5c decomposes that width into temporal signal and conditional observation variance",
        )
    return CheckResult(
        "C5c transmission",
        indicator,
        _transmit_value,
        _transmit_band,
        _trans_ok,
        f"the emission for {indicator} carries little temporal latent information relative "
        "to its conditional observation variance.",
        _diag,
        {
            "signal_fraction": _signal_fraction,
            "signal_variance": _signal_variance,
            "conditional_variance": _conditional_variance,
            "min_signal_fraction": min_signal_fraction,
        },
    )


def check_data_availability(indicator: str) -> CheckResult:
    """C5d — surface an emission channel with no observed values as prior-only."""
    return CheckResult(
        "C5d data availability",
        indicator,
        "0 observed values",
        "> 0 observed values",
        False,
        f"{indicator} has no observed values, so its emission and linked latent state are "
        "not empirically anchored in this panel.",
        (
            "the likelihood remains executable for forward simulation, but the current "
            "panel contributes no likelihood terms for this indicator",
            "treat trajectory and magnitude statements as prior-driven unless other measured "
            "states identify them through the structural model",
        ),
        {"n_obs": 0},
    )


# ---------------------------------------------------------------- severity (declarative)

CHECK_MODES = {
    "C1a finiteness": "hard",
    "C1b confinement": "soft",
    "C2 latent scale": "soft",
    "C3 resolvability": "soft",
    "C4b edge overwhelm": "soft",
    "C4c saturation": "soft",
    "C5a location reach": "soft",
    "C5b width": "soft",
    "C5c transmission": "soft",
    "C5d data availability": "soft",
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
    "C5a location reach": "{target}: little prior-predictive mass lies near the observed "
    "location; posterior adaptation may be prior-sensitive",
    "C5b width": "{target}: prior-predictive width imbalance accepted; expect weak "
    "regularization or slow warmup",
    "C5c transmission": "{target}: little prior-predictive variation comes from temporal "
    "movement in the emission mean; conditional observation variance dominates, so this "
    "construct's trajectory is weakly grounded in the data",
    "C5d data availability": "{target}: no observed values anchor this emission in the current "
    "panel; its contribution is prior-driven and must not be presented as empirically learned",
}


def stage_outcome(
    results: list[CheckResult],
    accepted: Mapping[tuple[str, str], str],
) -> tuple[str, tuple[str, ...]]:
    """Derive the admit / revise / accept verdict + carried annotations from the tables.

    Hard failure blocks (no override). An unaccepted soft failure needs a decision (revise or
    accept the consequence). Accepted soft failures become build-state annotations. The verdict
    is a returned string over the checks + the proposer's decisions, never an enum stored on an
    artifact.
    """
    _failed = [r for r in results if not r.passed]
    _hard = [r for r in _failed if CHECK_MODES[r.check] == "hard"]
    _pending = [
        r for r in _failed if CHECK_MODES[r.check] == "soft" and (r.check, r.target) not in accepted
    ]
    _annotations = tuple(
        CHECK_CONSEQUENCES[r.check].format(target=r.target)
        + f" [rationale: {accepted[(r.check, r.target)]}]"
        for r in _failed
        if CHECK_MODES[r.check] == "soft" and (r.check, r.target) in accepted
    )
    if _hard:
        return "BLOCKED — hard failure: revise the fragment (no override)", _annotations
    if _pending:
        _ids = ", ".join(f"{r.check} [{r.target}]" for r in _pending)
        return (
            f"NEEDS DECISION — revise the fragment or accept the consequence ({_ids})",
            _annotations,
        )
    if _annotations:
        return "ADMITTED with accepted consequences", _annotations
    return "ADMITTED", _annotations
