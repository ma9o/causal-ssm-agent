"""Rendering support for the blind case-study walkthroughs.

The case studies drive the *production* construct-admission engine
(:mod:`nof1_causal_lab.models.ssm.construct_admission`) and its reachability
battery (:mod:`nof1_causal_lab.models.ssm.reachability`). Those modules are
deliberately free of any plotting or notebook dependency, so the notebook-facing
presentation lives here: a markdown report table per admission attempt plus one
evidence figure per failed check family, reading the ``CheckResult.evidence``
dicts the battery attaches.

Severity modes and consequence texts are imported from the production tables —
this module renders them, it does not redefine them.
"""

from __future__ import annotations

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

from nof1_causal_lab.models.ssm.reachability import CHECK_CONSEQUENCES, CHECK_MODES

# ---------------------------------------------------------------- evidence figures


def _viz_confinement(ev):
    x, growth, dt = ev["x"], ev["growth"], ev["dt"]
    t = np.arange(x.shape[1]) * dt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.0))
    for row in x[:25]:
        ax0.plot(t, row, color="#c5c5c5", lw=0.6)
    for i in np.argsort(np.nan_to_num(growth, nan=np.inf))[-5:]:
        ax0.plot(t, x[i], color="#c0504d", lw=1.2)
    finite = x[np.isfinite(x)]
    if finite.size:
        lo_y, hi_y = np.percentile(finite, [0.1, 99.9])
        pad = 0.25 * (hi_y - lo_y + 1e-9)
        ax0.set_ylim(lo_y - pad, hi_y + pad)
    ax0.set_title("prior draws (gray) vs the 5 highest-growth draws (red)", fontsize=9)
    ax0.set_xlabel("day")
    finite_g = growth[np.isfinite(growth)]
    ax1.hist(np.clip(finite_g, 0, 20), bins=40, color="#3b6ea5")
    ax1.axvline(5.0, color="#c0504d", ls="--", label="growth gate ×5")
    ax1.set_title("late/early amplitude ratio per draw", fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_scale(ev):
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.hist(ev["sds"], bins=40, color="#3b6ea5")
    ax.axvline(ev["lo"], color="#c0504d", ls="--", label="band")
    ax.axvline(ev["hi"], color="#c0504d", ls="--")
    ax.axvline(ev["anchor"], color="#4a9d5b", lw=2, label="anchor")
    ax.set_title("per-draw stationary sd vs the scale-anchor band", fontsize=9)
    ax.set_xlabel("stationary sd")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_resolvability(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    tau = ev["tau"]
    hi_x = float(np.percentile(tau, 99))
    ax.hist(np.clip(tau, 0, hi_x), bins=50, color="#3b6ea5", label="prior τ = 1/decay")
    ax.axvspan(
        ev["lo"], min(ev["hi"], hi_x), color="#4a9d5b", alpha=0.12, label="resolvable window"
    )
    ax.axvline(ev["lo"], color="#c0504d", ls="--", label="cadence/3 floor")
    ax.axvline(ev["hi"], color="#7d6bb0", ls=":", label="span/4 ceiling")
    ax.axvline(ev["cadence"], color="#333333", ls="-", lw=0.8, label="observation cadence")
    ax.set_xlabel("self-relaxation τ (days)")
    ax.set_title("prior timescale vs the design's resolvable window", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_edge(ev):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.0))
    idx = np.arange(ev["on"].size)
    ax0.plot(idx, ev["on"], color="#3b6ea5", lw=1.4, label="edge on")
    ax0.plot(idx, ev["off"], color="#e08a3c", lw=1.4, ls="--", label="edge off (same noise)")
    ax0.set_xlabel("observation #")
    ax0.set_title("high-displacement draw: how much the edges move the child", fontsize=9)
    ax0.legend(frameon=False, fontsize=8)
    hi = float(np.percentile(ev["e"], 99))
    ax1.hist(np.clip(ev["e"], 0, hi), bins=40, color="#3b6ea5")
    ax1.axvline(0.95, color="#7d6bb0", ls=":", label="overwhelm cap")
    ax1.set_title("per-draw displacement / child scale", fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_saturation(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    parent = np.asarray(ev["parent"]).ravel()
    ax.hist(parent, bins=50, density=True, color="#c5c5c5", label="parent prior mass")
    ax.hist(ev["ec50"], bins=40, density=True, color="#3b6ea5", alpha=0.6, label="EC50 prior")
    ax.axvline(ev["p10"], color="#c0504d", ls="--", label="parent 10–90% range")
    ax.axvline(ev["p90"], color="#c0504d", ls="--")
    ax.set_title("Hill EC50 prior vs the parent's realized range", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_coverage(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.hist(ev["pp"], bins=60, density=True, color="#c5c5c5", label="prior predictive (pooled)")
    ax.hist(
        ev["signal"],
        bins=60,
        density=True,
        color="#4a9d5b",
        alpha=0.4,
        label="signal only (noise-free)",
    )
    ax.hist(ev["y_obs"], bins=20, density=True, color="#3b6ea5", alpha=0.6, label="observed")
    ax.axvline(ev["lo"], color="#c0504d", ls="--", label="pp [1,99]% band")
    ax.axvline(ev["hi"], color="#c0504d", ls="--")
    ax.set_title("prior predictive vs observed — location, width, transmission", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


CHECK_VIZ = {
    "C1a finiteness": _viz_confinement,
    "C1b confinement": _viz_confinement,
    "C2 latent scale": _viz_scale,
    "C3 resolvability": _viz_resolvability,
    "C4b edge overwhelm": _viz_edge,
    "C4c saturation": _viz_saturation,
    "C5a location reach": _viz_coverage,
    "C5b width": _viz_coverage,
    "C5c transmission": _viz_coverage,
}

_PATTERN_HINTS = (
    (
        {"C2 latent scale", "C5c transmission"},
        "shared input — both depend on where the latent's mass lands on the link: C2's "
        "band comes from the indicator's inverse-link spread, and C5c's transmitted signal "
        "scales with the same geometry. Their joint failure places the inconsistency on the "
        "emission side.",
    ),
)

# ---------------------------------------------------------------- report rendering


def render_report(title, report):
    """Render one :class:`AdmissionReport` as a marimo table + evidence figures."""
    results = report.results
    rows = "\n".join(
        f"| {r.check} | {CHECK_MODES[r.check]} | {r.target} | {r.value} | {r.band} | "
        f"{'✅' if r.passed else '❌'} |"
        for r in results
    )
    failed = [r for r in results if not r.passed]
    fb = []
    for r in failed:
        mode = CHECK_MODES[r.check]
        fb.append(f"- **{r.check}** ({mode}) — {r.note}")
        fb.extend(f"    - {line}" for line in r.diagnosis)
        if mode == "soft":
            fb.append(
                "    - *accepting means:* " + CHECK_CONSEQUENCES[r.check].format(target=r.target)
            )
    failed_ids = {r.check for r in failed}
    fb.extend(f"- **differential** — {txt}" for pat, txt in _PATTERN_HINTS if pat <= failed_ids)
    if failed:
        fb.append(
            "- *diagnostics are measurements, not recommendations: the revision "
            "decision belongs to the proposer, and any revised contribution is re-verified "
            "by the exact checks*"
        )
    notes = "\n\n**Feedback to the proposer:**\n" + "\n".join(fb) if failed else ""
    ann = (
        "\n\n**Annotations attached to the build state:**\n"
        + "\n".join(f"- {a}" for a in report.annotations)
        if report.annotations
        else ""
    )
    md = mo.md(
        f"### {title}\n\n"
        "| check | mode | target | prior-predictive value | band | verdict |\n"
        "|---|---|---|---|---|---|\n" + rows + f"\n\n**Outcome: {report.outcome}**" + notes + ann
    )
    figs = []
    seen = set()
    for r in failed:
        fn = CHECK_VIZ.get(r.check)
        if fn is not None and id(fn) not in seen and r.evidence is not None:
            figs.append(mo.as_html(fn(r.evidence)))
            seen.add(id(fn))
    return mo.vstack([md, *figs])
