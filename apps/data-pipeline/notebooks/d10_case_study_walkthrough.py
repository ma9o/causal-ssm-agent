import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    from pathlib import Path

    import gradual_build_tools as gbt
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, gbt, jax, jnp, np, plt


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # A blind D = 10 case study, built with the staged checks

    This notebook stress-tests the incremental model-building workflow — the fragments,
    exact-simulation checks C1–C6, declarative severity, and evidence-only diagnostics
    devised in `gradual_model_building_lab.py` — on a **larger, blind** problem, using the
    exact same tooling imported from `gradual_build_tools.py`.

    **The blind protocol.** A separate agent designed a hidden D = 10 continuous-time
    **nonlinear, non-Gaussian** ground truth (a single-subject behavioral/physiological
    story), generated 120 days of irregular data, and wrote a study brief. That agent
    operated behind an information firewall: everything below is built from the brief and
    from *legitimate summaries of the observed data only*. The generator and its parameters
    live in `data/d10_case_study/hidden/` and were **never opened** — so the priors here are
    a genuine blind elicitation, not reverse-engineered from the answer.

    **What "success" means here.** Passing all checks does **not** mean the priors match the
    hidden truth — we cannot see it. It means the priors are internally consistent and
    data-reachable *before any fit*: every construct is on a plausible scale, its dynamics
    are visible at the sampling cadence, its edges are detectable without overwhelming it,
    and its indicator carries information about it. That is exactly what a prior-predictive
    gate can certify, and all this notebook claims. Recovery is a separate, post-fit question.

    The trail is left intact, including a real revision episode where a check caught a
    genuine mistake in the first-pass priors.
    """)
    return


@app.cell(hide_code=True)
def firewall_md(mo):
    mo.md(r"""
    ## 1. The brief (all the modeler is allowed to see)

    Below is the verbatim study brief. It gives the constructs, the causal DAG, the indicator
    families, and the observation design — and deliberately **no** parameter values, scales,
    timescales, or hints about where the nonlinearities live.
    """)
    return


@app.cell
def brief_render(Path, mo):
    _brief = Path("notebooks/data/d10_case_study/brief.md")
    if not _brief.exists():
        _brief = Path("data/d10_case_study/brief.md")
    brief_text = _brief.read_text()
    mo.accordion({"📄 brief.md (click to expand)": mo.md(brief_text)})
    return (brief_text,)


@app.cell
def dag_spec():
    EDGES = [
        ("CaffeineIntake", "SleepQuality"),
        ("AutonomicArousal", "PerceivedStress"),
        ("AutonomicArousal", "SleepQuality"),
        ("AutonomicArousal", "MusculoskeletalPain"),
        ("PerceivedStress", "SleepQuality"),
        ("PerceivedStress", "NegativeMood"),
        ("PerceivedStress", "Fatigue"),
        ("SleepQuality", "Fatigue"),
        ("Fatigue", "MusculoskeletalPain"),
        ("Fatigue", "PhysicalActivity"),
        ("Fatigue", "CognitiveFocus"),
        ("MusculoskeletalPain", "PhysicalActivity"),
        ("PhysicalActivity", "NegativeMood"),
        ("NegativeMood", "SocialEngagement"),
        ("NegativeMood", "CognitiveFocus"),
    ]
    ORDER = [
        "CaffeineIntake",
        "AutonomicArousal",
        "PerceivedStress",
        "SleepQuality",
        "Fatigue",
        "MusculoskeletalPain",
        "PhysicalActivity",
        "NegativeMood",
        "CognitiveFocus",
        "SocialEngagement",
    ]
    UNOBSERVED = {"AutonomicArousal"}
    return EDGES, ORDER, UNOBSERVED


@app.cell
def dag_diagram(EDGES, ORDER, UNOBSERVED, mo, plt):
    _parents = {n: [] for n in ORDER}
    for _u, _v in EDGES:
        _parents[_v].append(_u)
    _depth = {}
    for _n in ORDER:
        _depth[_n] = 0 if not _parents[_n] else 1 + max(_depth[_p] for _p in _parents[_n])
    _by_d = {}
    for _n in ORDER:
        _by_d.setdefault(_depth[_n], []).append(_n)
    _pos = {}
    for _d, _ns in _by_d.items():
        for _k, _n in enumerate(_ns):
            _pos[_n] = (_d * 1.9, (_k - (len(_ns) - 1) / 2) * 1.7)

    _fig, _ax = plt.subplots(figsize=(11.5, 5.2))
    for _u, _v in EDGES:
        _x0, _y0 = _pos[_u]
        _x1, _y1 = _pos[_v]
        _ax.annotate(
            "",
            xy=(_x1, _y1),
            xytext=(_x0, _y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#9a9a9a",
                lw=1.2,
                shrinkA=20,
                shrinkB=20,
                connectionstyle="arc3,rad=0.12",
            ),
        )
    for _n, (_x, _y) in _pos.items():
        _unobs = _n in UNOBSERVED
        _ax.scatter(
            [_x],
            [_y],
            s=2400,
            facecolor="white" if _unobs else "#3b6ea5",
            edgecolor="#c0504d" if _unobs else "#3b6ea5",
            linewidth=2.2 if _unobs else 1.5,
            zorder=3,
        )
        _ax.text(
            _x,
            _y,
            _n.replace("Intake", "\nIntake")
            .replace("Arousal", "\nArousal")
            .replace("Stress", "\nStress")
            .replace("Quality", "\nQuality")
            .replace("Pain", "\nPain")
            .replace("Activity", "\nActivity")
            .replace("Mood", "\nMood")
            .replace("Focus", "\nFocus")
            .replace("Engagement", "\nEngagement"),
            ha="center",
            va="center",
            fontsize=7.0,
            color="#c0504d" if _unobs else "white",
            fontweight="bold",
            zorder=4,
        )
    _ax.set_title(
        "The posited DAG — depth = longest path from a root. "
        "AutonomicArousal (red, hollow) is the unobserved confounder.",
        fontsize=11,
        fontweight="bold",
    )
    _ax.axis("off")
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def load_data(Path, gbt, jax, jnp, np):
    _csv = Path("notebooks/data/d10_case_study/observations.csv")
    if not _csv.exists():
        _csv = Path("data/d10_case_study/observations.csv")
    _raw = np.genfromtxt(_csv, delimiter=",", names=True)
    obs_times = np.asarray(_raw["t"], dtype=float)
    data = {n: np.asarray(_raw[n], dtype=float) for n in _raw.dtype.names if n != "t"}

    _dt = 0.05
    _span = float(np.ceil(obs_times.max()) + 0.5)
    t_grid = jnp.linspace(0.0, _span, round(_span / _dt) + 1)
    obs_idx = np.round(obs_times / _dt).astype(int)

    admit = gbt.make_admitter(
        admit_key=jax.random.key(20260703),
        n_draws=200,
        t_grid=t_grid,
        obs_times=obs_times,
        obs_idx=obs_idx,
        data=data,
    )
    return admit, data, obs_times


@app.cell(hide_code=True)
def eda_md(mo):
    mo.md(r"""
    ## 2. Legitimate exploratory summaries

    From the observed indicators only — location, spread, and lag-1 rank autocorrelation
    (a model-free memory probe). These summaries, plus the brief's semantics, are the entire
    basis for the priors.
    """)
    return


@app.cell
def eda_table(data, mo, np, obs_times):
    def _rank1(v):
        _u = np.argsort(np.argsort(v[:-1])).astype(float)
        _w = np.argsort(np.argsort(v[1:])).astype(float)
        return float(np.corrcoef(_u, _w)[0, 1])

    _rows = []
    for _k, _v in data.items():
        _qs = np.percentile(_v, [25, 50, 75])
        _rows.append(
            f"| `{_k}` | {_v.mean():.2f} | {_v.std():.2f} | "
            f"{_qs[0]:.1f} / {_qs[1]:.1f} / {_qs[2]:.1f} | {_rank1(_v):+.2f} |"
        )
    _hdr = (
        f"Single subject · **{obs_times.size} retained days** over "
        f"{obs_times.min():.0f}–{obs_times.max():.0f} d · median gap "
        f"{np.median(np.diff(obs_times)):.2f} d.\n\n"
        "| indicator | mean | sd | q25 / q50 / q75 | lag-1 rank-corr |\n"
        "|---|---|---|---|---|\n"
    )
    mo.md(_hdr + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def elicitation_md(mo):
    mo.md(r"""
    ## 3. Elicitation strategy

    Three rules turn the brief + summaries into priors, with **no** reference to any hidden
    value:

    - **Standardized latents.** Aim each construct's stationary sd near 1, set by the
      diffusion/stiffness balance (OU guide sd ≈ diffusion/√(2·stiffness)). This is the C2
      convention; the loadings carry the scale.
    - **Stiffness from attenuation-corrected persistence.** For a noisily observed relaxing
      process, lag-1 rank-corr ≈ reliability · exp(−Δt/τ). Correcting for measurement
      attenuation (≈ 0.84 for continuous, ≈ 0.7 for sliders, ≈ 0.5 for counts) gives τ per
      node, hence a stiffness median 1/τ. One important correction is applied during the
      build (see the revision below): a *downstream* node's persistence is partly **inherited**
      from slow upstream drivers, so its raw τ over-estimates its own relaxation.
    - **Emissions from inverse-link data quantiles.** Loading median ≈ 0.9·(observed sd) for
      identity links; slider and count loadings from logit- and log-scale IQRs. Intercepts
      from the inverse-link median. Gaussian noise ≈ 0.4·(observed sd); counts are Poisson
      (noise tied to the rate, no separate noise prior).

    Edge-weight priors are `Normal(0, s)` with `s` **tied to the child's relaxation rate**
    (`s ∝ √a_child`): a slow child integrates parent input over a long memory, so it needs a
    tighter edge prior to stay self-driven. This detail was added *because a check caught its
    absence* — see §5.
    """)
    return


@app.cell
def fragment_builders(gbt, np):
    _L = np.log

    def node(name, tau, sd_self, edges=(), emission=None):
        _a = 1.0 / tau
        _diff = float(np.sqrt(2 * _a) * sd_self)
        return gbt.NodeFragment(
            name=name,
            stiffness=("lognormal", float(_L(_a)), 0.4),
            center=("normal", 0.0, 0.5),
            quartic=("halfnormal", 0.0, 0.2),
            diffusion=("lognormal", float(_L(_diff)), 0.4),
            x0=("normal", 0.0, 1.0),
            edges_in=edges,
            emission=emission,
        )

    def edge(parent, scale):
        return gbt.EdgeFragment(parent, ("normal", 0.0, scale))

    def edges_for(tau, parents, base):
        _scale = float(np.clip(base * np.sqrt((1.0 / tau) / 0.5), 0.2, 0.6))
        return tuple(edge(p, _scale) for p in parents)

    def em_ident(ind, obs_sd, obs_mean):
        return gbt.EmissionFragment(
            ind,
            "identity",
            ("lognormal", float(_L(0.9 * obs_sd)), 0.3),
            ("normal", float(obs_mean), float(0.3 * obs_sd)),
            ("lognormal", float(_L(0.4 * obs_sd)), 0.3),
        )

    def em_slider(ind, med01, noise=8.0):
        return gbt.EmissionFragment(
            ind,
            "sigmoid100",
            ("lognormal", 0.0, 0.3),
            ("normal", float(_L(med01 / (1 - med01))), 0.4),
            ("lognormal", float(_L(noise)), 0.3),
        )

    def em_count(ind, med_rate):
        return gbt.EmissionFragment(
            ind,
            "exp",
            ("lognormal", float(_L(0.5)), 0.4),
            ("normal", float(_L(med_rate)), 0.4),
            ("delta", 1.0, 0.0),
            family="poisson",
        )

    return edges_for, em_count, em_ident, em_slider, node


@app.cell
def fragments(edges_for, em_count, em_ident, em_slider, node):
    # medians from EDA + brief semantics; deep-node τ moderated for inherited
    # persistence; edge base scale 0.5 (see the revision episode in §5)
    FRAGS = {
        "CaffeineIntake": node(
            "CaffeineIntake", 0.67, 0.9, emission=em_count("caffeine_servings", 3.0)
        ),
        "AutonomicArousal": node("AutonomicArousal", 2.0, 0.9),
        "PerceivedStress": node(
            "PerceivedStress",
            3.3,
            0.7,
            edges_for(3.3, ["AutonomicArousal"], 0.5),
            em_slider("stress_vas", 0.523),
        ),
        "SleepQuality": node(
            "SleepQuality",
            1.4,
            0.6,
            edges_for(1.4, ["CaffeineIntake", "AutonomicArousal", "PerceivedStress"], 0.5),
            em_slider("sleep_quality_vas", 0.567),
        ),
        "Fatigue": node(
            "Fatigue",
            2.8,
            0.7,
            edges_for(2.8, ["PerceivedStress", "SleepQuality"], 0.5),
            em_ident("fatigue_score", 2.20, 5.04),
        ),
        "MusculoskeletalPain": node(
            "MusculoskeletalPain",
            2.5,
            0.7,
            edges_for(2.5, ["AutonomicArousal", "Fatigue"], 0.5),
            em_ident("pain_nrs", 1.57, 3.50),
        ),
        "PhysicalActivity": node(
            "PhysicalActivity",
            1.2,
            0.7,
            edges_for(1.2, ["Fatigue", "MusculoskeletalPain"], 0.5),
            em_ident("active_minutes", 10.46, 49.89),
        ),
        "NegativeMood": node(
            "NegativeMood",
            2.5,
            0.6,
            edges_for(2.5, ["PerceivedStress", "PhysicalActivity"], 0.5),
            em_ident("irritability_index", 1.00, 0.11),
        ),
        "CognitiveFocus": node(
            "CognitiveFocus",
            2.5,
            0.6,
            edges_for(2.5, ["Fatigue", "NegativeMood"], 0.5),
            em_ident("reaction_time_ms", 28.33, 328.37),
        ),
        "SocialEngagement": node(
            "SocialEngagement",
            1.8,
            0.9,
            edges_for(1.8, ["NegativeMood"], 0.5),
            em_count("social_contacts", 3.0),
        ),
    }
    return (FRAGS,)


@app.cell
def run_build(FRAGS, ORDER, admit, gbt):
    _state = gbt.BuildState()
    build_results = {}
    build_states = {}
    for _nm in ORDER:
        _state, _res, _art = admit(_state, FRAGS[_nm])
        build_results[_nm] = (_res, _art)
        build_states[_nm] = _state
    final_state = _state
    return build_results, build_states, final_state


@app.cell(hide_code=True)
def build_md(mo):
    mo.md(r"""
    ## 4. The staged build

    Each construct is admitted in topological order, its checks run on the cumulative partial
    model by exact Euler–Maruyama simulation. Roots and the unobserved node come first; every
    observed construct brings its emission (and thus its data anchor) at admission. Green
    reports are shown compactly; the one construct that needed a revision is called out in §5.
    """)
    return


@app.cell
def r_caffeine(build_results, gbt):
    gbt.render_report(
        "CaffeineIntake — root, Poisson count indicator", *build_results["CaffeineIntake"]
    )
    return


@app.cell(hide_code=True)
def note_caffeine(mo):
    mo.md(r"""
    *Caffeine's rank-corr@cadence (0.12) is the lowest in the model — caffeine intake is
    nearly day-specific — but it clears the 0.05 floor, so its (fast) dynamics remain
    marginally identifiable rather than white noise. The Poisson count also gives the weakest
    link signal (C6 ≈ 0.67), as expected for small counts.*
    """)
    return


@app.cell
def r_arousal(build_results, gbt):
    gbt.render_report(
        "AutonomicArousal — UNOBSERVED confounder (no emission ⇒ C1–C4 only)",
        *build_results["AutonomicArousal"],
    )
    return


@app.cell(hide_code=True)
def note_arousal(mo):
    mo.md(r"""
    *The unobserved node runs only C1–C4; its scale is the convention anchor (1.0), flagged in
    the C2 band as `convention: no indicator` — an irreducible convention surfaced rather than
    hidden. It has children but no parents, so C4 does not apply.*
    """)
    return


@app.cell
def r_stress(build_results, gbt):
    gbt.render_report(
        "PerceivedStress — slider indicator, one parent", *build_results["PerceivedStress"]
    )
    return


@app.cell
def r_sleep(build_results, gbt):
    gbt.render_report(
        "SleepQuality — slider indicator, three parents", *build_results["SleepQuality"]
    )
    return


@app.cell(hide_code=True)
def revision_md(mo):
    mo.md(r"""
    ## 5. A revision the checks forced: Fatigue and edge overwhelm

    The first-pass priors used a **fixed** edge-weight scale (`Normal(0, 0.6)`) for every edge,
    and set Fatigue's stiffness from its indicator's raw lag-1 (0.72 ⇒ τ ≈ 6.8 d). Admitting
    Fatigue against that first-pass fragment produces a real red — reproduced here:
    """)
    return


@app.cell
def fatigue_v1(em_ident, gbt, node):
    fatigue_v1_frag = node(
        "Fatigue",
        6.8,  # raw attenuation-corrected τ, before the inherited-persistence correction
        0.7,
        (
            gbt.EdgeFragment("PerceivedStress", ("normal", 0.0, 0.6)),
            gbt.EdgeFragment("SleepQuality", ("normal", 0.0, 0.6)),
        ),
        em_ident("fatigue_score", 2.20, 5.04),
    )
    return (fatigue_v1_frag,)


@app.cell
def fatigue_v1_report(admit, build_states, fatigue_v1_frag, gbt):
    _sim = admit(build_states["SleepQuality"], fatigue_v1_frag)
    gbt.render_report("Fatigue — FIRST ATTEMPT (fixed 0.6 edges, τ = 6.8 d)", _sim[1], _sim[2])
    return


@app.cell(hide_code=True)
def revision_reasoning(mo):
    mo.md(r"""
    **Reading the diagnostic.** C4b reports that, for the median prior draw, the two parents
    (PerceivedStress, SleepQuality) displace Fatigue's path by ~118 % of its own variation —
    parents dominate, self-dynamics vanish. The dependence line says the statistic *rises when
    the child's own stiffness/diffusion contribute little*. Two things were wrong, both fixed
    without ever consulting the hidden truth:

    - **The edge prior ignored the child's timescale.** A slow node low-pass-integrates parent
      fluctuations, so a fixed edge scale swamps it. Fix: tie the edge scale to the child's
      relaxation rate (`s ∝ √a_child`) and tighten the base to 0.5.
    - **Fatigue's τ was over-estimated.** Its indicator's high persistence is partly
      *inherited* from its slow drivers (stress, poor sleep), not its own relaxation.
      Attributing all of it to self-dynamics inflated τ to 6.8 d; the coherent intrinsic value
      is faster (≈ 2.8 d), with the parents supplying the rest of the sluggishness — which also
      raises the self-variance and resolves the overwhelm.

    The revised fragment (used in §4 above) admits cleanly, at C4b ≈ 88 %: Fatigue is still
    substantially parent-driven — physiologically correct — but now under the overwhelm cap.
    The same two corrections were applied to the other deep multi-parent nodes
    (Pain, NegativeMood, CognitiveFocus), which sat just over the line for the same reason.
    Had the effect been irreducibly dominant, the honest move would have been to *accept* the
    C4b consequence (own-dynamics weakly informed) rather than distort τ — here revision was
    the better fit.
    """)
    return


@app.cell
def r_fatigue(build_results, gbt):
    gbt.render_report(
        "Fatigue — REVISED (relaxation-tied edges, τ = 2.8 d)", *build_results["Fatigue"]
    )
    return


@app.cell(hide_code=True)
def rest_md(mo):
    mo.md(r"""
    ## 6. The remaining constructs

    The rest of the topological order, all admitted clean with the revised elicitation. Note
    `reaction_time_ms` for CognitiveFocus: higher focus reads as *faster* reaction time, but
    since latent orientation is a free normalization, a positive loading with the edge signs
    absorbing the direction is fine — no check depends on the sign.
    """)
    return


@app.cell
def r_pain(build_results, gbt):
    gbt.render_report(
        "MusculoskeletalPain — continuous indicator", *build_results["MusculoskeletalPain"]
    )
    return


@app.cell
def r_activity(build_results, gbt):
    gbt.render_report(
        "PhysicalActivity — continuous indicator (tens scale)", *build_results["PhysicalActivity"]
    )
    return


@app.cell
def r_mood(build_results, gbt):
    gbt.render_report(
        "NegativeMood — continuous indicator (near zero)", *build_results["NegativeMood"]
    )
    return


@app.cell
def r_focus(build_results, gbt):
    gbt.render_report(
        "CognitiveFocus — continuous indicator (reaction time, ms)",
        *build_results["CognitiveFocus"],
    )
    return


@app.cell
def r_social(build_results, gbt):
    gbt.render_report(
        "SocialEngagement — Poisson count indicator", *build_results["SocialEngagement"]
    )
    return


@app.cell(hide_code=True)
def summary_md(mo):
    mo.md(r"""
    ## 7. Outcome
    """)
    return


@app.cell
def summary_table(build_results, mo):
    _rows = []
    for _nm, (_res, _art) in build_results.items():
        _reds = [r.check for r in _res if not r.passed]
        _rows.append(
            f"| {_nm} | {len(_res)} | {_art['outcome'].split('—')[0].strip()} | "
            f"{', '.join(_reds) if _reds else '—'} |"
        )
    mo.md("| construct | # checks | outcome | reds |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def closing(mo):
    mo.md(r"""
    ## 8. What this run demonstrates

    - **The tooling scales.** The same fragments, checks, severity table, and evidence-only
      diagnostics from the D = 3 lab drove a D = 10 build with heterogeneous emissions
      (Gaussian identity, saturating slider, Poisson count) and an unobserved confounder — no
      changes beyond adding the Poisson family and a `make_admitter` factory.
    - **The checks caught a real error blind.** With no access to the truth, C4b flagged that
      the first-pass edge prior overwhelmed slow-relaxing children, and its dependence
      diagnostic pointed at both the edge scale and an over-estimated timescale. The fix was a
      genuine modeling insight (persistence is inherited down a causal chain), not a tweak to
      pass a threshold.
    - **The workflow is honest about what it certifies.** Every green here is a statement about
      the *prior*: on-scale, dynamics visible at cadence, edges detectable but not
      overwhelming, links informative — all before a single fit. Whether these priors
      *recover* the hidden parameters is the next question, answerable only by fitting and
      comparing against `hidden/` — which this exercise deliberately never opened.

    The staged workflow held up at D = 10: a blind, nonlinear, non-Gaussian problem, modeled
    to a full green board through one principled revision, with the complete trail above.
    """)
    return


if __name__ == "__main__":
    app.run()
