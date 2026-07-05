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
    import numpy as np

    return Path, gbt, jax, jnp, np


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # A second blind case study — the *revised* battery, on a lake ecosystem

    This notebook re-runs the staged prior-building workflow of `gradual_build_tools.py` on a
    fresh **blind** problem in a new domain, to exercise the **revised** check battery.

    **What changed in the battery.** A literature audit found that three of the original
    checks reached past *reachability* (the recognized remit of prior-predictive checking)
    into *practical identifiability* — the question of whether the data will pin a parameter,
    which the field answers **post-fit** (posterior contraction, power-scaling), not before.
    The tell was uniform: the offenders carried "prior-dominated" consequences and
    `2/√n_obs` signal-to-noise floors. So the battery was re-scoped:

    - **C3** is now a schedule-only **design-resolvability** screen — *is the prior's
      self-relaxation τ inside the window this sampling design can resolve, `[cadence/3,
      span/4]`?* It reads only the observation timestamps and the prior, so it is immune to
      the self-vs-inherited persistence confound that made the old persistence-based C3
      unreliable. It does not estimate τ.
    - **C4a** (edge detectability, an SNR floor) is **dropped**; only **C4b** (overwhelm, a
      degeneracy check) remains.
    - **C6** (link Fisher-information SNR) is **dropped**; its structural kernel survives as
      **C5c transmission** — a noise-free "is the link saturated / in a dead zone?" test.

    The pre-fit battery is now **C1, C2, C3-resolvability, C4b, C5, C5c** — reachability plus
    one design-observability screen. Estimability verdicts are deferred to the post-fit gate.

    **The blind protocol (unchanged).** A separate agent designed a hidden continuous-time
    **nonlinear, non-Gaussian** ground truth in a domain it chose, generated the data, and
    wrote a brief. It worked behind a firewall: everything below is built from the brief and
    from *legitimate summaries of the observed data only*. The generator and its parameters in
    `data/lake_ecosystem_case_study/hidden/` were **never opened**.

    **What "success" means.** Not that the priors match the hidden truth — we cannot see it —
    but that they are internally consistent and data-reachable before any fit, and that where
    the design *cannot* resolve a construct's dynamics, the workflow says so honestly rather
    than pretending otherwise.
    """)
    return


@app.cell(hide_code=True)
def brief_md(mo):
    mo.md(r"""
    ## 1. The brief (domain expert's account)

    **Unit.** One small, productive (mesotrophic–eutrophic) freshwater lake, monitored at a
    single mid-lake station through one ~60-day summer stratification season. A buoy sonde
    logs the physical/chemical suite; a technician takes paired grab samples on the same
    (weather-dependent, irregular) visits — 100 visits, median gap ≈ 0.47 d, some stretched
    to ~2.5 d.

    **Constructs (9; one latent).**

    | # | construct | indicator | type | units |
    |---|---|---|---|---|
    | 1 | **CatchmentLoading** | *latent — no sensor* | — | intensity of storm/runoff material loading |
    | 2 | WaterTemperature | `water_temp_C` | continuous | °C |
    | 3 | Nitrate | `nitrate_mgL` | continuous | mg/L N |
    | 4 | Turbidity | `turbidity_NTU` | continuous | NTU |
    | 5 | CDOM | `fdom_QSU` | continuous | QSU |
    | 6 | Phytoplankton | `chl_a_ugL` | continuous | µg/L chl-a |
    | 7 | DissolvedOxygen | `do_sat_pct` | bounded 0–100 | % saturation |
    | 8 | pH | `ph` | continuous | pH |
    | 9 | Zooplankton | `zoop_count` | count | individuals / tow |

    **Causal structure (directions only; magnitudes are ours to elicit).**

    - `CatchmentLoading → Nitrate, Turbidity, CDOM` — a storm pulse washes in all three at
      once; because loading is **unmeasured**, their storm-driven co-movement is a shared
      hidden common cause, *not* direct links among them (the confounder to watch).
    - `WaterTemperature → Nitrate, Phytoplankton, DissolvedOxygen, Zooplankton`
    - `Nitrate → Phytoplankton`; `Turbidity → Phytoplankton` *(saturating: light limit)*;
      `CDOM → Phytoplankton` *(shading)*
    - `Phytoplankton → DissolvedOxygen, pH, Zooplankton` *(the last saturating: grazer
      satiation)*

    **Timescales genuinely differ** (the expert is explicit): turbidity settles and oxygen
    exchanges in **hours**; water temperature turns over ~a day; nitrate, colour, and algal
    biomass carry multi-day memory; **zooplankton respond over a generational, weekly lag**.
    Algae and grazers **self-limit** (crowding / carrying capacity) — genuine nonlinearity.
    """)
    return


@app.cell
def load_data(Path, gbt, jax, jnp, np):
    _csv = Path("notebooks/data/lake_ecosystem_case_study/observations.csv")
    if not _csv.exists():
        _csv = Path("data/lake_ecosystem_case_study/observations.csv")
    _raw = np.genfromtxt(_csv, delimiter=",", names=True)
    obs_times = np.asarray(_raw["t"], dtype=float)
    data = {n: np.asarray(_raw[n], dtype=float) for n in _raw.dtype.names if n != "t"}

    _dt = 0.025  # fine grid: some constructs relax in hours
    _span = float(np.ceil(obs_times.max()) + 0.5)
    t_grid = jnp.linspace(0.0, _span, round(_span / _dt) + 1)
    obs_idx = np.round(obs_times / _dt).astype(int)

    admit = gbt.make_admitter(
        admit_key=jax.random.key(7),
        n_draws=200,
        t_grid=t_grid,
        obs_times=obs_times,
        obs_idx=obs_idx,
        data=data,
    )
    _cad = float(np.median(np.diff(obs_times)))
    _spn = float(np.ptp(obs_times))
    design = {"cadence": _cad, "span": _spn, "lo": _cad / 3.0, "hi": _spn / 4.0}
    return admit, data, design, obs_times


@app.cell(hide_code=True)
def eda(data, design, mo, np):
    _rows = []
    for _k, _v in data.items():
        _q = np.percentile(_v, [25, 50, 75])
        _rows.append(
            f"| `{_k}` | {_v.mean():.2f} | {_v.std():.2f} | {_v.min():.2f} | "
            f"{_q[1]:.2f} | {_v.max():.2f} |"
        )
    _tbl = "\n".join(_rows)
    mo.md(
        "## 2. Legitimate exploratory summaries\n\n"
        "Location and spread of the observed indicators — these summaries plus the brief's "
        "semantics are the entire basis for the priors. Note we do **not** read timescales "
        "off indicator autocorrelation: a downstream indicator's serial dependence mixes the "
        "construct's own relaxation with inherited parent persistence, an unidentified split. "
        "Timescales are elicited from domain knowledge as **wide** priors instead.\n\n"
        f"Sampling design: cadence (median gap) **{design['cadence']:.3f} d**, span "
        f"**{design['span']:.1f} d** ⇒ resolvable window "
        f"`[cadence/3, span/4] = [{design['lo']:.2f}, {design['hi']:.1f}] d`.\n\n"
        "| indicator | mean | sd | min | median | max |\n|---|---|---|---|---|---|\n" + _tbl
    )
    return


@app.cell(hide_code=True)
def strategy_md(mo):
    mo.md(r"""
    ## 3. Elicitation strategy

    Same skeleton as any staged build, tuned by the revised battery:

    - **Latent scale — standardize to sd ≈ 1** by convention; the emission **loading** carries
      the physical scale, read from the indicator's data quantiles through the inverse link
      (`C2` checks the standardized latent lands in the data-implied band). Loading and latent
      scale trade off (only their product is identified), so this is a normalization choice,
      not a claim.
    - **Timescale — wide domain priors.** Each construct's self-relaxation τ is set from the
      brief's physical account (hours / a day / days / weekly), with a wide lognormal
      (log-sd ≈ 0.55). We let **C3-resolvability** check τ against the sampling design rather
      than trying to pin it from the data — because the marginal cannot separate a
      construct's own relaxation from persistence inherited through its edges.
    - **Loadings / intercepts / noise** from inverse-link data quantiles (identity: loading
      ≈ 0.9·sd, intercept = mean, noise ≈ 0.4·sd; sigmoid & count analogues).
    - **Edges** `Normal(0, s)` with a moderate scale; `C4b` guards against a parent swamping
      the child. The two "saturating" edges are modelled linearly (the engine's edges are
      linear); the linear term matches the saturating one near the origin.
    - **Quartic self-limiting** on the two constructs the brief flags (Phytoplankton,
      Zooplankton).
    """)
    return


@app.cell
def builders(data, gbt, np):
    _L = np.log

    def node(name, tau, edges=(), em=None, q=0.15, sd_self=0.9, tau_logsd=0.55):
        _a = 1.0 / tau
        _diff = float(np.sqrt(2 * _a) * sd_self)
        return gbt.NodeFragment(
            name,
            ("lognormal", float(_L(_a)), tau_logsd),
            ("normal", 0.0, 0.5),
            ("halfnormal", 0.0, q),
            ("lognormal", float(_L(_diff)), 0.4),
            ("normal", 0.0, 1.0),
            edges,
            em,
        )

    def edge(p, s=0.4):
        return gbt.EdgeFragment(p, ("normal", 0.0, s))

    def em_ident(ind, noise_frac=0.4):
        _sd = float(np.std(data[ind]))
        _mn = float(np.mean(data[ind]))
        return gbt.EmissionFragment(
            ind,
            "identity",
            ("lognormal", float(_L(0.9 * _sd)), 0.3),
            ("normal", _mn, 0.3 * _sd),
            ("lognormal", float(_L(noise_frac * _sd)), 0.3),
        )

    def em_slider(ind, loading=1.0, noise=5.0):
        _p = float(np.median(data[ind]) / 100.0)
        return gbt.EmissionFragment(
            ind,
            "sigmoid100",
            ("lognormal", float(_L(loading)), 0.3),
            ("normal", float(_L(_p / (1 - _p))), 0.4),
            ("lognormal", float(_L(noise)), 0.3),
        )

    def em_count(ind, loading=0.4):
        _r = float(np.median(data[ind]))
        return gbt.EmissionFragment(
            ind,
            "exp",
            ("lognormal", float(_L(loading)), 0.4),
            ("normal", float(_L(_r)), 0.4),
            ("delta", 1.0, 0.0),
            family="poisson",
        )

    return edge, em_count, em_ident, em_slider, node


@app.cell
def frags(edge, em_count, em_ident, em_slider, node):
    # topological order along the causal arrows; τ from the brief's physical timescales.
    # Phytoplankton edges 0.25 and DissolvedOxygen loading 0.5 are the post-diagnosis values
    # (the first-attempt values and why they were tightened are shown further down).
    FRAGS = {
        "CatchmentLoading": node("CatchmentLoading", 2.5, sd_self=1.0),
        "WaterTemperature": node("WaterTemperature", 1.2, em=em_ident("water_temp_C")),
        "Nitrate": node(
            "Nitrate",
            3.5,
            (edge("CatchmentLoading"), edge("WaterTemperature")),
            em_ident("nitrate_mgL"),
        ),
        "Turbidity": node(
            "Turbidity", 0.15, (edge("CatchmentLoading"),), em_ident("turbidity_NTU")
        ),
        "CDOM": node("CDOM", 6.0, (edge("CatchmentLoading"),), em_ident("fdom_QSU")),
        "Phytoplankton": node(
            "Phytoplankton",
            4.5,
            (
                edge("Nitrate", 0.25),
                edge("WaterTemperature", 0.25),
                edge("Turbidity", 0.25),
                edge("CDOM", 0.25),
            ),
            em_ident("chl_a_ugL"),
            q=0.2,
        ),
        "DissolvedOxygen": node(
            "DissolvedOxygen",
            0.25,
            (edge("Phytoplankton"), edge("WaterTemperature")),
            em_slider("do_sat_pct", loading=0.5),
        ),
        "pH": node("pH", 0.5, (edge("Phytoplankton"),), em_ident("ph")),
        "Zooplankton": node(
            "Zooplankton",
            8.0,
            (edge("Phytoplankton"), edge("WaterTemperature")),
            em_count("zoop_count"),
            q=0.2,
        ),
    }
    ORDER = list(FRAGS)
    # Turbidity settles in hours — below what half-daily sampling resolves. That is a design
    # limit, not a fixable prior, so we keep the honest fast prior and accept the consequence.
    ACCEPTED = {
        "Turbidity": {
            "C3 resolvability": "suspended-sediment settling is genuinely sub-daily (hours); "
            "the ~half-daily station cadence cannot resolve it — keeping the physically-honest "
            "fast prior and accepting that turbidity's self-timescale is design-limited "
            "(confirm post-fit)."
        }
    }
    return ACCEPTED, FRAGS, ORDER


@app.cell
def run_build(ACCEPTED, FRAGS, ORDER, admit, gbt):
    _state = gbt.BuildState()
    build = {}
    for _nm in ORDER:
        _before = _state
        _state, _res, _art = admit(_state, FRAGS[_nm], ACCEPTED.get(_nm))
        build[_nm] = {"res": _res, "art": _art, "before": _before}
    final_state = _state
    return build, final_state


@app.cell(hide_code=True)
def roots_md(mo):
    mo.md(r"""
    ## 4. The build, one construct at a time

    ### Roots — the latent driver and the physical pace-setter
    `CatchmentLoading` has no indicator (a convention scale-anchor of 1; only C1/C2/C3 apply)
    and `WaterTemperature` is the physical root. Both resolve cleanly.
    """)
    return


@app.cell(hide_code=True)
def r_loading(build, gbt):
    gbt.render_report(
        "1 · CatchmentLoading — latent storm-driven confounder (no sensor)",
        build["CatchmentLoading"]["res"],
        build["CatchmentLoading"]["art"],
    )
    return


@app.cell(hide_code=True)
def r_temp(build, gbt):
    gbt.render_report(
        "2 · WaterTemperature — physical root (τ ≈ a day)",
        build["WaterTemperature"]["res"],
        build["WaterTemperature"]["art"],
    )
    return


@app.cell(hide_code=True)
def r_nitrate(build, gbt):
    gbt.render_report(
        "3 · Nitrate — loading + temperature drivers (multi-day pool)",
        build["Nitrate"]["res"],
        build["Nitrate"]["art"],
    )
    return


@app.cell(hide_code=True)
def turbidity_md(mo):
    mo.md(r"""
    ### Turbidity — where the design-resolvability screen earns its keep

    The brief is explicit that suspended sediment settles in **hours**, so the honest prior
    puts τ ≈ 0.15 d. But the station is visited roughly twice a day, and `cadence/3 ≈ 0.16 d`:
    the process relaxes faster than the sampling can follow. **C3-resolvability flags it** —
    only ~50% of the prior τ mass is inside the resolvable window.

    This is not a prior to "fix" — inflating τ to clear the floor would contradict the physics.
    It is a **design limitation**: no fit of *this* schedule can inform turbidity's relaxation
    time. So we keep the honest fast prior and **accept the consequence** (recorded on the
    build state), to be confirmed post-fit. This is exactly the honest outcome the reframed C3
    is meant to produce, and the reason the persistence-estimating version was retired.
    """)
    return


@app.cell(hide_code=True)
def r_turbidity(build, gbt):
    gbt.render_report(
        "4 · Turbidity — C3 fires (sub-cadence settling), accepted as a design limit",
        build["Turbidity"]["res"],
        build["Turbidity"]["art"],
    )
    return


@app.cell(hide_code=True)
def r_cdom(build, gbt):
    gbt.render_report(
        "5 · CDOM — coloured dissolved organics (multi-day, resolvable)",
        build["CDOM"]["res"],
        build["CDOM"]["art"],
    )
    return


@app.cell(hide_code=True)
def phyto_md(mo):
    mo.md(r"""
    ### Phytoplankton — C4b as a diagnostic even when it passes

    Phytoplankton has four drivers (nitrate, temperature, turbidity, CDOM). At the first-pass
    edge scale (0.30 each) the parents jointly account for **92%** of the child's path
    variation — under the 95% overwhelm cap, so it *passes*, but it leaves almost no room for
    the self-limiting bloom dynamics the brief emphasizes. Reading C4b as a diagnostic rather
    than a gate, we tighten the four edges to 0.25, which drops overwhelm to ~85% and lets the
    quartic self-limitation express. The first attempt is shown below the final report.
    """)
    return


@app.cell(hide_code=True)
def r_phyto(build, gbt):
    gbt.render_report(
        "6 · Phytoplankton — four drivers + self-limiting bloom (edges tightened to 0.25)",
        build["Phytoplankton"]["res"],
        build["Phytoplankton"]["art"],
    )
    return


@app.cell
def phyto_v1(edge, em_ident, node):
    phyto_v1_frag = node(
        "Phytoplankton",
        4.5,
        (
            edge("Nitrate", 0.3),
            edge("WaterTemperature", 0.3),
            edge("Turbidity", 0.3),
            edge("CDOM", 0.3),
        ),
        em_ident("chl_a_ugL"),
        q=0.2,
    )
    return (phyto_v1_frag,)


@app.cell(hide_code=True)
def r_phyto_v1(admit, build, gbt, phyto_v1_frag):
    _sim = admit(build["Phytoplankton"]["before"], phyto_v1_frag)
    gbt.render_report(
        "Phytoplankton — FIRST ATTEMPT (edges 0.30): C4b at the overwhelm cap",
        _sim[1],
        _sim[2],
    )
    return


@app.cell(hide_code=True)
def do_md(mo):
    mo.md(r"""
    ### DissolvedOxygen — the bounded sigmoid, and C5c transmission

    Oxygen is a bounded 0–100 % index, so it uses the `sigmoid100` link. Its observed band is
    narrow (≈ 56–90 %), so a unit loading over-transmits: the first attempt (loading 1.0)
    pushed the prior-predictive IQR to **3.8×** the data and put C2 near its band ceiling.
    Halving the loading to 0.5 narrows the sigmoid swing to the observed band (C5b → 2.4, C2
    comfortably mid-band). Critically, **C5c transmission stays well above its floor** — the
    link is *not* in a saturated dead zone; it sits in its responsive region. That dead-zone
    test is the structural kernel salvaged from the retired C6, without C6's SNR overreach.
    """)
    return


@app.cell(hide_code=True)
def r_do(build, gbt):
    gbt.render_report(
        "7 · DissolvedOxygen — bounded sigmoid, loading tuned to the observed band",
        build["DissolvedOxygen"]["res"],
        build["DissolvedOxygen"]["art"],
    )
    return


@app.cell(hide_code=True)
def r_ph(build, gbt):
    gbt.render_report(
        "8 · pH — tracks biology within a day (fast, still resolvable)",
        build["pH"]["res"],
        build["pH"]["art"],
    )
    return


@app.cell(hide_code=True)
def zoop_md(mo):
    mo.md(r"""
    ### Zooplankton — the slow end of the timescale spread

    Grazers respond over a generational, roughly weekly lag: the honest prior puts τ ≈ 8 d,
    with the wide prior's tail reaching ~16 d — brushing the `span/4 ≈ 15 d` ceiling. The
    median stays inside, so C3 passes, but the report shows how close the slowest construct
    sits to the window's far edge (the mirror of turbidity at the fast edge). The count
    indicator uses the Poisson/exp link.
    """)
    return


@app.cell(hide_code=True)
def r_zoop(build, gbt):
    gbt.render_report(
        "9 · Zooplankton — weekly grazer lag, Poisson counts",
        build["Zooplankton"]["res"],
        build["Zooplankton"]["art"],
    )
    return


@app.cell(hide_code=True)
def summary(build, design, mo):
    _order = [
        "CatchmentLoading",
        "WaterTemperature",
        "Nitrate",
        "Turbidity",
        "CDOM",
        "Phytoplankton",
        "DissolvedOxygen",
        "pH",
        "Zooplankton",
    ]
    _rows = []
    for _nm in _order:
        _c3 = next(r for r in build[_nm]["res"] if r.check == "C3 resolvability")
        _out = build[_nm]["art"]["outcome"].replace(
            "ADMITTED with accepted consequences", "ADMITTED*"
        )
        _rows.append(
            f"| {_nm} | {_c3.value.split(';')[0]} | {'✅' if _c3.passed else '⚠️ accept'} | {_out} |"
        )
    _tbl = "\n".join(_rows)
    mo.md(
        "## 5. Summary — the whole build\n\n"
        f"Resolvable window `[{design['lo']:.2f}, {design['hi']:.1f}] d`. All nine constructs "
        "admitted; one accepted design-consequence (Turbidity); two diagnostic-driven "
        "refinements (Phytoplankton edges, DissolvedOxygen loading).\n\n"
        "| construct | prior τ (C3) | C3 | outcome |\n|---|---|---|---|\n" + _tbl + "\n\n"
        "\\* admitted with an accepted consequence recorded on the build state.\n\n"
        "The **timescale gradient** is the story: C3 places turbidity (hours) *below* the "
        "resolvable floor and zooplankton (weekly) near the *ceiling*, with everything else "
        "comfortably inside — read straight off the schedule and the priors, no data values "
        "and no persistence estimate involved."
    )
    return


@app.cell(hide_code=True)
def closing(final_state, mo):
    mo.md(
        "## 6. What this blind run demonstrates\n\n"
        "- **Reachability held across a heterogeneous suite** — continuous, bounded-sigmoid, "
        "and count indicators; a latent confounder; linear and (linearized) saturating edges; "
        "quartic self-limitation — with C1/C2/C5/C5c all clean after two small refinements.\n"
        "- **C3-resolvability did its one honest job**: it flagged the construct the sampling "
        "design genuinely cannot resolve (turbidity's sub-daily settling) and stayed silent on "
        "the resolvable ones, purely from the schedule — no persistence estimate, no exposure "
        "to the self-vs-inherited confound.\n"
        "- **C5c caught the question C5 alone cannot** — whether the bounded link transmits or "
        "is saturated — without C6's identifiability overreach.\n"
        "- **The verdicts the old battery over-claimed are simply absent.** Whether "
        "turbidity's timescale, the four phytoplankton edges, or any construct's trajectory is "
        "*data-informed* is not decided here; that is a post-fit contraction question. The "
        "pre-fit gate certifies only that the priors can generate and cover this data, and "
        "flags where the design cannot help.\n\n"
        f"Final build state: **{len(final_state.nodes)} constructs admitted**, "
        f"**{len(final_state.annotations)} accepted consequence(s)** carried forward for "
        "post-fit follow-up."
    )
    return


if __name__ == "__main__":
    app.run()
