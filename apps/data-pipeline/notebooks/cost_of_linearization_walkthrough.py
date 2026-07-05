import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # The cost of linearization, priced before you pay it

    This project's posterior runs through *exact* engines — particle/SMC over the true
    emission, Euler–Maruyama over the true nonlinear drift — and never lets a linearized
    Gaussian surrogate stand in on a path that produces a reported number. That is the
    [linearization-is-init-only policy](../../../AGENTS.md): a Gaussian approximation
    discards curvature and silently biases the answer, so it is allowed only to *warm-start*
    the samplers, where the exact MCMC/SMC then corrects it.

    But the policy raises a practical question it does not answer: **for this model, in this
    regime, how badly *would* a linear/Gaussian approximation bias me?** Sometimes the honest
    answer is "barely" — and knowing that, cheaply, is worth a great deal before committing
    a GPU-week to particle inference.

    This notebook prices that, using **only simulation and O(T) Gaussian filtering — it never
    fits the nonlinear model.** The trick is to make the *simulated truth the yardstick*:

    1. generate a trajectory from a model we *declare* to be ground truth;
    2. fit a deliberately **relaxed** candidate with the cheapest valid Gaussian filter;
    3. check where the *known* simulated truth lands inside the candidate's distributions.

    If the candidate is faithful, the truth lands uniformly — calibrated. The *way* it
    departs from uniform names the cost, and sweeping a single regime knob shows exactly when
    "linear is fine" holds and when it shatters. It is the cheap pre-flight for an
    approximation; support code is in `cost_of_linearization_support.py`.
    """)
    return


@app.cell
def imports_lab():
    import cost_of_linearization_support as lab

    return (lab,)


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## 1. The toy truth, and its one regime knob

    The ground truth is a damped pendulum in discrete time, state $z=(\theta,\omega)$:

    $$\theta_{t+1}=\theta_t+\Delta\,\omega_t,\qquad
    \omega_{t+1}=\omega_t-\Delta\,\omega_0^2\,\sin\theta_t-\Delta\,\gamma\,\omega_t+\text{(noise)},
    \qquad y_t=\sin\theta_t+\text{(noise)}.$$

    Two nonlinearities and one heavy tail make this "the real physics": the restoring force
    $\sin\theta$, the **folding** sensor $y=\sin\theta$ (it reads the *height* of the bob, so
    it cannot tell $+\theta$ from $-\theta$), and Student-t noise. The textbook
    *small-angle* approximation $\sin\theta\approx\theta$ is exactly the linearization whose
    cost we want to price — and that cost is governed by one knob: the **swing amplitude**.

    Drag it. At small amplitude the dashed small-angle stand-ins lie on top of the truth: the
    restoring force is nearly linear and the readout barely folds. Wind it up toward $\pi$ and
    the truth pulls away — the force weakens ($\sin\theta<\theta$) and the readout turns over
    while $\theta$ keeps climbing (the red band $|y|\le 1$ is all the folding sensor can ever
    report). This is the regime axis every later panel sweeps.
    """)
    return


@app.cell
def traj_controls(mo):
    traj_amp = mo.ui.slider(
        start=0.2, stop=3.0, step=0.1, value=1.5, label="swing amplitude θ₀ (rad)", show_value=True
    )
    return (traj_amp,)


@app.cell
def traj_fig(lab, mo, traj_amp):
    mo.vstack([traj_amp, lab.fig_trajectories(traj_amp.value)])
    return


@app.cell(hide_code=True)
def relaxations_md(mo):
    mo.md(r"""
    ## 2. Four model spots, plus the filter

    The candidate model is built by turning the truth's hard parts into their easy stand-ins,
    one spot at a time. There are **four orthogonal model spots**, paired into two kinds:

    - **dynamics** — the restoring torque $-\omega_0^2\sin\theta \to -\omega_0^2\,\theta$;
    - **measurement** — the readout $\sin\theta \to \theta$;
    - **process noise** — Student-t $\to$ a *variance-matched* Gaussian;
    - **observation noise** — Student-t $\to$ a variance-matched Gaussian.

    (The panel below draws the two *nonlinearity* spots, identical sin-vs-line curves, and the
    shared noise shape — Student-t vs Gaussian — which applies to both noise spots.) Relax
    **all four** and the model is an exact linear-Gaussian SSM, so the cheapest valid fit is the
    **exact Kalman filter**. Keep a nonlinear piece and the cheapest valid Gaussian fit is the
    **extended Kalman filter** (EKF), which straightens that piece to its tangent each step
    (mechanics in `filtering_anatomy_walkthrough.py`). Three static parameters are inferred
    throughout: $q$, $r$, $\gamma$.

    There is also a **fifth, different kind of spot** — *how* you compute the state posterior
    given the model: KF / EKF / UKF / particle filter. That is the inference axis, taken up in
    §13; the four spots here are about the *model* the candidate believes.

    The green strip marks the small-angle window where the nonlinearity spots cost almost
    nothing. The noise spots are different in kind: they match in the bulk but, on a log scale,
    the Gaussian's tail plunges while the Student-t's stays fat — the disagreement lives
    entirely in the rare, large events.
    """)
    return


@app.cell
def relaxations_fig(lab):
    lab.fig_relaxations()
    return


@app.cell(hide_code=True)
def harness_md(mo):
    mo.md(r"""
    ## 3. The cheap harness

    One function does all the work — `crossgen(truth_cfg, cand_cfg, amp, …)` — and every
    diagnostic below is a different pair of configs handed to it. For each of a few hundred
    replicates it:

    1. **draws a truth** $\theta^\star=(q,r,\gamma)$ from the prior and **simulates** a
       length-$T$ trajectory from `truth_cfg`;
    2. **fits** `cand_cfg` with its cheapest valid Gaussian filter, turning the filter's
       marginal likelihood $p(y\mid\theta)$ and the prior into a posterior over $(q,r,\gamma)$
       on a small log-space grid (low-dimensional, so the grid is exact and instant);
    3. holds out the last $H$ steps, forms the **$H$-step-ahead predictive**, and records the
       **PIT** of the known true future observation;
    4. records the **fractional rank** of each true parameter — its posterior CDF at the truth.

    The replicate loop is one batched, JIT-compiled `jax.vmap`, so hundreds of refits finish
    in a fraction of a second and the sliders stay live. Nothing here ever runs the nonlinear
    model's *own* inference — that is the whole point: this is the part you can afford to run
    before deciding whether the expensive part is even necessary.

    Two readouts come back, and §6 explains why they are **not** equally trustworthy: the
    predictive PIT (target = an observable) is the trusted one; the parameter ranks are a
    rougher, confounded probe.
    """)
    return


@app.cell(hide_code=True)
def gate_md(mo):
    mo.md(r"""
    ## 4. The gate: is the harness itself honest?

    Before reading any *cost*, prove the instrument is calibrated. Set
    `truth_cfg == cand_cfg ==` the fully-linear-Gaussian model: the candidate is the exact
    Kalman filter fitting *its own* generative model, so there is no model mismatch and no
    linearization error. Every PIT **must** be uniform.

    And it is — the three parameter fractional ranks and the predictive PIT all scatter inside
    the grey 95% simultaneous band (the binning-free version, calibrated under the uniform
    null exactly as in `sbc_walkthrough.py`). This is simulation-based calibration used as an
    acceptance test for the *measurement device*: it certifies that the grid posterior, the
    likelihood, and the rank/PIT plumbing are faithful, so that any non-uniformity in the
    panels that follow is a real cost of relaxation, not an artifact of the harness.
    """)
    return


@app.cell
def gate_fig(lab):
    lab.fig_gate()
    return


@app.cell(hide_code=True)
def costs_md(mo):
    mo.md(r"""
    ## 5. The dictionary of costs — one spot at a time

    Now relax exactly **one** of the four spots in the truth away from the (fixed) fully-linear
    Kalman candidate, so each panel isolates that spot's cost — and read the **predictive PIT**.
    The gate stays flat; each spot signs its name in the SBC dictionary's vocabulary, and the
    four are revealingly *different*:

    - **readout → linear** — the violent one. The candidate believes $y=\theta$ while reality
      folds to $y=\sin\theta$, so as the swing grows the histogram collapses to one side
      (χ² in the hundreds). The small-angle pendulum's, and the folding sensor's, revenge.
    - **dynamics → small-angle** — a gentle tilt that grows with amplitude as the linear
      restoring force drifts out of phase with the true one.
    - **process noise → Student-t** — **almost free** (χ² near the floor). Heavy kicks enter
      the *velocity* and are integrated and smeared by the dynamics before they reach the
      observed signal, so the forecast barely notices them.
    - **observation noise → Student-t** — a small, **diffuse** cost (no clean shape). Heavy tails
      hit the measurement directly, but because the filter *estimates* its noise scale, the
      in-sample outliers inflate that estimate and partly self-correct. What is left is a cost
      that **χ² registers but a Kolmogorov–Smirnov statistic or a per-bin band can miss** — a
      reminder that the scalar you choose decides what you can see. Both noise costs are roughly
      **amplitude-independent** — heavy tails do not care how far the pendulum swings.

    The clean separation of the two noise spots is the payoff of splitting them: *where* the
    heavy tail enters (hidden velocity vs observed signal) decides whether it costs anything.
    Drag the amplitude and watch the noise panels hold steady while the readout panel detonates.
    """)
    return


@app.cell
def cost_controls(mo):
    cost_amp = mo.ui.slider(
        start=0.2, stop=2.8, step=0.1, value=1.3, label="swing amplitude θ₀ (rad)", show_value=True
    )
    return (cost_amp,)


@app.cell
def cost_fig(cost_amp, lab, mo):
    mo.vstack([cost_amp, lab.fig_cost_gallery(cost_amp.value)])
    return


@app.cell(hide_code=True)
def ecdf_md(mo):
    mo.md(r"""
    ## 6. The same costs, without choosing bins

    A histogram makes you pick a bin count. The binning-free view (Säilynoja, Bürkner &
    Vehtari 2022, arXiv [2103.10522](https://arxiv.org/abs/2103.10522)) plots the empirical
    CDF of the PITs **minus** the uniform diagonal, so calibration is the flat line at $0$
    inside the band. Each defect bends the curve its own way: a slope is a bias (the readout
    relaxation, growing as it skews), an S that crosses zero in the middle is a mis-scaled
    spread (the observation-noise relaxation's faint S). The process-noise curve barely leaves
    the band at all. Same five configurations as §5, at full resolution, no bins to second-guess
    — it shares the amplitude slider above.
    """)
    return


@app.cell
def ecdf_fig(cost_amp, lab):
    lab.fig_ecdf_costs(cost_amp.value)
    return


@app.cell(hide_code=True)
def rough_md(mo):
    mo.md(r"""
    ## 7. Two readouts, only one to trust

    Now fit the **full** nonlinear/Student-t truth with the **fully-linear** Kalman candidate
    — every relaxation at once — and look at both diagnostics together.

    The three parameter fractional ranks (top) are the **rough, confounded probe**. Under
    model relaxation the candidate is not estimating the true $(q,r,\gamma)$ at all; it
    estimates the *pseudo-true* values that make the wrong model fit the data least badly. Its
    ranks therefore blend genuine miscalibration with that projection bias, and a parameter
    can look "off" merely because its linear-model meaning differs from its nonlinear one —
    so do not over-read these shapes.

    The predictive PIT (bottom) is the **trusted readout**. Its target is an *observable* — a
    real future measurement — for which there is no pseudo-true ambiguity: either the
    forecast distribution covers it at the right rate or it does not. When the two disagree,
    believe the predictive PIT. This is why the workflow prices reliability on *held-out
    observables*, not on parameter recovery.
    """)
    return


@app.cell
def rough_controls(mo):
    rough_amp = mo.ui.slider(
        start=0.3, stop=2.8, step=0.1, value=1.6, label="swing amplitude θ₀ (rad)", show_value=True
    )
    return (rough_amp,)


@app.cell
def rough_fig(lab, mo, rough_amp):
    mo.vstack([rough_amp, lab.fig_rough_vs_trusted(rough_amp.value)])
    return


@app.cell(hide_code=True)
def floor_md(mo):
    mo.md(r"""
    ## 8. The EKF's own floor

    A subtlety the gate hides. The gate used the *exact* Kalman filter, which is exact for a
    linear model at any amplitude — its left panel stays flat no matter how hard you swing.
    But if you keep the nonlinear pieces and filter them with the **EKF** — and set
    `truth_cfg == cand_cfg` so there is again *no model mismatch* — a residual non-uniformity
    appears, and it is the **EKF's own linearization error**, nothing else.

    At gentle swings it is invisible (the tangent is a fine local stand-in); wind the
    amplitude up and a mild shape emerges as the straightened readout drifts from the curved
    truth — precisely the curvature term the EKF drops, the same effect that
    `filtering_anatomy_walkthrough.py` shows breaking the EKF on the $x^2$ and $x^3$ sensors.
    The honest move is to **surface** this floor, not fix it: it is the irreducible price of
    using *any* Gaussian filter on a curved model, and it sets the baseline the next panel's
    cheapest-faithful candidate can hope to reach.
    """)
    return


@app.cell
def floor_controls(mo):
    floor_amp = mo.ui.slider(
        start=0.3, stop=2.8, step=0.1, value=2.0, label="swing amplitude θ₀ (rad)", show_value=True
    )
    return (floor_amp,)


@app.cell
def floor_fig(floor_amp, lab, mo):
    mo.vstack([floor_amp, lab.fig_ekf_floor(floor_amp.value)])
    return


@app.cell(hide_code=True)
def heatmap_md(mo):
    mo.md(r"""
    ## 9. The cost map

    Put it together. The truth is now the full nonlinear/Student-t pendulum throughout; the
    rows are candidate **strategies** — fully-linear Kalman, or an EKF that keeps the
    dynamics, the readout, or both nonlinear — and the columns sweep the swing amplitude. Each
    cell is the predictive-PIT χ² (the trusted scalar), log-coloured.

    Two readings jump out. **Down the left edge** the four strategies *converge*: at small
    swings the nonlinearity costs vanish, so what is left is the cost they all share — the
    **Student-t noise floor**, the price any Gaussian filter pays for modelling heavy-tailed
    noise as light-tailed. That is why the left column sits at a similar, modest value rather
    than at zero, and it is exactly the regime where linearizing the *dynamics* is nearly free
    — the licence the project's policy leans on for warm-starting. **Across the rows**, the
    divergence with amplitude *is* the nonlinearity cost, and it is governed by *which* piece
    you keep nonlinear: the two strategies that keep the **folding readout** (bottom rows)
    track the floor at every amplitude, while the two that linearize it (top rows) explode —
    and keeping only the *dynamics* nonlinear barely helps, because the readout is the
    load-bearing nonlinearity here. The cheapest *faithful* filter is the EKF that keeps both;
    its irreducible residual blends that Student-t noise floor with the EKF's own
    linearization error from §8 once the swing grows.
    """)
    return


@app.cell
def heatmap_fig(lab):
    lab.fig_regime_heatmap()
    return


@app.cell(hide_code=True)
def curves_md(mo):
    mo.md(r"""
    ## 10. When does linear stop being free?

    The same numbers as curves make the ordering explicit. The dotted line is the 95% uniform
    threshold; above it, a relaxation is costing measurable reliability. The two
    readout-keeping strategies stay near the noise floor across the whole sweep — never far
    from the threshold — while the two that linearize the readout cross it early and climb
    without bound once the swing leaves the small-angle regime. This curve *is* the deliverable
    — read off the amplitude where your candidate leaves the floor and you have priced the
    approximation for your regime, for the cost of a few hundred Kalman filters.
    """)
    return


@app.cell
def curves_fig(lab):
    lab.fig_regime_curves()
    return


@app.cell(hide_code=True)
def vague_md(mo):
    mo.md(r"""
    ## 11. Calibrated, and still vague

    Everything so far leaned on the predictive PIT, so be precise about what that screen
    certifies. A flat PIT means the forecaster is **honest relative to the information it
    used** — not that it is **accurate**. This gap is not an edge case; it is the generic
    failure mode, and it has a name: *variance laundering*. A linear model that cannot resolve
    the state structure driving the conditional mean sees those swings as unpredictable and,
    to stay honest, **widens** its predictive to cover them — in this fit, by inflating its
    estimated process- and observation-noise scales. Its variance climbs toward the *marginal*
    spread of the target, $\mathbb E[\mathrm{Var}(y\mid x)] + \mathrm{Var}(\mathbb E[y\mid x])$,
    while the true predictive is only $\mathbb E[\mathrm{Var}(y\mid x)]$. The excess width is
    precisely the signal it failed to resolve, relabelled as noise — and because that width is
    *correct relative to the impoverished model*, the PIT stays flat. The limiting case is
    climatology: forecast the long-run average every day and you are perfectly calibrated and
    perfectly useless.

    The candidate below keeps the folding readout but linearizes the dynamics, so it is
    squarely **PIT-calibrated** (middle: flat, in the band). Yet one of its forecasts (left) is
    far wider than the **oracle** — the sharpest forecast physically possible, which we can
    write down because we simulated the data and so know the true state and parameters. Both
    distributions cover the realized value, so neither PIT is extreme. And it is systematic
    (right): *every* forecast is wider than the oracle. PIT cannot see this, for two structural
    reasons — you score against a single point realization, which carries no information about
    the spread of the distribution it came from, and you never even build the candidate's
    predictive *distribution* in the test, only one draw from it. Sharpening the nonlinear
    prior would not rescue it: that shrinks parameter uncertainty, not the process/observation
    noise that dominates predictive width, and a point sample still cannot reveal sharpness.
    """)
    return


@app.cell
def vague_controls(mo):
    vague_amp = mo.ui.slider(
        start=0.6, stop=2.4, step=0.1, value=1.3, label="swing amplitude θ₀ (rad)", show_value=True
    )
    return (vague_amp,)


@app.cell
def vague_fig(lab, mo, vague_amp):
    mo.vstack([vague_amp, lab.fig_calibrated_but_vague(vague_amp.value)])
    return


@app.cell(hide_code=True)
def oracle_md(mo):
    mo.md(r"""
    ## 12. The oracle score sees what calibration cannot

    The width gap *should* be detectable — we have the wrong instrument, not the wrong idea.
    And the fix stays in the cheap regime. Because the data is simulated, the **oracle
    predictive** $p(y_{t+H}\mid x_t,\theta_{\text{true}})$ is free: forward-propagate the known
    true state through the true model, no inference at all. Now drop PIT and use a **proper
    scoring rule** — CRPS, estimated bandwidth-free from forecast samples, which penalizes
    *width* as well as miscentring. The wide-but-calibrated candidate scores strictly worse
    than the sharp oracle, and the gap $\text{CRPS}_{\text{candidate}}-\text{CRPS}_{\text{oracle}}$
    is a direct, sharpness-aware measure of the accuracy lost — with no nonlinear fit anywhere.

    The top panel is the cheap PIT screen; the readout-keeping strategies stay in the band,
    *calibrated*. The bottom panel is the accuracy screen: every candidate sits **above** the
    oracle floor, and that vertical gap is the sharpness the calibration screen rates as
    perfect. The gap has the right logical shape for a pre-flight. The oracle conditions on the
    *true* state and parameters, so it beats any nonlinear model you could actually fit (which
    carries its own state- and parameter-inference error); the gap therefore **overstates** the
    loss a real nonlinear fit would recover. A small, flat gap thus certifies that linear is
    genuinely fine — you have bounded the recoverable loss from above and it is negligible —
    while a gap that widens with the regime is the trigger to spend on the exact engine.
    Calibration tells you the forecast is not lying; the oracle score gap tells you whether it
    is worth anything. You need both, and only the second is the accuracy number.
    """)
    return


@app.cell
def oracle_fig(lab):
    lab.fig_oracle_screen()
    return


@app.cell(hide_code=True)
def filter_md(mo):
    mo.md(r"""
    ## 13. The fourth spot: how you compute the state

    Everything so far relaxed the *model*. There is one more place an approximation lives — the
    one your question about "states" points at — and it is a different kind: **how you compute
    the posterior over the latent state**, holding the model fixed. Four engines, in increasing
    fidelity, all run here on the *full* nonlinear model at the *true* parameters (so only the
    state inference differs), against a Gaussian-noise truth (so heavy tails are not a confound
    and this isolates state representation alone):

    - **KF** — linearize the whole model to fully-linear, then the exact Kalman filter;
    - **EKF** — keep the model, linearize each step at the mean (first-order Jacobian);
    - **UKF** — keep the model, match moments through deterministic sigma points (no Jacobian);
    - **PF** — keep the model, carry the state belief as a particle cloud (no Gaussian assumption).

    On a linear-Gaussian model all four coincide exactly (verified to machine precision — the KF
    is the reference the EKF/UKF/PF are checked against). On the nonlinear model the lines fan
    out, and each gap names a spot:

    - **KF → EKF/UKF** — the cost of linearizing the *model* (large, grows with swing);
    - **EKF → UKF** — the linearization *quality* (Jacobian vs sigma points): small here, since
      the curvature is mild — an honest near-tie;
    - **UKF → PF** — the cost of the **Gaussian-state** assumption: the particle filter keeps the
      whole (and, under the folding readout, increasingly non-Gaussian) state posterior, so it
      pulls ahead as the swing grows. This is the "states" spot the other knobs never touch;
    - **PF → oracle** — the **irreducible** uncertainty of inferring the state from finite data
      rather than knowing it. It is large and roughly flat.

    That last decomposition sharpens §11–§12: the oracle gap a candidate shows is
    *recoverable* (candidate → PF, what a better engine could claw back) **plus** *irreducible*
    (PF → oracle, which no engine can). A cheap filter is worth replacing only to the extent of
    the first.
    """)
    return


@app.cell
def filter_fig(lab):
    lab.fig_filter_comparison()
    return


@app.cell(hide_code=True)
def bridge_md(mo):
    mo.md(r"""
    ## 14. Back to the engine

    This notebook is the **cheap front end** of a decision the expensive engine then makes
    properly. The mapping is direct:

    | in this demo | in the codebase |
    |---|---|
    | the relaxed Gaussian candidate (KF/EKF/UKF) | a linearized surrogate (IEKS / Laplace / local-linear) |
    | the particle filter (§13) | `cuthbert`'s `smc.particle_filter` over the true emission |
    | the full nonlinear/Student-t truth | the true continuous-time nonlinear SSM |
    | the exact production posterior | particle/SMC over the true emission, Euler–Maruyama over the true drift |
    | a flat predictive PIT in this regime | a regime where linear warm-starting is safe and cheap |
    | a detonating predictive PIT | a regime where only the exact engine may produce a reported number |
    | the oracle predictive (true state + params) | the best-possible forecast — a simulation-only yardstick, no fittable analogue |
    | the CRPS gap to the oracle | the sharpness-aware accuracy loss the PIT cannot see |
    | the PF→oracle gap (§13) | the irreducible part of that loss — what *no* engine recovers |

    The policy survives this demo intact. A linearized filter is *fine to initialise with* —
    the §9 left edge shows the *nonlinearity* cost is nearly free where curvature is small,
    which is exactly where warm-starting lives — but it **biases a reported result** wherever
    the cost map is hot, and only an exact engine (particle/SMC, Euler–Maruyama, the
    exactly-corrected `amala_exact` proposal — never the biased `amala` / `amala_plus`) may
    stand on a reported path. The guard test
    `tests/models/ssm/test_linearization_init_only.py` enforces that the linearization backend
    is importable only from the warmup path. The persistent noise floor in §9's left column is
    a second face of the same principle: it is the price a *Gaussian emission* pays against the
    truth's heavy tails, and it is exactly why the production filter runs particle/SMC over the
    **true** emission density rather than a Gaussian surrogate — a cost no amount of clever
    linearization of the dynamics can remove.

    What this notebook adds is a **price tag, computed before the expense** — and it is *two*
    numbers, because §11–§12 showed calibration alone is not enough. Run the cross-generation
    on your model and regime, then ask both questions. **Is the cheap fit lying?** — the
    predictive PIT. **Is it worth anything?** — the oracle CRPS gap. A cheap Gaussian fit is
    defensible only when the PIT stays in the band *and* the oracle gap is small and flat: then
    a particle-week is genuinely unnecessary. If the PIT detonates, or it stays flat while the
    oracle gap widens with the regime, you have learned — for the cost of a few hundred Kalman
    filters, with no nonlinear inference at all — that the expensive engine is not optional.
    """)
    return


if __name__ == "__main__":
    app.run()
