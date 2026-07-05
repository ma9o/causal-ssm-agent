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
    # Simulation-Based Calibration, from scratch

    You wrote a model and a prior, pointed a sampler at some data, and got a posterior
    back. **How do you know the sampler returned the posterior you asked for, and not some
    confidently-wrong neighbour of it?** Convergence diagnostics ($\hat R$, ESS) catch a
    chain that failed to *mix*; they say nothing about a sampler that mixes beautifully
    toward the *wrong stationary distribution*. Posterior-predictive checks ask whether the
    fitted model resembles the data — a question about the *model*, not the *algorithm*.

    Simulation-Based Calibration (Talts, Betancourt, Simpson, Vehtari & Gelman 2018, arXiv
    [1804.06788](https://arxiv.org/abs/1804.06788)) fills exactly that gap. It is a single
    self-consistency check on the *computation*: average the posterior over datasets drawn
    from the prior predictive and you must recover the prior,

    $$\int\!\!\int p(\theta \mid y)\, p(y \mid \theta')\, p(\theta')\; dy\, d\theta' \;=\; p(\theta).$$

    Turn that identity into a test. Draw a parameter from the prior, simulate data from it,
    and draw $L$ samples from your posterior. The prior draw and the $L$ posterior draws are
    then **exchangeable** — $L{+}1$ samples from one and the same distribution — so the
    **rank** of the prior draw among them is uniform on $\{0, 1, \dots, L\}$. Do it
    thousands of times and histogram the ranks. A correct sampler gives a *flat* histogram.
    A wrong one is not just non-flat: the **shape** of the deviation names the defect.

    To learn to read those shapes we need a sampler we can make correct *and* deliberately
    break, on demand. So we pick a toy whose posterior is known in closed form — then we
    play the exact sampler (uniform ranks, guaranteed) and four broken ones, and watch each
    failure sign its name. Support code lives in `sbc_support.py`.
    """)
    return


@app.cell
def imports_lab():
    import sbc_support as lab

    return (lab,)


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## 1. The toy, and the loop

    A conjugate Gaussian: an unknown mean $\theta$ under a Gaussian prior, observed through
    Gaussian noise of *known* scale.

    $$\theta \sim \mathcal N(\mu_0,\, \tau_0^2), \qquad y_i \sim \mathcal N(\theta,\, \sigma^2),\quad i = 1,\dots,N.$$

    The posterior is Gaussian and closed-form — only the sample mean $\bar y$ matters, since
    it is sufficient and $\bar y \mid \theta \sim \mathcal N(\theta, \sigma^2/N)$:

    $$\frac{1}{v} = \frac{1}{\tau_0^2} + \frac{N}{\sigma^2}, \qquad m = v\!\left(\frac{\mu_0}{\tau_0^2} + \frac{N\,\bar y}{\sigma^2}\right), \qquad \theta \mid y \sim \mathcal N(m,\, v).$$

    Knowing $m$ and $v$ exactly means we can be the *perfect* sampler, and we can corrupt it
    in named ways — widen it, narrow it, shift it, correlate its draws. One SBC simulation is
    four steps:

    1. **draw a truth** &nbsp; $\tilde\theta \sim p(\theta)$ &nbsp;— the parameter this run is secretly about;
    2. **simulate data** &nbsp; $\tilde y \sim p(y \mid \tilde\theta)$;
    3. **sample the posterior** &nbsp; $\theta^{(1)},\dots,\theta^{(L)} \sim q(\theta \mid \tilde y)$ &nbsp;— $q$ is the *candidate* under test;
    4. **rank the truth** &nbsp; $r = \#\{\,l : \theta^{(l)} < \tilde\theta\,\} \in \{0,\dots,L\}$.

    When $q$ is the true posterior, $r$ is uniform. One more sightline: as $L\to\infty$ the
    normalised rank $r/L$ converges to the posterior CDF evaluated at the truth,
    $F_q(\tilde\theta)$ — the **probability integral transform**, uniform precisely when $q$
    is calibrated. SBC is the PIT, made out of finitely many samples.
    """)
    return


@app.cell(hide_code=True)
def one_run_md(mo):
    mo.md(r"""
    ## 2. One simulation, in slow motion

    Before the thousands, watch a single run. A truth $\tilde\theta$ falls out of the prior
    (grey); $N$ data points are drawn from it (orange ticks); the posterior (blue) contracts
    around them. We then take $L=99$ posterior draws and split them at $\tilde\theta$: the
    ones **below** it (green) are counted — that count *is* the rank.

    Drag **dataset** to redraw the whole run, and **N** to change how much data each dataset
    carries. More data tightens the posterior, but notice the rank does not drift toward any
    particular value as you do so — over many redraws it stays equally likely to land
    anywhere in $\{0,\dots,99\}$. That stubborn uniformity, run after run, is the entire
    signal SBC listens for.
    """)
    return


@app.cell
def one_run_controls(mo):
    one_run_seed = mo.ui.slider(start=0, stop=40, step=1, value=3, label="dataset", show_value=True)
    one_run_n = mo.ui.slider(
        start=1, stop=40, step=1, value=8, label="N (data points)", show_value=True
    )
    return one_run_n, one_run_seed


@app.cell
def one_run_fig(lab, mo, one_run_n, one_run_seed):
    mo.vstack(
        [
            mo.hstack([one_run_seed, one_run_n], justify="start", gap=2),
            lab.fig_one_run(one_run_seed.value, one_run_n.value),
        ]
    )
    return


@app.cell(hide_code=True)
def calibrated_md(mo):
    mo.md(r"""
    ## 3. Many runs: what 'calibrated' looks like

    Now run $S$ simulations against the **exact** posterior and histogram the ranks. The
    dotted line is the uniform expectation $S/\text{bins}$; the grey ribbon is a **95%
    simultaneous band** — the region a genuinely-uniform histogram stays inside, *jointly
    across all bars*, 95% of the time. (Naïve per-bar intervals would flag a false alarm
    somewhere almost every time you have twenty bars; the simultaneous band, calibrated by
    Monte-Carlo under the uniform null, is the honest reference.)

    Every bar sits inside the ribbon, scattering around the line like coin flips. This is the
    *only* outcome a correct sampler produces, and it is the picture every later panel is
    measured against. Push $S$ up: the bars do not flatten so much as the *band tightens
    around them* — more simulations buy resolution, the power to see ever-smaller defects.
    """)
    return


@app.cell
def calibrated_controls(mo):
    calib_s = mo.ui.slider(
        start=200, stop=8000, step=200, value=2000, label="S (simulations)", show_value=True
    )
    calib_bins = mo.ui.radio(
        options={"10 bins": 10, "20 bins": 20, "25 bins": 25, "50 bins": 50},
        value="20 bins",
        label="bins",
    )
    return calib_bins, calib_s


@app.cell
def calibrated_fig(calib_bins, calib_s, lab, mo):
    mo.vstack(
        [
            mo.hstack([calib_s, calib_bins], justify="start", gap=2),
            lab.fig_calibrated(S=calib_s.value, bins=calib_bins.value),
        ]
    )
    return


@app.cell(hide_code=True)
def gallery_md(mo):
    mo.md(r"""
    ## 4. The dictionary of shapes

    Here is the payoff — six candidate samplers, the same prior draws, six histograms. Learn
    these and you can diagnose a sampler from across the room:

    - **calibrated** — flat. The exact posterior; ranks uniform.
    - **overconfident** (posterior too *narrow*) — a **∪ valley**. A too-tight posterior puts
      the truth in its tails too often, so ranks pile at *both* ends.
    - **underconfident** (posterior too *wide*) — a **∩ dome**. A bloated posterior swallows
      the truth near its centre, so ranks crowd the *middle*.
    - **biased high** (posterior centred *above* the truth) — a **↘ ramp**. Most draws sit
      above $\tilde\theta$, so few fall below it: ranks crowd *low*.
    - **biased low** — the mirror **↗ ramp**, ranks crowd *high*.
    - **under-thinned** (correct posterior, but autocorrelated draws) — a **∪ valley** again.
      The shape collides with overconfidence on purpose; §8 untangles them.

    Spread and location are the two axes of error, and SBC separates them cleanly:
    miscalibrated *width* bends the histogram into a curve (∪ or ∩), miscalibrated *centre*
    tilts it into a ramp.
    """)
    return


@app.cell
def gallery_fig(lab):
    lab.fig_gallery()
    return


@app.cell(hide_code=True)
def dial_md(mo):
    mo.md(r"""
    ## 5. Dial in your own defect

    Two knobs, live. **scale** multiplies the posterior's standard deviation; **bias** slides
    its mean, in units of a posterior standard deviation. The left panel is the rank
    histogram; the right is the same evidence as an **ECDF-minus-uniform** curve (the topic of
    §6); the title is an automatic read of the shape.

    Things worth finding by hand: a tiny `scale = 0.9` is nearly invisible, while `0.5` is a
    canyon — calibration error grows fast as a posterior tightens. Combine a narrow scale with
    a bias and the ∪ tilts into a lopsided wedge: real samplers usually fail on *both* axes at
    once, and the histogram superposes the two signatures rather than hiding either.
    """)
    return


@app.cell
def dial_controls(mo):
    dial_scale = mo.ui.slider(
        start=0.3, stop=2.5, step=0.05, value=0.7, label="scale (σ multiplier)", show_value=True
    )
    dial_bias = mo.ui.slider(
        start=-1.2, stop=1.2, step=0.05, value=0.3, label="bias (in σ)", show_value=True
    )
    return dial_bias, dial_scale


@app.cell
def dial_fig(dial_bias, dial_scale, lab, mo):
    mo.vstack(
        [
            mo.hstack([dial_scale, dial_bias], justify="start", gap=2),
            lab.fig_dial(scale=dial_scale.value, bias=dial_bias.value),
        ]
    )
    return


@app.cell(hide_code=True)
def ecdf_md(mo):
    mo.md(r"""
    ## 6. Reading it without bins

    A histogram makes you choose a bin count, and that choice is not free (§7). The modern
    SBC diagnostic (Säilynoja, Bürkner & Vehtari 2022, arXiv
    [2103.10522](https://arxiv.org/abs/2103.10522)) drops the choice entirely: plot the
    empirical CDF of the ranks **minus** the uniform diagonal, so calibration is the
    horizontal line $0$, surrounded by a simultaneous band.

    The deviations are now directional. The curve rises wherever the candidate piles up *too
    much* cumulative mass at low ranks and dips where it piles up too little, so each defect
    traces its own gentle path out of the band:

    - **bias** — a single hump, *above* the line for biased-high (excess of low ranks),
      *below* for biased-low;
    - **mis-scaled spread** — an **S** that crosses zero in the middle: overconfident leaves
      then *re-enters* from the opposite side (excess at both tails), underconfident does the
      reverse.

    Same information as the histograms, but every rank contributes at full resolution and
    there is no knob to second-guess.
    """)
    return


@app.cell
def ecdf_fig(lab):
    lab.fig_ecdf_gallery()
    return


@app.cell(hide_code=True)
def binning_md(mo):
    mo.md(r"""
    ## 7. Why the bin count is not innocent

    The same overconfident ranks, drawn three ways. Too **coarse** (10 bins) blurs the ∪
    into a shallow dish and can hide a localised defect inside a wide bar; too **fine** (50
    bins) thins each bar to a handful of counts, so honest sampling noise starts to look like
    structure. A practical rule of thumb keeps $L{+}1$ divisible by the bin count (here
    $100$ splits evenly into $10, 20, 25, 50$) so equal-width bins hold equal numbers of rank
    values — otherwise the binning alone stamps a sawtooth onto a *perfectly uniform*
    histogram. The ECDF view in §6 sidesteps the whole question.
    """)
    return


@app.cell
def binning_fig(lab):
    lab.fig_binning()
    return


@app.cell(hide_code=True)
def samplesize_md(mo):
    mo.md(r"""
    ## 8. How many simulations is enough?

    SBC has *power*, and power costs simulations. Here is one **mild** defect — a posterior
    only 20% too narrow — at four budgets. At $S=100$ the band is so wide the ∪ hides inside
    it: you would sign off on a broken sampler. By $S=8000$ the band has closed to a sliver
    and the valley is undeniable. The band narrows like $1/\sqrt S$, so the smallest defect
    you can resolve shrinks the same way.

    The asymmetry matters for how you read a *pass*: a flat histogram at small $S$ is not
    evidence of calibration, only absence of evidence against it. SBC can prove a sampler
    *wrong*; it can only fail to catch one that is *subtly* wrong, and how subtle you can
    afford to miss is set by how many fits you can pay for.
    """)
    return


@app.cell
def samplesize_fig(lab):
    lab.fig_sample_size()
    return


@app.cell(hide_code=True)
def autocorr_md(mo):
    mo.md(r"""
    ## 9. The trap that matters for MCMC: autocorrelation

    Back to that collision in §4 — why does a *correct* sampler with *correlated* draws fake
    a ∪? The draws here are stationary at the true posterior; the only flaw is that
    consecutive ones are autocorrelated, as every MCMC chain's are. Within a single run the
    $L$ draws then bunch together and explore only part of the posterior, so a fresh truth
    $\tilde\theta$ tends to land entirely above or entirely below the bunch — ranks at the
    extremes, a ∪ that is **indistinguishable from overconfidence**. In the limit $\rho\to 1$
    the chain is one value repeated $L$ times and the rank is $0$ or $L$ and nothing else.

    The fix is not more compute, it is **thinning**: keep every $k$-th draw until what remains
    is effectively independent. Raise $\rho$ to manufacture the valley, then raise the
    thinning until the right panel flattens back to uniform — the effective-sample-size
    fraction in each subtitle is what you are really buying. This is the single most common
    way a *correct* MCMC sampler fails SBC, and the reason ranks must be built from
    near-independent draws before their histogram means anything.
    """)
    return


@app.cell
def autocorr_controls(mo):
    ac_rho = mo.ui.slider(
        start=0.0, stop=0.97, step=0.01, value=0.9, label="ρ (autocorrelation)", show_value=True
    )
    ac_thin = mo.ui.slider(start=1, stop=40, step=1, value=12, label="thinning k", show_value=True)
    return ac_rho, ac_thin


@app.cell
def autocorr_fig(ac_rho, ac_thin, lab, mo):
    mo.vstack(
        [
            mo.hstack([ac_rho, ac_thin], justify="start", gap=2),
            lab.fig_autocorrelation(rho=ac_rho.value, thin=ac_thin.value),
        ]
    )
    return


@app.cell(hide_code=True)
def bridge_md(mo):
    mo.md(r"""
    ## 10. Back to the codebase

    SBC is the acceptance test for an inference *engine*, and it answers a different question
    from the calibration check already in the pipeline. The posterior-predictive
    `_check_calibration` in `models/posterior_predictive.py` asks *does the fitted model cover
    the real data?* — a verdict on the **model**, given one dataset. SBC asks *does the
    algorithm recover parameters it was simulated from?* — a verdict on the **computation**,
    averaged over the prior, needing no real data at all. A model can be wrong and pass SBC;
    an engine can be broken and pass posterior-predictive checks. You want both.

    The mapping into this project's continuous-time nonlinear SSM is direct:

    | in this toy | in the codebase |
    |---|---|
    | the candidate posterior $q$ under test | the particle/SMC posterior from `cuthbert` `smc.particle_filter` via `filtering.filter` |
    | the exact sampler → uniform ranks | the exact engines: particle/SMC over the true emission, Euler–Maruyama over the true drift, and the exactly-corrected `amala_exact` proposal |
    | a deformed sampler → non-uniform ranks | the biased *uncorrected* proposals `amala` / `amala_plus`, or any linearized surrogate (IEKS/Laplace) standing in on a *reported* path |
    | the under-thinned ρ chain → a fake ∪ | an MCMC/SMC posterior read off without enough thinning |

    Two takeaways land straight on the project's invariants. First, the
    [linearization-is-init-only policy](../../../AGENTS.md): a linearized engine biases the
    posterior on a reported path, and SBC is exactly the end-to-end instrument that would
    catch that bias as non-uniform ranks — the same reason `amala` / `amala_plus` stay
    non-default and "must never gate a reported result." Second, because the production
    posterior is sampled by MCMC/SMC, §9 is not a curiosity: ranks must be built from
    near-independent draws or the histogram lies in the direction of overconfidence.

    The catch is cost. One SBC run is $S$ *full* posterior fits, which is why SBC sits
    downstream of the fit in the pipeline (`specification_funnel_walkthrough.py` flags it as a
    post-fit diagnostic) rather than inside the hot loop. This toy is the cheap sandbox where
    the shapes are learned for free; the engine is where they are spent.
    """)
    return


if __name__ == "__main__":
    app.run()
