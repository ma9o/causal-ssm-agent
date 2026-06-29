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
    # Two geometries of marginalization — the funnel it *kills* and the ridge it *reveals*

    The companion notebook (`confounder_marginalization_walkthrough.py`) shows *where*
    an unobserved confounder goes when you marginalize it: it collapses into a single
    **covariance** between the things it touched. This notebook is about the other half
    of that story — **why that collapse helps the sampler**, and why "helps inference"
    and "identifies the effect" are two *different* things that get conflated.

    The whole thing rests on one fact you already know:

    > For a Gaussian latent, the **only** imprint $U$ leaves on the likelihood is the
    > covariance it induces among its children. So when a sampler wiggles $U$'s values,
    > everything the data can feel about that wiggle is summarized by that covariance.
    > Marginalizing = jump straight to the covariance and skip the wiggling.

    That single move shows up as **two** distinct pieces of likelihood geometry, and
    the point of this notebook is to *see* both and keep them apart:

    1. **The wiggle is invisible.** Rescaling the latent and compensating the loadings
       leaves the data untouched — a whole curve of latent parameters, one dataset.
    2. **The funnel marginalization always kills.** That invisible direction is a flat
       trench in $(\text{loading}, \text{scale})$ space — a sampler crawls it forever.
       Switch to the one identified product and it's a single peak. This is the
       *unconditional* inference win: it happens even with no confounding at all.
    3. **The ridge marginalization only reveals.** Once $U$ is a residual covariance
       $c$, the causal slope $\beta$ and $c$ trade off along a flat ridge whenever they
       land in the same observable cell. Marginalizing does not remove this — it makes
       it an honest, visible plateau. Identification is a *separate* question: whichever
       constraint cuts across the ridge.
    4. **The synthesis.** One move, two geometries: the funnel goes away for free, the
       ridge is exposed for inspection.

    Support code (simulators reused from `confounder_lab`, the likelihood surfaces, all
    plotting) lives in `inference_geometry_lab.py`; the ideas are inline.
    """)
    return


@app.cell
def imports_lab():
    import confounder_lab as clab
    import inference_geometry_lab as lab
    import numpy as np

    return clab, lab, np


@app.cell(hide_code=True)
def wiggle_md(mo):
    mo.md(r"""
    ## 1. Wiggling the latent is just redrawing one covariance

    Start with the back-door world — a confounder $U$ of scale $\tau = \mathrm{sd}(U)$
    that loads on the treatment ($U \to X$, loading $a$) and on the outcome ($U \to Y$,
    loading $b$), on top of the effect we actually want ($X \to Y$, slope $\beta$):
    """)
    return


@app.cell(hide_code=True)
def wiggle_dag(mo):
    mo.mermaid("""
    graph LR
      U(("U<br/>scale τ")):::latent
      X["X — treatment"]:::obs
      Y["Y — outcome"]:::obs
      U -->|a| X
      U -->|b| Y
      X -->|β| Y
      classDef latent fill:#f3f4f6,stroke:#6b7280,stroke-width:1px,stroke-dasharray:5 4;
      classDef obs fill:#dbeafe,stroke:#2563eb,stroke-width:1px;
    """)
    return


@app.cell(hide_code=True)
def wiggle_math_md(mo):
    mo.md(r"""
    Write the structural equations and watch what the data can actually feel:

    $$
    X = aU + \varepsilon_X, \qquad
    Y = \beta X + bU + \varepsilon_Y, \qquad
    U \sim \mathcal{N}(0, \tau^2).
    $$

    $U$ reaches the data only through the products $aU$ and $bU$, and the law of $aU$
    depends only on $a\tau$ (likewise $b\tau$). So the entire one-parameter family

    $$
    (a,\; b,\; \tau) \;\longmapsto\; \left(\tfrac{a}{k},\; \tfrac{b}{k},\; k\tau\right),
    \qquad k > 0,
    $$

    yields an **identical** observable covariance: only $a\tau$ and $b\tau$ are
    identified, the overall latent scale is free. That free direction is the redundancy
    everything below exploits.

    Drag the slider. The left panel is the latent $U$ the sampler resamples — its scale
    $\tau$ breathes — while the right panel, the data cloud $(X, Y)$ the likelihood
    evaluates, **does not move**. A whole curve of latent parameters, one frozen dataset.
    """)
    return


@app.cell
def wiggle_controls(mo):
    wiggle = mo.ui.slider(
        start=-1.3,
        stop=1.3,
        step=0.05,
        value=0.0,
        label="rescale the latent U (slide along the redundant direction)",
        show_value=True,
    )
    return (wiggle,)


@app.cell
def wiggle_fig(lab, mo, wiggle):
    mo.vstack([wiggle, lab.fig_wiggle_invariance(wiggle.value)])
    return


@app.cell(hide_code=True)
def funnel_md(mo):
    mo.md(r"""
    ## 2. The funnel marginalization *always* kills

    Make that invisible direction a likelihood surface. Take one latent child,
    $W = \lambda U + \varepsilon_W$ with $U \sim \mathcal{N}(0, \tau^2)$. The data hand
    you exactly one number, $\mathrm{Var}(W) = \lambda^2\tau^2 + \sigma^2$, so the
    likelihood over $(\lambda, \tau)$ depends *only* on the product
    $v = \lambda^2\tau^2$. The result (left) is a **flat trench** along
    $\lambda\tau = \text{const}$ — a straight diagonal in log-log. Every point on that
    white dashed line fits identically; a sampler in $(\lambda, \tau)$ coordinates can
    only crawl *along* the trench, never converging on either coordinate. This is the
    classic latent×loading **funnel**.

    $$
    \mathrm{Var}(W) = \lambda^2\tau^2 + \sigma_W^2
    \;\;\Longrightarrow\;\;
    \text{data identify only } v = \lambda^2\tau^2,
    \qquad \text{trench: } \lambda\tau = \sqrt{v} = \text{const.}
    $$

    Now marginalize: reparameterize by the single identified product $v$ (right). The
    trench collapses to one clean peak at the true $v_0$. Nothing was lost — the data
    only ever knew $v$ — and the pathological geometry is simply gone.

    The key point: **this collapse has nothing to do with confounding.** It is pure
    inference economy — fewer dimensions, no funnel, lower Monte-Carlo variance — and it
    pays off in *every* world, on-path or off. It is the unconditional half of "why
    marginalize."
    """)
    return


@app.cell(hide_code=True)
def funnel_dag(mo):
    mo.mermaid("""
    graph LR
      U(("U<br/>scale τ")):::latent
      W["W — observed child"]:::obs
      U -->|λ| W
      classDef latent fill:#f3f4f6,stroke:#6b7280,stroke-width:1px,stroke-dasharray:5 4;
      classDef obs fill:#dbeafe,stroke:#2563eb,stroke-width:1px;
    """)
    return


@app.cell
def funnel_fig(lab):
    lab.fig_funnel()
    return


@app.cell(hide_code=True)
def ridge_md(mo):
    mo.md(r"""
    ## 3. The ridge marginalization only *reveals*

    Now actually *do* the marginalization, on the very §1 equations. Collect the two
    $U$-bearing terms into residuals $\eta_X = aU + \varepsilon_X$ and
    $\eta_Y = bU + \varepsilon_Y$. The explicit model

    $$
    X = aU + \varepsilon_X, \qquad
    Y = \beta X + bU + \varepsilon_Y, \qquad
    U \sim \mathcal{N}(0, \tau^2)
    $$

    then becomes one in which $U$ has disappeared entirely — its whole footprint is a
    single covariance between those residuals:

    $$
    X = \eta_X, \qquad
    Y = \beta X + \eta_Y, \qquad
    \begin{pmatrix} \eta_X \\ \eta_Y \end{pmatrix} \sim
    \mathcal{N}\!\left( \mathbf{0},\;
    \begin{pmatrix} \sigma_X^2 & c \\ c & \sigma_Y^2 \end{pmatrix} \right),
    $$

    $$
    \sigma_X^2 = a^2\tau^2 + s_X^2, \qquad
    \sigma_Y^2 = b^2\tau^2 + s_Y^2, \qquad
    c = ab\,\tau^2 .
    $$

    That is the entire rewrite. The shared term $bU$ and the scale $\tau^2$ do **not**
    survive as themselves: $b^2\tau^2$ is swallowed into the diagonal variance
    $\sigma_Y^2$, where it is indistinguishable from ordinary noise $s_Y^2$ (this is why
    §2's funnel disappears — $a, b, \tau$ only ever surface as those lumped sums). The
    *one* thing that escapes the diagonal — the only surviving trace that $X$ and $Y$
    ever shared a cause — is the off-diagonal $c = ab\,\tau^2$. Five latent-side numbers
    $(a, b, \tau^2, s_X^2, s_Y^2)$ collapse to three identified ones
    $(\sigma_X^2, \sigma_Y^2, c)$, and $U$ is gone from the state.

    *Where* that $c$ lands is decided by the graph. On-path ($U \to X$, so $a \neq 0$)
    it is nonzero and falls in the $X$–$Y$ cell, colliding with $\beta$. Off-path
    ($X \perp U$, so $a = 0$) it is $c = ab\tau^2 = 0$ there — $S_{XY} = \beta\,S_{XX}$
    exactly — and the confounder's covariance reappears in a *different* cell,
    $\mathrm{Cov}(Y, S) = b\,g\,\tau^2$ (with $g$ the $U \to S$ loading). Same
    confounder, same strength; only the cell it occupies moves.

    Now fit the two free knobs $(\beta, c)$ to the observable covariance of $(X, Y)$.
    The model feels them only through the single combination $\beta\,S_{XX} + c$ (the
    $X$–$Y$ cell), so the likelihood is a **flat ridge** along

    $$
    \beta\,S_{XX} + c \;=\; S_{XY}.
    $$

    Both panels below are the *same* ridge — $(X, Y)$ alone can never split the causal
    slope from the handshake. What differs is the **knife**: an independent constraint
    on $c$ that cuts across the ridge and pins $\beta$ at the crossing.

    - **Off-path (§7).** The graph hands you the knife for free: with no $U \to X$ edge,
      the residuals of $X$ and $Y$ are uncorrelated, so $c \equiv 0$ structurally. The
      ridge is cut at the true $\beta$ — identified, with $U$ never observed.
    - **On-path (§1).** Nothing observable supplies a knife. The only way to learn $c$
      is to **measure $U$** (an anchor / proxy indicators); absent that, the entire
      ridge is admissible and you must report the equivalence class, not a point — the
      **drop**.

    Marginalizing did not create this ridge and cannot remove it; it *revealed* it,
    trading a high-dimensional funnel for an honest 2-parameter plateau you can read.
    """)
    return


@app.cell(hide_code=True)
def ridge_dag(mo):
    mo.mermaid("""
    graph LR
      subgraph onpath["§1 on-path — handshake hits the X-Y cell"]
        Xa["X"]:::obs
        Ya["Y"]:::obs
        Xa -->|β| Ya
        Xa -. "c (handshake)" .- Ya
      end
      subgraph offpath["§7 off-path — handshake lands on Y-S"]
        Xb["X"]:::obs
        Yb["Y"]:::obs
        Sb["S"]:::obs
        Xb -->|β| Yb
        Yb -. "c (handshake)" .- Sb
      end
      classDef obs fill:#dbeafe,stroke:#2563eb,stroke-width:1px;
    """)
    return


@app.cell
def ridge_fig(clab, lab, np):
    _on = clab.simulate_confounded(np.random.default_rng(7))
    _off = clab.simulate_offpath(np.random.default_rng(11))
    _s2_on = clab.sample_cov(_on, ["X", "Y"])
    _s2_off = clab.sample_cov(_off, ["X", "Y"])
    lab.fig_ridge_and_cuts(_s2_on, _s2_off)
    return


@app.cell(hide_code=True)
def synthesis_md(mo):
    mo.md(r"""
    ## 4. The synthesis: one move, two geometries

    Marginalizing a Gaussian confounder is a single act — **switch to the sufficient
    (covariance) parameterization** — but it does two unrelated things, and reading it
    as one is exactly the conflation to avoid:

    - **It always kills the funnel.** The redundant $(\text{loading}, \text{scale})$
      direction, and the entire $O(T)$ latent trajectory behind it, vanish. The sampler
      stops crawling a trench. This is *unconditional* — it is why you marginalize even
      when there is no confounding to worry about.
    - **It only reveals the ridge.** Where the surviving covariance lands in the
      observable cells is fixed by the graph, not by the parameterization. If it lands
      on the estimand's cell, $\beta$ and $c$ form a flat ridge — genuine,
      information-theoretic non-identifiability. Marginalizing makes that plateau
      *visible* and *cheap to detect*; it does not make it go away.

    So the honest summary of our whole exchange: marginalizing buys the funnel removal
    every time (cheaper, lower-variance, funnel-free inference). Whether you *also* get
    an identified effect is the separate question of whether a knife — the graph's own
    $c = 0$, or an observable anchor on $U$ — cuts across the ridge. And even when no
    knife exists, marginalizing still earns its keep: the sampler finds the flat
    plateau *instantly* instead of thrashing in a high-dimensional funnel chasing a $U$
    it could never have pinned down. The flatness is the honest signal that there was
    nothing there to find.
    """)
    return


if __name__ == "__main__":
    app.run()
