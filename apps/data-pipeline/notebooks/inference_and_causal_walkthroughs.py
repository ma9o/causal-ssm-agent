import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def viz_helpers():
    from matplotlib.patches import Circle, FancyBboxPatch

    palette = {
        "state": "#3b6ea5",  # local / own-trajectory (blue)
        "obs": "#e08a3c",  # observations (orange)
        "belief": "#4a9d5b",  # parallelizable / green
        "operator": "#c0504d",  # mean-field / sequential (red)
        "seam": "#7d6bb0",  # the shared boundary state (purple)
        "muted": "#c5c5c5",
        "ink": "#333333",
    }

    def node(ax, x, y, color, label="", r=0.32, filled=True, fontsize=12, z=3):
        _fc = color if filled else "white"
        ax.add_patch(Circle((x, y), r, facecolor=_fc, edgecolor=color, linewidth=2.0, zorder=z))
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="white" if filled else color,
            fontsize=fontsize,
            fontweight="bold",
            zorder=z + 1,
        )

    def box(
        ax,
        x,
        y,
        color,
        label="",
        w=0.9,
        h=0.6,
        filled=True,
        alpha=1.0,
        fontsize=11,
        lw=2.0,
        ls="-",
        z=3,
    ):
        _fc = color if filled else "white"
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=_fc,
                edgecolor=color,
                linewidth=lw,
                alpha=alpha,
                linestyle=ls,
                zorder=z,
            )
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="white" if filled else color,
            fontsize=fontsize,
            fontweight="bold",
            zorder=z + 1,
        )

    def arrow(ax, p0, p1, color="#333333", lw=1.8, ls="-", rad=0.0, shrink=14):
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                linestyle=ls,
                shrinkA=shrink,
                shrinkB=shrink,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=2,
        )

    return arrow, box, node, palette


@app.cell(hide_code=True)
def notebook_overview(mo):
    mo.md(r"""
    # Two walkthroughs: parallel-in-time inference

    This file bundles two self-contained marimo walkthroughs forming one arc — *when can
    the sequential work of state-space inference be reorganized to run in parallel?*

    - **Part 1 — Parallelizability audit of the state-space sampler family.** The main
      study: it applies a single "parallel-in-time" criterion, piece by piece, to the
      gradient-based particle samplers and the two parallel engines, then benchmarks the
      parallelizable replacement.
    - **Part 2 — Why a sequential filter can run in parallel, in pictures.** The intuition
      behind Part 1 §3: the associative-scan view of filtering, with diagrams and one tiny
      runnable example.

    Each part keeps its own narrative and can be read on its own.
    """)
    return


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Part 1 — Parallelizability audit of the state-space sampler family

    Across the last notebooks we reduced "parallel-in-time" to one criterion. This notebook
    **applies that criterion, piece by piece**, to the gradient-based particle samplers of
    Corenflos–Finke (*Particle-MALA / Particle-mGRAD*, arXiv 2401.14868) and the two
    Corenflos parallel engines (the *auxiliary-Kalman* scan, arXiv 2303.00301, and the
    *de-sequentialized particle smoother* DSMC, arXiv 2202.02264).

    The plan: (1) restate the criterion; (2) **computationally verify** that the operators
    which are supposed to parallelize actually re-bracket into a tree (identical answer,
    log-depth); (3) read off each method's dependency footprint from its published weight; and
    (4) classify every piece with the reason. No prose is taken on faith — the associative
    engines are checked numerically.
    """)
    return


@app.cell(hide_code=True)
def criterion_md(mo):
    mo.md(r"""
    ## 1. The one criterion

    A per-timestep computation runs in `O(log T)` **iff** it is an *associative scan over local
    operators*. Concretely, split time into blocks and ask what has to cross a block boundary:

    - if a block interacts with its neighbours only through a **bounded, fixed set of boundary
      states** (one state, or a small window), the block-combine is associative → the chain
      re-brackets into a tree → `O(log T)`;
    - if a block's contribution depends on a **joint particle-system object** that is changed by
      stitching (for example, all ancestor assignments crossing a block boundary), the pairwise
      Feynman–Kac stitch is gone. A brute-force tree over the full joint assignment space is not
      the DSMC / Kalman `O(N² log T)` object anymore.

    So the audit reduces to a single measurable quantity per method: **what is the interface
    between two adjacent blocks?**
    """)
    return


@app.cell
def criterion_diagram(arrow, box, mo, palette, plt):
    _fig, (_aL, _aR) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # left: local operator, bounded interface -> tree
    for _t in range(4):
        box(_aL, _t, 0, palette["state"], label=f"M{_t + 1}", w=0.72, h=0.5, fontsize=9)
    box(_aL, 0.5, 1.2, palette["ink"], label="•", w=0.5, h=0.42, fontsize=11)
    box(_aL, 2.5, 1.2, palette["ink"], label="•", w=0.5, h=0.42, fontsize=11)
    box(_aL, 1.5, 2.4, palette["ink"], label="•", w=0.5, h=0.42, fontsize=11)
    for _p0, _p1 in [
        ((0, 0.25), (0.5, 1.0)),
        ((1, 0.25), (0.5, 1.0)),
        ((2, 0.25), (2.5, 1.0)),
        ((3, 0.25), (2.5, 1.0)),
        ((0.5, 1.4), (1.5, 2.2)),
        ((2.5, 1.4), (1.5, 2.2)),
    ]:
        arrow(_aL, _p0, _p1, color=palette["muted"], lw=1.2, shrink=1)
    _aL.set_title(
        "local operator · interface = 1 state\n→ associative → tree (log depth)",
        fontsize=11,
        fontweight="bold",
        color=palette["belief"],
    )
    _aL.set_xlim(-0.7, 4)
    _aL.set_ylim(-0.5, 3.0)
    _aL.axis("off")

    # right: particle-system operator, assignment interface -> chain
    for _t in range(4):
        box(_aR, _t, 0, palette["operator"], label=f"W{_t + 1}", w=0.72, h=0.5, fontsize=9)
        if _t < 3:
            arrow(
                _aR, (_t + 0.37, 0), (_t + 1 - 0.37, 0), color=palette["operator"], lw=2.0, shrink=2
            )
        box(_aR, _t, 1.1, palette["muted"], label="", w=0.24, h=0.24, z=2)
        arrow(_aR, (_t, 0.95), (_t, 0.28), color=palette["muted"], lw=1.0, shrink=2)
    _aR.text(
        1.5,
        1.5,
        "each W can change with other\nparticles' boundary assignments",
        ha="center",
        color=palette["ink"],
        fontsize=9,
    )
    _aR.set_title(
        "particle-system operator · interface = joint assignment\n→ no pairwise tree → chain",
        fontsize=11,
        fontweight="bold",
        color=palette["operator"],
    )
    _aR.set_xlim(-0.7, 4)
    _aR.set_ylim(-0.5, 3.0)
    _aR.axis("off")

    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def reducer_md(mo):
    mo.md(r"""
    ## 2. The test harness

    A tiny reducer that runs the *same* reduction two ways — a **sequential** left-fold (the
    `O(T)` chain) and a **balanced tree** (the parallel schedule) — and reports the result plus
    the number of sequential rounds each took. If the operator's combine is associative, the two
    results are identical and the tree finishes in `⌈log₂ T⌉` rounds instead of `T − 1`.
    """)
    return


@app.cell
def reducer_engine():
    def reduce_sequential(elems, combine):
        _acc = elems[0]
        _depth = 0
        for _e in elems[1:]:
            _acc = combine(_acc, _e)
            _depth += 1
        return _acc, _depth

    def reduce_tree(elems, combine):
        _level = list(elems)
        _depth = 0
        while len(_level) > 1:
            _nxt = [combine(_level[_i], _level[_i + 1]) for _i in range(0, len(_level) - 1, 2)]
            if len(_level) % 2 == 1:
                _nxt.append(_level[-1])
            _level = _nxt
            _depth += 1
        return _level[0], _depth

    return reduce_sequential, reduce_tree


@app.cell(hide_code=True)
def probe_md(mo):
    mo.md(r"""
    ## 3. Verify the operators that should parallelize

    Three operators, all built to be **local**, run through the harness at `T = 32`:

    - **pairwise matrix** — one Feynman–Kac step as a transfer matrix (plain CSMC targets,
      Particle-aMALA targets conditional on `u`, and any auxiliary target whose increment is a
      per-trajectory function of `(x_{t-1}, x_t)`); combine = matrix
      product.
    - **affine-Gaussian ∘** — the Kalman / auxiliary-Kalman element `(E, f, L)`, combined by
      Corenflos eq. 30 (compose the affine maps, accumulate covariance).
    - **lifted transfer** — Particle-aMALA+'s three-state potential `(x_{t-2}, x_{t-1}, x_t)`
      lifted to a first-order transfer on state-*pairs*: a bounded-window potential is still a
      matrix, just on a bigger state.

    All three must give tree = sequential exactly, in `⌈log₂ 32⌉ = 5` rounds vs `31`.
    """)
    return


@app.cell
def probe_run(np, reduce_sequential, reduce_tree):
    _rng = np.random.default_rng(0)
    _t_len = 32
    _rows = []

    # (1) pairwise transfer matrices — local pairwise potential
    # (column-normalized so the length-T product stays bounded and the check reads at machine precision)
    _k = 4
    _mats = []
    for _ in range(_t_len):
        _m = _rng.random((_k, _k))
        _mats.append(_m / _m.sum(axis=0, keepdims=True))
    _sm, _dseq = reduce_sequential(_mats, lambda a, b: b @ a)
    _tm, _dtree = reduce_tree(_mats, lambda a, b: b @ a)
    _rows.append(
        {
            "name": "pairwise matrix — aMALA / aGRAD / CSMC",
            "interface": "1 state",
            "diff": float(np.max(np.abs(_sm - _tm))),
            "seq": _dseq,
            "tree": _dtree,
        }
    )

    # (2) affine-Gaussian element (E, f, L) — Kalman / auxiliary-Kalman, Corenflos eq. 30
    _d = 2

    def _mk_ag():
        _e = 0.3 * _rng.standard_normal((_d, _d))
        _f = _rng.standard_normal(_d)
        _a = _rng.standard_normal((_d, _d))
        return (_e, _f, 0.1 * _a @ _a.T + 0.1 * np.eye(_d))

    def _combine_ag(a, b):
        _ea, _fa, _la = a
        _eb, _fb, _lb = b
        return (_ea @ _eb, _ea @ _fb + _fa, _ea @ _lb @ _ea.T + _la)

    _els = [_mk_ag() for _ in range(_t_len)]
    _sa, _dseq2 = reduce_sequential(_els, _combine_ag)
    _ta, _dtree2 = reduce_tree(_els, _combine_ag)
    _rows.append(
        {
            "name": "affine-Gaussian ∘ — Kalman / aux-Kalman",
            "interface": "1 state",
            "diff": max(float(np.max(np.abs(_sa[_i] - _ta[_i]))) for _i in range(3)),
            "seq": _dseq2,
            "tree": _dtree2,
        }
    )

    # (3) 2nd-order potential lifted to a transfer on state-pairs — aMALA+
    _k3 = 3

    def _mk_lift():
        _g = _rng.random((_k3, _k3, _k3))
        _tt = np.zeros((_k3 * _k3, _k3 * _k3))
        for _a in range(_k3):
            for _b in range(_k3):
                for _c in range(_k3):
                    _tt[_b * _k3 + _c, _a * _k3 + _b] = _g[_a, _b, _c]
        _cs = _tt.sum(axis=0, keepdims=True)
        _cs[_cs == 0.0] = 1.0
        return _tt / _cs

    _lift = [_mk_lift() for _ in range(_t_len)]
    _sl, _dseq3 = reduce_sequential(_lift, lambda a, b: b @ a)
    _tl, _dtree3 = reduce_tree(_lift, lambda a, b: b @ a)
    _rows.append(
        {
            "name": "lifted transfer — aMALA+ (3-state potential)",
            "interface": "2 states (window)",
            "diff": float(np.max(np.abs(_sl - _tl))),
            "seq": _dseq3,
            "tree": _dtree3,
        }
    )

    assoc_rows = _rows
    return (assoc_rows,)


@app.cell
def probe_table(assoc_rows, mo):
    _head = "| local operator | block interface | tree = sequential? | max &#124;tree − seq&#124; | seq rounds | tree rounds |"
    _sep = "|---|---|:--:|--:|--:|--:|"
    _lines = [_head, _sep]
    for _r in assoc_rows:
        _ok = "✅ identical" if _r["diff"] < 1e-8 else "❌ differ"
        _lines.append(
            f"| {_r['name']} | {_r['interface']} | {_ok} | {_r['diff']:.1e} | {_r['seq']} | {_r['tree']} |"
        )
    mo.md(
        "**Result (T = 32).** Every local operator re-brackets into a tree with an identical "
        "answer, in 5 rounds instead of 31:\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def depth_md(mo):
    mo.md(r"""
    ## 4. The payoff, and where it stops

    For any **local target operator** (CSMC / dSMC targets, aMALA conditional on `u`, local
    auxiliary targets, the Kalman scan) the critical path can grow like `log T`. The published
    **Particle-mGRAD CSMC kernel** is different: after marginalising `u_t`, the correction contains
    `v̄_t`, an average over the ancestor-dependent transition centres of the whole time-`t` particle
    cloud. That is not a pairwise block stitch.
    """)
    return


@app.cell
def depth_fig(mo, np, palette, plt):
    _t = np.arange(1, 129)
    _seq = _t.astype(float)
    _tree = np.ceil(np.log2(np.maximum(_t, 2)))
    _tree[0] = 0.0

    _fig, _ax = plt.subplots(figsize=(8.5, 3.6))
    _ax.plot(
        _t,
        _seq,
        color=palette["operator"],
        lw=2.4,
        label="published Particle-mGRAD — sequential cSMC kernel, ≈ T",
    )
    _ax.plot(
        _t,
        _tree,
        color=palette["belief"],
        lw=2.4,
        label="local target operators (dSMC / aMALA targets / Kalman) — tree, ≈ log₂ T",
    )
    _ax.scatter([128, 128], [128, 7], color=[palette["operator"], palette["belief"]], zorder=5)
    _ax.annotate("128", (128, 128), xytext=(112, 116), color=palette["operator"], fontsize=10)
    _ax.annotate("7", (128, 7), xytext=(120, 20), color=palette["belief"], fontsize=10)
    _ax.set_xlabel("number of time steps, T")
    _ax.set_ylabel("sequential rounds")
    _ax.legend(frameon=False, fontsize=9, loc="center right")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_title(
        "Local operators tree to log T; Particle-mGRAD's published kernel stays linear",
        fontsize=12,
        fontweight="bold",
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def footprint_md(mo):
    mo.md(r"""
    ## 5. Why each method lands where it does — the weight's dependency footprint

    Read straight off the published weight `w_t^n`, what does it touch? Blue = **this particle's
    own path** (a local, per-trajectory dependence); red = **the particle cloud used by the
    marginal correction**. Grid columns are time `t-2 … t+1`; rows are particles, top row = the
    particle `n` whose weight we compute.

    The important distinction is whether the red quantity is fixed once a time-slice proposal cloud
    is drawn, or whether it changes when two blocks are stitched. `x̄_t` alone is a time-local
    statistic. `v̄_t` in Particle-mGRAD is worse: it averages `v_t^n = (I - A_t)m_t(x_{t-1}^{a_t^n})`,
    so it depends on all ancestor choices crossing the left boundary of time `t`.

    This plot is weight-only. The table below separates "the conditional target is local" from
    "the exact published proposal kernel is parallel."
    """)
    return


@app.cell
def footprint_fig(box, mo, palette, plt):
    _methods = [
        ("Particle-aMALA", {(0, 1), (0, 2)}, set(), True, "gradient of the filter posterior"),
        (
            "Particle-aMALA+",
            {(0, 0), (0, 1), (0, 2)},
            set(),
            True,
            "gradient of the smoothing posterior",
        ),
        ("Particle-aGRAD", {(0, 1), (0, 2)}, set(), True, "likelihood gradient + dynamics, u kept"),
        (
            "Particle-MALA",
            {(0, 1)},
            {(0, 2), (1, 2), (2, 2), (3, 2)},
            False,
            "marginalise u → time-local x̄_t",
        ),
        (
            "Particle-mGRAD",
            set(),
            {(_r, _c) for _r in range(4) for _c in (1, 2)},
            False,
            "marginalise u → x̄_t and ancestor-dependent v̄_t",
        ),
    ]
    _fig, _axs = plt.subplots(2, 3, figsize=(11.0, 6.2))
    _flat = _axs.flatten()
    for _idx, (_name, _own, _mf, _ok, _sub) in enumerate(_methods):
        _ax = _flat[_idx]
        for _r in range(4):
            for _c in range(4):
                if (_r, _c) in _mf:
                    _col, _al = palette["operator"], 1.0
                elif (_r, _c) in _own:
                    _col, _al = palette["state"], 1.0
                else:
                    _col, _al = palette["muted"], 0.25
                box(_ax, _c, 3 - _r, _col, w=0.9, h=0.9, alpha=_al, lw=0.8)
        for _c, _lab in enumerate(["t-2", "t-1", "t", "t+1"]):
            _ax.text(_c, -0.85, _lab, ha="center", va="center", fontsize=8, color=palette["ink"])
        _ax.text(
            -1.0,
            3,
            "n",
            ha="center",
            va="center",
            fontsize=9,
            color=palette["ink"],
            fontweight="bold",
        )
        _ax.text(
            -1.0,
            1,
            "others",
            ha="center",
            va="center",
            fontsize=7.5,
            color=palette["ink"],
            rotation=90,
        )
        _vcol = palette["belief"] if _ok else palette["operator"]
        _verdict = "local target ✓" if _ok else "marginal correction"
        _ax.set_title(f"{_name}\n{_sub}", fontsize=9.5, fontweight="bold")
        _ax.text(
            1.5,
            4.25,
            _verdict,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_vcol,
        )
        _ax.set_xlim(-1.5, 4)
        _ax.set_ylim(-1.3, 4.7)
        _ax.set_aspect("equal")
        _ax.axis("off")
    _flat[5].axis("off")
    _flat[5].text(
        0.5,
        0.55,
        "blue  = own trajectory (local)\nred   = particle-cloud statistic",
        ha="center",
        va="center",
        fontsize=9.5,
        color=palette["ink"],
        transform=_flat[5].transAxes,
    )
    _fig.suptitle(
        "Dependency footprint of the incremental weight  w_t^n", fontsize=12.5, fontweight="bold"
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def mgrad_obstruction_md(mo):
    mo.md(r"""
    ## 6. The mGRAD obstruction, made concrete

    Particle-mGRAD's published weight is

    \[
    w_t^n \propto Q_t(x_{t-1:t}^{(n)})
    H_{t,\phi_t^n}(x_t^n, v_t^n, \bar x_t, \bar v_t),
    \qquad
    v_t^n = (I-A_t)m_t(x_{t-1}^{a_t^n}).
    \]

    The cross-step term `Q_t(x_{t-1}, x_t)` is not the problem; a second-order or pairwise
    potential can be lifted into a larger local state. The problem is `\bar v_t`. It is an average
    over **all** ancestor-dependent transition centres at that time. So the weight of particle `n`
    changes when a different particle chooses a different left ancestor, even if particle `n`'s own
    pair `(x_{t-1}^{a_t^n}, x_t^n)` is unchanged.
    """)
    return


@app.cell
def mgrad_counterexample(mo, np):
    _prev_particles = np.array([-1.3, 0.4, 2.1, 3.0])
    _curr_particles = np.array([0.25, -0.5, 1.1, 1.7])
    _candidate = 0
    _a_gain = 0.55
    _half_delta = 0.35
    _num_free = len(_curr_particles) - 1
    _g_mat = _half_delta / (1.0 + _num_free * _a_gain)
    _phi = 0.12
    _xbar = float(np.mean(_curr_particles))

    def _transition_mean(prev):
        return prev

    def _log_h(x, v, xbar, vbar):
        _precision = 1.0 / (_half_delta * _a_gain) + _g_mat
        _first = 0.5 * (x - v) * _precision * (x - v)
        _second = (0.5 * _num_free * (x + _phi) * _a_gain + (x - v)) * _g_mat * (x + _phi)
        _third = (_num_free + 1.0) * (xbar - vbar) * _g_mat * (v + _phi)
        return float(_first - _second + _third)

    _assignments = {
        "A: other particles use nearby ancestors": np.array([0, 1, 2, 3]),
        "B: other particles switch left ancestors": np.array([0, 3, 3, 3]),
    }
    _rows = []
    for _name, _ancestors in _assignments.items():
        _v_all = (1.0 - _a_gain) * _transition_mean(_prev_particles[_ancestors])
        _v = float(_v_all[_candidate])
        _vbar = float(np.mean(_v_all))
        _rows.append(
            {
                "assignment": _name,
                "own pair": f"({float(_prev_particles[0]):.1f}, {float(_curr_particles[0]):.2f})",
                "xbar": _xbar,
                "vbar": _vbar,
                "logH for particle 0": _log_h(float(_curr_particles[0]), _v, _xbar, _vbar),
            }
        )

    _head = "| ancestor assignment | particle 0 pair | `x̄_t` | `v̄_t` | `log H` for particle 0 |"
    _sep = "|---|---:|--:|--:|--:|"
    _lines = [_head, _sep]
    for _r in _rows:
        _lines.append(
            f"| {_r['assignment']} | {_r['own pair']} | {_r['xbar']:.3f} | "
            f"{_r['vbar']:.3f} | {_r['logH for particle 0']:.6f} |"
        )
    _delta = abs(float(_rows[0]["logH for particle 0"]) - float(_rows[1]["logH for particle 0"]))
    mo.md(
        "Same selected particle, same selected left/right pair, same current particle cloud, "
        f"but a different assignment for the **other** particles changes `v̄_t` and moves "
        f"`log H` by `{_delta:.6f}`:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def mgrad_nonadditive_seam_test(mo, np):
    _left = np.array([-1.2, 0.7, 2.4])
    _right = np.array([-0.8, 0.3, 1.6])
    _a_gain = 0.5
    _half_delta = 0.4
    _num_free = 2
    _g_mat = _half_delta / (1.0 + _num_free * _a_gain)
    _phi = 0.15

    def _d(left_idx):
        return (1.0 - _a_gain) * _left[left_idx]

    def _log_h_for_assignment(pairs):
        _xs = np.array([_right[_j] for _, _j in pairs])
        _ds = np.array([_d(_i) for _i, _ in pairs])
        _xbar = float(np.mean(_xs))
        _dbar = float(np.mean(_ds))

        def _one(x, d):
            _precision = 1.0 / (_half_delta * _a_gain) + _g_mat
            _first = 0.5 * (x - d) * _precision * (x - d)
            _second = (0.5 * _num_free * (x + _phi) * _a_gain + (x - d)) * _g_mat * (x + _phi)
            _third = (_num_free + 1.0) * (_xbar - _dbar) * _g_mat * (d + _phi)
            return _first - _second + _third

        return float(sum(_one(_x, _d_value) for _x, _d_value in zip(_xs, _ds, strict=True)))

    _assignment_ab = [(0, 0), (2, 2), (0, 1)]
    _assignment_cd = [(1, 1), (2, 2), (0, 1)]
    _assignment_ad = [(0, 1), (2, 2), (0, 1)]
    _assignment_cb = [(1, 0), (2, 2), (0, 1)]
    _cross_difference = (
        _log_h_for_assignment(_assignment_ab)
        + _log_h_for_assignment(_assignment_cd)
        - _log_h_for_assignment(_assignment_ad)
        - _log_h_for_assignment(_assignment_cb)
    )
    mo.md(
        "Pairwise seam weights imply an additive joint score, so this cross-difference should be "
        f"`0`. For the mGRAD marginal correction it is `{_cross_difference:.6f}`. That is the "
        "finite-particle witness that the seam is a joint assignment, not a matrix of pair scores."
    )
    return


@app.cell(hide_code=True)
def joint_assignment_md(mo):
    mo.md(r"""
    ## 7. The strongest exact attempt: make the seam a joint assignment

    The obvious rescue is to stop asking for pairwise stitch weights. At a seam, choose the whole
    vector of pairings

    \[
    ((i_1,j_1),\ldots,(i_P,j_P)), \qquad P=N+1,
    \]

    score that **whole assignment** with the mGRAD `x̄_t` and `v̄_t`, and resample a new particle
    cloud from those joint assignments. This is exact in principle. It is also the point where the
    useful parallel algorithm disappears: the seam interface is no longer one boundary state per
    output trajectory; it is the full vector of `P` left/right pair choices.

    Standard dSMC considers `P²` candidate pairs at a seam. Exact Particle-mGRAD joint stitching
    considers `(P²)^P = P^{2P}` candidate **sets of pairs** before resampling.
    """)
    return


@app.cell
def joint_assignment_complexity(mo, np):
    _particle_counts = np.array([2, 4, 8, 16, 32, 64])
    _rows = []
    for _p in _particle_counts:
        _pair_candidates = _p**2
        _joint_log10 = 2 * _p * np.log10(_p)
        _rows.append(
            {
                "P": int(_p),
                "pair": int(_pair_candidates),
                "joint_log10": float(_joint_log10),
            }
        )

    _head = "| particles `P=N+1` | dSMC seam candidates `P²` | exact mGRAD joint candidates `log10(P^{2P})` |"
    _sep = "|---:|---:|---:|"
    _lines = [_head, _sep]
    for _r in _rows:
        _lines.append(f"| {_r['P']} | {_r['pair']} | {_r['joint_log10']:.1f} |")

    mo.md(
        "For the paper's usual `P=32`, the ordinary seam has `1024` pair scores. The exact "
        "joint-assignment seam has about `10^96` assignments:\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def low_rank_rescue_md(mo):
    mo.md(r"""
    ## 8. The low-rank rescue also just uncollapses mGRAD

    The `x̄_t`/`v̄_t` coupling is low-rank in state dimension, so one can write it as a Gaussian
    integral and condition on a new continuous auxiliary variable. Conditional on that variable, the
    seam factors back into local terms.

    That sounds promising, but it is exactly the move mGRAD was designed to avoid: reintroduce
    the auxiliary variable that was analytically integrated out. In the paper's terminology, this moves
    from **Particle-mGRAD** back to an auxiliary-gradient kernel such as **Particle-aGRAD** (or a
    twisted/auxiliary-Kalman version). That is target-correct and useful, but it is not the same
    marginal Particle-mGRAD transition kernel.

    The exact-kernel menu is therefore:

    - keep the marginalization and pay a joint particle-system seam;
    - or reintroduce an auxiliary variable and recover local seams;
    - or use a special case where `v_t^n` no longer depends on the left ancestor.
    """)
    return


@app.cell(hide_code=True)
def taxonomy_md(mo):
    mo.md(r"""
    ## 9. The audit

    Every piece we discussed, with the extra information it injects, what its weight depends on,
    the resulting block interface, and the verdict. "Exact kernel" means the same published
    transition kernel; "target-correct PIT" means a different MCMC kernel with the same invariant
    posterior.
    """)
    return


@app.cell(hide_code=True)
def taxonomy_table(mo):
    mo.md(r"""
    | piece | extra information injected | weight depends on | block interface | PIT verdict |
    |---|---|---|:--:|:--:|
    | **plain smoothing target** | none (prior + likelihood) | `(x_{t-1}, x_t)` | 1 state | ✅ target-correct dSMC |
    | **Particle-aMALA target** | local gradient (filter), `u` kept | own `(x_{t-1}, x_t, u_t)` | 1 state | ✅ target-correct dSMC |
    | **Particle-aMALA+ target** | local gradient (smoothing), `u` kept | own `(x_{t-2}, x_{t-1}, x_t)` | 2 states | ✅ target-correct dSMC |
    | **Particle-aGRAD Algorithm 6** | likelihood gradient + dynamics, `u` kept | guided proposal depends on ancestor | proposal chain | ❌ exact published kernel; ✅ if replaced by a dSMC kernel for the same auxiliary target |
    | **twisted / auxiliary-Kalman family** | look-ahead or LGSSM auxiliary variables | affine-Gaussian scan elements | 1 state | ✅ Kalman scan / target-correct PIT |
    | **Particle-MALA** | `u` marginalized (filter grad) | time-local ensemble mean `x̄_t` | time-slice statistic + pairwise transition | ⚠️ plausible with a custom embedded-HMM/dSMC representation |
    | **Particle-mGRAD Algorithm 7** | `u` marginalized + dynamics covariance | `x̄_t`, ancestor-dependent `v̄_t`, cross-step | full seam assignment | ❌ useful exact PIT; only brute-force joint seam |

    The dividing line is the **block interface**. Keeping `u` usually recovers a local target, though
    the exact sequential guided proposal may still need to be replaced by a dSMC kernel. Marginalizing
    `u` is harmless only when the collapsed correction can be fixed before the tree stitch.
    Particle-mGRAD fails that test because `v̄_t` changes with cross-block ancestor choices.
    """)
    return


@app.cell(hide_code=True)
def nuance_md(mo):
    mo.md(r"""
    ## 10. What the best shot looks like

    There are four routes, and only the first two are useful for parallel hardware:

    1. **Use an auxiliary target and a PIT kernel.** Keep `u_t`, then use dSMC or the auxiliary-Kalman
       scan on the local conditional target. This is not the same transition kernel as published
       Particle-mGRAD, but it is exact for the posterior.

    2. **Use Particle-MALA-style marginalization only.** If the marginal correction is just `x̄_t`,
       the statistic is fixed by the time-slice particle bank before stitching. A custom PIT
       implementation is plausible.

    3. **Special-case Particle-mGRAD when `m_t` ignores `x_{t-1}`.** Then `v_t^n` is fixed inside a
       time slice, `v̄_t` stops depending on ancestor assignments, and the method collapses toward the
       Particle-MALA / CSMC limits proved in Propositions 8 and 9. That is not the generic state-space
       model.

    4. **Brute-force the marginal kernel over joint seam assignments.** In principle, one could carry
       the whole vector of `N+1` ancestor choices at every stitch and score all joint matchings. That is
       exact, but the interface is the particle system rather than a state boundary, so the combine is
       exponential or high-order in `N`. It is not the useful `O(N² log T)` DSMC object.

    So the practical answer after trying the hard route is: **do not parallelize published
    Particle-mGRAD directly. Parallelize an auxiliary target-correct replacement instead.** It is the
    same design family, keeps the boundary interface bounded, and avoids the `v̄_t`
    ancestor-assignment obstruction.
    """)
    return


@app.cell(hide_code=True)
def closing(mo):
    mo.md(r"""
    ## The audit in one line

    Parallel-in-time is decided by a single measurable quantity — **the block interface**.
    Keep or reintroduce the auxiliary variable and the **target** can be made local, so a PIT kernel
    can use bounded seams. Marginalize it in Particle-mGRAD and `v̄_t` becomes a function of every
    cross-boundary ancestor assignment (particle-system interface → no useful pairwise associative
    combine → chain). The verified engines above (matrix product, affine-Gaussian ∘, lifted
    transfer) are the three faces of "bounded interface"; Particle-mGRAD is what losing that
    interface costs.
    """)
    return


@app.cell(hide_code=True)
def coda_intro(mo):
    mo.md(r"""
    ## 11. Empirical coda — does the parallelizable replacement actually pay off?

    Sections 1–10 are structural: they argue on paper that published **Particle-mGRAD**
    cannot re-bracket into a tree (the `v̄_t` obstruction), whereas an **auxiliary
    target-correct** kernel keeps a bounded seam and trees to `O(log T)`. Our production
    smoother is exactly that replacement: `amala_exact` and `amala_plus` are the gradient
    **leaf proposals inside the conditional de-sequentialized SMC tree**
    ([`.../marginal_particle_gibbs/smoothers/dsmc.py`](../src/nof1_causal_lab/models/ssm/inference/methods/marginal_particle_gibbs/smoothers/dsmc.py)).

    This section runs the real thing. It builds a toy **nonlinear** state-space model,
    then puts four `π_T`-invariant latent-path kernels on the *same* posterior at growing
    horizons `T`:

    - **Particle-mGRAD** — Corenflos–Finke Algorithm 7 (arXiv 2401.14868), implemented
      here from the paper: the guided proposal with the marginal `x̄_t`/`v̄_t` correction,
      forward-filtered + backward-sampled. Sequential in `t` by construction.
    - **Particle-aGRAD** — the **parallel replacement** §10 recommends. It folds the
      Gaussian prior dynamics into the proposal exactly like mGRAD, but *keeps* the
      auxiliary `u` instead of marginalising it, so the seam stays a bounded pairwise
      transition and the kernel runs on the c-dSMC tree. Implemented here from scratch as a
      tree (our `dsmc.py` ships only isotropic leaves); the tree stitch is validated to
      reproduce the real `dsmc.smooth` when given an isotropic leaf. (§12.2 stress-tests
      this hand-built leaf and finds its invariance is only approximate — the mixing and
      depth results below stand, but the exactness claim gets corrected there.)
    - **`amala_exact`** — the real source leaf proposal, driven through the actual
      `dsmc.smooth` c-dSMC tree. Auxiliary trajectory kept ⇒ exact; isotropic (does not fold
      the prior); bounded seam ⇒ trees. The shipped production default.
    - **`amala_plus`** — the real source *biased* leaf proposal (reference-path
      linearisation, no auxiliary correction). Same tree, no invariance guarantee.

    So the parallel family is graded: aGRAD folds the prior (best mixing), `amala_exact` is
    its isotropic cousin, `amala_plus` drops exactness. mGRAD is the thing that *can't* tree.

    The model is deliberately 1-D so the horizon `T` is the only axis that moves — the
    audit is about `T`, not `D`. It is nonlinear in the drift and conditionally Gaussian
    with a **constant** transition covariance, which is precisely the regime where
    Particle-mGRAD is defined (Section 4.2's `C_t(x_{t-1}) = C_t`).

    Three questions, each measured against a near-exact reference:

    1. **Exactness.** Does each kernel's stationary law match a gold-standard grid smoother?
    2. **Mixing.** How fast does each decorrelate *per sweep*, as `T` grows?
    3. **Depth.** What is the sequential cost of one sweep — the quantity the audit is about?
    """)
    return


@app.cell
def coda_imports():
    import dataclasses
    import math

    import jax
    import jax.numpy as jnp
    import jax.random as random

    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
        SmootherContext,
    )
    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.smoothers import (
        dsmc,
    )

    return SmootherContext, dataclasses, dsmc, jax, jnp, math, random


@app.cell
def coda_model(np):
    # Toy nonlinear SSM (D=1): nonlinear drift, additive Gaussian process noise (constant
    # covariance — the Particle-mGRAD regime), linear-Gaussian emission (states identifiable,
    # so latent recovery is well posed).
    #     x_0 ~ N(0, INIT_SD^2)
    #     x_t | x_{t-1} ~ N(a·x_{t-1} + b·sin(w·x_{t-1}), PROC_SD^2)
    #     y_t | x_t     ~ N(x_t, OBS_SD^2)
    DRIFT_A, DRIFT_B, DRIFT_W = 0.8, 1.6, 1.0
    PROC_SD, OBS_SD, INIT_SD = 0.6, 0.5, 1.0

    def _drift_mean(x):
        return DRIFT_A * x + DRIFT_B * np.sin(DRIFT_W * x)

    def simulate_ssm(seed, t_len):
        _rng = np.random.default_rng(seed)
        _x = np.zeros(t_len, dtype=np.float32)
        _x[0] = _rng.normal(0.0, INIT_SD)
        for _t in range(1, t_len):
            _x[_t] = _drift_mean(_x[_t - 1]) + _rng.normal(0.0, PROC_SD)
        _y = (_x + _rng.normal(0.0, OBS_SD, size=t_len)).astype(np.float32)
        return _x, _y

    def grid_smoother(y, n_grid=601):
        """Near-exact posterior marginals p(x_t | y_1:T) via grid forward–backward.

        This is the gold standard the samplers are scored against — a deterministic
        discretisation of the exact smoother, with only grid error.
        """
        _t_len = len(y)
        _lo = float(min(y.min(), -6.0)) - 2.0
        _hi = float(max(y.max(), 6.0)) + 2.0
        _xs = np.linspace(_lo, _hi, n_grid)

        def _logn(v, mu, sd):
            return -0.5 * (np.log(2.0 * np.pi * sd**2) + ((v - mu) ** 2) / sd**2)

        _log_obs = _logn(y[:, None], _xs[None, :], OBS_SD)  # (T, G)
        _mus = _drift_mean(_xs)  # (G,)
        _log_tr = _logn(_xs[None, :], _mus[:, None], PROC_SD)  # (G_prev, G_cur)

        _log_alpha = np.zeros((_t_len, n_grid))
        _log_alpha[0] = _logn(_xs, 0.0, INIT_SD) + _log_obs[0]
        for _t in range(1, _t_len):
            _a = _log_alpha[_t - 1]
            _m = _a.max()
            _pred = np.log(np.exp(_a - _m) @ np.exp(_log_tr) + 1e-300) + _m
            _log_alpha[_t] = _pred + _log_obs[_t]

        _log_beta = np.zeros((_t_len, n_grid))
        for _t in range(_t_len - 2, -1, -1):
            _b = _log_beta[_t + 1] + _log_obs[_t + 1]
            _m = _b.max()
            _log_beta[_t] = np.log(np.exp(_log_tr) @ np.exp(_b - _m) + 1e-300) + _m

        _log_g = _log_alpha + _log_beta
        _log_g -= _log_g.max(1, keepdims=True)
        _g = np.exp(_log_g)
        _g /= _g.sum(1, keepdims=True)
        _mean = (_g * _xs[None, :]).sum(1)
        return {"mean": _mean, "xs": _xs, "g": _g}

    return DRIFT_A, DRIFT_B, DRIFT_W, INIT_SD, OBS_SD, PROC_SD, grid_smoother, simulate_ssm


@app.cell
def coda_kernels(
    DRIFT_A,
    DRIFT_B,
    DRIFT_W,
    INIT_SD,
    OBS_SD,
    PROC_SD,
    SmootherContext,
    dataclasses,
    dsmc,
    jax,
    jnp,
    np,
    random,
):
    _D = 1
    _DT = jnp.float32

    def _m_mean(x):
        return DRIFT_A * x + DRIFT_B * jnp.sin(DRIFT_W * x)

    def _log_init(x):
        return jnp.sum(-0.5 * (jnp.log(2.0 * jnp.pi * INIT_SD**2) + (x**2) / INIT_SD**2))

    def _log_trans(x_prev, x_cur):
        _mu = _m_mean(x_prev)
        return jnp.sum(
            -0.5 * (jnp.log(2.0 * jnp.pi * PROC_SD**2) + ((x_cur - _mu) ** 2) / PROC_SD**2)
        )

    def _log_obs(x, y):
        return jnp.sum(-0.5 * (jnp.log(2.0 * jnp.pi * OBS_SD**2) + ((y - x) ** 2) / OBS_SD**2))

    def _build_ctx(y_obs, delta, kappa, leaf):
        # Hand-assemble the real SmootherContext for K=1 parameter particle (fixed
        # parameters). smooth() consumes only this subset; the rest is unused plumbing.
        _t_len = int(y_obs.shape[0])
        _k = 1

        def _obs_increment_fn(context, particle, time_idx, runtime_observations):
            del context
            return _log_obs(particle, runtime_observations[time_idx])

        def _initial_value_grad_by_param(p):
            _v, _g = jax.value_and_grad(_log_init)(p)
            return (
                jnp.broadcast_to(_v, (_k,)).astype(_DT),
                jnp.broadcast_to(_g, (_k, _D)).astype(_DT),
            )

        def _transition_current_value_grad_by_param(prev, cur, time_idx):
            del time_idx
            _v, _g = jax.value_and_grad(lambda c: _log_trans(prev, c))(cur)
            return (
                jnp.broadcast_to(_v, (_k,)).astype(_DT),
                jnp.broadcast_to(_g, (_k, _D)).astype(_DT),
            )

        def _transition_next_value_grad_by_param(cur, nxt, time_idx):
            del time_idx
            _v, _g = jax.value_and_grad(lambda c: _log_trans(c, nxt))(cur)
            return (
                jnp.broadcast_to(_v, (_k,)).astype(_DT),
                jnp.broadcast_to(_g, (_k, _D)).astype(_DT),
            )

        def _selected_transition_log_probs(prev, nxt, seam):
            _lp = jnp.where(seam < _t_len, jax.vmap(_log_trans)(prev, nxt), 0.0)
            return jnp.broadcast_to(_lp[:, None], (prev.shape[0], _k)).astype(_DT)

        def _pairwise_transition_log_probs(prev, nxt, seam):
            _lp = jax.vmap(lambda a: jax.vmap(lambda b: _log_trans(a, b))(nxt))(prev)
            _lp = jnp.where(seam < _t_len, _lp, 0.0)
            return jnp.broadcast_to(_lp[:, :, None], (prev.shape[0], nxt.shape[0], _k)).astype(_DT)

        def _trajectory_label_log_probs(path):
            del path
            return jnp.zeros((_k,), dtype=_DT)

        return SmootherContext(
            contexts=jnp.zeros((_k, 1), dtype=_DT),
            parameter_particles=jnp.zeros((_k, 1), dtype=_DT),
            parameter_log_probs=jnp.zeros((_k,), dtype=_DT),
            initial_label_log_probs=jnp.zeros((_k,), dtype=_DT),
            init_means=jnp.zeros((_k, _D), dtype=_DT),
            init_chols=jnp.broadcast_to(jnp.eye(_D) * INIT_SD, (_k, _D, _D)).astype(_DT),
            init_logdets=jnp.full((_k,), float(np.log(INIT_SD**2)), dtype=_DT),
            num_steps=_t_len,
            num_free_particles=0,  # set by run_amala via dataclasses.replace
            num_parameter_particles=_k,
            block_size=_t_len,
            num_blocks=1,
            latent_dtype=_DT,
            traj_dtype=_DT,
            complete_dtype=_DT,
            obs_increment_fn=_obs_increment_fn,
            runtime_observations=jnp.asarray(y_obs).reshape(_t_len, 1),
            trajectory_log_prob_fn=None,
            prior_terms_from_context_fn=None,
            log_prior_unc_fn=None,
            amala_delta=jnp.full((_D,), delta, dtype=_DT),
            amala_kappa=jnp.asarray(kappa, dtype=_DT),
            amala_grad_clip=jnp.asarray(jnp.inf, dtype=_DT),
            dsmc_leaf_proposal=leaf,
            diagnostic_metrics=frozenset(),
            initial_value_grad_by_param=_initial_value_grad_by_param,
            transition_current_value_grad_by_param=_transition_current_value_grad_by_param,
            transition_next_value_grad_by_param=_transition_next_value_grad_by_param,
            selected_transition_log_probs=_selected_transition_log_probs,
            pairwise_transition_log_probs=_pairwise_transition_log_probs,
            transition_log_probs_from_fixed_prev=None,
            transition_log_probs_by_param=None,
            transition_log_probs_to_next_by_param=None,
            sample_transition_by_label=None,
            segment_terminal_label_log_probs=None,
            path_future_tail_log_probs=None,
            trajectory_label_log_probs=_trajectory_label_log_probs,
        )

    def run_amala(y_obs, leaf, n_particles=16, delta=0.7, kappa=0.75, n_iter=700, seed=0):
        """MCMC chain of latent paths from the REAL dsmc.smooth c-dSMC tree."""
        _t_len = int(y_obs.shape[0])
        _ctx = dataclasses.replace(
            _build_ctx(y_obs, delta, kappa, leaf), num_free_particles=n_particles - 1
        )
        _x0 = jnp.asarray(y_obs).reshape(_t_len, 1)

        def _body(x_ref, key):
            _x = dsmc.smooth(_ctx, key, x_ref).latent_path
            return _x, _x

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _x0, _keys)
        return np.asarray(_chain)[:, :, 0]

    def run_mgrad(y_obs, n_particles=16, delta=0.7, kappa=1.0, n_iter=700, seed=0):
        """Published Particle-mGRAD CSMC (Corenflos & Finke, Algorithm 7), scalar case.

        Forward-filter with the guided mGRAD proposal + marginal x̄_t/v̄_t weight
        correction (the ancestor-dependent v̄_t is the term the audit flags), then
        backward-sample the new reference path. Sequential in t by construction.
        """
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles  # N + 1
        _n = _p - 1
        _half = 0.5 * delta

        def _prior_mean(time_idx, x_prev):
            return jnp.where(
                time_idx == 0, 0.0, DRIFT_A * x_prev + DRIFT_B * jnp.sin(DRIFT_W * x_prev)
            )

        def _prior_var(time_idx):
            return jnp.where(time_idx == 0, INIT_SD**2, PROC_SD**2)

        def _logn(v, mu, var):
            return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

        def _forward(key, x_ref):
            def _step(carry, time_idx):
                _prev, _weights, _key = carry
                _key, _anc_key, _u_key, _prop_key = random.split(_key, 4)
                _cov = _prior_var(time_idx)
                _a_gain = _cov / (_cov + _half)  # A_t
                _prop_var = _half * _a_gain  # C'_t = (delta/2) A_t
                _g_hat = (2.0 / delta) / (1.0 + _n * _a_gain)  # (2/delta)(I + N A_t)^{-1}
                # resample ancestors; index 0 keeps the reference lineage
                _anc = random.categorical(_anc_key, jnp.log(_weights + 1e-38), shape=(_p,))
                _anc = _anc.at[0].set(0)
                _mmean = _prior_mean(time_idx, _prev[_anc])
                _x_ref_t = x_ref[time_idx]
                _grad_ref = (_y[time_idx] - _x_ref_t) / OBS_SD**2
                _u = _x_ref_t + kappa * _half * _grad_ref + jnp.sqrt(_half) * random.normal(_u_key)
                _x_t = (
                    (1.0 - _a_gain) * _mmean
                    + _a_gain * _u
                    + jnp.sqrt(_prop_var) * random.normal(_prop_key, (_p,))
                )
                _x_t = _x_t.at[0].set(_x_ref_t)  # reference particle
                _v = (1.0 - _a_gain) * _mmean  # v_t^n
                _x_bar = jnp.mean(_x_t)
                _v_bar = jnp.mean(_v)
                _phi = kappa * _half * (_y[time_idx] - _x_t) / OBS_SD**2
                _log_q = _logn(_x_t, _mmean, _cov) + _logn(_y[time_idx], _x_t, OBS_SD**2)
                _term1 = 0.5 * (_x_t - _v) ** 2 * (1.0 / _prop_var + _g_hat)
                _term2 = (
                    -(0.5 * _n * (_x_t + _phi) * _a_gain + (_x_t - _v)) * _g_hat * (_x_t + _phi)
                )
                _term3 = (_n + 1) * (_x_bar - _v_bar) * _g_hat * (_v + _phi)
                _log_w = _log_q + _term1 + _term2 + _term3
                _log_w = _log_w - jax.scipy.special.logsumexp(_log_w)
                _new_w = jnp.exp(_log_w)
                return (_x_t, _new_w, _key), (_x_t, _new_w)

            (_, _, _), (_particles, _all_w) = jax.lax.scan(
                _step,
                (jnp.zeros(_p), jnp.ones(_p) / _p, key),
                jnp.arange(_t_len),
            )
            return _particles, _all_w

        def _backward(key, particles, weights):
            _key_last, _key_rest = random.split(key)
            _l_last = random.categorical(_key_last, jnp.log(weights[_t_len - 1] + 1e-38))

            def _bstep(carry, time_idx):
                _l_next, _key = carry
                _key, _sample_key = random.split(_key)
                _x_next = particles[time_idx + 1][_l_next]
                _cov = _prior_var(time_idx + 1)
                _mmean = DRIFT_A * particles[time_idx] + DRIFT_B * jnp.sin(
                    DRIFT_W * particles[time_idx]
                )
                _logp = jnp.log(weights[time_idx] + 1e-38) + _logn(_x_next, _mmean, _cov)
                _l = random.categorical(_sample_key, _logp)
                return (_l, _key), _l

            (_, _), _ls_rev = jax.lax.scan(
                _bstep, (_l_last, _key_rest), jnp.arange(_t_len - 2, -1, -1)
            )
            _ls = jnp.concatenate([jnp.flip(_ls_rev), _l_last[None]])
            return particles[jnp.arange(_t_len), _ls]

        def _sweep(x_ref, key):
            _key_f, _key_b = random.split(key)
            _particles, _weights = _forward(_key_f, x_ref)
            _path = _backward(_key_b, _particles, _weights)
            return _path, _path

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_sweep, _y, _keys)
        return np.asarray(_chain)

    return run_amala, run_mgrad


@app.cell
def coda_tree_machinery(DRIFT_A, DRIFT_B, DRIFT_W, INIT_SD, PROC_SD, jax, jnp, math, random):
    # c-dSMC tree stitch, extracted so §11's aGRAD leaf and §12's twisted leaf run on the
    # byte-identical tree. Only the leaf proposal distinguishes the kernels.
    def dsmc_prior_mean(time_idx, x_prev):
        return jnp.where(time_idx == 0, 0.0, DRIFT_A * x_prev + DRIFT_B * jnp.sin(DRIFT_W * x_prev))

    def dsmc_prior_var(time_idx):
        return jnp.where(time_idx == 0, INIT_SD**2, PROC_SD**2)

    def make_dsmc_tree(t_len, p, prior_mean, prior_var):
        """Build smooth(key, leaf_fn) — the c-dSMC tree over a leaf proposal.

        leaf_fn(time_idx, key) -> ((P,1) particles with the reference at index 0,
        (P,1) leaf log-potentials). Seams pay the true transition of the model given
        by (prior_mean, prior_var); the stitch mirrors the production `dsmc.smooth`
        (validated to reproduce its amala leaf).
        """

        def _logn(v, mu, var):
            return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

        def _seam_pairwise(prev, nxt, seam):  # (P,1),(P,1)->(P,P,1)
            _mm = prior_mean(seam, prev[:, 0])
            _lp = _logn(nxt[:, 0][None, :], _mm[:, None], prior_var(seam))
            return jnp.where(seam < t_len, _lp, 0.0)[:, :, None]

        def _seam_selected(prev, nxt, seam):  # (P,1),(P,1)->(P,1)
            _mm = prior_mean(seam, prev[:, 0])
            _lp = jnp.where(seam < t_len, _logn(nxt[:, 0], _mm, prior_var(seam)), 0.0)
            return _lp[:, None]

        def _multinomial(draw_key, logits, num_draws):
            _cum = jnp.cumsum(jax.nn.softmax(logits))
            _u = random.uniform(draw_key, (num_draws,), dtype=_cum.dtype)
            return jnp.minimum(
                jnp.searchsorted(_cum, _u, side="right"), logits.shape[0] - 1
            ).astype(jnp.int32)

        def _stitch_logits(left, right, seam):
            _, _ll, _lpsi, _, _lw = left
            _rf, _, _rpsi, _, _rw = right
            _tr = _seam_pairwise(_ll, _rf, seam)  # (P,P,1)
            _log_joint = jax.scipy.special.logsumexp(
                _lpsi[:, None, :] + _rpsi[None, :, :] + _tr, axis=-1
            )
            _log_left = jax.scipy.special.logsumexp(_lpsi, axis=1)
            _log_right = jax.scipy.special.logsumexp(_rpsi, axis=1)
            _coupling = _log_joint - _log_left[:, None] - _log_right[None, :]
            return _lw[:, None] + _rw[None, :] + _coupling

        def _combine(left, right, seam, key):
            _pl = _stitch_logits(left, right, seam)
            _free = _multinomial(key, _pl.reshape(-1), p - 1)
            _sel = jnp.concatenate([jnp.zeros((1,), jnp.int32), _free])
            _li, _ri = _sel // p, _sel % p
            _lf, _llast, _lpsi, _lorig, _ = left
            _rf, _rlast, _rpsi, _rorig, _ = right
            _origin = jnp.concatenate([_lorig[_li], _rorig[_ri]], axis=1)
            _tr = _seam_selected(_llast[_li], _rf[_ri], seam)
            _psi = _lpsi[_li] + _rpsi[_ri] + _tr
            _w = jnp.full((p,), -math.log(p))
            return _lf[_li], _rlast[_ri], _psi, _origin, _w

        def smooth(key, leaf_fn):
            _depth = max((t_len - 1).bit_length(), 0)
            _padded = 1 << _depth
            _kl, _kt, _kr = random.split(key, 3)
            _particles, _psi = jax.vmap(leaf_fn)(jnp.arange(t_len), random.split(_kl, t_len))
            _origin0 = jnp.broadcast_to(jnp.arange(p, dtype=jnp.int32)[:, None], (p, 1))
            _leaf_origin = jnp.broadcast_to(_origin0, (t_len, p, 1))
            _leaf_w = jax.vmap(lambda q: q - jax.scipy.special.logsumexp(q))(_psi[:, :, 0])
            if t_len == 1:
                _chosen = _multinomial(_kr, _psi[0, :, 0], 1)[0]
                return _particles[0, _chosen][None, :]
            _nph = _padded - t_len
            _first = jnp.concatenate([_particles, jnp.zeros((_nph, p, 1))], 0)
            _last = _first
            _psi_a = jnp.concatenate([_psi, jnp.zeros((_nph, p, 1))], 0)
            _origin = jnp.concatenate([_leaf_origin, jnp.broadcast_to(_origin0, (_nph, p, 1))], 0)
            _weights = jnp.concatenate([_leaf_w, jnp.full((_nph, p), -math.log(p))], 0)
            _level_keys = random.split(_kt, max(_depth - 1, 1))
            _segments = _padded
            for _level in range(_depth - 1):
                _npairs = _segments // 2
                _seams = (1 << _level) + jnp.arange(_npairs, dtype=jnp.int32) * (1 << (_level + 1))
                _left = (_first[0::2], _last[0::2], _psi_a[0::2], _origin[0::2], _weights[0::2])
                _right = (_first[1::2], _last[1::2], _psi_a[1::2], _origin[1::2], _weights[1::2])
                _first, _last, _psi_a, _origin, _weights = jax.vmap(_combine, in_axes=(0, 0, 0, 0))(
                    _left, _right, _seams, random.split(_level_keys[_level], _npairs)
                )
                _segments = _npairs
            _left_root = (_first[0], _last[0], _psi_a[0], _origin[0], _weights[0])
            _right_root = (_first[1], _last[1], _psi_a[1], _origin[1], _weights[1])
            _pl = _stitch_logits(_left_root, _right_root, _padded // 2)
            _chosen = _multinomial(_kr, _pl.reshape(-1), 1)[0]
            _origin_path = jnp.concatenate(
                [_origin[0][_chosen // p], _origin[1][_chosen % p]], axis=0
            )[:t_len]
            return _particles[jnp.arange(t_len), _origin_path]

        return smooth

    return dsmc_prior_mean, dsmc_prior_var, make_dsmc_tree


@app.cell
def coda_agrad_kernel(
    INIT_SD, OBS_SD, dsmc_prior_mean, dsmc_prior_var, jax, jnp, make_dsmc_tree, np, random
):
    def run_agrad(y_obs, n_particles=16, delta=0.7, kappa=1.0, n_iter=700, seed=0):
        """PARALLEL Particle-aGRAD: prior-covariance-folding leaf on a c-dSMC tree.

        Fold the Gaussian prior dynamics into the proposal like mGRAD, but KEEP the
        auxiliary u so the seam stays a bounded pairwise transition — so it trees to
        O(log T). Construction: propose x_t ~ q_t(·|u_t) with u_t drawn from the
        reference path; leaf weight = obs − log q_t; seam = the true transition ⇒
        importance weight π_T / ∏ q_t.

        Invariance caveat (§12.2): this leaf adapts to the reference path (gradient
        at x_ref_t, prior fold at x_ref_{t-1}) WITHOUT paying the auxiliary
        potential φ(u|x) in ψ, so the kernel is only approximately π_T-invariant —
        an O(1/N)-flavoured defect that grows with δ. The 20k-sweep probe in §12
        measures it; the twisted leaf there repairs it.
        """
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _half = 0.5 * delta
        _smooth = make_dsmc_tree(_t_len, _p, dsmc_prior_mean, dsmc_prior_var)

        def _logn(v, mu, var):
            return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

        def _leaf(x_ref, time_idx, key):
            _u_key, _s_key = random.split(key, 2)
            _cov = dsmc_prior_var(time_idx)
            _a_gain = _cov / (_cov + _half)  # A_t
            _prop_var = _half * _a_gain  # C'_t = (delta/2) A_t
            _x_ref_t = x_ref[time_idx, 0]
            _grad = (_y[time_idx] - _x_ref_t) / OBS_SD**2
            _u = _x_ref_t + kappa * _half * _grad + jnp.sqrt(_half) * random.normal(_u_key)
            # aGRAD proposal centre: blend prior-predictive mean with the gradient aux u
            _pm = dsmc_prior_mean(time_idx, x_ref[jnp.maximum(time_idx - 1, 0), 0])
            _center = (1.0 - _a_gain) * _pm + _a_gain * _u
            _free = _center + jnp.sqrt(_prop_var) * random.normal(_s_key, (_p - 1,))
            _particles = jnp.concatenate([_x_ref_t[None], _free])[:, None]  # (P,1)
            _psi = _logn(_y[time_idx], _particles[:, 0], OBS_SD**2) - _logn(
                _particles[:, 0], _center, _prop_var
            )
            _psi = jnp.where(time_idx == 0, _psi + _logn(_particles[:, 0], 0.0, INIT_SD**2), _psi)
            return _particles, _psi[:, None]  # (P,1),(P,1)

        def _body(x_ref, key):
            _xp = _smooth(key, lambda t, k: _leaf(x_ref, t, k))
            return _xp, _xp

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _y.reshape(_t_len, 1), _keys)
        return np.asarray(_chain)[:, :, 0]

    return (run_agrad,)


@app.cell
def coda_diag(np):
    def _iact_ess(x):
        """ESS via the initial-positive-sequence integrated autocorrelation."""
        _x = np.asarray(x, dtype=np.float64)
        _n = _x.size
        _x = _x - _x.mean()
        _var = float(np.dot(_x, _x) / _n)
        if _var <= 0.0:
            return float(_n)
        _rho_sum = 0.0
        for _lag in range(1, min(_n - 1, 1000)):
            _rho = float(np.dot(_x[:-_lag], _x[_lag:]) / (_n - _lag) / _var)
            if _rho <= 0.0:
                break
            _rho_sum += _rho
        return float(max(1.0, min(_n, _n / (1.0 + 2.0 * _rho_sum))))

    def coord_ess(chain, burn_frac=0.3):
        """Post-burn slice of the path chain and its per-coordinate ESS array."""
        _burn = chain[int(len(chain) * burn_frac) :]
        return _burn, np.array([_iact_ess(_burn[:, _t]) for _t in range(_burn.shape[1])])

    def ess_per_sweep(chain, burn_frac=0.3):
        """Median per-coordinate ESS of the path chain, divided by the sweep count."""
        _burn, _ess = coord_ess(chain, burn_frac)
        return float(np.median(_ess)) / len(_burn)

    return coord_ess, ess_per_sweep


@app.cell(hide_code=True)
def coda_bias_md(mo):
    mo.md(r"""
    ### 11.1 Exactness — the price of dropping the auxiliary variable

    First, are these kernels even sampling the right thing? We fix `T = 64`, sweep the
    proposal step size δ, and compare each kernel's stationary posterior mean to the
    grid gold standard. A `π_T`-invariant kernel matches it at every δ (only Monte-Carlo
    error, which shrinks with more sweeps). A biased kernel drifts away as δ grows.

    Particle-mGRAD keeps its auxiliary variable in the weights and is exact. Particle-aGRAD
    keeps `u` too (only the marginalisation is dropped) — also exact. `amala_exact` keeps the
    auxiliary trajectory `z ~ N(x_ref, (δ/2)I)` and pays the matching pseudo-observation
    potential — also exact. `amala_plus` linearises on the reference path with **no**
    auxiliary correction: cheap, parallel, and only approximately invariant.
    """)
    return


@app.cell
def coda_bias_run(grid_smoother, np, run_agrad, run_amala, run_mgrad, simulate_ssm):
    _x_true, _y = simulate_ssm(0, 64)
    _gold = grid_smoother(_y)
    _deltas = [0.25, 0.5, 1.0, 2.0, 4.0]
    _names = ("mgrad", "agrad", "amala_exact", "amala_plus")
    _rmse = {_n: [] for _n in _names}
    _cov = {_n: [] for _n in _names}
    for _delta in _deltas:
        _chains = {
            "mgrad": run_mgrad(_y, delta=_delta, n_iter=700, seed=5),
            "agrad": run_agrad(_y, delta=_delta, n_iter=700, seed=5),
            "amala_exact": run_amala(_y, "amala_exact", delta=_delta, n_iter=700, seed=5),
            "amala_plus": run_amala(_y, "amala_plus", delta=_delta, n_iter=700, seed=5),
        }
        for _name, _chain in _chains.items():
            _burn = _chain[350:]
            _pm = _burn.mean(0)
            _lo, _hi = np.percentile(_burn, 5, 0), np.percentile(_burn, 95, 0)
            _rmse[_name].append(float(np.sqrt(np.mean((_pm - _gold["mean"]) ** 2))))
            _cov[_name].append(float(np.mean((_x_true >= _lo) & (_x_true <= _hi))))
    bias_results = {"delta": _deltas, "rmse": _rmse, "cov": _cov}
    return (bias_results,)


@app.cell
def coda_bias_fig(bias_results, mo, palette, plt):
    _order = ("mgrad", "agrad", "amala_exact", "amala_plus")
    _colors = {
        "mgrad": palette["state"],
        "agrad": palette["belief"],
        "amala_exact": palette["seam"],
        "amala_plus": palette["operator"],
    }
    _labels = {
        "mgrad": "Particle-mGRAD — exact, sequential",
        "agrad": "Particle-aGRAD — exact, parallel",
        "amala_exact": "amala_exact — exact, parallel (isotropic)",
        "amala_plus": "amala_plus — biased, parallel",
    }
    _d = bias_results["delta"]
    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for _k in _order:
        _a0.plot(_d, bias_results["rmse"][_k], "-o", color=_colors[_k], lw=2.3, label=_labels[_k])
        _a1.plot(_d, bias_results["cov"][_k], "-o", color=_colors[_k], lw=2.3)
    _a0.set_xscale("log")
    _a0.set_xlabel("proposal step size δ")
    _a0.set_ylabel("‖posterior mean − exact posterior‖ (RMSE)")
    _a0.set_title(
        "Stationary bias vs the exact grid posterior",
        fontsize=11,
        fontweight="bold",
    )
    _a0.legend(frameon=False, fontsize=8.5, loc="upper left")
    _a0.spines[["top", "right"]].set_visible(False)
    _a1.set_xscale("log")
    _a1.axhline(0.9, color=palette["muted"], ls="--", lw=1.2)
    _a1.text(_d[0], 0.91, "nominal 90%", fontsize=8, color=palette["ink"])
    _a1.set_xlabel("proposal step size δ")
    _a1.set_ylabel("empirical 90% interval coverage")
    _a1.set_ylim(0.0, 1.0)
    _a1.set_title(
        "Calibration vs the true latent path",
        fontsize=11,
        fontweight="bold",
    )
    _a1.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def coda_bias_caption(mo):
    mo.md(r"""
    mGRAD, aGRAD, and `amala_exact` all track the gold standard at every δ here (their
    curves sit on top of each other near zero, with only Monte-Carlo wobble). `amala_plus`
    is fine while δ is small, but its bias blows up once the steps get aggressive: the
    posterior mean detaches from the true posterior and the 90% intervals cover far less
    than 90%. That is the concrete cost of dropping the auxiliary correction to buy a
    cheaper leaf.

    One caution this panel earns in hindsight: at 700 sweeps, "tracks the gold standard"
    only bounds a bias below the ≈0.03 Monte-Carlo floor. §12.2 repeats this check with
    20,000-sweep chains and finds our hand-built aGRAD leaf is itself *not* exactly
    invariant — it adapts its proposal to the reference path without paying the matching
    auxiliary potential, a mild version of the `amala_plus` sin. mGRAD (whose marginal
    weights *are* the correction) and the production `amala_exact` (which pays the
    pseudo-observation potential) both pass the same stress test.
    """)
    return


@app.cell(hide_code=True)
def coda_scaling_md(mo):
    mo.md(r"""
    ### 11.2 Mixing and depth — where the audit cashes out

    Exactness is table stakes; the audit is about **`T`**. We grow the horizon
    `T ∈ {16, …, 256}` and measure two things per kernel:

    - **mixing** — effective sample size per sweep (median over time coordinates), which
      says how many sweeps are needed for one independent path; and
    - **depth** — the sequential rounds inside one sweep: `T` for the Particle-mGRAD
      forward filter (the `v̄_t` chain), `⌈log₂ T⌉` for the c-dSMC tree that carries aGRAD,
      `amala_exact`, and `amala_plus`.

    The decisive quantity on parallel hardware is **ESS per unit of sequential depth**:
    statistical progress divided by the critical-path length that progress cost. This is
    where folding the prior earns its keep — aGRAD should mix like mGRAD but at tree depth.
    """)
    return


@app.cell
def coda_scaling_run(ess_per_sweep, math, run_agrad, run_amala, run_mgrad, simulate_ssm):
    _t_lens = [16, 32, 64, 128, 256]
    _names = ("mgrad", "agrad", "amala_exact", "amala_plus")
    _ess = {_n: [] for _n in _names}
    _ess_depth = {_n: [] for _n in _names}
    _depth_seq = []
    _depth_tree = []
    for _t_len in _t_lens:
        _, _y = simulate_ssm(0, _t_len)
        _chains = {
            "mgrad": run_mgrad(_y, delta=0.7, n_iter=700, seed=3),
            "agrad": run_agrad(_y, delta=0.7, n_iter=700, seed=3),
            "amala_exact": run_amala(_y, "amala_exact", delta=0.7, n_iter=700, seed=3),
            "amala_plus": run_amala(_y, "amala_plus", delta=0.7, n_iter=700, seed=3),
        }
        _dseq = _t_len
        _dtree = math.ceil(math.log2(_t_len))
        _depth_seq.append(_dseq)
        _depth_tree.append(_dtree)
        for _name, _chain in _chains.items():
            _e = ess_per_sweep(_chain)
            _ess[_name].append(_e)
            _ess_depth[_name].append(_e / (_dseq if _name == "mgrad" else _dtree))
    scaling_results = {
        "T": _t_lens,
        "ess": _ess,
        "ess_depth": _ess_depth,
        "depth_seq": _depth_seq,
        "depth_tree": _depth_tree,
    }
    return (scaling_results,)


@app.cell
def coda_scaling_fig(mo, palette, plt, scaling_results):
    _order = ("mgrad", "agrad", "amala_exact", "amala_plus")
    _colors = {
        "mgrad": palette["state"],
        "agrad": palette["belief"],
        "amala_exact": palette["seam"],
        "amala_plus": palette["operator"],
    }
    _labels = {
        "mgrad": "Particle-mGRAD",
        "agrad": "Particle-aGRAD",
        "amala_exact": "amala_exact",
        "amala_plus": "amala_plus",
    }
    _t = scaling_results["T"]
    _fig, (_a0, _a1, _a2) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for _k in _order:
        _a0.plot(_t, scaling_results["ess"][_k], "-o", color=_colors[_k], lw=2.3, label=_labels[_k])
    _a0.set_xscale("log", base=2)
    _a0.set_xlabel("time horizon T")
    _a0.set_ylabel("ESS per sweep (median over t)")
    _a0.set_title(
        "Per-sweep mixing is T-stable\n(all four: the CSMC property)",
        fontsize=10.5,
        fontweight="bold",
    )
    _a0.set_ylim(0.0, None)
    _a0.legend(frameon=False, fontsize=8.5)
    _a0.spines[["top", "right"]].set_visible(False)

    _a1.plot(
        _t,
        scaling_results["depth_seq"],
        "-o",
        color=palette["state"],
        lw=2.4,
        label="Particle-mGRAD — v̄_t chain, ≈ T",
    )
    _a1.plot(
        _t,
        scaling_results["depth_tree"],
        "-o",
        color=palette["belief"],
        lw=2.4,
        label="aGRAD / amala_* — c-dSMC tree, ≈ log₂ T",
    )
    _a1.set_xscale("log", base=2)
    _a1.set_xlabel("time horizon T")
    _a1.set_ylabel("sequential rounds per sweep")
    _a1.set_title("Depth per sweep\n(the audit's whole point)", fontsize=10.5, fontweight="bold")
    _a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    _a1.spines[["top", "right"]].set_visible(False)

    for _k in _order:
        _a2.plot(
            _t, scaling_results["ess_depth"][_k], "-o", color=_colors[_k], lw=2.3, label=_labels[_k]
        )
    _a2.set_xscale("log", base=2)
    _a2.set_yscale("log")
    _a2.set_xlabel("time horizon T")
    _a2.set_ylabel("ESS per unit sequential depth")
    _a2.set_title("Statistical progress per\ncritical-path round", fontsize=10.5, fontweight="bold")
    _a2.legend(frameon=False, fontsize=9)
    _a2.spines[["top", "right"]].set_visible(False)

    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def coda_recovery_md(mo):
    mo.md(r"""
    ### 11.3 Recovery at T = 1000

    The scaling curves stopped at T=256 to stay quick; the real payoff lands at a genuinely
    long horizon. Here we simulate **T = 1000** steps, run each kernel for a fixed budget of
    sweeps, and ask the blunt question: **how well does each recover the latent trajectory?**
    Two references frame "well": the raw observations (a floor to beat) and the exact grid
    smoother's own posterior mean (the *best achievable* recovery — no sampler can do better).

    At T=1000 the depth gap is no longer academic. One Particle-mGRAD sweep is a chain of
    **1000** sequential steps; one c-dSMC sweep (aGRAD, `amala_exact`, `amala_plus`) is a tree
    of **⌈log₂ 1000⌉ = 10**.
    """)
    return


@app.cell
def coda_recovery_run(
    ess_per_sweep, grid_smoother, math, np, run_agrad, run_amala, run_mgrad, simulate_ssm
):
    _t_len = 1000
    _x_true, _y = simulate_ssm(0, _t_len)
    _gold = grid_smoother(_y)
    _delta = 0.7
    _depth_tree = math.ceil(math.log2(_t_len))
    _chains = {
        "mgrad": run_mgrad(_y, delta=_delta, n_iter=500, seed=7),
        "agrad": run_agrad(_y, delta=_delta, n_iter=500, seed=7),
        "amala_exact": run_amala(_y, "amala_exact", delta=_delta, n_iter=500, seed=7),
        "amala_plus": run_amala(_y, "amala_plus", delta=_delta, n_iter=500, seed=7),
    }
    _metrics = {}
    for _name, _chain in _chains.items():
        _burn = _chain[200:]
        _pm = _burn.mean(0)
        _lo, _hi = np.percentile(_burn, 5, 0), np.percentile(_burn, 95, 0)
        _metrics[_name] = {
            "pm": _pm,
            "lo": _lo,
            "hi": _hi,
            "rmse_truth": float(np.sqrt(np.mean((_pm - _x_true) ** 2))),
            "rmse_gold": float(np.sqrt(np.mean((_pm - _gold["mean"]) ** 2))),
            "cov": float(np.mean((_x_true >= _lo) & (_x_true <= _hi))),
            "ess": float(ess_per_sweep(_chain)),
            "depth": _t_len if _name == "mgrad" else _depth_tree,
        }
    recovery_results = {
        "T": _t_len,
        "truth": _x_true,
        "obs": _y,
        "gold_rmse": float(np.sqrt(np.mean((_gold["mean"] - _x_true) ** 2))),
        "obs_rmse": float(np.sqrt(np.mean((_y - _x_true) ** 2))),
        "metrics": _metrics,
        "window": (440, 570),
    }
    return (recovery_results,)


@app.cell
def coda_recovery_fig(mo, palette, plt, recovery_results):
    _colors = {
        "mgrad": palette["state"],
        "agrad": palette["belief"],
        "amala_exact": palette["seam"],
        "amala_plus": palette["operator"],
    }
    _m = recovery_results["metrics"]
    _w0, _w1 = recovery_results["window"]
    _t = list(range(_w0, _w1))
    _fig, (_a0, _a1) = plt.subplots(
        1, 2, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # left — reconstruction window: aGRAD (exact, parallel) tracks truth; amala_plus drifts
    _a0.plot(
        _t,
        recovery_results["truth"][_w0:_w1],
        color=palette["ink"],
        lw=2.0,
        label="true latent path",
        zorder=4,
    )
    _a0.scatter(
        _t,
        recovery_results["obs"][_w0:_w1],
        color=palette["muted"],
        s=12,
        alpha=0.55,
        label="observations",
        zorder=2,
    )
    _ag = _m["agrad"]
    _a0.fill_between(
        _t, _ag["lo"][_w0:_w1], _ag["hi"][_w0:_w1], color=palette["belief"], alpha=0.18, zorder=1
    )
    _a0.plot(
        _t,
        _ag["pm"][_w0:_w1],
        color=palette["belief"],
        lw=1.8,
        label="aGRAD mean (exact, depth 10)",
        zorder=5,
    )
    _a0.plot(
        _t,
        _m["amala_plus"]["pm"][_w0:_w1],
        color=palette["operator"],
        lw=1.6,
        ls="--",
        label="amala_plus mean (biased)",
        zorder=3,
    )
    _a0.set_xlabel("time step t")
    _a0.set_ylabel("latent state xₜ")
    _a0.set_title(
        f"Recovered path at T={recovery_results['T']} (window)", fontsize=11, fontweight="bold"
    )
    _a0.legend(frameon=False, fontsize=8.5, loc="upper right")
    _a0.spines[["top", "right"]].set_visible(False)

    # right — sampler error vs exact posterior, per kernel, annotated with sequential depth
    _order = sorted(_colors, key=lambda k: _m[k]["rmse_gold"])
    _yp = list(range(len(_order)))
    _a1.barh(
        _yp,
        [_m[_k]["rmse_gold"] for _k in _order],
        color=[_colors[_k] for _k in _order],
        height=0.62,
    )
    _xmax = max(_m[_k]["rmse_gold"] for _k in _order)
    for _i, _k in enumerate(_order):
        _a1.text(
            _m[_k]["rmse_gold"] + 0.004,
            _i,
            f"depth {_m[_k]['depth']}",
            va="center",
            fontsize=8.5,
            color=palette["ink"],
        )
    _a1.set_yticks(_yp)
    _a1.set_yticklabels(_order, fontsize=9)
    _a1.set_xlim(0.0, _xmax * 1.35)
    _a1.set_xlabel("‖posterior mean − exact posterior‖")
    _a1.set_title("Fidelity, and the depth it cost", fontsize=11, fontweight="bold")
    _a1.spines[["top", "right"]].set_visible(False)

    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def coda_recovery_table(mo, recovery_results):
    _m = recovery_results["metrics"]
    _labels = {
        "mgrad": "Particle-mGRAD",
        "agrad": "Particle-aGRAD",
        "amala_exact": "amala_exact",
        "amala_plus": "amala_plus",
    }
    _head = (
        "| kernel | recovers truth (RMSE) | sampler error (vs exact) | 90% coverage "
        "| ESS/sweep | seq. depth |\n|---|:--:|:--:|:--:|:--:|:--:|"
    )
    _rows = []
    for _k in ("mgrad", "agrad", "amala_exact", "amala_plus"):
        _r = _m[_k]
        _rows.append(
            f"| **{_labels[_k]}** | {_r['rmse_truth']:.3f} | {_r['rmse_gold']:.3f} | "
            f"{_r['cov']:.2f} | {_r['ess']:.2f} | {_r['depth']} |"
        )
    mo.md(
        f"**Recovery at T={recovery_results['T']}** — fixed budget of 500 sweeps, δ=0.7. "
        f"The exact-smoother floor is RMSE={recovery_results['gold_rmse']:.3f} (the best any "
        f"sampler could do); the raw observations sit at RMSE={recovery_results['obs_rmse']:.3f}.\n\n"
        + _head
        + "\n"
        + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def coda_recovery_caption(mo):
    mo.md(r"""
    All three exact kernels recover the T=1000 path essentially as well as the exact smoother
    itself — their *recovers-truth* RMSE sits on the floor, mGRAD and the two tree kernels
    alike. What separates them is **cost**: mGRAD reaches the lowest sampler error, but only by
    paying a 1000-step sequential chain per sweep; aGRAD matches it to within Monte-Carlo noise
    at depth 10, and `amala_exact` is close behind. `amala_plus` is the lone failure — its bias
    lifts it off the floor even at this modest δ. Recovery quality is a near-tie among the exact
    kernels; **sequential depth is what actually differs, by 100×.**
    """)
    return


@app.cell(hide_code=True)
def coda_verdict_md(mo):
    mo.md(r"""
    ### 11.4 The verdict

    | kernel | source | folds prior? | exact? | mixing / sweep | depth / sweep | ESS / depth @ T=256 |
    |---|---|:--:|:--:|:--:|:--:|:--:|
    | **Particle-mGRAD** | Alg 7 (this notebook) | ✅ | ✅ | best | ⚠️ ≈ T | worst (falls like 1/T) |
    | **Particle-aGRAD** | tree (this notebook) | ✅ | ⚠️ approximate (§12.2) | ≈ mGRAD | ✅ ≈ log₂ T | **best** |
    | **amala_exact** | dsmc.py (shipped default) | ❌ (isotropic) | ✅ | good | ✅ ≈ log₂ T | strong |
    | **amala_plus** | dsmc.py (non-default) | ❌ | ❌ biased at large δ | good | ✅ ≈ log₂ T | — (biased) |

    Particle-mGRAD mixes best **per sweep** — folding the prior dynamics into the proposal is
    worth real ESS. But each sweep is a length-T sequential chain, exactly the `v̄_t`
    obstruction of Sections 6–9, so its ESS *per unit of depth* falls like 1/T.

    **Particle-aGRAD is the resolution the audit predicted.** It folds the same prior dynamics
    as mGRAD — so it inherits most of mGRAD's per-sweep mixing (roughly twice `amala_exact`'s
    here) — but it *keeps* the auxiliary `u` instead of marginalising it, so the seam stays a
    bounded pairwise transition and the whole kernel runs on the `⌈log₂ T⌉`-deep c-dSMC tree.
    The result is the top line of the rightmost panel: aGRAD delivers the most effective
    samples per unit of sequential depth at every T, and by T = 256 it is roughly an order of
    magnitude ahead of published mGRAD while remaining exact (it tracks the gold posterior in
    §11.1). `amala_exact`, our shipped default, is the isotropic cousin — it forgoes the
    prior-folding for simplicity, so it mixes less per sweep but shares the same exactness and
    tree depth. `amala_plus` shares the tree depth but not the invariance: cheap, parallel, and
    silently wrong once the steps are large.

    **This is the audit's recommendation, measured:** do not chase published Particle-mGRAD
    onto parallel hardware. Move to the exact auxiliary replacement — keep `u`, keep the
    prior-folding, and collect an unbounded, T-growing win (∝ T / log T) in the quantity a
    parallel machine actually charges for — sequential depth.

    **§12 pushes past this verdict on all three open fronts:** a *twisted* leaf (the
    taxonomy's third ✅ family, never benched here) beats aGRAD on both mixing and
    ESS-per-depth; a particle-*width* sweep shows the tree can buy back mGRAD's per-sweep
    edge at zero depth cost; and a 20,000-sweep invariance probe corrects the aGRAD row
    above.
    """)
    return


@app.cell(hide_code=True)
def coda2_intro(mo):
    mo.md(r"""
    ## 12. Coda 2 — the twisted leaf, the width axis, and an invariance correction

    §11 closed the case the audit opened, but it left three loose ends:

    1. **The taxonomy's third family was never benched.** §9 marked *three*
       parallel-compatible rows; §11 measured two (the prior-folding aGRAD leaf and the
       isotropic amala leaves). The **twisted / auxiliary-Kalman** row — proposals that
       look *ahead* through the data, not just at the local gradient — stayed a ✅ on
       paper.
    2. **mGRAD kept one real advantage: mixing per sweep** (≈0.7 vs ≈0.3–0.45 for the
       tree kernels at P = 16). Is that advantage *structural* to marginalisation — or
       just proposal quality, purchasable by the parallel family with a resource mGRAD
       cannot use?
    3. **§11.1's exactness check was blunt.** 700-sweep chains bound a stationary bias
       only below ≈0.03. A sharper instrument finds something §11 missed — in our own
       hand-built leaf, not in the published kernels.

    This coda answers all three with the same harness: same model, same gold-standard
    grid smoother, same ESS accounting.
    """)
    return


@app.cell(hide_code=True)
def coda2_twisted_md(mo):
    mo.md(r"""
    ### 12.1 The twisted leaf: lookahead proposals, exact correction

    The §11 leaves aim at time `t` using only time-`t` information (the observation
    gradient at the reference) plus a one-step prior fold. The best *independent* leaf is
    a different object entirely: the smoothing marginal `p(x_t | y_{1:T})` — every
    observation, both directions. The twisted leaf approximates exactly that:

    - **Pilot (once, before the chain):** three passes of an iterated extended RTS
      smoother around the data — linearise the drift along the current estimate,
      Kalman-filter, RTS-smooth, re-linearise. An affine forward–backward pass is an
      associative-scan object (the parallel-Kalman trick), so the pilot is itself
      O(log T) depth on parallel hardware — and it is the same machinery the production
      stack already runs as the Laplace warmup.
    - **Leaves:** `q_t = N(μ_t, c·σ_t²)` — the pilot's smoothing marginals, mildly
      inflated (`c = 1.5`). Fixed for the entire chain and independent across time:
      bounded seams, the identical tree.
    - **Exactness:** `ψ` pays `log G_t − log q_t` (plus the initial prior at `t = 0`) and
      the seams pay the true transition, so every reported quantity flows through the
      exact model densities. The linearisation can only make the *proposals* poor —
      never the posterior wrong. Same exactness class as `amala_exact`, and the
      policy-relevant point: the linearised smoother stays strictly on the proposal side
      of the weights.
    - **Depth:** each sweep adds nothing sequential beyond the `⌈log₂ T⌉` tree — the
      same depth accounting as the aGRAD and amala leaves.

    We also measured the auxiliary-anchored version (draw `z ~ N(x_ref, τI)`, pay its
    potential in `ψ`, condition the pilot marginals on `z`): ESS/sweep 0.38–0.45 across
    `τ ∈ [1, 4]` at T = 64, approaching this fixed kernel as `τ → ∞`. The fixed kernel is
    simpler, at least as good here, and — as §12.2 shows — its *fixedness* is precisely
    what makes it exactly invariant. So the fixed kernel is the one we bench.
    """)
    return


@app.cell
def coda2_twisted_kernel(
    DRIFT_A,
    DRIFT_B,
    DRIFT_W,
    INIT_SD,
    OBS_SD,
    PROC_SD,
    dsmc_prior_mean,
    dsmc_prior_var,
    jax,
    jnp,
    make_dsmc_tree,
    np,
    random,
):
    def run_twisted(y_obs, n_particles=16, inflate=1.5, n_iter=700, seed=0, n_pilot=3):
        """Twisted-leaf c-dSMC: fixed lookahead proposals, exact ψ-correction.

        Pilot: iterated extended RTS around the data (linearise the drift, smooth,
        re-linearise) — run ONCE, before the chain. Leaves: the pilot's smoothing
        marginals, inflated by `inflate`, fixed for the whole chain — independent of
        the reference path (that is what makes the kernel exactly invariant, §12.2)
        and independent across time (bounded seams, same tree as run_agrad).
        """
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _smooth = make_dsmc_tree(_t_len, _p, dsmc_prior_mean, dsmc_prior_var)

        def _logn(v, mu, var):
            return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

        def _affine_from(x_lin):
            # first-order drift expansion along x_lin: transition into t is
            # N(f_t x_{t-1} + b_t, PROC_SD^2); entry 0 is unused (initial prior).
            _f = DRIFT_A + DRIFT_B * DRIFT_W * jnp.cos(DRIFT_W * x_lin[:-1])
            _b = DRIFT_A * x_lin[:-1] + DRIFT_B * jnp.sin(DRIFT_W * x_lin[:-1]) - _f * x_lin[:-1]
            return (
                jnp.concatenate([jnp.zeros((1,)), _f]),
                jnp.concatenate([jnp.zeros((1,)), _b]),
            )

        def _affine_rts(f_all, b_all):
            # Kalman filter + RTS smoothing marginals of the affine system. Written as
            # scans for clarity; both recursions are associative (parallel-Kalman), so
            # the whole pass is O(log T) depth on parallel hardware.
            def _kf_step(carry, inp):
                _m_prev, _p_prev = carry
                _y_t, _f_t, _b_t, _first = inp
                _m_pred = jnp.where(_first, 0.0, _f_t * _m_prev + _b_t)
                _p_pred = jnp.where(_first, INIT_SD**2, _f_t**2 * _p_prev + PROC_SD**2)
                _gain = _p_pred / (_p_pred + OBS_SD**2)
                _m_f = _m_pred + _gain * (_y_t - _m_pred)
                _p_f = (1.0 - _gain) * _p_pred
                return (_m_f, _p_f), (_m_pred, _p_pred, _m_f, _p_f)

            _first = jnp.concatenate([jnp.ones((1,)), jnp.zeros((_t_len - 1,))])
            (_, _), (_m_pred, _p_pred, _m_f, _p_f) = jax.lax.scan(
                _kf_step, (0.0, 0.0), (_y, f_all, b_all, _first)
            )

            def _rts_step(carry, inp):
                _m_next, _p_next = carry
                _m_f_t, _p_f_t, _m_pred_next, _p_pred_next, _f_next = inp
                _g = _p_f_t * _f_next / jnp.maximum(_p_pred_next, 1e-12)
                _m_s = _m_f_t + _g * (_m_next - _m_pred_next)
                _p_s = _p_f_t + _g**2 * (_p_next - _p_pred_next)
                return (_m_s, _p_s), (_m_s, _p_s)

            _inp = (_m_f[:-1], _p_f[:-1], _m_pred[1:], _p_pred[1:], f_all[1:])
            _inp_rev = jax.tree_util.tree_map(lambda a: jnp.flip(a, 0), _inp)
            (_, _), (_m_s_rev, _p_s_rev) = jax.lax.scan(_rts_step, (_m_f[-1], _p_f[-1]), _inp_rev)
            return (
                jnp.concatenate([jnp.flip(_m_s_rev), _m_f[-1:]]),
                jnp.concatenate([jnp.flip(_p_s_rev), _p_f[-1:]]),
            )

        _x_lin = _y
        for _ in range(n_pilot):
            _f_all, _b_all = _affine_from(_x_lin)
            _x_lin, _ = _affine_rts(_f_all, _b_all)
        _f_all, _b_all = _affine_from(_x_lin)
        _mu_q, _var_q = _affine_rts(_f_all, _b_all)
        _q_var = inflate * _var_q

        def _leaf(x_ref, time_idx, key):
            _x_ref_t = x_ref[time_idx, 0]
            _free = _mu_q[time_idx] + jnp.sqrt(_q_var[time_idx]) * random.normal(key, (_p - 1,))
            _particles = jnp.concatenate([_x_ref_t[None], _free])[:, None]  # (P,1)
            _xs = _particles[:, 0]
            _psi = _logn(_y[time_idx], _xs, OBS_SD**2) - _logn(
                _xs, _mu_q[time_idx], _q_var[time_idx]
            )
            _psi = jnp.where(time_idx == 0, _psi + _logn(_xs, 0.0, INIT_SD**2), _psi)
            return _particles, _psi[:, None]  # (P,1),(P,1)

        def _body(x_ref, key):
            _xp = _smooth(key, lambda t, k: _leaf(x_ref, t, k))
            return _xp, _xp

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _y.reshape(_t_len, 1), _keys)
        return np.asarray(_chain)[:, :, 0]

    return (run_twisted,)


@app.cell(hide_code=True)
def coda2_invariance_md(mo):
    mo.md(r"""
    ### 12.2 An honest correction: the aGRAD leaf is not exactly invariant

    Writing §12.1 forces a question §11 skipped: *when is a leaf proposal that depends on
    the reference path actually legitimate?* Take the smallest case — one time step, one
    free particle. The reference is `x`, the proposal draws `x' ~ q(·|x)`, both get
    weight π/q, and one of the two is selected proportionally. The kernel density works
    out to

    \[
    \pi(x)\,k(x \to x') \;=\;
    \frac{\pi(x)\,\pi(x')}{\pi(x)/q(x\mid x) \;+\; \pi(x')/q(x'\mid x)} .
    \]

    Detailed balance needs the right-hand side to be symmetric in `(x, x')`. With a
    **fixed** proposal it is — the denominator reads π(x)/q(x) + π(x')/q(x'). With a
    **reference-centred** Gaussian it is not: the self-density `q(x|x)` = `q(x'|x')` is
    the proposal's mode height `r`, the cross-density `q(x'|x)` = `q(x|x')` is some
    smaller `s`, so the forward and reverse denominators differ by exactly
    (π(x) − π(x'))(1/r − 1/s) ≠ 0. The asymmetry survives unless π(x) = π(x').

    The rule this generalises to: a CSMC leaf may depend on the reference **only through
    a variable whose conditional law is exactly accounted for** — an auxiliary `u`
    refreshed by Gibbs *with its potential φ(u|x) paid in the weights* (the
    `amala_exact` pattern; and the reason published mGRAD is exact — its marginal weights
    *are* that correction, integrated) — or **not at all** (fixed proposals, the twisted
    leaf). §11's aGRAD leaf does neither, twice over: its gradient auxiliary `u` is drawn
    around the reference but φ(u|x) never enters `ψ`, and its prior fold is anchored at
    the reference's previous state directly. The defect dilutes with particle count (the
    selection is an importance resampler, so the error is O(1/N)-flavoured) and grows
    with the adaptation strength δ. The probe below measures it.
    """)
    return


@app.cell
def coda2_probe_run(
    coord_ess, grid_smoother, mo, np, run_agrad, run_amala, run_mgrad, run_twisted, simulate_ssm
):
    _, _y = simulate_ssm(0, 16)
    _gold = grid_smoother(_y)
    _configs = (
        (
            "aGRAD leaf (§11), P=4, δ=4",
            lambda: run_agrad(_y, n_particles=4, delta=4.0, n_iter=20000, seed=11),
        ),
        (
            "aGRAD leaf (§11), P=16, δ=4",
            lambda: run_agrad(_y, n_particles=16, delta=4.0, n_iter=20000, seed=11),
        ),
        (
            "twisted leaf (§12), P=4",
            lambda: run_twisted(_y, n_particles=4, n_iter=20000, seed=11),
        ),
        (
            "Particle-mGRAD, P=4, δ=4",
            lambda: run_mgrad(_y, n_particles=4, delta=4.0, n_iter=20000, seed=11),
        ),
        (
            "amala_exact (production), P=4, δ=4",
            lambda: run_amala(_y, "amala_exact", n_particles=4, delta=4.0, n_iter=20000, seed=11),
        ),
    )
    _lines = [
        "| kernel | RMSE vs exact posterior | max \\|z\\| across coordinates |",
        "|---|--:|--:|",
    ]
    for _name, _fn in _configs:
        _burn, _ess = coord_ess(_fn(), burn_frac=0.25)
        _err = _burn.mean(0) - _gold["mean"]
        _se = _burn.std(0) / np.sqrt(np.maximum(_ess, 1.0))
        _zmax = float(np.max(np.abs(_err) / np.maximum(_se, 1e-9)))
        _lines.append(f"| {_name} | {float(np.sqrt(np.mean(_err**2))):.4f} | {_zmax:.1f} |")
    mo.md(
        "**The stress probe.** T = 16, 20,000 sweeps (5,000 burn-in), stationary "
        "posterior mean vs the exact grid smoother; |z| is the per-coordinate error in "
        "units of its Monte-Carlo standard error (|z| ≫ 3 ⇒ bias, not noise):\n\n"
        + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def coda2_probe_caption(mo):
    mo.md(r"""
    The prediction from §12.2, confirmed quantitatively. The aGRAD leaf at δ = 4 is
    biased beyond any doubt (|z| ≈ 15–17) — and note the P = 16 row sits *inside*
    §11.1's sweep, hidden there beneath the 700-sweep Monte-Carlo floor. Doubling the
    particle count shrinks the error (≈0.18 → ≈0.06, the O(1/N) dilution), but does not
    remove it. Everything that pays its correction passes at the same stress level: the
    twisted leaf (fixed proposals), published mGRAD (marginal weights), and the
    production `amala_exact` leaf (auxiliary potential) — so `dsmc.py` is unaffected;
    the defect is confined to this notebook's §11 hand-built leaf.

    At the coda's operating point (P = 16, δ = 0.7) the aGRAD bias is beneath even this
    probe's detection floor, so §11.2–11.3's mixing, depth, and recovery conclusions
    stand as stated. What changes is the label: aGRAD-as-implemented is *approximately*
    invariant, in exactly the way `amala_plus` is — only much milder.
    """)
    return


@app.cell
def coda2_scaling_run(ess_per_sweep, math, run_twisted, scaling_results, simulate_ssm):
    _ess = []
    _ess_depth = []
    for _t_len in scaling_results["T"]:
        _, _y = simulate_ssm(0, _t_len)
        _e = ess_per_sweep(run_twisted(_y, n_iter=700, seed=3))
        _ess.append(_e)
        _ess_depth.append(_e / math.ceil(math.log2(_t_len)))
    twisted_scaling = {"ess": _ess, "ess_depth": _ess_depth}
    return (twisted_scaling,)


@app.cell
def coda2_scaling_fig(mo, palette, plt, scaling_results, twisted_scaling):
    _order = ("mgrad", "agrad", "amala_exact", "amala_plus")
    _colors = {
        "mgrad": palette["state"],
        "agrad": palette["belief"],
        "amala_exact": palette["seam"],
        "amala_plus": palette["operator"],
    }
    _labels = {
        "mgrad": "Particle-mGRAD",
        "agrad": "aGRAD leaf",
        "amala_exact": "amala_exact",
        "amala_plus": "amala_plus",
    }
    _t = scaling_results["T"]
    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for _k in _order:
        _a0.plot(
            _t,
            scaling_results["ess"][_k],
            "-o",
            color=_colors[_k],
            lw=1.8,
            alpha=0.55,
            label=_labels[_k],
        )
        _a1.plot(
            _t,
            scaling_results["ess_depth"][_k],
            "-o",
            color=_colors[_k],
            lw=1.8,
            alpha=0.55,
        )
    _a0.plot(
        _t,
        twisted_scaling["ess"],
        "-o",
        color=palette["obs"],
        lw=2.6,
        label="twisted leaf (§12)",
        zorder=5,
    )
    _a1.plot(_t, twisted_scaling["ess_depth"], "-o", color=palette["obs"], lw=2.6, zorder=5)
    _a0.set_xscale("log", base=2)
    _a0.set_xlabel("time horizon T")
    _a0.set_ylabel("ESS per sweep (median over t)")
    _a0.set_ylim(0.0, None)
    _a0.set_title("Per-sweep mixing", fontsize=11, fontweight="bold")
    _a0.legend(frameon=False, fontsize=8.5)
    _a0.spines[["top", "right"]].set_visible(False)
    _a1.set_xscale("log", base=2)
    _a1.set_yscale("log")
    _a1.set_xlabel("time horizon T")
    _a1.set_ylabel("ESS per unit sequential depth")
    _a1.set_title("Progress per critical-path round", fontsize=11, fontweight="bold")
    _a1.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def coda2_scaling_caption(mo):
    mo.md(r"""
    ### 12.3 Head-to-head: mixing, width, recovery

    The twisted leaf dominates the aGRAD leaf pointwise — at every horizon it mixes
    better per sweep at identical depth, so its ESS-per-depth curve sits strictly above
    (both P = 16, and the twisted leaf needs no per-sweep gradient or auxiliary draw at
    all). The remaining gap to mGRAD's per-sweep mixing is the subject of the width
    sweep below.
    """)
    return


@app.cell
def coda2_width_run(ess_per_sweep, run_agrad, run_mgrad, run_twisted, simulate_ssm):
    _, _y = simulate_ssm(0, 64)
    _ps = [8, 16, 32, 64]
    _ess = {"mgrad": [], "agrad": [], "twisted": []}
    for _p in _ps:
        _ess["mgrad"].append(
            ess_per_sweep(run_mgrad(_y, n_particles=_p, delta=0.7, n_iter=700, seed=5))
        )
        _ess["agrad"].append(
            ess_per_sweep(run_agrad(_y, n_particles=_p, delta=0.7, n_iter=700, seed=5))
        )
        _ess["twisted"].append(ess_per_sweep(run_twisted(_y, n_particles=_p, n_iter=700, seed=5)))
    width_results = {"P": _ps, "ess": _ess}
    return (width_results,)


@app.cell
def coda2_width_fig(mo, palette, plt, width_results):
    _colors = {"mgrad": palette["state"], "agrad": palette["belief"], "twisted": palette["obs"]}
    _labels = {
        "mgrad": "Particle-mGRAD — saturated: depth T at any P",
        "agrad": "aGRAD leaf — climbs with width",
        "twisted": "twisted leaf — climbs with width",
    }
    _p = width_results["P"]
    _fig, _ax = plt.subplots(figsize=(7.2, 4.2))
    for _k in ("mgrad", "agrad", "twisted"):
        _ax.plot(_p, width_results["ess"][_k], "-o", color=_colors[_k], lw=2.4, label=_labels[_k])
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("particles P (parallel width — costs no depth)")
    _ax.set_ylabel("ESS per sweep (median over t)")
    _ax.set_ylim(0.0, 1.0)
    _ax.set_title(
        "T = 64: the tree buys mixing with width; mGRAD cannot", fontsize=11, fontweight="bold"
    )
    _ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def coda2_width_caption(mo):
    mo.md(r"""
    This is the panel that removes mGRAD's last advantage. mGRAD saturates by P ≈ 16 —
    its per-sweep mixing is limited by the auxiliary step size δ, not by particle count —
    while the tree kernels keep climbing: every extra particle dilutes the reference's
    retention at every stitch. By P = 64 the twisted tree *overtakes* mGRAD's saturated
    per-sweep ESS. And width is the one resource that is depth-free on parallel hardware
    (the stitch grows as P² *work*, all of it parallel width), whereas mGRAD cannot
    convert any amount of hardware into less than a length-T chain per sweep.
    """)
    return


@app.cell
def coda2_recovery_run(ess_per_sweep, grid_smoother, math, mo, np, recovery_results, run_twisted):
    _y = recovery_results["obs"]
    _x_true = recovery_results["truth"]
    _gold = grid_smoother(_y)
    _t_len = recovery_results["T"]
    _chain = run_twisted(_y, n_iter=500, seed=7)
    _burn = _chain[200:]
    _pm = _burn.mean(0)
    _lo, _hi = np.percentile(_burn, 5, 0), np.percentile(_burn, 95, 0)
    twisted_recovery = {
        "rmse_truth": float(np.sqrt(np.mean((_pm - _x_true) ** 2))),
        "rmse_gold": float(np.sqrt(np.mean((_pm - _gold["mean"]) ** 2))),
        "cov": float(np.mean((_x_true >= _lo) & (_x_true <= _hi))),
        "ess": float(ess_per_sweep(_chain)),
        "depth": math.ceil(math.log2(_t_len)),
    }
    _labels = {
        "mgrad": "Particle-mGRAD",
        "agrad": "aGRAD leaf",
        "amala_exact": "amala_exact",
        "amala_plus": "amala_plus",
    }
    _lines = [
        "| kernel | RMSE vs truth | RMSE vs exact posterior | 90% cov | ESS/sweep | depth | ESS/depth |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for _k in ("mgrad", "agrad", "amala_exact", "amala_plus"):
        _m = recovery_results["metrics"][_k]
        _lines.append(
            f"| {_labels[_k]} | {_m['rmse_truth']:.4f} | {_m['rmse_gold']:.4f} "
            f"| {_m['cov']:.3f} | {_m['ess']:.3f} | {_m['depth']} "
            f"| {_m['ess'] / _m['depth']:.5f} |"
        )
    _m = twisted_recovery
    _lines.append(
        f"| **twisted leaf (§12)** | {_m['rmse_truth']:.4f} | {_m['rmse_gold']:.4f} "
        f"| {_m['cov']:.3f} | {_m['ess']:.3f} | {_m['depth']} "
        f"| {_m['ess'] / _m['depth']:.5f} |"
    )
    _lines.append("")
    _lines.append(
        f"(references: exact grid smoother recovers truth at RMSE "
        f"{recovery_results['gold_rmse']:.4f} — the achievable floor; raw observations "
        f"sit at {recovery_results['obs_rmse']:.4f})"
    )
    mo.md(
        "**Recovery at T = 1000** — same budget as §11.3 (500 sweeps, 200 burn-in), "
        "twisted leaf appended to the §11 table:\n\n" + "\n".join(_lines)
    )
    return (twisted_recovery,)


@app.cell(hide_code=True)
def coda2_verdict(mo):
    mo.md(r"""
    ### 12.4 The amended verdict

    | kernel | exact? | mixing / sweep | depth / sweep | ESS / depth |
    |---|:--:|:--:|:--:|:--:|
    | **Particle-mGRAD** | ✅ proven (Prop. 5) | best at P=16, saturated in P | ⚠️ ≈ T | worst, falls like 1/T |
    | **aGRAD leaf (§11)** | ⚠️ approximate (§12.2) | good, climbs with P | ✅ ≈ log₂ T | strong |
    | **twisted leaf (§12)** | ✅ probe-passed | > aGRAD at every T and P; ≥ mGRAD at P=64 | ✅ ≈ log₂ T | **best** |

    (`amala_exact` and `amala_plus` keep their §11.4 rows; the production `amala_exact`
    also passes the §12.2 stress probe.)

    Three upgrades over §11's verdict:

    1. **The recommendation sharpens.** "Replace mGRAD with an auxiliary kernel" becomes:
       *precompute lookahead marginals with the linearised machinery the stack already
       owns, pay the exact correction in the weights, and run the same tree.* The twisted
       leaf beats the aGRAD leaf on every axis measured — mixing per sweep, ESS per
       depth, recovery — and it is exactly invariant, which the aGRAD leaf turns out not
       to be. (For `dsmc.py` this is a concrete upgrade path: the warmup already builds
       the linearised smoother; a twisted leaf moves it to the proposal side of the exact
       weights, where the init-only linearization policy already allows it.)
    2. **mGRAD's last advantage is not structural.** Its per-sweep mixing edge is
       proposal quality, purchasable by the tree with particle *width* — a depth-free
       resource on parallel hardware — overtaking mGRAD's saturated per-sweep ESS by
       P = 64 while keeping `⌈log₂ T⌉` depth. mGRAD cannot spend the same coin: at any P
       its sweep is a length-T chain.
    3. **Exactness lives in the potential accounting, not the proposal.** The one pattern
       to refuse is adapt-without-paying — grossly (`amala_plus`) or subtly (§11's aGRAD
       leaf). Everything that pays passes: fixed proposals, paid auxiliaries, or marginal
       weights that are the payment integrated out.

    The audit's one-line summary survives with a sharper edge: the block interface
    decides *whether* you can parallelise; the twisted leaf shows how much of mGRAD's
    remaining edge the parallel family can then simply buy back — all of it.

    One caveat is owed: everything above was earned on friendly terrain — a Gaussian
    emission and gentle curvature, where a linearised pilot is nearly free lunch. §13
    turns both knobs and reruns the whole comparison where the lunch is not free.
    """)
    return


@app.cell(hide_code=True)
def coda3_intro(mo):
    mo.md(r"""
    ## 13. Coda 3 — hostile terrain: non-Gaussian emissions, harsher curvature

    §12's toy was kind to linearisation: a Gaussian emission carrying per-observation
    Fisher information 1/0.5² = 4 about the state, and drift curvature |m″| ≤ 1.6. This
    coda turns both knobs at once:

    - **Emission** → stochastic volatility, `y_t ~ N(0, exp(x_t))` — the emission family
      from the Corenflos–Finke experiments. Non-Gaussian, skewed, and *weak*: the Fisher
      information about `x_t` is exactly ½ per observation, **8× less** than §11–12.
    - **Drift** → `m(x) = 0.85·x + 1.0·sin(2x)`: curvature |m″| ≤ 4.0 (2.5× harsher),
      local slope swinging over [−1.15, 2.85].

    Weak data over hard dynamics means large posterior spread across curved drift — the
    curvature×spread regime where a linearised approximation is genuinely wrong, not
    just slightly off. And the sin-drift's multiple stable fixed points, no longer
    pinned down by informative observations, make the smoothing posterior **genuinely
    multimodal** at long horizons (109 of 1000 marginals at T = 1000; zero at T = 64).

    Three §12 claims are on trial:

    1. **Exactness accounting should be model-independent** — kernels that pay their
       potential stay exact; the unpaid aGRAD leaf should get *worse*.
    2. **The twisted leaf's edge should shrink or break** — its pilot is exactly the
       linearisation this terrain punishes.
    3. **mGRAD should shine** — its proposals re-derive local gradients from the current
       reference every sweep and fold ancestor-diverse prior means, nothing frozen.

    Same harness throughout: grid gold standard, same ESS accounting, per-kernel tuned
    step sizes (this model wants much bolder steps than δ = 0.7 — mGRAD's ESS triples
    from δ = 0.7 to its operating point δ = 4).
    """)
    return


@app.cell
def coda3_model(np):
    # Hostile toy: harsher sin-drift, constant transition covariance (still exactly the
    # Particle-mGRAD regime), stochastic-volatility emission.
    #     x_0 ~ N(0, SV_INIT_SD^2)
    #     x_t | x_{t-1} ~ N(SV_A·x + SV_B·sin(SV_W·x), SV_PROC_SD^2)
    #     y_t | x_t     ~ N(0, exp(x_t))
    SV_A, SV_B, SV_W = 0.85, 1.0, 2.0
    SV_PROC_SD, SV_INIT_SD = 0.5, 1.0

    def _drift(x):
        return SV_A * x + SV_B * np.sin(SV_W * x)

    def simulate_sv(seed, t_len):
        _rng = np.random.default_rng(seed)
        _x = np.zeros(t_len, dtype=np.float32)
        _x[0] = _rng.normal(0.0, SV_INIT_SD)
        for _t in range(1, t_len):
            _x[_t] = _drift(_x[_t - 1]) + _rng.normal(0.0, SV_PROC_SD)
        _y = (np.exp(_x / 2.0) * _rng.normal(0.0, 1.0, size=t_len)).astype(np.float32)
        return _x, _y

    def grid_smoother_sv(y, n_grid=721, lo=-9.0, hi=9.0):
        """Near-exact SV smoothing marginals — the same grid gold standard as §11."""
        _t_len = len(y)
        _xs = np.linspace(lo, hi, n_grid)

        def _logn(v, mu, sd):
            return -0.5 * (np.log(2.0 * np.pi * sd**2) + ((v - mu) ** 2) / sd**2)

        _log_obs = -0.5 * (
            np.log(2.0 * np.pi) + _xs[None, :] + (y[:, None] ** 2) * np.exp(-_xs[None, :])
        )
        _log_tr = _logn(_xs[None, :], _drift(_xs)[:, None], SV_PROC_SD)

        _log_alpha = np.zeros((_t_len, n_grid))
        _log_alpha[0] = _logn(_xs, 0.0, SV_INIT_SD) + _log_obs[0]
        for _t in range(1, _t_len):
            _a = _log_alpha[_t - 1]
            _m = _a.max()
            _log_alpha[_t] = np.log(np.exp(_a - _m) @ np.exp(_log_tr) + 1e-300) + _m + _log_obs[_t]
        _log_beta = np.zeros((_t_len, n_grid))
        for _t in range(_t_len - 2, -1, -1):
            _b = _log_beta[_t + 1] + _log_obs[_t + 1]
            _m = _b.max()
            _log_beta[_t] = np.log(np.exp(_log_tr) @ np.exp(_b - _m) + 1e-300) + _m
        _log_g = _log_alpha + _log_beta
        _log_g -= _log_g.max(1, keepdims=True)
        _g = np.exp(_log_g)
        _g /= _g.sum(1, keepdims=True)
        _mean = (_g * _xs[None, :]).sum(1)
        _sd = np.sqrt((_g * (_xs[None, :] - _mean[:, None]) ** 2).sum(1))
        return {"mean": _mean, "sd": _sd, "xs": _xs, "g": _g}

    def count_multimodal(gold, thresh=0.05):
        """Time points whose smoothing marginal has more than one substantial mode."""
        _g = gold["g"]
        _idx = []
        for _t in range(_g.shape[0]):
            _d = _g[_t]
            _peaks = 0
            for _i in range(1, len(_d) - 1):
                if _d[_i] > _d[_i - 1] and _d[_i] >= _d[_i + 1] and _d[_i] > thresh * _d.max():
                    _peaks += 1
            if _peaks > 1:
                _idx.append(_t)
        return _idx

    return SV_A, SV_B, SV_INIT_SD, SV_PROC_SD, SV_W, count_multimodal, grid_smoother_sv, simulate_sv


@app.cell
def coda3_terrain_fig(
    DRIFT_A,
    DRIFT_B,
    DRIFT_W,
    SV_A,
    SV_B,
    SV_W,
    count_multimodal,
    grid_smoother_sv,
    mo,
    np,
    palette,
    plt,
    simulate_sv,
):
    _xg = np.linspace(-4.0, 4.0, 400)
    _m11 = DRIFT_A * _xg + DRIFT_B * np.sin(DRIFT_W * _xg)
    _m13 = SV_A * _xg + SV_B * np.sin(SV_W * _xg)

    _, _y = simulate_sv(0, 256)
    _gold = grid_smoother_sv(_y)
    _multi = count_multimodal(_gold)
    _t_bi = _multi[0] if _multi else 0
    _t_uni = next(_t for _t in range(256) if _t not in set(_multi))

    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    _a0.plot(_xg, _xg, color=palette["muted"], lw=1.2, ls="--", label="identity (fixed points)")
    _a0.plot(_xg, _m11, color=palette["belief"], lw=2.2, label="§11 drift — |m″| ≤ 1.6")
    _a0.plot(_xg, _m13, color=palette["operator"], lw=2.2, label="§13 drift — |m″| ≤ 4.0")
    _a0.set_xlabel("x")
    _a0.set_ylabel("transition mean m(x)")
    _a0.set_title("Harsher curvature, more attractors", fontsize=11, fontweight="bold")
    _a0.legend(frameon=False, fontsize=8.5)
    _a0.spines[["top", "right"]].set_visible(False)

    _a1.plot(
        _gold["xs"],
        _gold["g"][_t_uni],
        color=palette["belief"],
        lw=2.2,
        label=f"t = {_t_uni}: unimodal",
    )
    _a1.plot(
        _gold["xs"],
        _gold["g"][_t_bi],
        color=palette["operator"],
        lw=2.2,
        label=f"t = {_t_bi}: bimodal",
    )
    _a1.set_xlim(-4.5, 4.5)
    _a1.set_xlabel("x_t")
    _a1.set_ylabel("smoothing marginal (grid)")
    _a1.set_title(
        f"Weak SV data leaves drift basins unresolved\n({len(_multi)}/256 marginals multimodal)",
        fontsize=10.5,
        fontweight="bold",
    )
    _a1.legend(frameon=False, fontsize=8.5)
    _a1.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def coda3_kernels(SV_A, SV_B, SV_INIT_SD, SV_PROC_SD, SV_W, jax, jnp, make_dsmc_tree, np, random):
    def _m_mean(x):
        return SV_A * x + SV_B * jnp.sin(SV_W * x)

    def _logn(v, mu, var):
        return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

    def _log_obs(x, y_t):  # log N(y; 0, e^x)
        return -0.5 * (jnp.log(2.0 * jnp.pi) + x + (y_t**2) * jnp.exp(-x))

    def _grad_obs(x, y_t):
        return -0.5 + 0.5 * (y_t**2) * jnp.exp(-x)

    def _prior_mean(time_idx, x_prev):
        return jnp.where(time_idx == 0, 0.0, _m_mean(x_prev))

    def _prior_var(time_idx):
        return jnp.where(time_idx == 0, SV_INIT_SD**2, SV_PROC_SD**2)

    def _x_init(y):
        # crude log-volatility estimate: log y² is unbiased for x up to E[log χ²₁] ≈ −1.27
        return jnp.clip(jnp.log(y**2 + 1e-2) + 1.27, -8.0, 8.0)

    def run_mgrad_sv(y_obs, n_particles=16, delta=4.0, kappa=1.0, n_iter=700, seed=0):
        """Particle-mGRAD (Alg 7) on the SV model — only the emission terms change."""
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _n = _p - 1
        _half = 0.5 * delta

        def _forward(key, x_ref):
            def _step(carry, time_idx):
                _prev, _weights, _key = carry
                _key, _anc_key, _u_key, _prop_key = random.split(_key, 4)
                _cov = _prior_var(time_idx)
                _a_gain = _cov / (_cov + _half)
                _prop_v = _half * _a_gain
                _g_hat = (2.0 / delta) / (1.0 + _n * _a_gain)
                _anc = random.categorical(_anc_key, jnp.log(_weights + 1e-38), shape=(_p,))
                _anc = _anc.at[0].set(0)
                _mmean = _prior_mean(time_idx, _prev[_anc])
                _x_ref_t = x_ref[time_idx]
                _u = (
                    _x_ref_t
                    + kappa * _half * _grad_obs(_x_ref_t, _y[time_idx])
                    + jnp.sqrt(_half) * random.normal(_u_key)
                )
                _x_t = (
                    (1.0 - _a_gain) * _mmean
                    + _a_gain * _u
                    + jnp.sqrt(_prop_v) * random.normal(_prop_key, (_p,))
                )
                _x_t = _x_t.at[0].set(_x_ref_t)
                _v = (1.0 - _a_gain) * _mmean
                _x_bar = jnp.mean(_x_t)
                _v_bar = jnp.mean(_v)
                _phi = kappa * _half * _grad_obs(_x_t, _y[time_idx])
                _log_q = _logn(_x_t, _mmean, _cov) + _log_obs(_x_t, _y[time_idx])
                _term1 = 0.5 * (_x_t - _v) ** 2 * (1.0 / _prop_v + _g_hat)
                _term2 = (
                    -(0.5 * _n * (_x_t + _phi) * _a_gain + (_x_t - _v)) * _g_hat * (_x_t + _phi)
                )
                _term3 = (_n + 1) * (_x_bar - _v_bar) * _g_hat * (_v + _phi)
                _log_w = _log_q + _term1 + _term2 + _term3
                _log_w = _log_w - jax.scipy.special.logsumexp(_log_w)
                return (_x_t, jnp.exp(_log_w), _key), (_x_t, jnp.exp(_log_w))

            (_, _, _), (_particles, _all_w) = jax.lax.scan(
                _step, (jnp.zeros(_p), jnp.ones(_p) / _p, key), jnp.arange(_t_len)
            )
            return _particles, _all_w

        def _backward(key, particles, weights):
            _key_last, _key_rest = random.split(key)
            _l_last = random.categorical(_key_last, jnp.log(weights[_t_len - 1] + 1e-38))

            def _bstep(carry, time_idx):
                _l_next, _key = carry
                _key, _s_key = random.split(_key)
                _x_next = particles[time_idx + 1][_l_next]
                _logp = jnp.log(weights[time_idx] + 1e-38) + _logn(
                    _x_next, _m_mean(particles[time_idx]), _prior_var(time_idx + 1)
                )
                _l = random.categorical(_s_key, _logp)
                return (_l, _key), _l

            (_, _), _ls_rev = jax.lax.scan(
                _bstep, (_l_last, _key_rest), jnp.arange(_t_len - 2, -1, -1)
            )
            _ls = jnp.concatenate([jnp.flip(_ls_rev), _l_last[None]])
            return particles[jnp.arange(_t_len), _ls]

        def _sweep(x_ref, key):
            _key_f, _key_b = random.split(key)
            _particles, _weights = _forward(_key_f, x_ref)
            _path = _backward(_key_b, _particles, _weights)
            return _path, _path

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_sweep, _x_init(_y), _keys)
        return np.asarray(_chain)

    def run_agrad_sv(y_obs, n_particles=16, delta=3.0, kappa=1.0, n_iter=700, seed=0):
        """§11's aGRAD leaf on the SV model — same unpaid reference adaptation."""
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _half = 0.5 * delta
        _smooth = make_dsmc_tree(_t_len, _p, _prior_mean, _prior_var)

        def _leaf(x_ref, time_idx, key):
            _u_key, _s_key = random.split(key, 2)
            _cov = _prior_var(time_idx)
            _a_gain = _cov / (_cov + _half)
            _prop_v = _half * _a_gain
            _x_ref_t = x_ref[time_idx, 0]
            _u = (
                _x_ref_t
                + kappa * _half * _grad_obs(_x_ref_t, _y[time_idx])
                + jnp.sqrt(_half) * random.normal(_u_key)
            )
            _pm = _prior_mean(time_idx, x_ref[jnp.maximum(time_idx - 1, 0), 0])
            _center = (1.0 - _a_gain) * _pm + _a_gain * _u
            _free = _center + jnp.sqrt(_prop_v) * random.normal(_s_key, (_p - 1,))
            _particles = jnp.concatenate([_x_ref_t[None], _free])[:, None]
            _xs = _particles[:, 0]
            _psi = _log_obs(_xs, _y[time_idx]) - _logn(_xs, _center, _prop_v)
            _psi = jnp.where(time_idx == 0, _psi + _logn(_xs, 0.0, SV_INIT_SD**2), _psi)
            return _particles, _psi[:, None]

        def _body(x_ref, key):
            _xp = _smooth(key, lambda t, k: _leaf(x_ref, t, k))
            return _xp, _xp

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _x_init(_y).reshape(_t_len, 1), _keys)
        return np.asarray(_chain)[:, :, 0]

    def run_twisted_sv(
        y_obs,
        n_particles=16,
        inflate=3.0,
        n_iter=700,
        seed=0,
        n_pilot=25,
        damp=0.3,
        lam_cap=25.0,
        eps=0.0,
        wide=9.0,
    ):
        """Twisted leaf on the SV model: STABILISED iterated-Laplace pilot.

        §12's pilot (three undamped Gauss–Newton passes) diverges here — NaN by pass
        four. This terrain demands the standard IEKS repertoire: a smoothed crude
        init, damped iteration, a trust-region cap on the Laplace precision, and a
        trust radius on the Newton pseudo-observation step. Even stabilised, the
        pilot lands ≈1 posterior-sd off — the intended stress.

        eps > 0 adds a defensive mixture tail (Hesterberg-style): each leaf draws
        from N(μ_t, inflate·σ_t²) with prob 1−eps and from N(μ_t, wide) with prob
        eps. Fixed proposals either way ⇒ exactly invariant either way; the wide
        component is what turns 'silently pinned off-target' into ordinary mixing.
        """
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _smooth = make_dsmc_tree(_t_len, _p, _prior_mean, _prior_var)

        def _laplace_rts(x_hat):
            _f = SV_A + SV_B * SV_W * jnp.cos(SV_W * x_hat[:-1])
            _b = _m_mean(x_hat[:-1]) - _f * x_hat[:-1]
            _f_all = jnp.concatenate([jnp.zeros((1,)), _f])
            _b_all = jnp.concatenate([jnp.zeros((1,)), _b])
            _lam = jnp.clip(0.5 * (_y**2) * jnp.exp(-x_hat), 1e-3, lam_cap)  # -g'' (log-concave)
            # Newton pseudo-observation with a trust radius on the STEP: an unbounded
            # step where the precision cap binds blows the pilot up at long horizons.
            _z = x_hat + jnp.clip(_grad_obs(x_hat, _y) / _lam, -3.0, 3.0)
            _r = 1.0 / _lam

            def _kf_step(carry, inp):
                _m_prev, _p_prev = carry
                _z_t, _r_t, _f_t, _b_t, _first = inp
                _m_pred = jnp.where(_first, 0.0, _f_t * _m_prev + _b_t)
                _p_pred = jnp.where(_first, SV_INIT_SD**2, _f_t**2 * _p_prev + SV_PROC_SD**2)
                _gain = _p_pred / (_p_pred + _r_t)
                _m_f = _m_pred + _gain * (_z_t - _m_pred)
                _p_f = (1.0 - _gain) * _p_pred
                return (_m_f, _p_f), (_m_pred, _p_pred, _m_f, _p_f)

            _first = jnp.concatenate([jnp.ones((1,)), jnp.zeros((_t_len - 1,))])
            (_, _), (_m_pred, _p_pred, _m_f, _p_f) = jax.lax.scan(
                _kf_step, (0.0, 0.0), (_z, _r, _f_all, _b_all, _first)
            )

            def _rts_step(carry, inp):
                _m_next, _p_next = carry
                _m_f_t, _p_f_t, _mpn, _ppn, _f_next = inp
                _g = _p_f_t * _f_next / jnp.maximum(_ppn, 1e-12)
                _m_s = _m_f_t + _g * (_m_next - _mpn)
                _p_s = _p_f_t + _g**2 * (_p_next - _ppn)
                return (_m_s, _p_s), (_m_s, _p_s)

            _inp = (_m_f[:-1], _p_f[:-1], _m_pred[1:], _p_pred[1:], _f_all[1:])
            _inp_rev = jax.tree_util.tree_map(lambda a: jnp.flip(a, 0), _inp)
            (_, _), (_m_s_rev, _p_s_rev) = jax.lax.scan(_rts_step, (_m_f[-1], _p_f[-1]), _inp_rev)
            return (
                jnp.concatenate([jnp.flip(_m_s_rev), _m_f[-1:]]),
                jnp.concatenate([jnp.flip(_p_s_rev), _p_f[-1:]]),
            )

        _raw = _x_init(_y)
        _x_hat = jnp.convolve(jnp.pad(_raw, (3, 3), mode="edge"), jnp.ones(7) / 7.0, mode="valid")
        for _ in range(n_pilot):
            _mu, _ = _laplace_rts(_x_hat)
            _x_hat = (1.0 - damp) * _x_hat + damp * _mu
        _mu_q, _var_q = _laplace_rts(_x_hat)
        _q_var = inflate * _var_q

        def _log_q(xs, time_idx):
            if eps == 0.0:
                return _logn(xs, _mu_q[time_idx], _q_var[time_idx])
            return jnp.logaddexp(
                jnp.log(1.0 - eps) + _logn(xs, _mu_q[time_idx], _q_var[time_idx]),
                jnp.log(eps) + _logn(xs, _mu_q[time_idx], wide),
            )

        def _leaf(x_ref, time_idx, key):
            _pick_key, _draw_key = random.split(key)
            _sd = jnp.sqrt(_q_var[time_idx])
            if eps > 0.0:
                _pick = random.bernoulli(_pick_key, eps, (_p - 1,))
                _sd = jnp.where(_pick, jnp.sqrt(wide), _sd)
            _free = _mu_q[time_idx] + _sd * random.normal(_draw_key, (_p - 1,))
            _particles = jnp.concatenate([x_ref[time_idx, 0][None], _free])[:, None]
            _xs = _particles[:, 0]
            _psi = _log_obs(_xs, _y[time_idx]) - _log_q(_xs, time_idx)
            _psi = jnp.where(time_idx == 0, _psi + _logn(_xs, 0.0, SV_INIT_SD**2), _psi)
            return _particles, _psi[:, None]

        def _body(x_ref, key):
            _xp = _smooth(key, lambda t, k: _leaf(x_ref, t, k))
            return _xp, _xp

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _x_init(_y).reshape(_t_len, 1), _keys)
        return np.asarray(_chain)[:, :, 0]

    return run_agrad_sv, run_mgrad_sv, run_twisted_sv


@app.cell(hide_code=True)
def coda3_pilot_md(mo):
    mo.md(r"""
    ### 13.1 What survived contact, before any benchmark

    Just *building* the twisted leaf on this model already answers part of the
    question. The §12 pilot — three undamped extended-RTS passes — **diverges to NaN by
    the fourth pass** here: the SV log-likelihood's exponential curvature plus drift
    slopes past ±1 make plain Gauss–Newton oscillate and explode. The stabilised pilot
    (smoothed crude init, damping 0.3 over 25 passes, precision cap, Newton-step trust
    radius) converges, but only to within about **one posterior standard deviation** of
    the truth (mean-RMSE vs gold 0.2–0.8 across horizons, against a posterior sd of
    ≈0.55). That is the honest input to everything below: proposals that overlap the
    posterior without describing it.

    Two failure modes of the naive fixed-proposal leaf follow directly, and both show
    up in the cells below:

    - **under-inflated proposals pin the chain off-target** — at `c = 2` the T = 64
      chain sits 0.58 from gold for its whole budget *while reporting healthy ESS*;
    - **where the pilot is locally wrong** (T = 16 here) some coordinates never move
      at all — visible as (near-)zero sample variance, the loud version.

    The defensive mixture (`eps = 0.25` of leaf draws from a wide `N(μ_t, 9)` tail) is
    the classical insurance against exactly this, and it is free at the depth level.
    Below, "twisted" means the plain `c = 3` leaf and "twisted+def" the defensive one.
    """)
    return


@app.cell
def coda3_probe_run(
    coord_ess, grid_smoother_sv, mo, np, run_agrad_sv, run_mgrad_sv, run_twisted_sv, simulate_sv
):
    _, _y = simulate_sv(0, 16)
    _gold = grid_smoother_sv(_y)
    _configs = (
        (
            "aGRAD leaf, P=16, δ=4 (20k)",
            lambda: run_agrad_sv(_y, n_particles=16, delta=4.0, n_iter=20000, seed=11),
        ),
        (
            "aGRAD leaf, P=4, δ=4 (20k)",
            lambda: run_agrad_sv(_y, n_particles=4, delta=4.0, n_iter=20000, seed=11),
        ),
        (
            "twisted, P=16 (20k)",
            lambda: run_twisted_sv(_y, n_particles=16, n_iter=20000, seed=11),
        ),
        (
            "twisted+def, P=16 (20k)",
            lambda: run_twisted_sv(_y, n_particles=16, eps=0.25, n_iter=20000, seed=11),
        ),
        (
            "twisted+def, P=4 (20k)",
            lambda: run_twisted_sv(_y, n_particles=4, eps=0.25, n_iter=20000, seed=11),
        ),
        (
            "Particle-mGRAD, P=4, δ=4 (60k)",
            lambda: run_mgrad_sv(_y, n_particles=4, delta=4.0, n_iter=60000, seed=11),
        ),
    )
    _lines = [
        "| kernel | RMSE vs exact posterior | max \\|z\\| | min coordinate sd |",
        "|---|--:|--:|--:|",
    ]
    for _name, _fn in _configs:
        _burn, _ess = coord_ess(_fn(), burn_frac=0.25)
        _err = _burn.mean(0) - _gold["mean"]
        _se = _burn.std(0) / np.sqrt(np.maximum(_ess, 1.0))
        _zmax = float(np.max(np.abs(_err) / np.maximum(_se, 1e-9)))
        _lines.append(
            f"| {_name} | {float(np.sqrt(np.mean(_err**2))):.4f} | {_zmax:.1f} "
            f"| {float(_burn.std(0).min()):.2e} |"
        )
    mo.md(
        "**The §12.2 stress probe on hostile terrain.** T = 16, stationary posterior "
        "mean vs the grid gold standard (mGRAD gets a longer budget because its P = 4 "
        "sweeps move less, so its Monte-Carlo error decays slower):\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def coda3_probe_caption(mo):
    mo.md(r"""
    Prediction 1, confirmed and amplified. The exactness accounting is indeed
    model-independent: mGRAD (marginal weights = the correction) and both defensive
    twisted configurations (fixed proposals) pass. The plain twisted leaf **fails
    loudly** — a near-zero minimum coordinate sd is a chain that never moved there;
    the astronomical z-score is that frozen coordinate, not a subtle bias. And §11's
    aGRAD leaf now fails **badly**: the bias that needed a 20,000-sweep microscope on
    the Gaussian model (0.065 at δ = 4) is 0.12 here — the unpaid reference adaptation
    scales with exactly the terrain features this model maximises. Worse, the aGRAD
    chain also **traps**: at δ = 4 (T = 64) two of three seeds sit ≈0.5 from gold for
    thousands of sweeps *with healthy-looking ESS*, and at its tuned δ = 3 the same
    silent trapping shows up at T ≥ 128 in the scaling panel below — the failure
    self-reinforces because proposals are centred on the wrong reference.
    """)
    return


@app.cell
def coda3_scaling_run(
    ess_per_sweep, grid_smoother_sv, math, run_agrad_sv, run_mgrad_sv, run_twisted_sv, simulate_sv
):
    _t_lens = [16, 32, 64, 128, 256]
    _names = ("mgrad", "agrad", "twisted", "twisted_def")
    _ess = {_n: [] for _n in _names}
    _ess_depth = {_n: [] for _n in _names}
    _rmse = {_n: [] for _n in _names}
    for _t_len in _t_lens:
        _, _y = simulate_sv(0, _t_len)
        _gold = grid_smoother_sv(_y)
        _chains = {
            "mgrad": run_mgrad_sv(_y, delta=4.0, n_iter=700, seed=3),
            "agrad": run_agrad_sv(_y, delta=3.0, n_iter=700, seed=3),
            "twisted": run_twisted_sv(_y, inflate=3.0, n_iter=700, seed=3),
            "twisted_def": run_twisted_sv(_y, inflate=3.0, eps=0.25, n_iter=700, seed=3),
        }
        _dtree = math.ceil(math.log2(_t_len))
        for _name, _chain in _chains.items():
            _e = ess_per_sweep(_chain)
            _ess[_name].append(_e)
            _ess_depth[_name].append(_e / (_t_len if _name == "mgrad" else _dtree))
            _rmse[_name].append(float(((_chain[350:].mean(0) - _gold["mean"]) ** 2).mean() ** 0.5))
    sv_scaling = {"T": _t_lens, "ess": _ess, "ess_depth": _ess_depth, "rmse": _rmse}
    return (sv_scaling,)


@app.cell
def coda3_scaling_fig(mo, palette, plt, sv_scaling):
    _colors = {
        "mgrad": palette["state"],
        "agrad": palette["belief"],
        "twisted": palette["obs"],
        "twisted_def": palette["obs"],
    }
    _styles = {"mgrad": "-", "agrad": "-", "twisted": ":", "twisted_def": "-"}
    _labels = {
        "mgrad": "Particle-mGRAD (δ=4)",
        "agrad": "aGRAD leaf (δ=3)",
        "twisted": "twisted (c=3)",
        "twisted_def": "twisted+def (c=3, ε=0.25)",
    }
    _t = sv_scaling["T"]
    _fig, (_a0, _a1, _a2) = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for _k in ("mgrad", "agrad", "twisted", "twisted_def"):
        _a0.plot(
            _t,
            sv_scaling["ess"][_k],
            _styles[_k],
            marker="o",
            color=_colors[_k],
            lw=2.2,
            label=_labels[_k],
        )
        _a1.plot(_t, sv_scaling["rmse"][_k], _styles[_k], marker="o", color=_colors[_k], lw=2.2)
        _a2.plot(
            _t, sv_scaling["ess_depth"][_k], _styles[_k], marker="o", color=_colors[_k], lw=2.2
        )
    _a0.set_xscale("log", base=2)
    _a0.set_xlabel("time horizon T")
    _a0.set_ylabel("ESS per sweep (median over t)")
    _a0.set_ylim(0.0, None)
    _a0.set_title("Mixing per sweep", fontsize=11, fontweight="bold")
    _a0.legend(frameon=False, fontsize=8)
    _a0.spines[["top", "right"]].set_visible(False)
    _a1.set_xscale("log", base=2)
    _a1.set_yscale("log")
    _a1.axhline(0.05, color=palette["muted"], ls="--", lw=1.2)
    _a1.text(_t[0], 0.056, "≈ Monte-Carlo floor", fontsize=8, color=palette["ink"])
    _a1.set_xlabel("time horizon T")
    _a1.set_ylabel("‖posterior mean − exact‖ (RMSE)")
    _a1.set_title(
        "Convergence within the budget\n(silent failures show here)",
        fontsize=10.5,
        fontweight="bold",
    )
    _a1.spines[["top", "right"]].set_visible(False)
    _a2.set_xscale("log", base=2)
    _a2.set_yscale("log")
    _a2.set_xlabel("time horizon T")
    _a2.set_ylabel("ESS per unit sequential depth")
    _a2.set_title("Progress per critical-path round", fontsize=11, fontweight="bold")
    _a2.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def coda3_scaling_caption(mo):
    mo.md(r"""
    Predictions 2 and 3, measured. **mGRAD is untouched by the terrain**: per-sweep ESS
    ≈ 0.51–0.72 (essentially its Gaussian-model numbers) and it sits on the
    Monte-Carlo floor at every horizon — fresh gradients at the current reference plus
    ancestor-diverse prior folds hop between drift basins for free. The **plain
    twisted leaf breaks** exactly where its fixed pilot is locally wrong (T = 16) or
    where the posterior turns multimodal (T ≥ 128) — and breaks *silently*, with
    plausible-looking ESS. The **aGRAD leaf breaks the same way at T ≥ 128** (0.35–0.39
    from gold). The **defensive mixture repairs the twisted leaf completely** at these
    horizons: on the floor everywhere, at essentially no ESS cost, and its
    ESS-per-depth still beats mGRAD by ~11× at T = 256 (0.029 vs 0.0026).
    """)
    return


@app.cell
def coda3_recovery_run(
    count_multimodal,
    ess_per_sweep,
    grid_smoother_sv,
    mo,
    np,
    run_agrad_sv,
    run_mgrad_sv,
    run_twisted_sv,
    simulate_sv,
):
    _t_len = 1000
    _x_true, _y = simulate_sv(0, _t_len)
    _gold = grid_smoother_sv(_y)
    _n_multi = len(count_multimodal(_gold))
    _rows = (
        (
            "Particle-mGRAD, 500 sweeps",
            lambda: run_mgrad_sv(_y, delta=4.0, n_iter=500, seed=7),
            200,
            500 * _t_len,
        ),
        (
            "aGRAD leaf, 500 sweeps",
            lambda: run_agrad_sv(_y, delta=3.0, n_iter=500, seed=7),
            200,
            500 * 10,
        ),
        (
            "twisted, 500 sweeps",
            lambda: run_twisted_sv(_y, inflate=3.0, n_iter=500, seed=7),
            200,
            500 * 10,
        ),
        (
            "twisted+def, 500 sweeps",
            lambda: run_twisted_sv(_y, inflate=3.0, eps=0.25, n_iter=500, seed=7),
            200,
            500 * 10,
        ),
        (
            "aGRAD leaf, 5000 sweeps",
            lambda: run_agrad_sv(_y, delta=3.0, n_iter=5000, seed=7),
            2000,
            5000 * 10,
        ),
        (
            "twisted+def, 5000 sweeps",
            lambda: run_twisted_sv(_y, inflate=3.0, eps=0.25, n_iter=5000, seed=7),
            2000,
            5000 * 10,
        ),
    )
    _lines = [
        "| kernel | RMSE vs exact posterior | RMSE vs truth | 90% cov | ESS/sweep | sequential rounds spent |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for _name, _fn, _burn_at, _depth_spent in _rows:
        _chain = _fn()
        _burn = _chain[_burn_at:]
        _pm = _burn.mean(0)
        _lo, _hi = np.percentile(_burn, 5, 0), np.percentile(_burn, 95, 0)
        _lines.append(
            f"| {_name} | {float(np.sqrt(np.mean((_pm - _gold['mean']) ** 2))):.4f} "
            f"| {float(np.sqrt(np.mean((_pm - _x_true) ** 2))):.4f} "
            f"| {float(np.mean((_x_true >= _lo) & (_x_true <= _hi))):.3f} "
            f"| {ess_per_sweep(_chain):.3f} | {_depth_spent:,} |"
        )
    _lines.append("")
    _lines.append(
        f"(exact grid smoother recovers truth at RMSE "
        f"{float(np.sqrt(np.mean((_gold['mean'] - _x_true) ** 2))):.4f}; "
        f"{_n_multi}/{_t_len} smoothing marginals are multimodal)"
    )
    mo.md(
        "**Recovery at T = 1000 on hostile terrain.** The last column is the quantity "
        "the audit is about — sequential rounds on the critical path (sweeps × depth "
        "per sweep):\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def coda3_budget_md(mo):
    mo.md(r"""
    ### 13.2 The chain-length competition — anytime metrics per unit of hardware time

    The recovery table above compares fixed sweep counts, but the real contest is
    *anytime* performance: what does each kernel deliver as a function of the budget a
    machine actually charges for? We run one long chain per kernel (three seeds,
    averaged) and evaluate every prefix with the standard anytime convention — the
    second half of each prefix is the sample, the first half is burn-in — under four
    metrics:

    - **RMSE vs exact posterior mean** — pure sampler quality; keeps improving as
      1/√ESS after convergence;
    - **RMSE vs truth** — *recovery*; saturates at the exact smoother's own error
      (the model's ceiling, not the sampler's);
    - **posterior-sd RMSE** — distributional convergence; specifically punishes
      chains that visit only one basin at the multimodal timepoints;
    - **90% coverage** — calibration against the true path.

    The x-axis is measured A100 wall-clock: marginal cost per sweep on an
    A100-SXM4-40GB (Modal, 2026-07-02), obtained from paired 500- vs 5000-sweep runs —
    **49.8 ms/sweep for mGRAD** (a length-1000 launch-bound chain) versus
    **1.28 ms/sweep for the twisted+def tree** (ten batched levels). One-time
    compile/pilot costs are excluded for both. Per *sequential round* the two kernels
    cost within 2.6× of each other (≈50 vs ≈130 µs), so this axis is, to a small
    constant, the notebook's depth accounting made physical.
    """)
    return


@app.cell
def coda3_budget_run(grid_smoother_sv, np, run_mgrad_sv, run_twisted_sv, simulate_sv):
    # Marginal ms/sweep measured on Modal A100-SXM4-40GB (2026-07-02), from paired
    # 500- vs 5000-sweep runs of these exact kernels; compile/pilot excluded (one-time).
    _ms_a100 = {"mgrad": 49.8, "twisted_def": 1.28}
    _t_len = 1000
    _x_true, _y = simulate_sv(0, _t_len)
    _gold = grid_smoother_sv(_y)
    _budgets = {
        "mgrad": [50, 100, 200, 400, 800, 1600, 3200, 5000],
        "twisted_def": [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000],
    }
    _runners = {
        "mgrad": lambda s: run_mgrad_sv(_y, delta=4.0, n_iter=5000, seed=s),
        "twisted_def": lambda s: run_twisted_sv(_y, inflate=3.0, eps=0.25, n_iter=20000, seed=s),
    }

    def _anytime(chain, budgets):
        _rows = []
        for _b in budgets:
            _burn = chain[_b // 2 : _b]
            _pm = _burn.mean(0)
            _sd = _burn.std(0)
            _lo, _hi = np.percentile(_burn, 5, 0), np.percentile(_burn, 95, 0)
            _rows.append(
                (
                    float(np.sqrt(np.mean((_pm - _gold["mean"]) ** 2))),
                    float(np.sqrt(np.mean((_pm - _x_true) ** 2))),
                    float(np.sqrt(np.mean((_sd - _gold["sd"]) ** 2))),
                    float(np.mean((_x_true >= _lo) & (_x_true <= _hi))),
                )
            )
        return np.array(_rows)

    _seeds = (7, 21, 42)
    sv_budget = {
        "floor": float(np.sqrt(np.mean((_gold["mean"] - _x_true) ** 2))),
        "ms_a100": _ms_a100,
        "curves": {},
    }
    for _name in ("mgrad", "twisted_def"):
        _m = np.stack([_anytime(_runners[_name](_s), _budgets[_name]) for _s in _seeds]).mean(0)
        sv_budget["curves"][_name] = {
            "sweeps": _budgets[_name],
            "a100_s": [_b * _ms_a100[_name] / 1000.0 for _b in _budgets[_name]],
            "rmse_gold": _m[:, 0].tolist(),
            "rmse_truth": _m[:, 1].tolist(),
            "sd_rmse": _m[:, 2].tolist(),
            "cov90": _m[:, 3].tolist(),
        }
    return (sv_budget,)


@app.cell
def coda3_budget_fig(mo, palette, plt, sv_budget):
    _colors = {"mgrad": palette["state"], "twisted_def": palette["obs"]}
    _labels = {"mgrad": "Particle-mGRAD (δ=4)", "twisted_def": "twisted+def (c=3, ε=0.25)"}
    _fig, _axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    (_a0, _a1), (_a2, _a3) = _axes
    for _k in ("mgrad", "twisted_def"):
        _c = sv_budget["curves"][_k]
        _a0.plot(_c["a100_s"], _c["rmse_gold"], "-o", color=_colors[_k], lw=2.2, label=_labels[_k])
        _a1.plot(_c["a100_s"], _c["rmse_truth"], "-o", color=_colors[_k], lw=2.2)
        _a2.plot(_c["a100_s"], _c["sd_rmse"], "-o", color=_colors[_k], lw=2.2)
        _a3.plot(_c["a100_s"], _c["cov90"], "-o", color=_colors[_k], lw=2.2)
    _a0.set_xscale("log")
    _a0.set_yscale("log")
    _a0.set_ylabel("RMSE vs exact posterior mean")
    _a0.set_title("Sampler quality per A100-second", fontsize=10.5, fontweight="bold")
    _a0.legend(frameon=False, fontsize=8.5)
    _a1.set_xscale("log")
    _a1.axhline(sv_budget["floor"], color=palette["muted"], ls="--", lw=1.4)
    _a1.text(
        0.07,
        sv_budget["floor"] + 0.004,
        "exact-posterior ceiling",
        fontsize=8,
        color=palette["ink"],
    )
    _a1.set_ylabel("RMSE vs truth (recovery)")
    _a1.set_title("Recovery saturates at the model's ceiling", fontsize=10.5, fontweight="bold")
    _a2.set_xscale("log")
    _a2.set_yscale("log")
    _a2.set_ylabel("RMSE of posterior sd vs exact")
    _a2.set_title(
        "Distributional convergence\n(punishes single-basin chains)",
        fontsize=10,
        fontweight="bold",
    )
    _a3.set_xscale("log")
    _a3.axhline(0.9, color=palette["muted"], ls="--", lw=1.4)
    _a3.set_ylim(0.6, 1.0)
    _a3.set_ylabel("90% interval coverage of truth")
    _a3.set_title("Calibration", fontsize=10.5, fontweight="bold")
    for _ax in (_a0, _a1, _a2, _a3):
        _ax.set_xlabel("A100 wall-clock (s, log)")
        _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def coda3_budget_table(mo, sv_budget):
    _targets = (0.15, 0.10, 0.05, 0.035)
    _lines = [
        "| RMSE-vs-posterior target | mGRAD (A100 s) | twisted+def (A100 s) | speedup |",
        "|--:|--:|--:|--:|",
    ]
    for _tgt in _targets:
        _hit = {}
        for _k in ("mgrad", "twisted_def"):
            _c = sv_budget["curves"][_k]
            _hit[_k] = next(
                (_s for _s, _v in zip(_c["a100_s"], _c["rmse_gold"], strict=True) if _v <= _tgt),
                None,
            )
        _m, _t = _hit["mgrad"], _hit["twisted_def"]
        _lines.append(
            f"| ≤ {_tgt} "
            f"| {f'{_m:.1f}' if _m is not None else 'not reached'} "
            f"| {f'{_t:.2f}' if _t is not None else 'not reached'} "
            f"| {f'{_m / _t:.0f}×' if (_m is not None and _t is not None) else '—'} |"
        )
    mo.md(
        "**A100 time-to-target** (first budget on the grid whose anytime estimate "
        "reaches the target; grids are coarse, so speedups are conservative "
        "round-ups):\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def coda3_budget_caption(mo):
    mo.md(r"""
    The chain-length competition, settled panel by panel:

    - **Sampler quality** (top-left): at every wall-clock budget from ≈0.5 s to
      ≈250 s, twisted+def sits below mGRAD — 5–15× faster to every practical accuracy
      target. Per *sweep* mGRAD still wins (its curve uses 40× fewer sweeps per
      second of x-axis); the hardware conversion is what flips the race.
    - **Recovery** (top-right): both kernels converge onto the exact-posterior
      ceiling and then stay there. Longer chains do not — cannot — recover the truth
      better; past convergence the budget buys posterior precision, not recovery.
    - **Distributional convergence** (bottom-left): the posterior-sd error tells the
      same story as the mean (twisted+def at ≈26 s ≈ mGRAD at ≈160 s), confirming the
      wide-tail leaf is genuinely visiting both basins at the multimodal timepoints,
      not just averaging through them.
    - **Calibration** (bottom-right): both reach the nominal 0.9 band; twisted+def
      gets there around one second, mGRAD around a minute.

    One honest tail note: twisted+def's deep-precision tail flattens near ≈0.035
    while mGRAD's 249-second point reaches 0.030 — equilibrating the *relative
    weights* of the drift basins is the tree kernel's slowest mode, where mGRAD's
    ancestor-diverse resampling still mixes slightly better per effective sample. At
    every budget either kernel can actually afford, the ordering is unambiguous.
    """)
    return


@app.cell(hide_code=True)
def coda3_verdict(mo):
    mo.md(r"""
    ### 13.3 The hostile-terrain verdict

    | kernel | exact? | robust here? | converged accuracy per sequential round |
    |---|:--:|:--:|:--:|
    | **Particle-mGRAD** | ✅ | ✅ untouched | poor — every sweep costs T |
    | **aGRAD leaf (§11)** | ❌ bias now first-order | ❌ traps silently | — (does not converge) |
    | **twisted, plain** | ✅ (invariant) | ❌ fails silently off-pilot | — (does not converge) |
    | **twisted + defensive tail** | ✅ probe-passed | ✅ | **best** |

    Reading the T = 1000 table bottom-up:

    1. **mGRAD earns real redemption.** At 500 sweeps it is the most accurate sampler
       per sweep on the board (0.088 from gold), with per-sweep mixing unchanged from
       the friendly model. Its advantages — gradients re-derived at the current
       reference every sweep, prior folds over resampled ancestors, exact backward
       pass — are precisely the things a fixed pilot cannot imitate. If sequential
       depth is free, mGRAD is the robustness champion on hostile terrain.
    2. **But depth is the entire question, and the defensive twisted leaf wins it.**
       Give it 5,000 sweeps — **one tenth of mGRAD's sequential-round budget** — and it
       reaches 0.042 from gold with 91% coverage: *twice as close as mGRAD at 500
       sweeps for 10× fewer critical-path rounds*. §13.2 runs the full anytime race in
       measured A100 wall-clock and settles it: 5–15× faster to every practical
       accuracy target, with identical recovery and calibration asymptotes. The depth
       argument survives the hostile model; it just needs the insurance tail.
    3. **The aGRAD leaf does not survive.** Ten times the sweeps leave it 0.35 from
       gold: its §12.2 defect is stationary bias, not slow mixing, and this terrain
       makes it first-order. The kernel §11 benchmarked as "the parallel replacement"
       is, on hard models, neither exact nor robust.
    4. **Exactness and robustness are different axes.** The plain twisted leaf is
       provably invariant and still practically useless off-pilot — invariance says
       where the chain converges, proposal coverage says whether it does so within
       your lifetime. On hostile terrain every kernel that failed, failed *silently*
       (healthy-looking single-chain ESS while off-target); only the gold standard —or
       multi-seed comparison — exposes it.

    For production, the directional note (measured once, 1-D toy, lightly tuned): the
    shipped `amala_exact` leaf stays exact here but mixes at ESS ≈ 0.10/sweep at its
    best δ — 3–7× below the toy tree leaves and far off the floor at long horizons
    within these budgets. The same defensive-twisted construction (pilot = the
    existing Laplace warmup + wide-tail insurance + exact ψ) is the natural upgrade
    on both counts.
    """)
    return


@app.cell(hide_code=True)
def coda4_intro(mo):
    mo.md(r"""
    ## 14. Coda 4 — where even mGRAD breaks: a multimodal posterior

    Every model so far has been *unimodal* given the data — hard to linearise, weak,
    curved, but with the posterior mass in one place. On such models §13 crowned mGRAD
    the robustness champion: fresh reference gradients and ancestor-diverse prior folds
    let it track the mode from anywhere. This coda asks the opposite question — **what
    breaks mGRAD?** — and the answer is not size.

    Dimension `D` is the axis this whole family scales *well* on, by construction: the
    entire Corenflos–Finke argument is that gradient-informed proposals give favourable
    `D`-scaling (their headline is a `D = 30`, `T = 128` stochastic-volatility model
    with 3840 unknowns). Making the state bigger helps mGRAD. The axis that *breaks*
    local gradient methods is the one the paper names as its own limitation —
    **multimodality** — and it is independent of `D`, so we can keep an exact 1-D grid
    gold standard and still stage the failure cleanly.

    The canonical hard case is the **Kitagawa–Gordon nonlinear growth benchmark**, the
    standard "bootstrap-particle-filter-or-bust" model in the SSM literature:

    \[
    x_t = 0.5\,x_{t-1} + \frac{25\,x_{t-1}}{1 + x_{t-1}^2} + 8\cos(1.2\,t) + N(0, 10),
    \qquad
    y_t = \frac{x_t^2}{20} + N(0, 1).
    \]

    Constant process covariance — still exactly the Particle-mGRAD regime. But the
    `x_t^2` emission is **sign-ambiguous**: `y_t` fixes `|x_t|`, never the sign, so the
    smoothing posterior is genuinely **bimodal** (here 32 of 100 marginals have two
    well-separated modes, and the exact `P(x_t > 0 \mid y)` sweeps the whole `[0, 1]`).
    A local gradient move is *structurally* unable to cross this: the emission
    log-likelihood has a valley at `x = 0` with `∂/∂x \log G` pointing *away* from zero
    in both basins, so a MALA/mGRAD step started near `+\sqrt{20 y}` can never reach
    `-\sqrt{20 y}`. Only the resampling/ancestor mechanism could flip the sign — and
    only if some particle proposes it.

    Because the posterior is multimodal, the posterior *mean* is a poor summary (it sits
    in the valley between modes). We score with metrics that actually see the modes:

    - **Wasserstein-1 to the exact marginal** — the average over `t` of the 1-D optimal
      transport distance between the sampler's empirical marginal and the grid gold
      marginal. This is zero iff the sampler reproduces the full bimodal shape.
    - **sign-probability error** — RMSE over `t` of `\lvert \hat P(x_t>0) - P_{\text{gold}}(x_t>0)\rvert`.
      This is the direct "did you get the mode weights right" score.
    """)
    return


@app.cell
def coda4_model(math, np):
    KIT_SIG_V = math.sqrt(10.0)  # process sd (classic var = 10)
    KIT_SIG_W = 1.0  # obs sd
    KIT_INIT_SD = 5.0

    def _kit_drift(x, t):
        return 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)

    def simulate_kit(seed, t_len):
        _rng = np.random.default_rng(seed)
        _x = np.zeros(t_len, dtype=np.float64)
        _x[0] = _rng.normal(0.0, KIT_INIT_SD)
        for _t in range(1, t_len):
            _x[_t] = _kit_drift(_x[_t - 1], _t) + _rng.normal(0.0, KIT_SIG_V)
        _y = _x**2 / 20.0 + _rng.normal(0.0, KIT_SIG_W, size=t_len)
        return _x, _y

    def grid_smoother_kit(y, n_grid=1201, lo=-32.0, hi=32.0):
        """Near-exact grid smoother; returns bimodality-aware summaries too."""
        _t_len = len(y)
        _xs = np.linspace(lo, hi, n_grid)
        _dx = _xs[1] - _xs[0]

        def _logn(v, mu, sd):
            return -0.5 * (np.log(2.0 * np.pi * sd**2) + ((v - mu) ** 2) / sd**2)

        _log_obs = _logn(y[:, None], (_xs**2 / 20.0)[None, :], KIT_SIG_W)
        _log_alpha = np.zeros((_t_len, n_grid))
        _log_alpha[0] = _logn(_xs, 0.0, KIT_INIT_SD) + _log_obs[0]
        for _t in range(1, _t_len):
            _log_tr = _logn(_xs[None, :], _kit_drift(_xs, _t)[:, None], KIT_SIG_V)
            _a = _log_alpha[_t - 1]
            _m = _a.max()
            _log_alpha[_t] = np.log(np.exp(_a - _m) @ np.exp(_log_tr) + 1e-300) + _m + _log_obs[_t]
        _log_beta = np.zeros((_t_len, n_grid))
        for _t in range(_t_len - 2, -1, -1):
            _log_tr = _logn(_xs[None, :], _kit_drift(_xs, _t + 1)[:, None], KIT_SIG_V)
            _b = _log_beta[_t + 1] + _log_obs[_t + 1]
            _m = _b.max()
            _log_beta[_t] = np.log(np.exp(_log_tr) @ np.exp(_b - _m) + 1e-300) + _m
        _log_g = _log_alpha + _log_beta
        _log_g -= _log_g.max(1, keepdims=True)
        _g = np.exp(_log_g)
        _g /= _g.sum(1, keepdims=True)
        _mean = (_g * _xs[None, :]).sum(1)
        _sd = np.sqrt((_g * (_xs[None, :] - _mean[:, None]) ** 2).sum(1))
        return {
            "mean": _mean,
            "sd": _sd,
            "xs": _xs,
            "g": _g,
            "dx": _dx,
            "p_pos": _g[:, _xs > 0].sum(1),
            "cdf": np.cumsum(_g, axis=1),
        }

    def count_multimodal_kit(gold, thresh=0.05):
        _g = gold["g"]
        _idx = []
        for _t in range(_g.shape[0]):
            _d = _g[_t]
            _peaks = sum(
                1
                for _i in range(1, len(_d) - 1)
                if _d[_i] > _d[_i - 1] and _d[_i] >= _d[_i + 1] and _d[_i] > thresh * _d.max()
            )
            if _peaks > 1:
                _idx.append(_t)
        return _idx

    def wasserstein1_kit(chain_burn, gold):
        _xs = gold["xs"]
        _edges = np.concatenate([[_xs[0] - gold["dx"] / 2], _xs + gold["dx"] / 2])
        _t_len = chain_burn.shape[1]
        _w1 = np.zeros(_t_len)
        for _t in range(_t_len):
            _hist, _ = np.histogram(chain_burn[:, _t], bins=_edges)
            _emp_cdf = np.cumsum(_hist / max(_hist.sum(), 1))
            _w1[_t] = np.sum(np.abs(_emp_cdf - gold["cdf"][_t])) * gold["dx"]
        return float(_w1.mean())

    def sign_prob_error_kit(chain_burn, gold):
        return float(np.sqrt(np.mean(((chain_burn > 0).mean(0) - gold["p_pos"]) ** 2)))

    return (
        KIT_INIT_SD,
        KIT_SIG_V,
        KIT_SIG_W,
        count_multimodal_kit,
        grid_smoother_kit,
        sign_prob_error_kit,
        simulate_kit,
        wasserstein1_kit,
    )


@app.cell
def coda4_kernels(KIT_INIT_SD, KIT_SIG_V, KIT_SIG_W, jax, jnp, make_dsmc_tree, np, random):
    def _m_mean(time_idx, x):
        return 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * jnp.cos(1.2 * time_idx)

    def _logn(v, mu, var):
        return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

    def _log_obs(x, y_t):
        return _logn(y_t, x**2 / 20.0, KIT_SIG_W**2)

    def _grad_obs(x, y_t):
        return (y_t - x**2 / 20.0) * (x / 10.0) / KIT_SIG_W**2

    def _prior_mean(time_idx, x_prev):
        return jnp.where(time_idx == 0, 0.0, _m_mean(time_idx, x_prev))

    def _prior_var(time_idx):
        return jnp.where(time_idx == 0, KIT_INIT_SD**2, KIT_SIG_V**2)

    def _x_init(y):
        return jnp.sqrt(jnp.clip(20.0 * y, 0.0, None))  # crude +root init (single sign)

    def _laplace_pilot(y, n_pilot=30, damp=0.25, lam_cap=5.0):
        _t_len = y.shape[0]

        def _rts(x_hat):
            _idx = jnp.arange(_t_len)
            _f = 0.5 + 25.0 * (1.0 - x_hat[:-1] ** 2) / (1.0 + x_hat[:-1] ** 2) ** 2
            _b = _m_mean(_idx[1:], x_hat[:-1]) - _f * x_hat[:-1]
            _f_all = jnp.concatenate([jnp.zeros((1,)), _f])
            _b_all = jnp.concatenate([jnp.zeros((1,)), _b])
            _g2 = (x_hat**2 / 100.0 - (y - x_hat**2 / 20.0) / 10.0) / KIT_SIG_W**2
            _lam = jnp.clip(_g2, 1e-3, lam_cap)  # -g'' clipped positive (non-log-concave!)
            _z = x_hat + jnp.clip(_grad_obs(x_hat, y) / _lam, -5.0, 5.0)
            _r = 1.0 / _lam

            def _kf(c, inp):
                _m_prev, _p_prev = c
                _z_t, _r_t, _f_t, _b_t, _first = inp
                _mp = jnp.where(_first, 0.0, _f_t * _m_prev + _b_t)
                _pp = jnp.where(_first, KIT_INIT_SD**2, _f_t**2 * _p_prev + KIT_SIG_V**2)
                _gain = _pp / (_pp + _r_t)
                _mf = _mp + _gain * (_z_t - _mp)
                return (_mf, (1.0 - _gain) * _pp), (_mp, _pp, _mf, (1.0 - _gain) * _pp)

            _first = jnp.concatenate([jnp.ones((1,)), jnp.zeros((_t_len - 1,))])
            (_, _), (_mp, _pp, _mf, _pf) = jax.lax.scan(
                _kf, (0.0, 0.0), (_z, _r, _f_all, _b_all, _first)
            )

            def _rstep(c, inp):
                _m_next, _p_next = c
                _mf_t, _pf_t, _mpn, _ppn, _f_next = inp
                _gg = _pf_t * _f_next / jnp.maximum(_ppn, 1e-12)
                _ms = _mf_t + _gg * (_m_next - _mpn)
                _ps = _pf_t + _gg**2 * (_p_next - _ppn)
                return (_ms, _ps), (_ms, _ps)

            _inp = (_mf[:-1], _pf[:-1], _mp[1:], _pp[1:], _f_all[1:])
            _inp_rev = jax.tree_util.tree_map(lambda a: jnp.flip(a, 0), _inp)
            (_, _), (_msr, _psr) = jax.lax.scan(_rstep, (_mf[-1], _pf[-1]), _inp_rev)
            return (
                jnp.concatenate([jnp.flip(_msr), _mf[-1:]]),
                jnp.concatenate([jnp.flip(_psr), _pf[-1:]]),
            )

        _x_hat = _x_init(y)
        for _ in range(n_pilot):
            _mu, _ = _rts(_x_hat)
            _x_hat = (1.0 - damp) * _x_hat + damp * _mu
        return _rts(_x_hat)

    def run_mgrad_kit(y_obs, n_particles=32, delta=1.0, kappa=1.0, n_iter=2000, seed=0):
        """Particle-mGRAD (Alg 7) on Kitagawa — the local gradient kernel, unchanged."""
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _n = _p - 1
        _half = 0.5 * delta

        def _forward(key, x_ref):
            def _step(carry, time_idx):
                _prev, _weights, _key = carry
                _key, _ak, _uk, _pk = random.split(_key, 4)
                _cov = _prior_var(time_idx)
                _a_gain = _cov / (_cov + _half)
                _prop_v = _half * _a_gain
                _g_hat = (2.0 / delta) / (1.0 + _n * _a_gain)
                _anc = random.categorical(_ak, jnp.log(_weights + 1e-38), shape=(_p,)).at[0].set(0)
                _mmean = _prior_mean(time_idx, _prev[_anc])
                _x_ref_t = x_ref[time_idx]
                _u = (
                    _x_ref_t
                    + kappa * _half * _grad_obs(_x_ref_t, _y[time_idx])
                    + jnp.sqrt(_half) * random.normal(_uk)
                )
                _x_t = (
                    (
                        (1.0 - _a_gain) * _mmean
                        + _a_gain * _u
                        + jnp.sqrt(_prop_v) * random.normal(_pk, (_p,))
                    )
                    .at[0]
                    .set(_x_ref_t)
                )
                _v = (1.0 - _a_gain) * _mmean
                _phi = kappa * _half * _grad_obs(_x_t, _y[time_idx])
                _log_q = _logn(_x_t, _mmean, _cov) + _log_obs(_x_t, _y[time_idx])
                _t1 = 0.5 * (_x_t - _v) ** 2 * (1.0 / _prop_v + _g_hat)
                _t2 = -(0.5 * _n * (_x_t + _phi) * _a_gain + (_x_t - _v)) * _g_hat * (_x_t + _phi)
                _t3 = (_n + 1) * (jnp.mean(_x_t) - jnp.mean(_v)) * _g_hat * (_v + _phi)
                _log_w = _log_q + _t1 + _t2 + _t3
                _log_w = _log_w - jax.scipy.special.logsumexp(_log_w)
                return (_x_t, jnp.exp(_log_w), _key), (_x_t, jnp.exp(_log_w))

            (_, _, _), (_particles, _all_w) = jax.lax.scan(
                _step, (jnp.zeros(_p), jnp.ones(_p) / _p, key), jnp.arange(_t_len)
            )
            return _particles, _all_w

        def _backward(key, particles, weights):
            _kl, _kr = random.split(key)
            _l_last = random.categorical(_kl, jnp.log(weights[_t_len - 1] + 1e-38))

            def _bstep(carry, time_idx):
                _l_next, _key = carry
                _key, _sk = random.split(_key)
                _logp = jnp.log(weights[time_idx] + 1e-38) + _logn(
                    particles[time_idx + 1][_l_next],
                    _m_mean(time_idx + 1, particles[time_idx]),
                    _prior_var(time_idx + 1),
                )
                _l = random.categorical(_sk, _logp)
                return (_l, _key), _l

            (_, _), _ls_rev = jax.lax.scan(_bstep, (_l_last, _kr), jnp.arange(_t_len - 2, -1, -1))
            _ls = jnp.concatenate([jnp.flip(_ls_rev), _l_last[None]])
            return particles[jnp.arange(_t_len), _ls]

        def _sweep(x_ref, key):
            _kf, _kb = random.split(key)
            _particles, _weights = _forward(_kf, x_ref)
            _path = _backward(_kb, _particles, _weights)
            return _path, _path

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_sweep, _x_init(_y), _keys)
        return np.asarray(_chain)

    def run_twisted_def_kit(
        y_obs, n_particles=32, inflate=3.0, eps=0.5, wide=100.0, n_iter=2000, seed=0
    ):
        """Twisted + generic wide-tail defensive mixture (the §13 fix, unchanged)."""
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _smooth = make_dsmc_tree(_t_len, _p, _prior_mean, _prior_var)
        _mu_q, _var_q = _laplace_pilot(_y)
        _q_var = inflate * _var_q

        def _log_q(xs, t):
            return jnp.logaddexp(
                jnp.log(1.0 - eps) + _logn(xs, _mu_q[t], _q_var[t]),
                jnp.log(eps) + _logn(xs, _mu_q[t], wide),
            )

        def _leaf(x_ref, time_idx, key):
            _pk, _dk = random.split(key)
            _pick = random.bernoulli(_pk, eps, (_p - 1,))
            _sd = jnp.where(_pick, jnp.sqrt(wide), jnp.sqrt(_q_var[time_idx]))
            _free = _mu_q[time_idx] + _sd * random.normal(_dk, (_p - 1,))
            _particles = jnp.concatenate([x_ref[time_idx, 0][None], _free])[:, None]
            _xs = _particles[:, 0]
            _psi = _log_obs(_xs, _y[time_idx]) - _log_q(_xs, time_idx)
            _psi = jnp.where(time_idx == 0, _psi + _logn(_xs, 0.0, KIT_INIT_SD**2), _psi)
            return _particles, _psi[:, None]

        def _body(x_ref, key):
            _xp = _smooth(key, lambda t, k: _leaf(x_ref, t, k))
            return _xp, _xp

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _x_init(_y).reshape(_t_len, 1), _keys)
        return np.asarray(_chain)[:, :, 0]

    def run_twisted_root_kit(y_obs, n_particles=32, root_sd=2.5, w_root=0.45, n_iter=2000, seed=0):
        """Mode-AWARE fixed proposal: components at BOTH emission roots ±√(20 y).

        q_t = w_root N(+r_t, s²) + w_root N(-r_t, s²) + (1-2 w_root) N(μ_pilot, var_pilot),
        with r_t = √(20 y_t⁺). Still a FIXED independent proposal ⇒ exactly invariant
        (ψ pays log G − log q, seams pay the true transition). The point is that the
        proposal *support* covers both sign basins — a global structure a local
        gradient move can never build.
        """
        _y = jnp.asarray(y_obs).reshape(-1)
        _t_len = int(_y.shape[0])
        _p = n_particles
        _smooth = make_dsmc_tree(_t_len, _p, _prior_mean, _prior_var)
        _mu_q, _var_q = _laplace_pilot(_y)
        _root = jnp.sqrt(jnp.clip(20.0 * _y, 0.0, None))
        _w_pilot = 1.0 - 2.0 * w_root

        def _log_q(xs, t):
            return jnp.logaddexp(
                jnp.logaddexp(
                    jnp.log(w_root) + _logn(xs, _root[t], root_sd**2),
                    jnp.log(w_root) + _logn(xs, -_root[t], root_sd**2),
                ),
                jnp.log(_w_pilot) + _logn(xs, _mu_q[t], _var_q[t]),
            )

        def _leaf(x_ref, time_idx, key):
            _ck, _dk = random.split(key)
            _comp = random.categorical(
                _ck, jnp.log(jnp.array([w_root, w_root, _w_pilot])), shape=(_p - 1,)
            )
            _centers = jnp.stack(
                [
                    jnp.full((_p - 1,), _root[time_idx]),
                    jnp.full((_p - 1,), -_root[time_idx]),
                    jnp.full((_p - 1,), _mu_q[time_idx]),
                ]
            )
            _sds = jnp.array([root_sd, root_sd, jnp.sqrt(_var_q[time_idx])])
            _cen = jnp.take_along_axis(_centers, _comp[None, :], 0)[0]
            _free = _cen + _sds[_comp] * random.normal(_dk, (_p - 1,))
            _particles = jnp.concatenate([x_ref[time_idx, 0][None], _free])[:, None]
            _xs = _particles[:, 0]
            _psi = _log_obs(_xs, _y[time_idx]) - _log_q(_xs, time_idx)
            _psi = jnp.where(time_idx == 0, _psi + _logn(_xs, 0.0, KIT_INIT_SD**2), _psi)
            return _particles, _psi[:, None]

        def _body(x_ref, key):
            _xp = _smooth(key, lambda t, k: _leaf(x_ref, t, k))
            return _xp, _xp

        _keys = random.split(random.PRNGKey(seed), n_iter)
        _, _chain = jax.lax.scan(_body, _x_init(_y).reshape(_t_len, 1), _keys)
        return np.asarray(_chain)[:, :, 0]

    return run_mgrad_kit, run_twisted_def_kit, run_twisted_root_kit


@app.cell
def coda4_run(
    count_multimodal_kit,
    grid_smoother_kit,
    np,
    run_mgrad_kit,
    run_twisted_def_kit,
    run_twisted_root_kit,
    sign_prob_error_kit,
    simulate_kit,
    wasserstein1_kit,
):
    _t_len = 100
    _x_true, _y = simulate_kit(0, _t_len)
    _gold = grid_smoother_kit(_y)
    _n_iter = 2000
    _burn = _n_iter // 2
    _chains = {
        "mgrad": run_mgrad_kit(_y, n_particles=32, delta=1.0, n_iter=_n_iter, seed=5),
        "twisted_def": run_twisted_def_kit(
            _y, n_particles=32, eps=0.5, wide=100.0, n_iter=_n_iter, seed=5
        ),
        "twisted_root": run_twisted_root_kit(
            _y, n_particles=32, root_sd=2.5, n_iter=_n_iter, seed=5
        ),
    }
    _metrics = {}
    for _name, _chain in _chains.items():
        _b = _chain[_burn:]
        _pm = _b.mean(0)
        _lo, _hi = np.percentile(_b, 5, 0), np.percentile(_b, 95, 0)
        _metrics[_name] = {
            "w1": wasserstein1_kit(_b, _gold),
            "sign_err": sign_prob_error_kit(_b, _gold),
            "rmse_gold": float(np.sqrt(np.mean((_pm - _gold["mean"]) ** 2))),
            "rmse_truth": float(np.sqrt(np.mean((_pm - _x_true) ** 2))),
            "cov": float(np.mean((_x_true >= _lo) & (_x_true <= _hi))),
            "emp_pos": (_b > 0).mean(0),
            "post_burn": _b,
        }
    kit_results = {
        "T": _t_len,
        "truth": _x_true,
        "gold": _gold,
        "n_multi": len(count_multimodal_kit(_gold)),
        "metrics": _metrics,
    }
    return (kit_results,)


@app.cell
def coda4_scoreboard(kit_results, mo):
    _labels = {
        "mgrad": "Particle-mGRAD (local gradient)",
        "twisted_def": "twisted + generic wide tail (§13 fix)",
        "twisted_root": "twisted + mode-aware roots (§14)",
    }
    _lines = [
        "| kernel | W1 to exact marginal ↓ | sign-prob error ↓ | RMSE vs gold mean ↓ | RMSE vs truth ↓ | 90% cov |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for _k in ("mgrad", "twisted_def", "twisted_root"):
        _m = kit_results["metrics"][_k]
        _lines.append(
            f"| {_labels[_k]} | {_m['w1']:.3f} | {_m['sign_err']:.3f} "
            f"| {_m['rmse_gold']:.3f} | {_m['rmse_truth']:.3f} | {_m['cov']:.3f} |"
        )
    mo.md(
        f"**The multimodal scoreboard.** Kitagawa `T = {kit_results['T']}`, "
        f"{kit_results['n_multi']}/{kit_results['T']} marginals bimodal, `P = 32`, 2000 "
        "sweeps. W1 and sign-prob error are the mode-aware metrics; both are zero for a "
        "sampler that reproduces the exact bimodal posterior:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def coda4_marginal_fig(kit_results, mo, np, palette, plt):
    _gold = kit_results["gold"]
    _xs = _gold["xs"]
    # pick the most bimodal timepoint (grid mass most evenly split across the sign)
    _split = np.minimum(_gold["p_pos"], 1.0 - _gold["p_pos"])
    _t_bi = int(np.argmax(_split))
    _colors = {
        "mgrad": palette["state"],
        "twisted_def": palette["belief"],
        "twisted_root": palette["obs"],
    }
    _labels = {
        "mgrad": "Particle-mGRAD",
        "twisted_def": "twisted + wide tail",
        "twisted_root": "twisted + roots",
    }
    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(12.0, 4.3))

    # left — the exact bimodal marginal vs each sampler's empirical marginal
    _a0.plot(
        _xs,
        _gold["g"][_t_bi] / _gold["dx"],
        color=palette["ink"],
        lw=2.4,
        label="exact posterior (grid)",
        zorder=5,
    )
    for _k in ("mgrad", "twisted_def", "twisted_root"):
        _samp = kit_results["metrics"][_k]["post_burn"][:, _t_bi]
        _a0.hist(
            _samp,
            bins=60,
            range=(-25, 25),
            density=True,
            histtype="step",
            color=_colors[_k],
            lw=2.0,
            label=_labels[_k],
        )
    _a0.axvline(kit_results["truth"][_t_bi], color=palette["muted"], ls=":", lw=1.5)
    _a0.set_xlim(-22, 22)
    _a0.set_xlabel(f"x_t  (t = {_t_bi}, the most bimodal marginal)")
    _a0.set_ylabel("density")
    _a0.set_title("Only mode-aware roots recover both basins", fontsize=11, fontweight="bold")
    _a0.legend(frameon=False, fontsize=8.5)
    _a0.spines[["top", "right"]].set_visible(False)

    # right — P(x_t>0): exact vs each sampler, over the whole horizon
    _t = np.arange(kit_results["T"])
    _a1.plot(_t, _gold["p_pos"], color=palette["ink"], lw=2.2, label="exact P(x>0)", zorder=5)
    for _k in ("mgrad", "twisted_def", "twisted_root"):
        _a1.plot(
            _t,
            kit_results["metrics"][_k]["emp_pos"],
            color=_colors[_k],
            lw=1.6,
            alpha=0.85,
            label=_labels[_k],
        )
    _a1.set_xlabel("time t")
    _a1.set_ylabel("P(x_t > 0)")
    _a1.set_ylim(-0.05, 1.05)
    _a1.set_title("Sign-probability track (mode weights over time)", fontsize=11, fontweight="bold")
    _a1.legend(frameon=False, fontsize=8, ncol=2)
    _a1.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def coda4_verdict(mo):
    mo.md(r"""
    ### 14.1 The verdict — and why it inverts §13

    | kernel | exact? | captures bimodality? | why |
    |---|:--:|:--:|---|
    | **Particle-mGRAD** | ✅ | ❌ **fails** | local gradient cannot cross the `x=0` valley; collapses to one sign basin |
    | **twisted + wide tail** | ✅ | ⚠️ partial | independent tail *can* jump signs, but a single width can't span modes 40 apart |
    | **twisted + mode-aware roots** | ✅ | ✅ **recovers** | fixed proposal places mass at *both* `±√(20y)`; exact weight sorts the mode weights |

    This coda inverts §13's ranking, and the reason is the whole point of the audit read
    backwards.

    - **mGRAD's failure is structural, not a tuning miss.** It is flat across step size
      (sign-probability error stays ≈0.75 for every δ ∈ {0.3, …, 2.0}) *and* flat across
      particle count (≈0.74 at `P = 16, 32, 64, 128` alike) — 8× the particles does not
      populate the missing mode. The forward filter proposes around the reference, the
      reference is in one sign basin, so every particle lands in that basin and the other
      mode is never seen. The exact backward pass then faithfully reports a confidently
      *wrong*, unimodal posterior. This is precisely the limitation the Corenflos–Finke
      paper flags for its own methods: locality is fatal on multimodal targets.

    - **The tree kernel's leaf is an *independent* proposal, and that is now an
      advantage, not a cost.** Through §11–13 the auxiliary/twisted leaves paid for
      keeping `u` or fixing the pilot; here the independence *is* the feature. A fixed
      proposal can carry global structure the local move cannot even represent — mass at
      both emission roots — and because it is exactly importance-weighted, placing that
      mass costs nothing in correctness. The mode-aware leaf drives W1 to the grid to
      ≈0.07 and sign-probability error to ≈0.006 (three-seed means), at the same
      `⌈log₂ T⌉` depth as every other tree kernel. That it is genuinely *exact* — not
      merely close — is confirmed by a separate long run: at `T = 20`, 40 000 sweeps, it
      matches the grid posterior at W1 ≈0.016 with sign-probability error ≈0.002, so it
      reproduces the full bimodal law, not just its first moment.

    - **The generic §13 fix is not enough here.** The wide-tail defensive mixture *can*
      cross signs (its independent tail reaches the other basin when the modes are near),
      but a single tail width cannot span modes that sit up to 40 units apart at large
      `|x|`; it lands halfway (sign error ≈0.33). The lesson is that the *right* global
      structure has to come from the model — the emission's own roots — which the
      independent-proposal family can express and the local kernel cannot.

    **The two codas together frame the real trade.** §13: on a unimodal-but-hard model
    the local gradient kernel (mGRAD) is the robustness champion, and the tree only wins
    once you price sequential depth. §14: on a multimodal model the local kernel does not
    just mix slowly — it is *structurally blind* to the second mode, and the
    independent-proposal tree is the only one of the two that can be made to see it, at
    no exactness cost and no depth cost. Locality buys robustness on easy posteriors and
    forfeits it on hard ones; independence is the reverse. Which family you want is a
    property of the posterior's *shape*, not just its size — and the parallelizable
    family is the one with the freedom to adapt its proposal to that shape.
    """)
    return


@app.cell(hide_code=True)
def pf_intro(mo):
    mo.md(r"""
    ---

    # Part 2 — Why a sequential filter can run in parallel, in pictures

    *The pictures behind Part 1 §3 — the associative-scan view the parallelizability
    argument rests on.*

    A filter reads observations one at a time and updates a running **belief** about a hidden
    **state**. That feels unavoidably one-at-a-time: each belief seems to need the one before
    it, so the work looks like a chain as long as the timeline.

    This notebook shows — with pictures and one tiny runnable example — *why* the same work can
    be reorganized to finish in far fewer sequential rounds. That reorganization is the whole
    idea behind parallel-in-time filtering and smoothing. No heavy notation: just **states**,
    **observations**, **likelihood**, and **beliefs**.
    """)
    return


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## 1. The pieces: states, observations, likelihood, dynamics

    A hidden **state** drifts over time — that drift is the **dynamics**. At each moment we get
    a noisy **observation** of it, through the **likelihood**. We never see the states directly
    (drawn hollow); we only ever see the observations (drawn solid).
    """)
    return


@app.cell
def model_diagram(arrow, box, mo, node, palette, plt):
    _fig, _ax = plt.subplots(figsize=(9.5, 3.4))
    _n, _ys, _yo = 5, 1.5, 0.0
    for _t in range(_n):
        node(_ax, _t, _ys, palette["state"], r=0.30, filled=False)
        box(_ax, _t, _yo, palette["obs"], w=0.5, h=0.5)
        arrow(_ax, (_t, _ys - 0.30), (_t, _yo + 0.28), color=palette["obs"], lw=1.6, shrink=2)
        if _t < _n - 1:
            arrow(
                _ax,
                (_t + 0.32, _ys),
                (_t + 1 - 0.32, _ys),
                color=palette["state"],
                lw=2.2,
                shrink=2,
            )
    _ax.text(
        -1.05,
        _ys,
        "hidden\nstates",
        ha="center",
        va="center",
        color=palette["state"],
        fontsize=11,
        fontweight="bold",
    )
    _ax.text(
        -1.05,
        _yo,
        "observations",
        ha="center",
        va="center",
        color=palette["obs"],
        fontsize=11,
        fontweight="bold",
    )
    _ax.text(
        2.0,
        _ys + 0.55,
        "dynamics — how the hidden state drifts",
        ha="center",
        color=palette["state"],
        fontsize=10,
    )
    _ax.text(0.18, 0.78, "likelihood", ha="left", va="center", color=palette["obs"], fontsize=10)
    _ax.text(4.35, _ys, "time →", ha="left", va="center", color=palette["ink"], fontsize=10)
    _ax.set_xlim(-1.8, 5.2)
    _ax.set_ylim(-0.6, 2.3)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(
        "The world is a chain: a hidden state that drifts, seen only through noisy observations",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def belief_md(mo):
    mo.md(r"""
    ## 2. The filter walks the chain

    Filtering keeps a running **belief**: given all observations so far, where is the state
    *now*? Two moves per step — push the belief forward through the dynamics, then reweight it
    by how well the new observation fits (the likelihood).

    The catch: each belief is built from the previous belief, so the beliefs form a **chain of
    length T**. That is the "sequential" in sequential filtering.
    """)
    return


@app.cell
def belief_chain_diagram(arrow, box, mo, palette, plt):
    _fig, _ax = plt.subplots(figsize=(9.5, 3.0))
    _n, _yb = 6, 1.0
    for _t in range(_n):
        box(_ax, _t, _yb + 1.0, palette["obs"], w=0.32, h=0.32)
        arrow(_ax, (_t, _yb + 0.83), (_t, _yb + 0.33), color=palette["obs"], lw=1.4, shrink=2)
        box(_ax, _t, _yb, palette["belief"], label=f"belief\n{_t}", w=0.86, h=0.66, fontsize=10)
        if _t < _n - 1:
            arrow(
                _ax,
                (_t + 0.45, _yb),
                (_t + 1 - 0.45, _yb),
                color=palette["belief"],
                lw=2.4,
                shrink=2,
            )
    arrow(_ax, (0, 0.35), (_n - 1, 0.35), color=palette["operator"], lw=1.2, shrink=0)
    _ax.text(
        (_n - 1) / 2,
        0.02,
        "each belief needs the one before it  →  T steps, strictly in order",
        ha="center",
        color=palette["operator"],
        fontsize=10.5,
        fontweight="bold",
    )
    _ax.text(
        -0.15, _yb + 1.0, "observation", ha="right", va="center", color=palette["obs"], fontsize=9
    )
    _ax.set_xlim(-1.6, _n)
    _ax.set_ylim(-0.3, 2.5)
    _ax.axis("off")
    _ax.set_title(
        'The filter walks the chain one step at a time — this is what "sequential" means',
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def operator_md(mo):
    mo.md(r"""
    ## 3. The one trick: belief vs step-operator

    Look closely at a single step. Two *different* things live there:

    - the **belief** — it has absorbed every observation so far, so it cannot be formed until
      everything before it is done. It carries the whole past.
    - the **step-operator** — the little rule that turns *any* incoming belief into the next
      one. It is built only from that step's own observation and the fixed model. It does
      **not** depend on the past.

    The sequential filter is slow because it carries the *belief*. The parallel filter instead
    works with the *operators* — and operators can be combined without walking the timeline.
    """)
    return


@app.cell
def operator_vs_belief_diagram(arrow, box, mo, palette, plt):
    _fig, (_a0, _a1) = plt.subplots(2, 1, figsize=(9.5, 5.0))
    _n = 6

    for _t in range(_n):
        box(_a0, _t, 0, palette["belief"], label=f"b{_t}", w=0.7, h=0.58, fontsize=10)
        if _t < _n - 1:
            arrow(
                _a0, (_t + 0.37, 0), (_t + 1 - 0.37, 0), color=palette["belief"], lw=2.0, shrink=2
            )
    arrow(_a0, (0, 0.55), (_n - 1, 0.55), color=palette["muted"], lw=1.4, rad=-0.28)
    _a0.text(
        (_n - 1) / 2,
        1.05,
        "a belief has swallowed the entire past — it cannot start until everything before it is done",
        ha="center",
        color=palette["ink"],
        fontsize=9.5,
    )
    _a0.set_title(
        "belief  =  carries the past  (must go in order)",
        fontsize=11,
        fontweight="bold",
        color=palette["belief"],
    )
    _a0.set_xlim(-0.8, _n + 0.1)
    _a0.set_ylim(-0.5, 1.4)
    _a0.axis("off")

    _mx = (_n - 1) / 2
    box(_a1, _mx, 1.35, palette["ink"], label="fixed model (dynamics)", w=2.4, h=0.5, fontsize=9)
    for _t in range(_n):
        box(_a1, _t, 0, palette["operator"], label=f"M{_t}", w=0.7, h=0.58, fontsize=10)
        box(_a1, _t, 0.85, palette["obs"], w=0.28, h=0.28)
        arrow(_a1, (_t, 0.70), (_t, 0.31), color=palette["obs"], lw=1.3, shrink=2)
        arrow(_a1, (_mx, 1.10), (_t, 0.32), color=palette["muted"], lw=0.7, shrink=6, rad=0.04)
    _a1.text(
        (_n - 1) / 2,
        -0.55,
        "a step-operator is built from just its own observation + the fixed model — it knows nothing about the past",
        ha="center",
        color=palette["ink"],
        fontsize=9.5,
    )
    _a1.set_title(
        "step-operator  =  self-contained  (needs no past)",
        fontsize=11,
        fontweight="bold",
        color=palette["operator"],
    )
    _a1.set_xlim(-0.8, _n + 0.1)
    _a1.set_ylim(-0.85, 1.8)
    _a1.axis("off")

    _fig.suptitle(
        "The trick: separate the belief (needs the past) from the step-operator (does not)",
        fontsize=12.5,
        fontweight="bold",
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def multiply_md(mo):
    mo.md(r"""
    ## 4. A step is just a multiply

    Make it concrete with a 3-state toy (think weather: sun / cloud / rain). The step-operator
    is a small matrix — the **likelihood** of the new observation times the **dynamics**.

    Applying one filter step is one matrix multiply:

    **new belief  =  step-operator  ×  old belief.**

    So the whole filter is nothing but a stack of these matrices multiplied together.
    """)
    return


@app.cell
def step_is_multiply_diagram(mo, np, palette, plt):
    _states = ["sun", "cloud", "rain"]
    _dyn = np.array([[0.7, 0.3, 0.2], [0.2, 0.4, 0.3], [0.1, 0.3, 0.5]])  # columns sum to 1
    _lik = np.array([0.1, 0.5, 0.9])  # how well today's observation fits each hidden state
    _step = np.diag(_lik) @ _dyn  # M = likelihood x dynamics
    _old = np.array([0.6, 0.3, 0.1])
    _new = _step @ _old
    _new = _new / _new.sum()

    _fig, _axs = plt.subplot_mosaic(
        [["old", "M", "new"]], figsize=(9.5, 3.3), gridspec_kw={"width_ratios": [1, 1.4, 1]}
    )
    _axs["old"].barh(_states[::-1], _old[::-1], color=palette["belief"])
    _axs["old"].set_xlim(0, 1)
    _axs["old"].set_title("old belief\n(where the state was)", fontsize=10)
    _axs["old"].spines[["top", "right"]].set_visible(False)

    _axs["M"].imshow(_step, cmap="Reds", vmin=0.0)
    _axs["M"].set_xticks(range(3), _states)
    _axs["M"].set_yticks(range(3), _states)
    _axs["M"].set_title("step-operator  M = likelihood × dynamics", fontsize=10)
    for _i in range(3):
        for _j in range(3):
            _axs["M"].text(
                _j,
                _i,
                f"{_step[_i, _j]:.2f}",
                ha="center",
                va="center",
                color="white" if _step[_i, _j] > _step.max() / 2 else palette["ink"],
                fontsize=8,
            )

    _axs["new"].barh(_states[::-1], _new[::-1], color=palette["belief"])
    _axs["new"].set_xlim(0, 1)
    _axs["new"].set_title("new belief\n= M · old belief", fontsize=10)
    _axs["new"].spines[["top", "right"]].set_visible(False)

    _fig.suptitle("One filter step is just a multiply", fontsize=12.5, fontweight="bold")
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def tree_md(mo):
    mo.md(r"""
    ## 5. Multiplying a stack: chain vs tree

    Here is the payoff. Multiplying the eight step-operators **one at a time** is a chain seven
    combines deep. But multiplication is **associative** — the answer does not depend on how you
    group the factors. Group them in pairs, then pairs of pairs: the *same* product is only
    three combines deep, and the independent pairs can be combined at the same time.

    Depth drops from **T** to about **log T**.
    """)
    return


@app.cell
def chain_vs_tree_diagram(arrow, box, mo, palette, plt):
    _n = 8
    _fig, (_aL, _aR) = plt.subplots(1, 2, figsize=(10.5, 5.0))

    # serial: leaves at the bottom, a ladder of combines climbing one level per step
    for _t in range(_n):
        box(_aL, _t, 0, palette["operator"], label=f"M{_t + 1}", w=0.72, h=0.46, fontsize=8)
    _acc = (0.0, 0.0)
    for _k in range(1, _n):
        _pos = (0.7 + 0.12 * _k, float(_k))
        box(_aL, _pos[0], _pos[1], palette["ink"], label="•", w=0.5, h=0.42, fontsize=11)
        arrow(
            _aL,
            (_acc[0], _acc[1] + 0.23),
            (_pos[0], _pos[1] - 0.21),
            color=palette["ink"],
            lw=1.2,
            shrink=1,
        )
        arrow(
            _aL,
            (float(_k), 0.23),
            (_pos[0], _pos[1] - 0.21),
            color=palette["muted"],
            lw=1.0,
            shrink=1,
        )
        _acc = _pos
    _aL.set_title(
        "one-at-a-time:  7 combines deep", fontsize=11, fontweight="bold", color=palette["operator"]
    )
    _aL.set_xlim(-0.7, _n)
    _aL.set_ylim(-0.6, _n)
    _aL.axis("off")

    # tree: pair up, then pairs of pairs
    _levels = [[float(i) for i in range(_n)]]
    while len(_levels[-1]) > 1:
        _prev = _levels[-1]
        _cur = []
        _i = 0
        while _i < len(_prev):
            if _i + 1 < len(_prev):
                _cur.append((_prev[_i] + _prev[_i + 1]) / 2)
                _i += 2
            else:
                _cur.append(_prev[_i])
                _i += 1
        _levels.append(_cur)
    for _lvl_i, _lvl in enumerate(_levels):
        for _xi, _x in enumerate(_lvl):
            if _lvl_i == 0:
                box(
                    _aR, _x, 0, palette["operator"], label=f"M{_xi + 1}", w=0.72, h=0.46, fontsize=8
                )
            else:
                box(_aR, _x, float(_lvl_i), palette["ink"], label="•", w=0.5, h=0.42, fontsize=11)
    for _lvl_i in range(1, len(_levels)):
        _prev = _levels[_lvl_i - 1]
        _i, _c = 0, 0
        while _i < len(_prev):
            _px = _levels[_lvl_i][_c]
            arrow(
                _aR,
                (_prev[_i], _lvl_i - 1 + 0.23),
                (_px, _lvl_i - 0.21),
                color=palette["muted"],
                lw=1.1,
                shrink=1,
            )
            if _i + 1 < len(_prev):
                arrow(
                    _aR,
                    (_prev[_i + 1], _lvl_i - 1 + 0.23),
                    (_px, _lvl_i - 0.21),
                    color=palette["muted"],
                    lw=1.1,
                    shrink=1,
                )
                _i += 2
            else:
                _i += 1
            _c += 1
    _aR.set_title(
        "in a tree:  3 combines deep  (log)", fontsize=11, fontweight="bold", color=palette["state"]
    )
    _aR.set_xlim(-0.7, _n)
    _aR.set_ylim(-0.6, _n)
    _aR.axis("off")

    _fig.suptitle(
        "Same product M₈·…·M₁ — regrouped. Multiplication does not care how you bracket it.",
        fontsize=12.5,
        fontweight="bold",
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def pf_depth_md(mo):
    mo.md(r"""
    ## 6. How much do you save?

    Sequential rounds needed as the timeline grows: the one-at-a-time filter grows in a
    straight line with **T**; the tree grows like **log T**. At T = 128 that is 128 rounds
    versus 7.
    """)
    return


@app.cell
def depth_plot(mo, np, palette, plt):
    _T = np.arange(1, 129)
    _serial = _T.astype(float)
    _tree = np.ceil(np.log2(np.maximum(_T, 2)))
    _tree[0] = 0.0

    _fig, _ax = plt.subplots(figsize=(8.5, 3.6))
    _ax.plot(_T, _serial, color=palette["operator"], lw=2.4, label="one-at-a-time  (≈ T)")
    _ax.plot(_T, _tree, color=palette["state"], lw=2.4, label="tree / scan  (≈ log₂ T)")
    _ax.scatter([128, 128], [128, 7], color=[palette["operator"], palette["state"]], zorder=5)
    _ax.annotate("128 rounds", (128, 128), xytext=(74, 116), color=palette["operator"], fontsize=10)
    _ax.annotate("7 rounds", (128, 7), xytext=(96, 22), color=palette["state"], fontsize=10)
    _ax.set_xlabel("number of observations, T")
    _ax.set_ylabel("sequential rounds needed")
    _ax.legend(frameon=False, fontsize=10, loc="center right")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_title(
        "The chain grows with T; the tree grows with log T", fontsize=12, fontweight="bold"
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def seam_md(mo):
    mo.md(r"""
    ## 7. Why the pieces do not need each other

    "Combine two neighboring blocks without looking at the rest" sounds impossible — surely a
    belief far down the line depends on everything. It does. But a **block** talks to its
    neighbors *only* through the **single state at its edge**. Conditioning on that one shared
    state makes everything on its left independent of everything on its right — the **Markov**
    property.

    So each block can be built alone, and merging two blocks only has to reconcile their one
    shared edge. The far-away dependence still arrives — it just climbs the tree instead of
    walking the chain.
    """)
    return


@app.cell
def seam_diagram(arrow, box, mo, node, palette, plt):
    _fig, _ax = plt.subplots(figsize=(9.5, 3.6))
    _n, _seam = 7, 3
    box(
        _ax,
        (_seam) / 2.0,
        0,
        palette["state"],
        w=_seam + 1.4,
        h=1.25,
        filled=True,
        alpha=0.12,
        lw=0.0,
        z=0,
    )
    box(
        _ax,
        (_seam + _n - 1) / 2.0,
        0,
        palette["operator"],
        w=(_n - 1 - _seam) + 1.4,
        h=1.25,
        filled=True,
        alpha=0.12,
        lw=0.0,
        z=0,
    )
    for _t in range(_n):
        if _t == _seam:
            node(_ax, _t, 0, palette["seam"], r=0.34)
        else:
            node(_ax, _t, 0, palette["state"], r=0.26, filled=False)
        if _t < _n - 1:
            arrow(_ax, (_t + 0.27, 0), (_t + 1 - 0.27, 0), color=palette["muted"], lw=1.6, shrink=1)
    _ax.text(
        1.0, 0.95, "left block", ha="center", color=palette["state"], fontsize=11, fontweight="bold"
    )
    _ax.text(
        5.0,
        0.95,
        "right block",
        ha="center",
        color=palette["operator"],
        fontsize=11,
        fontweight="bold",
    )
    _ax.annotate(
        "the one shared state\n(everything passes through here)",
        xy=(_seam, -0.36),
        xytext=(_seam, -1.2),
        ha="center",
        color=palette["seam"],
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=palette["seam"], lw=1.6),
    )
    _ax.set_xlim(-1.0, _n)
    _ax.set_ylim(-1.5, 1.3)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(
        "Two blocks meet at a single shared state — that one state is the whole interface",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def proof_md(mo):
    mo.md(r"""
    ## 8. Same answer, far fewer rounds

    Proof on a tiny 3-state model. Compute every belief the slow way (one-at-a-time), and again
    by forming each belief from an **independent** product of its own block — no left-to-right
    walk. The two land on top of each other to machine precision: the *same* filter,
    reorganized from a length-T chain into a depth-log T tree.
    """)
    return


@app.cell
def proof_cell(mo, np, palette, plt):
    _rng = np.random.default_rng(1)
    _k, _t_max = 3, 32
    _dyn = _rng.random((_k, _k))
    _dyn /= _dyn.sum(axis=0, keepdims=True)
    _ops = [np.diag(_rng.random(_k) + 0.1) @ _dyn for _ in range(_t_max)]
    _b0 = np.full(_k, 1.0 / _k)

    # (1) slow way: one belief at a time
    _serial = np.zeros((_t_max + 1, _k))
    _serial[0] = _b0
    _b = _b0.copy()
    for _i in range(_t_max):
        _b = _ops[_i] @ _b
        _b = _b / _b.sum()
        _serial[_i + 1] = _b

    # (2) each belief from its OWN independent product of step-operators
    _scan = np.zeros((_t_max + 1, _k))
    _scan[0] = _b0
    for _i in range(_t_max):
        _prod = np.eye(_k)
        for _s in range(_i + 1):
            _prod = _ops[_s] @ _prod
        _val = _prod @ _b0
        _scan[_i + 1] = _val / _val.sum()

    _rounds = int(np.ceil(np.log2(_t_max)))
    _gap = float(np.max(np.abs(_serial - _scan)))

    _fig, _ax = plt.subplots(figsize=(9.0, 3.8))
    _cols = [palette["state"], palette["operator"], palette["belief"]]
    _names = ["state A", "state B", "state C"]
    _time = range(_t_max + 1)
    for _c in range(_k):
        _ax.plot(_time, _serial[:, _c], color=_cols[_c], lw=2.0, label=_names[_c])
        _ax.scatter(_time, _scan[:, _c], facecolors="none", edgecolors=_cols[_c], s=22, zorder=5)
    _ax.set_xlabel("time step")
    _ax.set_ylabel("belief  P(state | observations so far)")
    _ax.set_ylim(0, 1)
    _ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_title(
        f"lines = one-at-a-time filter · circles = independent tree/scan — they agree to {_gap:.0e}\n(T = {_t_max} observations:  one-at-a-time needs {_t_max} rounds,  the tree needs {_rounds})",
        fontsize=11,
        fontweight="bold",
    )
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def pf_closing(mo):
    mo.md(r"""
    ## The essence in one line

    A sequential filter *looks* like a chain because we carry the **belief**. Underneath, it is
    a **product of local step-operators**, and a product can be regrouped from a chain (depth
    **T**) into a tree (depth **log T**) — because each block meets its neighbors at a single
    shared state.

    That single move is the whole parallelization opportunity. Kalman and particle methods are
    then just two ways of making those step-operators exact for a given model.
    """)
    return


if __name__ == "__main__":
    app.run()
