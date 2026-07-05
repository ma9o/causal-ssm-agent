import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import math

    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import jax.random as random
    import matplotlib.pyplot as plt
    import numpy as np

    palette = {
        "state": "#3b6ea5",  # reference-local / baseline (blue)
        "obs": "#e08a3c",  # per-time flips (orange)
        "belief": "#4a9d5b",  # segment moves (green)
        "operator": "#c0504d",  # failure / stuck (red)
        "muted": "#999999",
        "ink": "#333333",
    }
    return jax, jnp, math, np, palette, plt, random


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Mode nucleation: segment flips for high-D multimodal smoothing

    **The open cell.** The D-loop study
    ([inference_and_causal_walkthroughs](inference_and_causal_walkthroughs.py) §16)
    ended on one regime nothing survives: state dimension `D ≥ 16` with far-separated
    posterior modes. On the Kitagawa-D benchmark (independent Kitagawa coordinates,
    `y = x²/20` per coordinate, so the smoothing posterior is sign-ambiguous in every
    coordinate and the *product of 1-D grid smoothers is an exact gold standard at any
    D*): local-gradient kernels are structurally sign-blind at every `D`; independence
    proposals die of weight degeneracy at `D ≈ 8`; and the per-time flip leaf — a
    sign-symmetric auxiliary that makes single-time flips affordable — pushed the
    frontier to `D ≈ 8` and no further, *at every flip rate tried*. The `p → 0` limit
    cleanly recovers the sign-stuck reference-local kernel, so the wall is not particle
    acceptance. It is **nucleation**: a sign change must hold over a contiguous time
    segment to be dynamically coherent, and independent per-time flips can only grow
    such a segment through rare domain-wall moves.

    **What the literature says.** This is a very old problem wearing new clothes.

    - In lattice physics it is *critical slowing down*, and the canonical cure is
      **cluster algorithms** — [Swendsen–Wang multi-cluster
      updates](https://arxiv.org/abs/1202.0635) and Wolff single-cluster moves, extended
      to continuous fields by *embedding* Ising variables into them (see the
      [cluster-algorithm lectures](https://hef.ru.nl/~tbudd/mct/lectures/cluster_algorithms.html)
      and [Kennedy's review](https://arxiv.org/abs/hep-lat/9704009)). The requirement is
      exactly what a sign-ambiguous emission provides: a `±` symmetry the energy (here,
      the emission likelihood) cannot see.
    - In statistics the same shape appears as **mode-jumping MCMC** —
      [Tjelmeland & Hegstad (2001)](https://www.semanticscholar.org/paper/Mode-Jumping-Proposals-in-MCMC-Tjelmeland-Hegstad/9eb5af8e47b44179c47a700643d81deb9becdf9f)
      mix a local kernel with a deterministic-jump kernel; the recent
      [multimodal-MCMC survey](https://arxiv.org/abs/2501.05908) frames the whole family
      (jumping, tempering, Wang–Landau, wormhole/darting variants).
    - In the SSM corner, [particle tempering](https://www.sciencedirect.com/science/article/abs/pii/S2452306222000843)
      and [grid-guided particle Gibbs](https://arxiv.org/abs/2501.03395) attack
      multimodality globally, at replica or grid cost — the structure-agnostic fallback.
    - **Closest prior art** (found by a post-hoc deep literature check):
      [Shestopaloff & Neal (2016)](https://arxiv.org/abs/1602.06030) add a
      *mirror-symmetric pool* to an embedded-HMM sampler for exactly this problem
      class (`y ~ Poisson(σ|x|)`, sign-multimodal smoothing posterior): their discrete
      forward-backward pass flips large contiguous trajectory segments, they state the
      exact free-flip symmetry condition, and they document that single-state
      Metropolis never crosses sign modes while PGBS needed 80 000 particles. Their
      flip negates **all coordinates at once**, has no per-coordinate move, no
      asymmetric-fold/Jacobian treatment, and no cSMC/tree composition — precisely the
      gaps this lab fills. In physics,
      [Gulbahce, Alexander & Johnson (2005)](https://arxiv.org/abs/cond-mat/0511242)
      ran embedded-Ising space-time cluster sign-flips on *measurement-conditioned*
      Langevin field histories — the cluster idea on a time-indexed posterior, with
      percolation-grown clusters on a lattice field theory rather than SSM smoothing
      within particle Gibbs.

    **The idea under test.** Stop trying to put mode moves *inside* the leaf. Compose
    the tree sweep with a separate π-invariant **segment sign-flip Metropolis–Hastings
    kernel** acting on the reference trajectory between sweeps: pick a coordinate and a
    contiguous time window from a fixed distribution, propose flipping that window's
    sign, and accept by the exact density ratio. Because the flip is an involution and
    the window law does not depend on the state, the acceptance is just `π(x')/π(x)` —
    the emission terms cancel under the fold symmetry, so the ratio reduces to the two
    domain walls plus the forcing interaction along the window: *precisely the cost the
    exact posterior itself assigns to a sign change*. This is the SSM specialisation of
    an embedded-Ising cluster move (a segment is a 1-D cluster), and it sidesteps the
    leaf's failure mode entirely: one MH decision per (coordinate, window) instead of a
    per-particle joint-coherence lottery over all `D` coordinates at once.

    The lab: (A) on the factorised Kitagawa-D with its exact gold, does
    tree + segment-MH crack the `D ≥ 16` wall that killed everything else? (B) does it
    remain exact when coordinates are *coupled* (validated against a joint 2-D grid
    gold), where per-coordinate sign problems no longer separate?
    """)
    return


@app.cell
def model_cell(jnp, np):
    # Kitagawa-D: per-coordinate Kitagawa dynamics, x^2/20 emission per coordinate,
    # optional mean-field drift coupling kappa*(x_bar - x_d) (kappa=0 => the posterior
    # factorises over coordinates and the product of 1-D grid smoothers is exact).
    KIT_SIG_V = float(np.sqrt(10.0))
    KIT_SIG_W = 1.0
    KIT_INIT_SD = 5.0

    def make_kit_d(dim, kappa=0.0, forcing=True):
        force = 8.0 if forcing else 0.0

        def drift(t, x):  # t broadcastable against x (..., D); forcing=False => ODD drift
            base = 0.5 * x + 25.0 * x / (1.0 + x**2) + force * jnp.cos(1.2 * t)
            if kappa == 0.0:
                return base
            return base + kappa * (jnp.mean(x, axis=-1, keepdims=True) - x)

        def log_obs(x, y):  # (..., D), (D,) -> (...,)
            return jnp.sum(
                -0.5 * (jnp.log(2.0 * jnp.pi * KIT_SIG_W**2) + (y - x**2 / 20.0) ** 2), axis=-1
            )

        def simulate(seed, t_len):
            rng = np.random.default_rng(seed)
            x = np.zeros((t_len, dim))
            x[0] = KIT_INIT_SD * rng.standard_normal(dim)
            for t in range(1, t_len):
                x[t] = np.asarray(drift(t, jnp.asarray(x[t - 1]))) + KIT_SIG_V * (
                    rng.standard_normal(dim)
                )
            y = x**2 / 20.0 + KIT_SIG_W * rng.standard_normal((t_len, dim))
            return x, y

        def obs_grad_1d(z, y_col):  # per-coordinate d/dz log g
            return (y_col - z**2 / 20.0) * (z / 10.0)

        return dict(
            dim=dim,
            kappa=kappa,
            drift=drift,
            log_obs=log_obs,
            obs_grad_1d=obs_grad_1d,
            simulate=simulate,
            sig_v=KIT_SIG_V,
            sig_w=KIT_SIG_W,
            init_sd=KIT_INIT_SD,
        )

    return KIT_INIT_SD, KIT_SIG_V, KIT_SIG_W, make_kit_d


@app.cell
def fold_model_cell(KIT_INIT_SD, KIT_SIG_V, KIT_SIG_W, jnp, np):
    # ASYMMETRIC fold: same Kitagawa dynamics, emission y = h(x) + N(0,1) with
    #   h(x) = (1 + beta*sign(x)) * |x| / 3
    # Two preimages per level with DIFFERENT slopes: the mirror map
    # rho(x) = -x (1+beta)/(1-beta) (from the + branch) has non-unit Jacobian
    # (1+beta)/(1-beta), and the mode locations are asymmetric. beta = 0 recovers the
    # symmetric |x| fold. This is the test bed for the branch-path FFBS with the
    # Jacobian correction.
    def make_fold_d(dim, beta, kappa=0.0):
        def drift(t, x):
            base = 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * jnp.cos(1.2 * t)
            if kappa == 0.0:
                return base
            return base + kappa * (jnp.mean(x, axis=-1, keepdims=True) - x)

        def h_fn(x):
            return (1.0 + beta * jnp.sign(x)) * jnp.abs(x) / 3.0

        def log_obs(x, y):
            return jnp.sum(
                -0.5 * (jnp.log(2.0 * jnp.pi * KIT_SIG_W**2) + (y - h_fn(x)) ** 2), axis=-1
            )

        def obs_grad_1d(z, y_col):
            hp = jnp.sign(z) * (1.0 + beta * jnp.sign(z)) / 3.0
            return (y_col - h_fn(z)) * hp

        def branch_rep(v):  # canonical (positive) preimage of h(v)
            return jnp.where(v >= 0, v, -v * (1.0 - beta) / (1.0 + beta))

        def branch_mirror(r):  # the negative preimage at the same level
            return -r * (1.0 + beta) / (1.0 - beta)

        def simulate(seed, t_len):
            rng = np.random.default_rng(seed)
            x = np.zeros((t_len, dim))
            x[0] = KIT_INIT_SD * rng.standard_normal(dim)
            for t in range(1, t_len):
                x[t] = np.asarray(drift(t, jnp.asarray(x[t - 1]))) + KIT_SIG_V * (
                    rng.standard_normal(dim)
                )
            y = np.asarray(h_fn(jnp.asarray(x))) + KIT_SIG_W * rng.standard_normal((t_len, dim))
            return x, y

        return dict(
            dim=dim,
            kappa=kappa,
            beta=beta,
            drift=drift,
            log_obs=log_obs,
            obs_grad_1d=obs_grad_1d,
            h_fn=h_fn,
            branch_rep=branch_rep,
            branch_mirror=branch_mirror,
            branch_log_jac=float(np.log((1.0 + beta) / (1.0 - beta))),
            simulate=simulate,
            sig_v=KIT_SIG_V,
            sig_w=KIT_SIG_W,
            init_sd=KIT_INIT_SD,
        )

    return (make_fold_d,)


@app.cell
def fold_gold_cell(KIT_INIT_SD, KIT_SIG_V, KIT_SIG_W, np):
    # Exact per-coordinate grid gold for the fold model (kappa = 0).
    def _grid_1d_fold(y_d, beta, n_grid=601, lo=-32.0, hi=32.0):
        t_len = len(y_d)
        xs = np.linspace(lo, hi, n_grid)

        def _logn(v, mu, sd):
            return -0.5 * (np.log(2.0 * np.pi * sd**2) + ((v - mu) ** 2) / sd**2)

        def _drift(t, x):
            return 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)

        h_xs = (1.0 + beta * np.sign(xs)) * np.abs(xs) / 3.0
        log_obs = _logn(y_d[:, None], h_xs[None, :], KIT_SIG_W)
        log_alpha = np.zeros((t_len, n_grid))
        log_alpha[0] = _logn(xs, 0.0, KIT_INIT_SD) + log_obs[0]
        for t in range(1, t_len):
            lt = _logn(xs[None, :], _drift(t, xs)[:, None], KIT_SIG_V)
            a = log_alpha[t - 1]
            m = a.max()
            log_alpha[t] = np.log(np.exp(a - m) @ np.exp(lt) + 1e-300) + m + log_obs[t]
        log_beta = np.zeros((t_len, n_grid))
        for t in range(t_len - 2, -1, -1):
            lt = _logn(xs[None, :], _drift(t + 1, xs)[:, None], KIT_SIG_V)
            b = log_beta[t + 1] + log_obs[t + 1]
            m = b.max()
            log_beta[t] = np.log(np.exp(lt) @ np.exp(b - m) + 1e-300) + m
        g = np.exp((log_alpha + log_beta) - (log_alpha + log_beta).max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        mean = (g * xs[None, :]).sum(1)
        sd = np.sqrt((g * (xs[None, :] - mean[:, None]) ** 2).sum(1))
        return {
            "xs": xs,
            "dx": xs[1] - xs[0],
            "cdf": np.cumsum(g, 1),
            "p_pos": g[:, xs > 0].sum(1),
            "sd": sd,
        }

    def product_gold_fold(y, beta):
        return [
            _grid_1d_fold(np.asarray(y[:, d], dtype=np.float64), beta) for d in range(y.shape[1])
        ]

    return (product_gold_fold,)


@app.cell
def gold_1d_cell(KIT_INIT_SD, KIT_SIG_V, KIT_SIG_W, np):
    # Exact gold for the factorised model: 1-D grid forward-backward per coordinate.
    # h_np overrides the emission mean function (default: the Kitagawa x²/20).
    def _grid_1d(y_d, h_np=None, n_grid=601, lo=-32.0, hi=32.0):
        t_len = len(y_d)
        xs = np.linspace(lo, hi, n_grid)

        def _logn(v, mu, sd):
            return -0.5 * (np.log(2.0 * np.pi * sd**2) + ((v - mu) ** 2) / sd**2)

        def _drift(t, x):
            return 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)

        h_xs = xs**2 / 20.0 if h_np is None else h_np(xs)
        log_obs = _logn(y_d[:, None], h_xs[None, :], KIT_SIG_W)
        log_alpha = np.zeros((t_len, n_grid))
        log_alpha[0] = _logn(xs, 0.0, KIT_INIT_SD) + log_obs[0]
        for t in range(1, t_len):
            lt = _logn(xs[None, :], _drift(t, xs)[:, None], KIT_SIG_V)
            a = log_alpha[t - 1]
            m = a.max()
            log_alpha[t] = np.log(np.exp(a - m) @ np.exp(lt) + 1e-300) + m + log_obs[t]
        log_beta = np.zeros((t_len, n_grid))
        for t in range(t_len - 2, -1, -1):
            lt = _logn(xs[None, :], _drift(t + 1, xs)[:, None], KIT_SIG_V)
            b = log_beta[t + 1] + log_obs[t + 1]
            m = b.max()
            log_beta[t] = np.log(np.exp(lt) @ np.exp(b - m) + 1e-300) + m
        g = np.exp((log_alpha + log_beta) - (log_alpha + log_beta).max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        mean = (g * xs[None, :]).sum(1)
        sd = np.sqrt((g * (xs[None, :] - mean[:, None]) ** 2).sum(1))
        return {
            "xs": xs,
            "dx": xs[1] - xs[0],
            "cdf": np.cumsum(g, 1),
            "p_pos": g[:, xs > 0].sum(1),
            "sd": sd,
            "g": g,
        }

    def product_gold(y, h_np=None):
        return [_grid_1d(np.asarray(y[:, d], dtype=np.float64), h_np) for d in range(y.shape[1])]

    def kit_metrics(chain_burn, golds):
        """Mean per-(t,d) W1 normalised by the gold's mean sd, plus sign error."""
        dim = chain_burn.shape[2]
        w1s, sderrs, sds = [], [], []
        for d in range(dim):
            g = golds[d]
            edges = np.concatenate([[g["xs"][0] - g["dx"] / 2], g["xs"] + g["dx"] / 2])
            for t in range(chain_burn.shape[1]):
                hist, _ = np.histogram(chain_burn[:, t, d], bins=edges)
                emp = np.cumsum(hist / max(hist.sum(), 1))
                w1s.append(np.sum(np.abs(emp - g["cdf"][t])) * g["dx"])
            sderrs.append(np.sqrt(np.mean(((chain_burn[:, :, d] > 0).mean(0) - g["p_pos"]) ** 2)))
            sds.append(g["sd"].mean())
        return {
            "w1_rel": float(np.mean(w1s) / np.mean(sds)),
            "sign_err": float(np.mean(sderrs)),
        }

    return kit_metrics, product_gold


@app.cell
def tree_cell(jax, jnp, math, random):
    # Compact multivariate c-dSMC tree (the same stitch as the audit notebook's §15
    # tree, with (P, D) particles and model-supplied multivariate seams).
    def make_tree(t_len, p, dim, seam_pair, seam_sel):
        def _multinomial(draw_key, logits, num_draws):
            cum = jnp.cumsum(jax.nn.softmax(logits))
            u = random.uniform(draw_key, (num_draws,), dtype=cum.dtype)
            return jnp.minimum(jnp.searchsorted(cum, u, side="right"), logits.shape[0] - 1).astype(
                jnp.int32
            )

        def _stitch_logits(left, right, seam):
            _, ll, lpsi, _, lw = left
            rf, _, rpsi, _, rw = right
            tr = seam_pair(ll, rf, seam)[:, :, None]
            log_joint = jax.scipy.special.logsumexp(
                lpsi[:, None, :] + rpsi[None, :, :] + tr, axis=-1
            )
            log_left = jax.scipy.special.logsumexp(lpsi, axis=1)
            log_right = jax.scipy.special.logsumexp(rpsi, axis=1)
            return lw[:, None] + rw[None, :] + log_joint - log_left[:, None] - log_right[None, :]

        def _combine(left, right, seam, key):
            pl = _stitch_logits(left, right, seam)
            free = _multinomial(key, pl.reshape(-1), p - 1)
            sel = jnp.concatenate([jnp.zeros((1,), jnp.int32), free])
            li, ri = sel // p, sel % p
            lf, llast, lpsi, lorig, _ = left
            rf, rlast, rpsi, rorig, _ = right
            origin = jnp.concatenate([lorig[li], rorig[ri]], axis=1)
            tr = seam_sel(llast[li], rf[ri], seam)[:, None]
            psi = lpsi[li] + rpsi[ri] + tr
            return lf[li], rlast[ri], psi, origin, jnp.full((p,), -math.log(p))

        def smooth(key, leaf_fn):
            depth = max((t_len - 1).bit_length(), 0)
            padded = 1 << depth
            kl, kt, kr = random.split(key, 3)
            particles, psi = jax.vmap(leaf_fn)(jnp.arange(t_len), random.split(kl, t_len))
            origin0 = jnp.broadcast_to(jnp.arange(p, dtype=jnp.int32)[:, None], (p, 1))
            leaf_w = jax.vmap(lambda q: q - jax.scipy.special.logsumexp(q))(psi[:, :, 0])
            nph = padded - t_len
            first = jnp.concatenate([particles, jnp.zeros((nph, p, dim))], 0)
            last = first
            psi_a = jnp.concatenate([psi, jnp.zeros((nph, p, 1))], 0)
            origin = jnp.concatenate(
                [jnp.broadcast_to(origin0, (t_len, p, 1)), jnp.broadcast_to(origin0, (nph, p, 1))],
                0,
            )
            weights = jnp.concatenate([leaf_w, jnp.full((nph, p), -math.log(p))], 0)
            level_keys = random.split(kt, max(depth - 1, 1))
            segments = padded
            for level in range(depth - 1):
                npairs = segments // 2
                seams = (1 << level) + jnp.arange(npairs, dtype=jnp.int32) * (1 << (level + 1))
                left = (first[0::2], last[0::2], psi_a[0::2], origin[0::2], weights[0::2])
                right = (first[1::2], last[1::2], psi_a[1::2], origin[1::2], weights[1::2])
                first, last, psi_a, origin, weights = jax.vmap(_combine, in_axes=(0, 0, 0, 0))(
                    left, right, seams, random.split(level_keys[level], npairs)
                )
                segments = npairs
            left_root = (first[0], last[0], psi_a[0], origin[0], weights[0])
            right_root = (first[1], last[1], psi_a[1], origin[1], weights[1])
            pl = _stitch_logits(left_root, right_root, padded // 2)
            chosen = _multinomial(kr, pl.reshape(-1), 1)[0]
            origin_path = jnp.concatenate([origin[0][chosen // p], origin[1][chosen % p]], axis=0)[
                :t_len
            ]
            return particles[jnp.arange(t_len), origin_path]

        return smooth

    def logn(v, mu, var):
        return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (v - mu) ** 2 / var)

    return logn, make_tree


@app.cell
def kernels_cell(jax, jnp, logn, make_tree, np, random):
    # The tree sweep with the (optionally sign-symmetric-auxiliary) reference-local
    # leaf: pflip = 0 gives plain amala_z (the sign-stuck baseline), pflip > 0 gives
    # the §16 per-time flip leaf. Plus the NEW segment-flip MH kernel and the composed
    # runner (tree sweep, then n_props segment proposals, both pi-invariant).
    def make_leaf_sweep(model, y, p, delta=2.0, pflip=0.0):
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        tau = 0.5 * delta
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]

        def seam_pair(prev, nxt, seam):
            mm = drift(seam, prev)
            lp = jnp.sum(logn(nxt[None, :, :], mm[:, None, :], sig_v2), axis=-1)
            return jnp.where(seam < t_len, lp, 0.0)

        def seam_sel(prev, nxt, seam):
            mm = drift(seam, prev)
            lp = jnp.sum(logn(nxt, mm, sig_v2), axis=-1)
            return jnp.where(seam < t_len, lp, 0.0)

        smooth = make_tree(t_len, p, dim, seam_pair, seam_sel)

        def g1(x, t):  # per-coordinate emission gradient
            return (y_j[t] - x**2 / 20.0) * (x / 10.0)

        t_idx = jnp.arange(t_len)
        log_pf = jnp.log(jnp.asarray(pflip)) if pflip > 0 else -jnp.inf
        log_1mpf = jnp.log1p(-pflip)

        def sweep(x_ref, key):
            kz1, kz2, kt = random.split(key, 3)
            z_flip = random.bernoulli(kz1, pflip, (t_len, dim))
            z_center = jnp.where(z_flip, -x_ref, x_ref)
            z = z_center + jnp.sqrt(tau) * random.normal(kz2, (t_len, dim))
            center = z + tau * jax.vmap(g1)(z, t_idx)

            def log_m(z_t, xs):
                lp = jnp.logaddexp(log_1mpf + logn(z_t, xs, tau), log_pf + logn(z_t, -xs, tau))
                return jnp.sum(lp, axis=-1)

            def log_q(xs, t):
                lp = jnp.logaddexp(
                    log_1mpf + logn(xs, center[t], tau), log_pf + logn(xs, -center[t], tau)
                )
                return jnp.sum(lp, axis=-1)

            def leaf(t, k):
                fk, dk = random.split(k)
                flip = random.bernoulli(fk, pflip, (p - 1, dim))
                cen = jnp.where(flip, -center[t], center[t])
                free = cen + jnp.sqrt(tau) * random.normal(dk, (p - 1, dim))
                parts = jnp.concatenate([x_ref[t][None], free], axis=0)
                psi = model["log_obs"](parts, y_j[t]) + log_m(z[t], parts) - log_q(parts, t)
                psi = jnp.where(t == 0, psi + jnp.sum(logn(parts, 0.0, init_var), -1), psi)
                return parts, psi[:, None]

            return smooth(kt, leaf)

        return sweep

    def make_seg_mh(model, y, n_props, mean_len=8.0):
        """Segment sign-flip MH on the reference trajectory.

        A proposal picks (coordinate d, start a, window) from a FIXED distribution
        (uniform d and a; with prob 1/2 a suffix window [a, T), else a geometric
        length), flips the sign of coordinate d over the window, and accepts with the
        exact ratio pi(x')/pi(x). The flip is an involution and the window law is
        state-independent, so no proposal correction is needed. The ratio is computed
        as the full masked trajectory-density difference (transitions touching the
        window, the initial density if a = 0, and the emission over the window — the
        latter cancels exactly under the x^2 fold, but is computed generally), which
        also stays exact when drift coupling makes coordinates interact."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]
        t_arr = jnp.arange(1, t_len)
        ts = jnp.arange(t_len)

        def trans_lp(xx):  # (T-1,) transition log-densities, summed over coordinates
            mu = drift(t_arr[:, None], xx[:-1])
            return jnp.sum(logn(xx[1:], mu, sig_v2), axis=-1)

        def obs_lp(xx):  # (T,)
            return jax.vmap(model["log_obs"])(xx, y_j)

        def one_prop(x, key):
            kd, ka, kl, ks, ku = random.split(key, 5)
            d = random.randint(kd, (), 0, dim)
            a = random.randint(ka, (), 0, t_len)
            suffix = random.bernoulli(ks, 0.5)
            length = 1 + jnp.floor(-mean_len * jnp.log(random.uniform(kl))).astype(jnp.int32)
            b = jnp.where(suffix, t_len - 1, jnp.minimum(a + length - 1, t_len - 1))
            win = (ts >= a) & (ts <= b)
            col = jnp.arange(dim) == d
            x_new = jnp.where(win[:, None] & col[None, :], -x, x)
            tmask = (t_arr >= jnp.maximum(a, 1)) & (t_arr <= jnp.minimum(b + 1, t_len - 1))
            delta = jnp.sum(tmask * (trans_lp(x_new) - trans_lp(x)))
            delta += jnp.sum(win * (obs_lp(x_new) - obs_lp(x)))
            delta += (a == 0) * jnp.sum(logn(x_new[0], 0.0, init_var) - logn(x[0], 0.0, init_var))
            accept = jnp.log(random.uniform(ku)) < delta
            return jnp.where(accept, x_new, x), accept

        def mh_sweep(x, key):
            def step(xx, k):
                return one_prop(xx, k)

            x_out, accs = jax.lax.scan(step, x, random.split(key, n_props))
            return x_out, jnp.mean(accs)

        return mh_sweep

    def run_chain(leaf_sweep, mh_sweep, x0, n_iter, seed):
        """Compose: tree sweep, then the MH segment proposals (both pi-invariant)."""
        if mh_sweep is None:

            def body(x, key):
                x = leaf_sweep(x, key)
                return x, (x, 0.0)
        else:

            def body(x, key):
                k1, k2 = random.split(key)
                x = leaf_sweep(x, k1)
                x, acc = mh_sweep(x, k2)
                return x, (x, acc)

        keys = random.split(random.PRNGKey(seed), n_iter)
        _, (chain, accs) = jax.jit(lambda ks: jax.lax.scan(body, x0, ks))(keys)
        return np.asarray(chain), float(np.mean(np.asarray(accs)))

    return make_leaf_sweep, make_seg_mh, run_chain


@app.cell
def sign_ffbs_cell(jax, jnp, logn, random):
    def make_sign_ffbs(model, y):
        """EXACT conditional Gibbs on each coordinate's sign path — the limit of
        adaptive windows.

        Conditioned on the magnitudes |x| (and the other coordinates), the sign path
        s_{0:T-1} of one coordinate is a two-state Markov chain in time: transitions
        contribute nearest-neighbour (s_{t-1}, s_t) potentials, the initial density
        and emissions contribute site potentials, and nothing longer-range exists. In
        1-D time the Swendsen-Wang cluster construction therefore collapses to a
        transfer matrix: forward-filter the two-state chain, backward-sample the WHOLE
        sign path from its exact conditional. No windows, no acceptance, no tuning.
        The 2x2 potentials are evaluated by brute force on the full transition density
        (all coordinates), so drift coupling between coordinates is priced exactly and
        the move stays a valid Gibbs step for coupled models; sweeping coordinates in
        fixed order is systematic-scan Gibbs, pi-invariant."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]
        t_arr = jnp.arange(1, t_len)

        def sweep(x, key):
            def per_coord(x_cur, inp):
                d, kd = inp
                m_d = jnp.abs(x_cur[:, d])
                xp = x_cur.at[:, d].set(m_d)
                xm = x_cur.at[:, d].set(-m_d)
                mu_p = drift(t_arr[:, None], xp[:-1])  # (T-1, D)
                mu_m = drift(t_arr[:, None], xm[:-1])

                def tl(xt, mu):  # transition log-density summed over coordinates
                    return jnp.sum(logn(xt, mu, sig_v2), axis=-1)

                lp = jnp.stack(
                    [
                        jnp.stack([tl(xm[1:], mu_m), tl(xp[1:], mu_m)], axis=-1),
                        jnp.stack([tl(xm[1:], mu_p), tl(xp[1:], mu_p)], axis=-1),
                    ],
                    axis=-2,
                )  # (T-1, s_prev, s_cur)
                e = jnp.stack(
                    [
                        jax.vmap(model["log_obs"])(xm, y_j),
                        jax.vmap(model["log_obs"])(xp, y_j),
                    ],
                    axis=-1,
                )  # (T, 2)
                alpha0 = (
                    jnp.stack(
                        [
                            jnp.sum(logn(xm[0], 0.0, init_var)),
                            jnp.sum(logn(xp[0], 0.0, init_var)),
                        ]
                    )
                    + e[0]
                )

                def fstep(alpha, inp2):
                    lp_t, e_t = inp2
                    a_new = e_t + jax.scipy.special.logsumexp(alpha[:, None] + lp_t, axis=0)
                    return a_new, a_new

                alpha_last, alphas = jax.lax.scan(fstep, alpha0, (lp, e[1:]))
                alphas_all = jnp.concatenate([alpha0[None], alphas[:-1]], axis=0)  # (T-1, 2)
                k_last, k_back = random.split(kd)
                s_last = random.categorical(k_last, alpha_last)

                def bstep(s_next, inp2):
                    alpha_t, lp_t, k_t = inp2
                    s_t = random.categorical(k_t, alpha_t + lp_t[:, s_next])
                    return s_t, s_t

                keys_b = random.split(k_back, t_len - 1)
                _, s_rev = jax.lax.scan(
                    bstep,
                    s_last,
                    (jnp.flip(alphas_all, 0), jnp.flip(lp, 0), keys_b),
                )
                s = jnp.concatenate([jnp.flip(s_rev), s_last[None]])  # (T,) in {0, 1}
                x_out = x_cur.at[:, d].set(jnp.where(s == 1, m_d, -m_d))
                return x_out, None

            keys = random.split(key, dim)
            x_new, _ = jax.lax.scan(per_coord, x, (jnp.arange(dim), keys))
            return x_new, jnp.asarray(1.0)

        return sweep

    return (make_sign_ffbs,)


@app.cell(hide_code=True)
def exp_a_md(mo):
    mo.md(r"""
    ## A. The factorised wall, attacked with segments

    Setup identical to §16's frontier: `T = 48`, `P = 16`, 2 500 sweeps, exact product
    gold. Three kernels, all through the byte-identical tree:

    - `amala_z` — reference-local paid leaf, no flips (the sign-stuck baseline);
    - `flip leaf` — the §16 per-time sign-symmetric-auxiliary flip (p = 0.1, the best
      configuration from the D-sweep);
    - `amala_z + segMH` — the same plain leaf, composed with `2D` segment sign-flip
      proposals per sweep (mean window length 8, half of them suffix windows).

    The nucleation hypothesis predicts: per-time flips fail for `D ≥ 16` (confirmed on
    Modal at scale), while segment moves — which propose the whole domain wall at once
    and pay only the two boundary terms plus the forcing interaction — should not care
    about `D` here at all, because each proposal is a *single* MH decision on one
    coordinate rather than a joint-coherence event across all `D`.
    """)
    return


@app.cell
def exp_a_run(kit_metrics, make_kit_d, make_leaf_sweep, make_seg_mh, np, product_gold, run_chain):
    _t_len, _p, _n_iter = 48, 16, 2500
    a_results = {}
    for _dim in (2, 8, 16, 30):
        _model = make_kit_d(_dim)
        _, _y = _model["simulate"](0, _t_len)
        _golds = product_gold(_y)
        _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
        _row = {}
        _leaf_plain = make_leaf_sweep(_model, _y, _p, delta=2.0, pflip=0.0)
        _leaf_flip = make_leaf_sweep(_model, _y, _p, delta=1.0, pflip=0.1)
        _mh = make_seg_mh(_model, _y, n_props=2 * _dim)
        for _name, _leaf, _mh_k in (
            ("amala_z", _leaf_plain, None),
            ("flip_leaf", _leaf_flip, None),
            ("amala_z+segMH", _leaf_plain, _mh),
        ):
            _chain, _acc = run_chain(_leaf, _mh_k, _x0, _n_iter, seed=5)
            _m = kit_metrics(_chain[_n_iter // 2 :], _golds)
            _m["mh_accept"] = _acc
            _row[_name] = _m
        a_results[_dim] = _row
    return (a_results,)


@app.cell
def exp_a_table(a_results, mo):
    _lines = [
        "| D | amala_z (stuck baseline) | flip leaf (per-time) | **amala_z + segMH** |",
        "|--:|---|---|---|",
    ]
    for _d, _row in a_results.items():
        _cells = []
        for _k in ("amala_z", "flip_leaf", "amala_z+segMH"):
            _m = _row[_k]
            _cells.append(f"{_m['w1_rel']:.3f} / {_m['sign_err']:.3f}")
        _acc = _row["amala_z+segMH"]["mh_accept"]
        _lines.append(f"| {_d} | {_cells[0]} | {_cells[1]} | **{_cells[2]}** (acc {_acc:.2f}) |")
    mo.md(
        "**Factorised Kitagawa-D vs the exact product gold** — `W1/σ̄` / sign-error "
        "(2 500 sweeps, single seed; the Modal D-sweep numbers for the first two "
        "kernels bracket these):\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_a_fig(a4_results, a_results, mo, palette, plt):
    _ds = sorted(a_results)
    _fig, _ax = plt.subplots(figsize=(7.2, 4.2))
    for _k, _label, _c in (
        ("amala_z", "amala_z (reference-local)", palette["state"]),
        ("flip_leaf", "per-time flip leaf", palette["obs"]),
        ("amala_z+segMH", "amala_z + segment MH", palette["belief"]),
    ):
        _v = [a_results[_d][_k]["sign_err"] for _d in _ds]
        _ax.plot(_ds, _v, "o-", color=_c, lw=2.0, ms=6, label=_label)
    _ax.plot(
        _ds,
        [a4_results[_d]["sign_err"] for _d in _ds],
        "o-",
        color=palette["ink"],
        lw=2.0,
        ms=6,
        label="amala_z + sign-path FFBS",
    )
    _ax.set_xscale("log", base=2)
    _ax.set_yscale("log")
    _ax.set_xticks(_ds, [str(_d) for _d in _ds])
    _ax.set_xlabel("state dimension D")
    _ax.set_ylabel("sign-probability error (exact gold)")
    _ax.axhline(0.02, color=palette["muted"], ls=":", lw=1.2)
    _ax.text(float(_ds[0]), 0.023, "≈ exact", fontsize=8, color=palette["muted"])
    _ax.set_title(
        "Segment moves do not care about D on the factorised wall",
        fontsize=11,
        fontweight="bold",
    )
    _ax.legend(frameon=False, fontsize=9)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def exp_a3_md(mo):
    mo.md(r"""
    ### A3. Feeding the sampler

    The A-table's residual gap at `D ∈ {16, 30}` comes with acceptance ≈ 1–2% and only
    `2D` proposals per sweep — each of the `T·D` sign variables receives roughly two
    accepted flips over the entire chain. If the gap is acceptance *starvation* rather
    than anything structural, multiplying the proposal count (they cost `O(T·D)` each —
    trivial next to the tree sweep) should close it. Same chains, `20D` proposals per
    sweep:
    """)
    return


@app.cell
def exp_a3_run(kit_metrics, make_kit_d, make_leaf_sweep, make_seg_mh, np, product_gold, run_chain):
    _t_len, _p, _n_iter = 48, 16, 2500
    a3_results = {}
    for _dim in (16, 30):
        _model = make_kit_d(_dim)
        _, _y = _model["simulate"](0, _t_len)
        _golds = product_gold(_y)
        _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
        _leaf = make_leaf_sweep(_model, _y, _p, delta=2.0, pflip=0.0)
        _mh = make_seg_mh(_model, _y, n_props=20 * _dim)
        _chain, _acc = run_chain(_leaf, _mh, _x0, _n_iter, seed=5)
        _m = kit_metrics(_chain[_n_iter // 2 :], _golds)
        _m["mh_accept"] = _acc
        a3_results[_dim] = _m
    return (a3_results,)


@app.cell
def exp_a3_table(a3_results, a_results, mo):
    _lines = [
        "| D | 2D proposals/sweep | 20D proposals/sweep |",
        "|--:|---|---|",
    ]
    for _d in (16, 30):
        _lo = a_results[_d]["amala_z+segMH"]
        _hi = a3_results[_d]
        _lines.append(
            f"| {_d} | {_lo['w1_rel']:.3f} / {_lo['sign_err']:.3f} "
            f"(acc {_lo['mh_accept']:.2f}) | "
            f"**{_hi['w1_rel']:.3f} / {_hi['sign_err']:.3f}** "
            f"(acc {_hi['mh_accept']:.2f}) |"
        )
    mo.md(
        "**A3 — the starvation test** (`W1/σ̄` / sign-error vs the exact product "
        "gold):\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def exp_a4_md(mo):
    mo.md(r"""
    ### A4. The exact-cluster limit: sign-path FFBS

    A3 says the binding constraint is window placement. Rather than *adapting* windows
    (the Swendsen–Wang route), one can go to the limit: in 1-D time, conditioned on the
    magnitudes, each coordinate's sign path is a **two-state Markov chain**, and its
    exact conditional is sampled in `O(T)` by discrete forward-filtering
    backward-sampling. This is the cluster construction with the stochasticity
    integrated out — perfect windows, no acceptance, no tuning. Composed with the tree
    sweep (magnitudes local, signs global), it should hit the exact floor at every `D`
    on the factorised wall.
    """)
    return


@app.cell
def exp_a4_run(
    kit_metrics, make_kit_d, make_leaf_sweep, make_sign_ffbs, np, product_gold, run_chain
):
    _t_len, _p, _n_iter = 48, 16, 2500
    a4_results = {}
    for _dim in (2, 8, 16, 30):
        _model = make_kit_d(_dim)
        _, _y = _model["simulate"](0, _t_len)
        _golds = product_gold(_y)
        _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
        _leaf = make_leaf_sweep(_model, _y, _p, delta=2.0, pflip=0.0)
        _ffbs = make_sign_ffbs(_model, _y)
        _chain, _ = run_chain(_leaf, _ffbs, _x0, _n_iter, seed=5)
        a4_results[_dim] = kit_metrics(_chain[_n_iter // 2 :], _golds)
    return (a4_results,)


@app.cell
def exp_a4_table(a3_results, a4_results, a_results, mo):
    _lines = [
        "| D | segMH (2D props) | segMH (20D props) | **sign-path FFBS** |",
        "|--:|---|---|---|",
    ]
    for _d in (2, 8, 16, 30):
        _seg = a_results[_d]["amala_z+segMH"]
        _seg20 = a3_results.get(_d)
        _f = a4_results[_d]
        _mid = f"{_seg20['w1_rel']:.3f} / {_seg20['sign_err']:.3f}" if _seg20 else "—"
        _lines.append(
            f"| {_d} | {_seg['w1_rel']:.3f} / {_seg['sign_err']:.3f} | {_mid} | "
            f"**{_f['w1_rel']:.3f} / {_f['sign_err']:.3f}** |"
        )
    mo.md(
        "**A4 — the exact-cluster limit vs the windowed approximations** "
        "(`W1/σ̄` / sign-error vs the exact product gold, same budget):\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def coord_leaf_cell(jnp, logn, make_tree, random):
    def make_coord_leaf_sweep(model, y, p, delta=2.0):
        """Coordinate-conditional tree sweep: each sweep picks ONE random coordinate d
        and proposes amala-z candidates for that coordinate only (all other
        coordinates copied from the reference into every particle).

        Every particle then differs from the reference in a single coordinate, so the
        stitch never faces the exponential-in-D joint-coherence lottery — it is cSMC
        on the coordinate-d conditional sub-model, with the full-transition seams
        automatically pricing every cross-coordinate coupling term. Random-scan over
        coordinates is pi-invariant; at kappa = 0 each sweep is exactly the D = 1
        kernel for its coordinate. The price: one sweep advances one coordinate, so
        chains need ~D x more sweeps for the same per-coordinate update count."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        tau = 0.5 * delta
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]

        def seam_pair(prev, nxt, seam):
            mm = drift(seam, prev)
            lp = jnp.sum(logn(nxt[None, :, :], mm[:, None, :], sig_v2), axis=-1)
            return jnp.where(seam < t_len, lp, 0.0)

        def seam_sel(prev, nxt, seam):
            mm = drift(seam, prev)
            lp = jnp.sum(logn(nxt, mm, sig_v2), axis=-1)
            return jnp.where(seam < t_len, lp, 0.0)

        smooth = make_tree(t_len, p, dim, seam_pair, seam_sel)

        def sweep(x_ref, key):
            kd, kz, kt = random.split(key, 3)
            d = random.randint(kd, (), 0, dim)
            z = x_ref[:, d] + jnp.sqrt(tau) * random.normal(kz, (t_len,))
            center = z + tau * model["obs_grad_1d"](z, y_j[:, d])  # (T,)

            def leaf(t, k):
                free = center[t] + jnp.sqrt(tau) * random.normal(k, (p - 1,))
                parts = jnp.broadcast_to(x_ref[t], (p, dim))
                parts = parts.at[1:, d].set(free)
                psi = (
                    model["log_obs"](parts, y_j[t])
                    + logn(z[t], parts[:, d], tau)
                    - logn(parts[:, d], center[t], tau)
                )
                psi = jnp.where(t == 0, psi + jnp.sum(logn(parts, 0.0, init_var), -1), psi)
                return parts, psi[:, None]

            return smooth(kt, leaf)

        return sweep

    return (make_coord_leaf_sweep,)


@app.cell
def branch_ffbs_cell(jax, jnp, logn, random):
    def make_branch_ffbs(model, y, include_jacobian=True):
        """Branch-path FFBS for a GENERAL two-branch emission fold.

        Parameterise trajectories by (canonical branch representative, branch path s):
        r_t = branch_rep(x_t) is the positive preimage of the emission level and
        z_t(1) = branch_mirror(r_t) the other one. The conditional of s given the
        representatives is a two-state chain whose weights carry the change-of-
        variables factor |d branch_mirror / d r| per mirrored site — the Jacobian
        SITE term. With it, sampling s by FFBS is an exact Gibbs step on the discrete
        component (the sign-symmetric case is beta = 0 with log-Jacobian 0);
        include_jacobian=False is the deliberate NEGATIVE CONTROL showing the
        correction is load-bearing."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]
        t_arr = jnp.arange(1, t_len)
        log_jac = model["branch_log_jac"] if include_jacobian else 0.0

        def sweep(x, key):
            def per_coord(x_cur, inp):
                d, kd = inp
                r_d = model["branch_rep"](x_cur[:, d])
                xp = x_cur.at[:, d].set(r_d)
                xm = x_cur.at[:, d].set(model["branch_mirror"](r_d))
                mu_p = drift(t_arr[:, None], xp[:-1])
                mu_m = drift(t_arr[:, None], xm[:-1])

                def tl(xt, mu):
                    return jnp.sum(logn(xt, mu, sig_v2), axis=-1)

                lp = jnp.stack(
                    [
                        jnp.stack([tl(xm[1:], mu_m), tl(xp[1:], mu_m)], axis=-1),
                        jnp.stack([tl(xm[1:], mu_p), tl(xp[1:], mu_p)], axis=-1),
                    ],
                    axis=-2,
                )  # (T-1, s_prev, s_cur) with s = 1 the canonical branch
                e = jnp.stack(
                    [
                        jax.vmap(model["log_obs"])(xm, y_j) + log_jac,
                        jax.vmap(model["log_obs"])(xp, y_j),
                    ],
                    axis=-1,
                )  # (T, 2)
                alpha0 = (
                    jnp.stack(
                        [
                            jnp.sum(logn(xm[0], 0.0, init_var)),
                            jnp.sum(logn(xp[0], 0.0, init_var)),
                        ]
                    )
                    + e[0]
                )

                def fstep(alpha, inp2):
                    lp_t, e_t = inp2
                    a_new = e_t + jax.scipy.special.logsumexp(alpha[:, None] + lp_t, axis=0)
                    return a_new, a_new

                alpha_last, alphas = jax.lax.scan(fstep, alpha0, (lp, e[1:]))
                alphas_all = jnp.concatenate([alpha0[None], alphas[:-1]], axis=0)
                k_last, k_back = random.split(kd)
                s_last = random.categorical(k_last, alpha_last)

                def bstep(s_next, inp2):
                    alpha_t, lp_t, k_t = inp2
                    s_t = random.categorical(k_t, alpha_t + lp_t[:, s_next])
                    return s_t, s_t

                keys_b = random.split(k_back, t_len - 1)
                _, s_rev = jax.lax.scan(
                    bstep, s_last, (jnp.flip(alphas_all, 0), jnp.flip(lp, 0), keys_b)
                )
                s = jnp.concatenate([jnp.flip(s_rev), s_last[None]])
                x_out = x_cur.at[:, d].set(jnp.where(s == 1, r_d, model["branch_mirror"](r_d)))
                return x_out, None

            keys = random.split(key, dim)
            x_new, _ = jax.lax.scan(per_coord, x, (jnp.arange(dim), keys))
            return x_new, jnp.asarray(1.0)

        return sweep

    return (make_branch_ffbs,)


@app.cell(hide_code=True)
def exp_a5_md(mo):
    mo.md(r"""
    ### A5. What the shared plateau is

    At `D ∈ {16, 30}` the three sign kernels — including the *exact* conditional
    FFBS — land on numerically identical residuals. An exact sign sampler cannot be
    beaten by an approximate one, so signs are no longer the bottleneck; the common
    plateau must be the remaining shared component: **the magnitude path through the
    tree sweep**, the ordinary unimodal high-D slowness measured in §16 (`amala_z`
    ESS/sweep ≈ 0.0025 at `D = 30`). Two readings are possible: slow-but-moving
    magnitudes (the residual is Monte-Carlo error and shrinks with chain length by the
    √-law) or effectively *frozen* magnitudes (the sign chain keeps seeing the same
    distorted wall costs — inflated small magnitudes from the `√(20 y⁺)` init at
    exactly the sign-flexible sites — and the residual is a fixed conditional-law
    offset that no amount of sign sampling repairs). 4× the sweeps discriminates:
    """)
    return


@app.cell
def exp_a5_run(
    a4_results,
    kit_metrics,
    make_kit_d,
    make_leaf_sweep,
    make_sign_ffbs,
    mo,
    np,
    product_gold,
    run_chain,
):
    _t_len, _p, _n_iter = 48, 16, 10000
    _model = make_kit_d(30)
    _, _y = _model["simulate"](0, _t_len)
    _golds = product_gold(_y)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _leaf = make_leaf_sweep(_model, _y, _p, delta=2.0, pflip=0.0)
    _ffbs = make_sign_ffbs(_model, _y)
    _chain, _ = run_chain(_leaf, _ffbs, _x0, _n_iter, seed=5)
    _m10k = kit_metrics(_chain[_n_iter // 2 :], _golds)
    _m25 = a4_results[30]
    mo.md(
        "**A5 — the discrimination** (D = 30, sign-path FFBS): "
        f"2 500 sweeps → `W1/σ̄` {_m25['w1_rel']:.3f} / sign-error "
        f"{_m25['sign_err']:.3f}; **10 000 sweeps → {_m10k['w1_rel']:.3f} / "
        f"{_m10k['sign_err']:.3f}**, against a √-law prediction of "
        f"≈ {_m25['sign_err'] / 2:.3f} for pure mixing error. The plateau barely "
        "moves: the magnitudes are effectively frozen at this (P, leaf, D), and the "
        "sign chain faithfully samples the sign law *conditioned on distorted "
        "magnitudes*. The sign problem is solved; the residual belongs entirely to "
        "the unimodal magnitude engine."
    )
    return


@app.cell(hide_code=True)
def exp_b_md(mo):
    mo.md(r"""
    ## B. Coupling — when the sign problems stop separating

    The factorised benchmark is the *easy* version of high dimension for a composed
    move: coordinates never interact, so each segment proposal is a private 1-D
    problem. Real models couple coordinates. Two checks:

    - **B1 (exactness under coupling, exact gold).** Add mean-field drift coupling
      `κ (x̄ − x_d)` at `D = 2`, `κ = 0.15`, and build a *joint* 2-D grid smoother
      (coarse but converged for these tolerances). The segment-MH ratio is computed
      from the full trajectory density, so coupling terms — including the effect of a
      coordinate-d flip on the *other* coordinates' transition densities — enter
      exactly. The kernel must stay exact.
    - **B2 (directional, no gold).** The same coupled model at `D = 30`: three seeds of
      tree + segMH, checked for cross-seed agreement of the per-(t, d) sign
      probabilities — the honest, gold-free probe of whether segment moves keep mixing
      signs when flips in one coordinate re-price every other coordinate's transitions.
    """)
    return


@app.cell
def gold_2d_cell(KIT_INIT_SD, KIT_SIG_V, np):
    def joint_gold_2d(model, y, n_grid=71, lo=-26.0, hi=26.0):
        """Joint 2-D grid smoother for the coupled model; returns per-coordinate
        marginal summaries in the same format as the product gold."""
        import jax.numpy as jnp

        t_len = y.shape[0]
        xs = np.linspace(lo, hi, n_grid)
        g1, g2 = np.meshgrid(xs, xs, indexing="ij")
        states = np.stack([g1.ravel(), g2.ravel()], axis=1)  # (N, 2)
        n_states = states.shape[0]

        def _logn(v, mu, sd):
            return -0.5 * (np.log(2.0 * np.pi * sd**2) + ((v - mu) ** 2) / sd**2)

        def _trans(t):
            mu = np.asarray(model["drift"](t, jnp.asarray(states)))  # (N, 2)
            lt = _logn(states[None, :, 0], mu[:, None, 0], KIT_SIG_V)
            lt += _logn(states[None, :, 1], mu[:, None, 1], KIT_SIG_V)
            return lt  # (N, N)

        log_obs = np.stack(
            [
                _logn(y[t, 0], states[:, 0] ** 2 / 20.0, model["sig_w"])
                + _logn(y[t, 1], states[:, 1] ** 2 / 20.0, model["sig_w"])
                for t in range(t_len)
            ]
        )
        log_alpha = np.zeros((t_len, n_states))
        log_alpha[0] = (
            _logn(states[:, 0], 0.0, KIT_INIT_SD)
            + _logn(states[:, 1], 0.0, KIT_INIT_SD)
            + log_obs[0]
        )
        for t in range(1, t_len):
            lt = _trans(t)
            a = log_alpha[t - 1]
            m = a.max()
            log_alpha[t] = np.log(np.exp(a - m) @ np.exp(lt) + 1e-300) + m + log_obs[t]
        log_beta = np.zeros((t_len, n_states))
        for t in range(t_len - 2, -1, -1):
            lt = _trans(t + 1)
            b = log_beta[t + 1] + log_obs[t + 1]
            m = b.max()
            log_beta[t] = np.log(np.exp(lt) @ np.exp(b - m) + 1e-300) + m
        g = np.exp((log_alpha + log_beta) - (log_alpha + log_beta).max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        g = g.reshape(t_len, n_grid, n_grid)
        golds = []
        for d in range(2):
            gm = g.sum(axis=2 - d)  # marginal over the OTHER coordinate
            mean = (gm * xs[None, :]).sum(1)
            sd = np.sqrt((gm * (xs[None, :] - mean[:, None]) ** 2).sum(1))
            golds.append(
                {
                    "xs": xs,
                    "dx": xs[1] - xs[0],
                    "cdf": np.cumsum(gm, 1),
                    "p_pos": gm[:, xs > 0].sum(1),
                    "sd": sd,
                }
            )
        return golds

    return (joint_gold_2d,)


@app.cell
def exp_b1_run(
    joint_gold_2d,
    kit_metrics,
    make_kit_d,
    make_leaf_sweep,
    make_seg_mh,
    make_sign_ffbs,
    np,
    run_chain,
):
    _t_len, _p, _n_iter = 32, 16, 3000
    _model = make_kit_d(2, kappa=0.15)
    _, _y = _model["simulate"](0, _t_len)
    _golds = joint_gold_2d(_model, _y)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _leaf = make_leaf_sweep(_model, _y, _p, delta=2.0, pflip=0.0)
    _mh = make_seg_mh(_model, _y, n_props=4)
    _ffbs = make_sign_ffbs(_model, _y)
    b1_results = {}
    for _name, _mh_k in (
        ("amala_z", None),
        ("amala_z+segMH", _mh),
        ("amala_z+signFFBS", _ffbs),
    ):
        _chain, _acc = run_chain(_leaf, _mh_k, _x0, _n_iter, seed=5)
        _m = kit_metrics(_chain[_n_iter // 2 :], _golds)
        _m["mh_accept"] = _acc
        b1_results[_name] = _m
    return (b1_results,)


@app.cell
def exp_b1_table(b1_results, mo):
    _lines = ["| kernel | W1/σ̄ vs joint 2-D gold | sign error | MH accept |", "|---|--:|--:|--:|"]
    for _k, _m in b1_results.items():
        _lines.append(
            f"| {_k} | {_m['w1_rel']:.3f} | {_m['sign_err']:.3f} | {_m['mh_accept']:.2f} |"
        )
    mo.md(
        "**B1 — coupled (κ = 0.15) D = 2 against the exact joint grid gold.** The "
        "segment-MH ratio includes every cross-coordinate coupling term, so exactness "
        "must survive coupling:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_b2_run(make_kit_d, make_leaf_sweep, make_seg_mh, make_sign_ffbs, np, run_chain):
    _t_len, _p, _n_iter = 48, 16, 2500
    _model = make_kit_d(30, kappa=0.15)
    _, _y = _model["simulate"](0, _t_len)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _leaf = make_leaf_sweep(_model, _y, _p, delta=2.0, pflip=0.0)
    _kernels = {
        "segMH": make_seg_mh(_model, _y, n_props=60),
        "signFFBS": make_sign_ffbs(_model, _y),
    }
    b2_results = {}
    for _name, _mh in _kernels.items():
        _sign_tracks = []
        for _seed in (5, 6, 7):
            _chain, _ = run_chain(_leaf, _mh, _x0, _n_iter, seed=_seed)
            _sign_tracks.append((_chain[_n_iter // 2 :] > 0).mean(0))  # (T, D)
        _tracks = np.stack(_sign_tracks)
        b2_results[_name] = {
            "seed_sign_rmse": float(np.sqrt(np.mean(_tracks.var(0)))),
            "frac_bimodal": float(np.mean((_tracks.mean(0) > 0.05) & (_tracks.mean(0) < 0.95))),
        }
    return (b2_results,)


@app.cell
def exp_b2_table(b2_results, mo):
    _lines = [
        "| sign kernel | cross-seed sign RMS ↓ | fraction sign-mixed |",
        "|---|--:|--:|",
    ]
    for _name, _m in b2_results.items():
        _lines.append(f"| {_name} | {_m['seed_sign_rmse']:.3f} | {_m['frac_bimodal']:.2f} |")
    mo.md(
        "**B2 — coupled (κ = 0.15) D = 30, three seeds, no gold (directional).** "
        "Cross-seed RMS deviation of the per-(t, d) sign probabilities is the "
        "gold-free convergence signal (small = the chains agree on the full sign "
        "law):\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def exp_c_md(mo):
    mo.md(r"""
    ## C. Closing the loop: unfreeze the magnitudes

    A5 localised the plateau in the magnitude engine. Two levers, both from first
    principles:

    - **Width** (`P = 64`): §16's width-axis result — tree kernels convert particles
      into mixing, because extra candidates dilute reference retention at every
      stitch. A blunt lever: the joint leaf still asks all `D` coordinates to be
      coherent at once.
    - **Coordinate-conditional sweeps** (`P = 16`): one random coordinate per sweep
      through the same tree with full seams. This removes the joint-coherence lottery
      entirely — at κ = 0 each sweep *is* the `D = 1` kernel, which sits at the gold
      floor — at the price of needing ~`D`× more sweeps to service every coordinate
      equally.

    Both compose with sign-path FFBS exactly as before.
    """)
    return


@app.cell
def exp_c1_run(
    kit_metrics,
    make_coord_leaf_sweep,
    make_kit_d,
    make_leaf_sweep,
    make_sign_ffbs,
    np,
    product_gold,
    run_chain,
):
    _t_len = 48
    _model = make_kit_d(30)
    _, _y = _model["simulate"](0, _t_len)
    _golds = product_gold(_y)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _ffbs = make_sign_ffbs(_model, _y)
    c1_results = {}
    _chain, _ = run_chain(
        make_leaf_sweep(_model, _y, 64, delta=2.0, pflip=0.0), _ffbs, _x0, 2500, seed=5
    )
    c1_results["joint P=64, 2.5k sweeps"] = kit_metrics(_chain[1250:], _golds)
    _chain, _ = run_chain(
        make_coord_leaf_sweep(_model, _y, 16, delta=2.0), _ffbs, _x0, 15000, seed=5
    )
    c1_results["coord-conditional P=16, 15k sweeps"] = kit_metrics(_chain[7500:], _golds)
    return (c1_results,)


@app.cell
def exp_c1_table(a4_results, c1_results, mo):
    _base = a4_results[30]
    _lines = [
        "| magnitude engine (all + sign-path FFBS) | W1/σ̄ | sign error |",
        "|---|--:|--:|",
        f"| joint P=16, 2.5k sweeps (A4 baseline) | {_base['w1_rel']:.3f} "
        f"| {_base['sign_err']:.3f} |",
    ]
    for _name, _m in c1_results.items():
        _lines.append(f"| {_name} | **{_m['w1_rel']:.3f}** | **{_m['sign_err']:.3f}** |")
    mo.md(
        "**C1 — factorised D = 30 against the exact product gold.** The frozen-"
        "magnitude diagnosis predicts the coordinate-conditional sweep (which reduces "
        "to the D = 1 kernel per coordinate) recovers the floor:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_c2_run(make_coord_leaf_sweep, make_kit_d, make_sign_ffbs, np, run_chain):
    _t_len, _n_iter = 48, 6000
    _model = make_kit_d(30, kappa=0.15)
    _, _y = _model["simulate"](0, _t_len)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _leaf = make_coord_leaf_sweep(_model, _y, 16, delta=2.0)
    _ffbs = make_sign_ffbs(_model, _y)
    _tracks = []
    for _seed in (5, 6):
        _chain, _ = run_chain(_leaf, _ffbs, _x0, _n_iter, seed=_seed)
        _tracks.append((_chain[_n_iter // 2 :] > 0).mean(0))
    _t = np.stack(_tracks)
    c2_results = {
        "seed_sign_rmse": float(np.sqrt(np.mean(_t.var(0)))),
        "frac_bimodal": float(np.mean((_t.mean(0) > 0.05) & (_t.mean(0) < 0.95))),
    }
    return (c2_results,)


@app.cell
def exp_c2_table(b2_results, c2_results, mo):
    mo.md(
        "**C2 — coupled (κ = 0.15) D = 30, coordinate-conditional + FFBS.** "
        f"Cross-seed sign RMS **{c2_results['seed_sign_rmse']:.3f}** "
        f"(B2's joint-leaf kernels: segMH {b2_results['segMH']['seed_sign_rmse']:.3f}, "
        f"signFFBS {b2_results['signFFBS']['seed_sign_rmse']:.3f}); fraction "
        f"sign-mixed {c2_results['frac_bimodal']:.2f}."
    )
    return


@app.cell(hide_code=True)
def exp_d_md(mo):
    mo.md(r"""
    ## D. Generalisation: asymmetric folds and the Jacobian

    Everything so far leaned on an *exactly symmetric* fold (`x → −x`, unit Jacobian).
    Production measurement links are not symmetric. The general construction: for a
    two-branch emission `h`, parameterise each coordinate's trajectory by its
    *canonical branch representative* `r_t` (the positive preimage of the emission
    level) and a branch path `s`; the conditional of `s` given the representatives is
    still a two-state chain, but its site weights must carry the change-of-variables
    factor `|d ρ/d r|` on mirrored sites — the **Jacobian site term**. With it, the
    branch-path FFBS is an exact Gibbs step for arbitrary folds.

    Test bed: the Kitagawa-D dynamics with `y = (1 + β·sign(x))·|x|/3 + N(0, 1)`,
    β = 0.3 — preimages at different slopes (mirror Jacobian `(1+β)/(1−β) ≈ 1.86`),
    asymmetric mode locations, exact per-coordinate grid gold. Three kernels, all on
    the coordinate-conditional magnitude engine from C:

    - no branch moves (the stuck baseline);
    - branch-path FFBS **with** the Jacobian term (the claim: exact);
    - branch-path FFBS **without** it (the negative control: an exact-looking sampler
      whose branch *weights* are systematically wrong — the failure a naive port
      would ship).
    """)
    return


@app.cell
def exp_d_run(
    kit_metrics,
    make_branch_ffbs,
    make_coord_leaf_sweep,
    make_fold_d,
    np,
    product_gold_fold,
    run_chain,
):
    _t_len, _p, _beta = 48, 16, 0.3
    d_results = {}
    for _dim, _n_iter in ((2, 6000), (16, 10000)):
        _model = make_fold_d(_dim, beta=_beta)
        _, _y = _model["simulate"](0, _t_len)
        _golds = product_gold_fold(_y, _beta)
        _x0 = np.abs(_y) * 3.0 / (1.0 + _beta)  # crude positive-branch init
        _leaf = make_coord_leaf_sweep(_model, _y, _p, delta=2.0)
        _row = {}
        for _name, _mh in (
            ("coord only (stuck)", None),
            ("+ branch FFBS (Jacobian)", make_branch_ffbs(_model, _y, True)),
            ("+ branch FFBS, NO Jacobian", make_branch_ffbs(_model, _y, False)),
        ):
            _chain, _ = run_chain(_leaf, _mh, _x0, _n_iter, seed=5)
            _row[_name] = kit_metrics(_chain[_n_iter // 2 :], _golds)
        d_results[_dim] = _row
    return (d_results,)


@app.cell
def exp_d_table(d_results, mo):
    _lines = [
        "| D | kernel | W1/σ̄ | branch-prob error |",
        "|--:|---|--:|--:|",
    ]
    for _dim, _row in d_results.items():
        for _name, _m in _row.items():
            _lines.append(f"| {_dim} | {_name} | {_m['w1_rel']:.3f} | {_m['sign_err']:.3f} |")
    mo.md(
        "**D — asymmetric fold (β = 0.3) vs the exact product gold.** "
        "`branch-prob error` is the RMSE of `P(x > 0)` — exactly the quantity a "
        "missing Jacobian corrupts:\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def exp_e_md(mo):
    mo.md(r"""
    ## E. Six low-hanging fruits

    With the loop closed, six cheap extensions — each one cell, each with its own
    exact check. E1 reunites the §15 1-D leaf toolbox with the any-D engine; E2 is the
    production-shaped factor-sign flip; E3 makes the sign pass parallel-in-time; E4
    finds the optimal coordinate-block size; E5 removes the δ tuning surface; E6
    generalises the branch chain to folds with more than two preimages.
    """)
    return


@app.cell
def block_leaf_cell(jax, jnp, logn, make_tree, np, random):
    def make_block_leaf_sweep(model, y, p, k):
        """Block-k coordinate-conditional amala-z tree sweep (k = 1 is the C engine;
        k = D is the joint sweep). tau is a traced argument so E5 can adapt it."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]

        def seam_pair(prev, nxt, seam):
            mm = drift(seam, prev)
            lp = jnp.sum(logn(nxt[None, :, :], mm[:, None, :], sig_v2), axis=-1)
            return jnp.where(seam < t_len, lp, 0.0)

        def seam_sel(prev, nxt, seam):
            mm = drift(seam, prev)
            return jnp.where(seam < t_len, jnp.sum(logn(nxt, mm, sig_v2), axis=-1), 0.0)

        smooth = make_tree(t_len, p, dim, seam_pair, seam_sel)

        def sweep(x_ref, key, tau):
            kd, kz, kt = random.split(key, 3)
            coords = random.permutation(kd, dim)[:k]
            cmask = jnp.zeros(dim, bool).at[coords].set(True)
            z = x_ref[:, coords] + jnp.sqrt(tau) * random.normal(kz, (t_len, k))
            center = z + tau * jax.vmap(
                lambda zz, yy: model["obs_grad_1d"](zz, yy), in_axes=(1, 1), out_axes=1
            )(z, y_j[:, coords])

            def leaf(t, kk):
                free = center[t] + jnp.sqrt(tau) * random.normal(kk, (p - 1, k))
                parts = jnp.broadcast_to(x_ref[t], (p, dim))
                parts = parts.at[1:, coords].set(free)
                psi = (
                    model["log_obs"](parts, y_j[t])
                    + jnp.sum(logn(z[t], parts[:, coords], tau), -1)
                    - jnp.sum(logn(parts[:, coords], center[t], tau), -1)
                )
                psi = jnp.where(t == 0, psi + jnp.sum(logn(parts, 0.0, init_var), -1), psi)
                return parts, psi[:, None]

            return smooth(kt, leaf), cmask

        return sweep

    def run_composed_e(sweep, ffbs, x0, n_iter, seed, tau):
        def body(x, key):
            k1, k2 = random.split(key)
            x, _ = sweep(x, k1, tau)
            if ffbs is not None:
                x = ffbs(x, k2)
            return x, x

        keys = random.split(random.PRNGKey(seed), n_iter)
        _, chain = jax.jit(lambda ks: jax.lax.scan(body, x0, ks))(keys)
        return np.asarray(chain)

    return make_block_leaf_sweep, run_composed_e


@app.cell
def mbranch_cell(jax, jnp, logn, random):
    def make_mbranch_ffbs(model, y, branch_values):
        """m-state branch-path FFBS: branch_values(v_col) -> (candidates (m, T),
        valid mask (m, T), per-branch log-Jacobians (m,)). Invalid branches (folds
        whose preimage count varies with level) are masked with -inf site weight.
        The 2-state sign chain is m = 2 with unit Jacobians."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        t_arr = jnp.arange(1, t_len)
        n_br = int(branch_values(jnp.zeros(3))[0].shape[0])

        def sweep(x, key):
            def per_coord(x_cur, inp):
                d, kd = inp
                cands, valid, log_jac = branch_values(x_cur[:, d])
                xvars = jax.vmap(lambda c: x_cur.at[:, d].set(c))(cands)  # (m, T, D)
                mus = jax.vmap(lambda xv: model["drift"](t_arr[:, None], xv[:-1]))(xvars)

                def tl(j, i):
                    return jnp.sum(logn(xvars[j][1:], mus[i], sig_v2), -1)

                lp = jax.vmap(lambda i: jax.vmap(lambda j: tl(j, i))(jnp.arange(n_br)))(
                    jnp.arange(n_br)
                )
                lp = jnp.transpose(lp, (2, 0, 1))  # (T-1, prev, cur)
                e = jax.vmap(lambda xv: jax.vmap(model["log_obs"])(xv, y_j))(xvars).T
                e = e + log_jac[None, :] + jnp.where(valid.T, 0.0, -1e30)
                alpha0 = jax.vmap(lambda xv: jnp.sum(logn(xv[0], 0.0, init_var)))(xvars) + e[0]

                def fstep(alpha, inp2):
                    lp_t, e_t = inp2
                    a_new = e_t + jax.scipy.special.logsumexp(alpha[:, None] + lp_t, axis=0)
                    return a_new, a_new

                alpha_last, alphas = jax.lax.scan(fstep, alpha0, (lp, e[1:]))
                alphas_all = jnp.concatenate([alpha0[None], alphas[:-1]], 0)
                k_last, k_back = random.split(kd)
                s_last = random.categorical(k_last, alpha_last)

                def bstep(s_next, inp2):
                    alpha_t, lp_t, k_t = inp2
                    s_t = random.categorical(k_t, alpha_t + lp_t[:, s_next])
                    return s_t, s_t

                keys_b = random.split(k_back, t_len - 1)
                _, s_rev = jax.lax.scan(
                    bstep, s_last, (jnp.flip(alphas_all, 0), jnp.flip(lp, 0), keys_b)
                )
                s = jnp.concatenate([jnp.flip(s_rev), s_last[None]])
                x_out = x_cur.at[:, d].set(jnp.take_along_axis(cands, s[None, :], 0)[0])
                return x_out, None

            keys = random.split(key, dim)
            x_new, _ = jax.lax.scan(per_coord, x, (jnp.arange(dim), keys))
            return x_new

        return sweep

    def sign_branches(v):
        r = jnp.abs(v)
        return jnp.stack([r, -r]), jnp.ones((2, v.shape[0]), bool), jnp.zeros(2)

    return make_mbranch_ffbs, sign_branches


@app.cell
def coord_root_cell(jax, jnp, logn, make_tree, np, random):
    def make_coord_root_sweep(model, y, p, root_sd=2.5, w_root=0.45, inflate=3.0):
        """E1: the coordinate-conditional engine with the §15 1-D twisted-ROOT leaf —
        fixed per-coordinate mixture at both emission roots plus a damped-Laplace
        pilot. Independence leaves die in JOINT D but each sweep here is a D = 1
        problem, where they were best-in-class. tau argument ignored (fixed leaf)."""
        y_j = jnp.asarray(y)
        t_len, dim = y.shape
        sig_v2, init_var = model["sig_v"] ** 2, model["init_sd"] ** 2
        drift = model["drift"]

        def seam_pair(prev, nxt, seam):
            mm = drift(seam, prev)
            lp = jnp.sum(logn(nxt[None, :, :], mm[:, None, :], sig_v2), axis=-1)
            return jnp.where(seam < t_len, lp, 0.0)

        def seam_sel(prev, nxt, seam):
            mm = drift(seam, prev)
            return jnp.where(seam < t_len, jnp.sum(logn(nxt, mm, sig_v2), axis=-1), 0.0)

        smooth = make_tree(t_len, p, dim, seam_pair, seam_sel)
        # vectorised per-coordinate damped-Laplace pilot (kit x²/20 emission)
        t_idx = jnp.arange(t_len)
        dj = jax.vmap(jax.vmap(jax.grad(drift, argnums=1), in_axes=(None, 0)), (0, 0))

        def g1(x):
            return (y_j - x**2 / 20.0) * (x / 10.0)

        def g2m(x):
            return x**2 / 100.0 - (y_j - x**2 / 20.0) / 10.0

        @jax.jit
        def rts(x_hat):
            f = dj(t_idx[1:], x_hat[:-1])
            b = jax.vmap(drift)(t_idx[1:], x_hat[:-1]) - f * x_hat[:-1]
            f_all = jnp.concatenate([jnp.zeros((1, dim)), f])
            b_all = jnp.concatenate([jnp.zeros((1, dim)), b])
            lam = jnp.clip(g2m(x_hat), 1e-3, 5.0)
            z = x_hat + jnp.clip(g1(x_hat) / lam, -5.0, 5.0)
            r = 1.0 / lam

            def kf(c, inp):
                m_prev, p_prev = c
                z_t, r_t, f_t, b_t, first = inp
                mp = jnp.where(first, 0.0, f_t * m_prev + b_t)
                pp = jnp.where(first, init_var, f_t**2 * p_prev + sig_v2)
                gain = pp / (pp + r_t)
                return (mp + gain * (z_t - mp), (1.0 - gain) * pp), (
                    mp,
                    pp,
                    mp + gain * (z_t - mp),
                    (1.0 - gain) * pp,
                )

            first = jnp.concatenate([jnp.ones((1,)), jnp.zeros((t_len - 1,))])
            (_, _), (mp, pp, mf, pf) = jax.lax.scan(
                kf, (jnp.zeros(dim), jnp.zeros(dim)), (z, r, f_all, b_all, first)
            )

            def rstep(c, inp):
                m_next, p_next = c
                mf_t, pf_t, mpn, ppn, f_next = inp
                gg = pf_t * f_next / jnp.maximum(ppn, 1e-12)
                return (mf_t + gg * (m_next - mpn), pf_t + gg**2 * (p_next - ppn)), (
                    mf_t + gg * (m_next - mpn),
                    pf_t + gg**2 * (p_next - ppn),
                )

            inp = (mf[:-1], pf[:-1], mp[1:], pp[1:], f_all[1:])
            inp_rev = jax.tree_util.tree_map(lambda a: jnp.flip(a, 0), inp)
            (_, _), (msr, psr) = jax.lax.scan(rstep, (mf[-1], pf[-1]), inp_rev)
            return (
                jnp.concatenate([jnp.flip(msr, 0), mf[-1:]], 0),
                jnp.concatenate([jnp.flip(psr, 0), pf[-1:]], 0),
            )

        raw = np.sqrt(np.clip(20.0 * np.asarray(y), 0.0, None))
        x_hat = jnp.asarray(
            np.stack(
                [
                    np.convolve(np.pad(raw[:, d], (3, 3), mode="edge"), np.ones(7) / 7.0, "valid")
                    for d in range(dim)
                ],
                1,
            )
        )
        for _ in range(30):
            mu, _ = rts(x_hat)
            x_hat = 0.75 * x_hat + 0.25 * mu
        mu_q, var_q = rts(x_hat)
        var_q = inflate * var_q
        root = jnp.sqrt(jnp.clip(20.0 * y_j, 0.0, None))
        log_w3 = jnp.log(jnp.asarray([w_root, w_root, 1.0 - 2.0 * w_root]))

        def sweep(x_ref, key, tau):
            del tau
            kd, kt = random.split(key)
            d = random.randint(kd, (), 0, dim)

            def log_q(xs, t):
                comp = jnp.stack(
                    [
                        log_w3[0] + logn(xs, root[t, d], root_sd**2),
                        log_w3[1] + logn(xs, -root[t, d], root_sd**2),
                        log_w3[2] + logn(xs, mu_q[t, d], var_q[t, d]),
                    ]
                )
                return jax.scipy.special.logsumexp(comp, axis=0)

            def leaf(t, kk):
                ck, dk = random.split(kk)
                comp = random.categorical(ck, log_w3, shape=(p - 1,))
                cen = jnp.stack([root[t, d], -root[t, d], mu_q[t, d]])[comp]
                sd = jnp.stack([root_sd, root_sd, jnp.sqrt(var_q[t, d])])[comp]
                free = cen + sd * random.normal(dk, (p - 1,))
                parts = jnp.broadcast_to(x_ref[t], (p, dim))
                parts = parts.at[1:, d].set(free)
                psi = model["log_obs"](parts, y_j[t]) - log_q(parts[:, d], t)
                psi = jnp.where(t == 0, psi + jnp.sum(logn(parts, 0.0, init_var), -1), psi)
                return parts, psi[:, None]

            return smooth(kt, leaf), jnp.zeros(dim, bool).at[d].set(True)

        return sweep

    return (make_coord_root_sweep,)


@app.cell
def exp_e1_run(
    kit_metrics,
    make_block_leaf_sweep,
    make_coord_root_sweep,
    make_kit_d,
    make_mbranch_ffbs,
    np,
    product_gold,
    run_composed_e,
    sign_branches,
):
    _t_len, _p, _n_iter = 48, 16, 15000
    _model = make_kit_d(30)
    _, _y = _model["simulate"](0, _t_len)
    _golds = product_gold(_y)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _ffbs = make_mbranch_ffbs(_model, _y, sign_branches)
    e1_results = {}
    for _name, _sweep, _mh in (
        ("coord-amala + FFBS (C1)", make_block_leaf_sweep(_model, _y, _p, 1), _ffbs),
        ("coord-ROOT leaf alone", make_coord_root_sweep(_model, _y, _p), None),
        ("coord-ROOT + FFBS", make_coord_root_sweep(_model, _y, _p), _ffbs),
    ):
        _chain = run_composed_e(_sweep, _mh, _x0, _n_iter, seed=5, tau=1.0)
        e1_results[_name] = kit_metrics(_chain[_n_iter // 2 :], _golds)
    return (e1_results,)


@app.cell
def exp_e1_table(e1_results, mo):
    _lines = ["| kernel (D = 30, 15k sweeps) | W1/σ̄ | sign error |", "|---|--:|--:|"]
    for _n, _m in e1_results.items():
        _lines.append(f"| {_n} | {_m['w1_rel']:.3f} | {_m['sign_err']:.3f} |")
    mo.md(
        "**E1 — the §15 toolbox transfers per coordinate.** The 1-D twisted-root leaf "
        "(which dies in joint D) is best-in-class inside the coordinate-conditional "
        "engine — its independence is safe again because each sweep is a D = 1 "
        "problem, and its root components do the sign moves themselves:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_e2_run(jax, jnp, logn, make_kit_d, make_tree, np, random):
    # E2: factor-SSM (y = λ_d x_d + noise) mini-Gibbs with the joint (x, λ) sign flip.
    # Odd drift (forcing off) + symmetric priors => the flip is an EXACT symmetry:
    # its MH delta is identically 0, and the true posterior has P(λ_d > 0) = 1/2.
    _t_len, _dim, _p, _n_iter = 48, 4, 16, 4000
    _sig_y, _sig_lam = 2.0, 2.0
    _lam_true = np.array([1.0, 0.7, -0.9, 0.5])
    _model = make_kit_d(_dim, forcing=False)
    _sig_v2, _init_var = _model["sig_v"] ** 2, _model["init_sd"] ** 2
    _rng = np.random.default_rng(0)
    _x_true = np.zeros((_t_len, _dim))
    _x_true[0] = _model["init_sd"] * _rng.standard_normal(_dim)
    for _t in range(1, _t_len):
        _x_true[_t] = np.asarray(_model["drift"](_t, jnp.asarray(_x_true[_t - 1]))) + _model[
            "sig_v"
        ] * _rng.standard_normal(_dim)
    _y = _lam_true * _x_true + _sig_y * _rng.standard_normal((_t_len, _dim))
    _y_j = jnp.asarray(_y)

    def _seam_pair(prev, nxt, seam):
        _mm = _model["drift"](seam, prev)
        return jnp.where(
            seam < _t_len, jnp.sum(logn(nxt[None, :, :], _mm[:, None, :], _sig_v2), -1), 0.0
        )

    def _seam_sel(prev, nxt, seam):
        _mm = _model["drift"](seam, prev)
        return jnp.where(seam < _t_len, jnp.sum(logn(nxt, _mm, _sig_v2), -1), 0.0)

    _smooth = make_tree(_t_len, _p, _dim, _seam_pair, _seam_sel)
    _tau = 1.0

    def _x_sweep(x_ref, lam, key):
        _kz, _kt = random.split(key)
        _z = x_ref + jnp.sqrt(_tau) * random.normal(_kz, (_t_len, _dim))
        _center = _z + _tau * lam * (_y_j - lam * _z) / _sig_y**2

        def _leaf(t, k):
            _free = _center[t] + jnp.sqrt(_tau) * random.normal(k, (_p - 1, _dim))
            _parts = jnp.concatenate([x_ref[t][None], _free], 0)
            _psi = (
                jnp.sum(logn(_y_j[t], lam * _parts, _sig_y**2), -1)
                + jnp.sum(logn(_z[t], _parts, _tau), -1)
                - jnp.sum(logn(_parts, _center[t], _tau), -1)
            )
            _psi = jnp.where(t == 0, _psi + jnp.sum(logn(_parts, 0.0, _init_var), -1), _psi)
            return _parts, _psi[:, None]

        return _smooth(_kt, _leaf)

    def _traj_logp(x, lam):
        _ta = jnp.arange(1, _t_len)
        _lp = jnp.sum(logn(x[1:], _model["drift"](_ta[:, None], x[:-1]), _sig_v2))
        _lp += jnp.sum(logn(x[0], 0.0, _init_var))
        _lp += jnp.sum(logn(_y_j, lam * x, _sig_y**2))
        return _lp + jnp.sum(logn(lam, 0.0, _sig_lam**2))

    def _run(with_flip, seed):
        def _body(carry, key):
            _x, _lam = carry
            _k1, _k2, _k3, _k4 = random.split(key, 4)
            _prec = 1.0 / _sig_lam**2 + jnp.sum(_x**2, 0) / _sig_y**2
            _lam = jnp.sum(_x * _y_j, 0) / _sig_y**2 / _prec + random.normal(
                _k1, (_dim,)
            ) / jnp.sqrt(_prec)
            _x = _x_sweep(_x, _lam, _k2)
            _delta = jnp.asarray(0.0)
            if with_flip:
                _d = random.randint(_k3, (), 0, _dim)
                _x_new = _x.at[:, _d].multiply(-1.0)
                _lam_new = _lam.at[_d].multiply(-1.0)
                _delta = _traj_logp(_x_new, _lam_new) - _traj_logp(_x, _lam)
                _acc = jnp.log(random.uniform(_k4)) < _delta
                _x = jnp.where(_acc, _x_new, _x)
                _lam = jnp.where(_acc, _lam_new, _lam)
            return (_x, _lam), (_lam, _delta)

        _x0 = jnp.asarray(np.abs(_x_true) + 0.1)
        _lam0 = jnp.abs(jnp.asarray(_lam_true)) + 0.1
        _keys = random.split(random.PRNGKey(seed), _n_iter)
        (_, _), (_lams, _deltas) = jax.jit(lambda ks: jax.lax.scan(_body, (_x0, _lam0), ks))(_keys)
        return np.asarray(_lams), np.asarray(_deltas)

    e2_results = {}
    for _name, _wf in (("gibbs only", False), ("gibbs + joint (x, λ) flip", True)):
        _p_pos, _abs_lam, _max_delta = [], [], 0.0
        for _seed in (5, 6, 7):
            _lams, _deltas = _run(_wf, _seed)
            _b = _lams[_n_iter // 2 :]
            _p_pos.append((_b > 0).mean(0))
            _abs_lam.append(np.abs(_b).mean(0))
            _max_delta = max(_max_delta, float(np.max(np.abs(_deltas))))
        e2_results[_name] = {
            "p_pos": np.mean(_p_pos, 0),
            "abs_lam": np.mean(_abs_lam, 0),
            "max_delta": _max_delta,
        }
    return (e2_results,)


@app.cell
def exp_e2_table(e2_results, mo, np):
    _lines = [
        "| sampler | P(λ_d > 0), exact answer = 0.5 | E\\|λ_d\\| | max \\|Δ\\| of flip |",
        "|---|---|---|--:|",
    ]
    for _n, _m in e2_results.items():
        _pp = ", ".join(f"{v:.2f}" for v in np.asarray(_m["p_pos"]))
        _al = ", ".join(f"{v:.2f}" for v in np.asarray(_m["abs_lam"]))
        _lines.append(f"| {_n} | [{_pp}] | [{_al}] | {_m['max_delta']:.1e} |")
    mo.md(
        "**E2 — the production-shaped move: joint (trajectory, loading) sign flips.** "
        "In factor measurement models the fold ambiguity is the factor-sign ambiguity "
        "— the posterior symmetry is (x_d, λ_d) → (−x_d, −λ_d). Under odd drift and "
        "symmetric priors the flip's MH delta is identically zero (acceptance 1), and "
        "the exact posterior has P(λ_d > 0) = 1/2 — so the diagnostic is exact by "
        "symmetry, no gold needed. Sign-invariant functionals (E|λ|) must agree "
        "between samplers:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_e3_run(jax, jnp, random):
    # E3: parallel-in-time FFBS. Forward messages are log-semiring products of 2x2
    # matrices (associative); backward sampling with pre-drawn uniforms is composition
    # of random threshold maps {0,1}->{0,1} (associative). Both become
    # jax.lax.associative_scan — O(log T) depth — and must reproduce the sequential
    # scan BITWISE given shared uniforms.
    def _logmatmul(a, b):
        return jax.scipy.special.logsumexp(a[..., :, :, None] + b[..., None, :, :], axis=-2)

    def _seq_path(alpha0, lp, e, u_last, us):
        def _fstep(alpha, inp):
            lp_t, e_t = inp
            a = e_t + jax.scipy.special.logsumexp(alpha[:, None] + lp_t, 0)
            return a, a

        _alpha_last, _alphas = jax.lax.scan(_fstep, alpha0, (lp, e))
        _alphas_all = jnp.concatenate([alpha0[None], _alphas[:-1]], 0)
        _s_last = (u_last < jax.nn.softmax(_alpha_last)[1]).astype(jnp.int32)

        def _bstep(s_next, inp):
            alpha_t, lp_t, u_t = inp
            s_t = (u_t < jax.nn.softmax(alpha_t + lp_t[:, s_next])[1]).astype(jnp.int32)
            return s_t, s_t

        _, _s_rev = jax.lax.scan(_bstep, _s_last, (jnp.flip(_alphas_all, 0), jnp.flip(lp, 0), us))
        return jnp.concatenate([jnp.flip(_s_rev), _s_last[None]]), _alphas_all

    def _par_path(alpha0, lp, e, u_last, us):
        _mats = lp + e[:, None, :]
        _prefix = jax.lax.associative_scan(_logmatmul, _mats)
        _alphas = jax.scipy.special.logsumexp(alpha0[None, :, None] + _prefix, axis=1)
        _alphas_all = jnp.concatenate([alpha0[None], _alphas[:-1]], 0)
        _s_last = (u_last < jax.nn.softmax(_alphas[-1])[1]).astype(jnp.int32)
        _pr = jax.nn.softmax(jnp.flip(_alphas_all, 0)[:, :, None] + jnp.flip(lp, 0), axis=1)
        _fmaps = (us[:, None] < jnp.transpose(_pr, (0, 2, 1))[:, :, 1]).astype(jnp.int32)

        def _compose(a, b):  # later block applied after earlier: g = b ∘ a
            return jnp.take_along_axis(b, a, axis=-1)

        _gmaps = jax.lax.associative_scan(_compose, _fmaps)
        _s_rev = jnp.take_along_axis(_gmaps, jnp.full((_gmaps.shape[0], 1), _s_last), -1)[:, 0]
        return jnp.concatenate([jnp.flip(_s_rev), _s_last[None]]), _alphas_all

    _t_len, _n_trials = 64, 300
    _n_ok, _max_diff = 0, 0.0
    for _trial in range(_n_trials):
        _k1, _k2, _k3, _k4, _k5 = random.split(random.PRNGKey(_trial), 5)
        _lp = random.normal(_k1, (_t_len - 1, 2, 2)) * 2.0
        _e = random.normal(_k2, (_t_len - 1, 2)) * 1.5
        _alpha0 = random.normal(_k3, (2,))
        _u_last = random.uniform(_k4)
        _us = random.uniform(_k5, (_t_len - 1,))
        _s_seq, _a_seq = _seq_path(_alpha0, _lp, _e, _u_last, _us)
        _s_par, _a_par = _par_path(_alpha0, _lp, _e, _u_last, _us)
        _max_diff = max(_max_diff, float(jnp.max(jnp.abs(_a_seq - _a_par))))
        _n_ok += int(jnp.all(_s_seq == _s_par))
    e3_results = {"match_rate": _n_ok / _n_trials, "max_alpha_diff": _max_diff}
    return (e3_results,)


@app.cell
def exp_e3_table(e3_results, mo):
    mo.md(
        "**E3 — the sign pass is parallel-in-time.** Over "
        f"{300} random two-state chains, the associative-scan FFBS reproduces the "
        f"sequential scan with path match rate **{e3_results['match_rate']:.3f}** and "
        f"forward-message max deviation {e3_results['max_alpha_diff']:.1e}. Both "
        "passes are `associative_scan`s (2×2 log-matrix products forward; composition "
        "of random threshold maps backward), so the composed kernel's sequential "
        "depth returns to `O(log T)` end to end."
    )
    return


@app.cell
def exp_e4_run(
    kit_metrics,
    make_block_leaf_sweep,
    make_kit_d,
    make_mbranch_ffbs,
    np,
    product_gold,
    run_composed_e,
    sign_branches,
):
    _t_len, _p, _n_iter = 48, 16, 6000
    _model = make_kit_d(30)
    _, _y = _model["simulate"](0, _t_len)
    _golds = product_gold(_y)
    _x0 = np.sqrt(np.clip(20.0 * _y, 0.0, None))
    _ffbs = make_mbranch_ffbs(_model, _y, sign_branches)
    e4_results = {}
    for _k in (1, 2, 4, 8, 16, 30):
        _sweep = make_block_leaf_sweep(_model, _y, _p, _k)
        _chain = run_composed_e(_sweep, _ffbs, _x0, _n_iter, seed=5, tau=1.0)
        e4_results[_k] = kit_metrics(_chain[_n_iter // 2 :], _golds)
    return (e4_results,)


@app.cell
def exp_e4_table(e4_results, mo):
    _lines = ["| block size k | W1/σ̄ | sign error |", "|--:|--:|--:|"]
    for _k, _m in e4_results.items():
        _lines.append(f"| {_k} | {_m['w1_rel']:.3f} | {_m['sign_err']:.3f} |")
    mo.md(
        "**E4 — the coordinate-block sweet spot** (D = 30, fixed 6 000-sweep budget, "
        "FFBS composed). Coverage (more coordinates serviced per sweep) beats "
        "joint-coherence cost up to k ≈ 2–4, then degeneracy takes over — k = D is "
        "the frozen joint sweep:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_e5_run(
    jax,
    jnp,
    kit_metrics,
    make_block_leaf_sweep,
    make_kit_d,
    make_mbranch_ffbs,
    np,
    product_gold,
    random,
    run_composed_e,
    sign_branches,
):
    _t_len, _p, _dim = 48, 16, 8
    _model = make_kit_d(_dim)
    _, _y = _model["simulate"](0, _t_len)
    _golds = product_gold(_y)
    _x0 = jnp.asarray(np.sqrt(np.clip(20.0 * _y, 0.0, None)))
    _ffbs = make_mbranch_ffbs(_model, _y, sign_branches)
    _sweep = make_block_leaf_sweep(_model, _y, _p, 1)
    _n_warm, _n_main, _target, _gamma = 1500, 6000, 0.5, 0.08

    @jax.jit
    def _one(x, key, tau):
        _k1, _k2 = random.split(key)
        _x_new, _cmask = _sweep(x, _k1, tau)
        _moved = jnp.sum(jnp.abs(_x_new - x) > 1e-12, axis=0)
        _mf = jnp.sum(jnp.where(_cmask, _moved, 0)) / _t_len
        return _ffbs(_x_new, _k2), _mf

    e5_results = {}
    for _name, _delta0 in (("adapted from δ₀ = 8", 8.0), ("adapted from δ₀ = 0.5", 0.5)):
        _x = _x0
        _log_tau = float(np.log(0.5 * _delta0))
        _key = random.PRNGKey(11)
        for _ in range(_n_warm):
            _key, _sub = random.split(_key)
            _x, _mf = _one(_x, _sub, jnp.exp(_log_tau))
            _log_tau += _gamma * (float(_mf) - _target)
        _tau_hat = float(np.exp(_log_tau))
        _chain = run_composed_e(_sweep, _ffbs, _x, _n_main, seed=12, tau=_tau_hat)
        _m = kit_metrics(_chain[_n_main // 2 :], _golds)
        _m["tau_hat"] = _tau_hat
        e5_results[_name] = _m
    _chain = run_composed_e(_sweep, _ffbs, _x0, _n_main, seed=12, tau=1.0)
    e5_results["hand-tuned τ = 1"] = kit_metrics(_chain[_n_main // 2 :], _golds)
    return (e5_results,)


@app.cell
def exp_e5_table(e5_results, mo):
    _lines = ["| schedule | τ̂ after warmup | W1/σ̄ | sign error |", "|---|--:|--:|--:|"]
    for _n, _m in e5_results.items():
        _th = f"{_m['tau_hat']:.2f}" if "tau_hat" in _m else "—"
        _lines.append(f"| {_n} | {_th} | {_m['w1_rel']:.3f} | {_m['sign_err']:.3f} |")
    mo.md(
        "**E5 — δ tunes itself.** Robbins–Monro warmup targeting a 0.5 moved-fraction "
        "(the anti-freezing signal), adaptation frozen before sampling (exactness "
        "untouched). Both deliberately bad initialisations converge to the same "
        "healthy τ̂ and match hand-tuned quality:\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def exp_e6_run(
    jnp,
    kit_metrics,
    make_block_leaf_sweep,
    make_kit_d,
    make_mbranch_ffbs,
    np,
    product_gold,
    run_composed_e,
):
    # E6: 4-preimage W-fold emission y = ||x| - a|/s + noise: preimages {±a ± L}
    # (only ±(a+L) valid when L > a — masked). Per-branch Jacobians are unit here
    # (piecewise-linear slopes ±1/s), exercising the m-state machinery with masking.
    _t_len, _p, _dim, _n_iter, _wa, _ws = 48, 16, 8, 10000, 8.0, 2.0
    _base = make_kit_d(_dim)

    def _h(x):
        return jnp.abs(jnp.abs(x) - _wa) / _ws

    _model = dict(_base)
    _model["log_obs"] = lambda x, y: jnp.sum(
        -0.5 * (jnp.log(2.0 * jnp.pi) + (y - _h(x)) ** 2), axis=-1
    )
    _model["obs_grad_1d"] = lambda z, y_col: (
        (y_col - _h(z)) * (jnp.sign(z) * jnp.sign(jnp.abs(z) - _wa) / _ws)
    )

    def _branches(v):
        _lvl = jnp.abs(jnp.abs(v) - _wa)
        _c = jnp.stack([_wa + _lvl, _wa - _lvl, -(_wa - _lvl), -(_wa + _lvl)])
        _valid = jnp.abs(jnp.abs(jnp.abs(_c) - _wa) - _lvl) < 1e-9
        return _c, _valid, jnp.zeros(4)

    _rng = np.random.default_rng(0)
    _x = np.zeros((_t_len, _dim))
    _x[0] = _base["init_sd"] * _rng.standard_normal(_dim)
    for _t in range(1, _t_len):
        _x[_t] = np.asarray(_base["drift"](_t, jnp.asarray(_x[_t - 1]))) + _base[
            "sig_v"
        ] * _rng.standard_normal(_dim)
    _y = np.asarray(_h(jnp.asarray(_x))) + _rng.standard_normal((_t_len, _dim))
    _golds = product_gold(_y, h_np=lambda xs: np.abs(np.abs(xs) - _wa) / _ws)

    def _region_rmse(chain_burn):
        _errs = []
        for _d in range(_dim):
            _g = _golds[_d]
            _xs = _g["xs"]
            _masks = [
                _xs < -_wa,
                (_xs >= -_wa) & (_xs < 0),
                (_xs >= 0) & (_xs < _wa),
                _xs >= _wa,
            ]
            _gp = np.stack([_g["g"][:, _mk].sum(1) for _mk in _masks], 1)
            _c = chain_burn[:, :, _d]
            _emp = np.stack(
                [
                    (_c < -_wa).mean(0),
                    ((_c >= -_wa) & (_c < 0)).mean(0),
                    ((_c >= 0) & (_c < _wa)).mean(0),
                    (_c >= _wa).mean(0),
                ],
                1,
            )
            _errs.append(np.sqrt(np.mean((_emp - _gp) ** 2)))
        return float(np.mean(_errs))

    _x0 = jnp.full((_t_len, _dim), _wa + 0.5)
    _sweep = make_block_leaf_sweep(_model, _y, _p, 1)
    _ffbs4 = make_mbranch_ffbs(_model, _y, _branches)
    e6_results = {}
    for _name, _mh in (("coord-amala only (stuck)", None), ("+ 4-branch FFBS", _ffbs4)):
        _chain = run_composed_e(_sweep, _mh, _x0, _n_iter, seed=5, tau=1.0)
        _b = _chain[_n_iter // 2 :]
        _m = kit_metrics(_b, _golds)
        _m["region_rmse"] = _region_rmse(_b)
        e6_results[_name] = _m
    return (e6_results,)


@app.cell
def exp_e6_table(e6_results, mo):
    _lines = [
        "| kernel | W1/σ̄ | 4-region branch-prob RMSE |",
        "|---|--:|--:|",
    ]
    for _n, _m in e6_results.items():
        _lines.append(f"| {_n} | {_m['w1_rel']:.3f} | {_m['region_rmse']:.3f} |")
    mo.md(
        "**E6 — beyond two branches.** The W-fold emission `y = ||x| − a|/s` has FOUR "
        "preimages per level (two masked where the level exceeds `a`); the branch "
        "chain becomes a 4-state transfer matrix and the same FFBS recovers a "
        "posterior with four modes per (t, d):\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def verdict(mo):
    mo.md(r"""
    ## Verdict: the multimodal problem reduces to the unimodal one

    The lab's arc, each step forced by a measurement:

    1. **Composition beats embedding.** Segment sign-flip MH breaches the wall where
       every D-sweep kernel is fully blind (sign-error 0.62–0.70): the first method to
       move sign mass at `D ≥ 16`.
    2. **A3 rejected the starvation hypothesis.** 10× more random windows barely
       helped — accepted flips concentrate at cheap-wall sites.
    3. **A4 went to the adaptive-window limit.** In 1-D time the cluster construction
       collapses to an *exact* conditional: sign-path FFBS. It is best-in-class at
       every `D` and near the floor at `D ≤ 8` — and at `D ∈ {16, 30}` all three sign
       kernels, exact FFBS included, land on numerically identical residuals.
    4. **The shared plateau is not a sign problem — and not chain length either.**
       An exact sign sampler cannot be beaten by an approximate one, and A5 shows 4×
       the sweeps barely moves the `D = 30` residual (0.217 → 0.197 vs a √-law 0.109).
       So the magnitudes are effectively *frozen* at this (P, leaf, D): the sign chain
       exactly samples the sign law conditioned on magnitudes that never relax from
       their distorted init at precisely the sign-flexible (small-`|x|`) sites.
    5. **Coupling does not break any of it** (B1 exact against the joint 2-D gold;
       B2's gold-free cross-seed diagnostic favours FFBS at coupled `D = 30`).

    6. **C closed the loop.** The width lever barely moves the plateau (`P = 64`:
       0.217 → 0.173 sign-error — joint coherence, not particle count, is the
       disease). The coordinate-conditional sweep removes the joint-coherence lottery
       and, composed with sign-path FFBS, lands at **sign-error 0.019 at `D = 30`** —
       the exact-floor territory (`D = 2` reads 0.026 at this budget) in the cell
       where every D-sweep kernel sits at 0.62–0.70 — with `W1/σ̄` 0.212 limited only
       by the per-coordinate update budget. Under coupling the cross-seed sign RMS
       drops to 0.021 (from 0.081–0.102 for the joint-leaf kernels).

    7. **D generalised it beyond symmetry.** For an asymmetric fold (β = 0.3, mirror
       Jacobian ≈ 1.86) the branch-path FFBS with the **Jacobian site term** recovers
       the exact gold at `D = 2` and `D = 16` (branch-probability error 0.042 / 0.063
       vs 0.56 / 0.61 stuck), while the no-Jacobian negative control ships a
       systematic ≈ 2× branch-weight bias — the quiet failure a naive port would
       carry. The construction needs only the emission's preimage map and its
       derivative, both compile-time derivable from a measurement link — the
       production-shaped version.

    **The closed claim:** the high-D far-mode cell — the one every paper in this
    family concedes — is solved for two-branch fold emissions (symmetric or not) by
    two composed, exactly-invariant, tuning-free moves through the same log-depth
    tree: branch structure by per-coordinate FFBS on the embedded two-state chain
    (with the Jacobian site term for asymmetric folds), magnitudes by
    coordinate-conditional tree sweeps that reduce to the `D = 1` kernel each. The
    price is sweeps scaling with `D` (one coordinate per sweep) — sequential-depth
    per sweep stays `⌈log₂T⌉`, and the moves cost `O(4·T·D)` and `O(P²·T·D)`
    respectively.

    8. **Part E hardened all of it.** (E1) the §15 1-D root leaf, dead in joint D,
       is best-in-class *inside* the coordinate-conditional engine — the program's
       best `D = 30` result — because each sweep is a `D = 1` problem where
       independence is safe again; (E2) the production-shaped joint (x, λ) factor-sign
       flip has MH delta identically 0 under the exact symmetry and restores
       `P(λ > 0) = 1/2` where plain Gibbs freezes at 0 or 1; (E3) both FFBS passes are
       associative scans (bitwise-equal to sequential), so the composed kernel is
       `O(log T)` depth end to end; (E4) coordinate blocks have a sweet spot at
       `k ≈ 2–4` — coverage beats coherence until it doesn't; (E5) δ self-tunes by
       moved-fraction-targeted warmup from bad initialisations in either direction;
       (E6) the branch chain generalises to `m`-preimage folds (4-state W-fold
       recovered with branch-probability RMSE at the floor).

    **Next steps, in order of expected value:**

    1. *Wire it production-shaped.* Paid-mix leaf + per-indicator branch-path FFBS
       (compile-time fold detection on the measurement links) + the joint (x, λ)
       flip from E2 + E5's self-tuning δ + frozen-fraction diagnostics — every piece
       now measured.
    2. *Extreme asymmetry / stronger coupling range.* β = 0.3 is exact; map where
       mixing degrades, with segment-MH (which prices any asymmetry in its ratio) and
       tempering as fallbacks.
    3. *Parallel coordinate scans.* E4's `k* ≈ 4` recovers part of the `D`× cost;
       at κ = 0 coordinates parallelise exactly, and checkerboard scans should
       recover most of the rest under coupling.
    """)
    return


if __name__ == "__main__":
    app.run()
