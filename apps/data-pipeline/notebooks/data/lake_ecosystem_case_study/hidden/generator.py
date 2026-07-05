"""Hidden ground-truth generator for the lake-ecosystem N-of-1 case study.

One physical unit: a single small, productive freshwater lake (one monitored
station on a mesotrophic pond/lake), sampled irregularly by an automated
multiparameter buoy plus grab samples over a summer stratification season.

Generative form (continuous-time latent SDE, one latent x_i per construct):

    dx_i = -( a_i (x_i - c_i) + q_i (x_i - c_i)^3 ) dt
           + sum_j f_ij(x_j) dt
           + sigma_i dW_i

with edges f_ij either linear (w * x_j) or saturating tanh (w * tanh(x_j)).

Emissions (one indicator per observed construct, conditionally independent
given the latent state, i.i.d. noise):

    gaussian + identity :   y = lam*x + b + N(0, sigma_e)
    gaussian + sigmoid100 : y = 100*sigmoid(lam*x + b) + N(0, sigma_e)
    poisson  + exp :        y ~ Poisson(exp(lam*x + b))

Everything is seeded. Running this script rewrites observations.csv and
hidden/truth.json deterministically.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 71413092
DT = 0.01  # internal Euler-Maruyama step, days
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent

# Independent RNG substreams derived from the single master seed.
_ss_times, _ss_freerun, _ss_sim, _ss_emit = np.random.SeedSequence(SEED).spawn(4)

# --------------------------------------------------------------------------
# Constructs (topological order: parents before children)
# --------------------------------------------------------------------------
# index 0 is the UNOBSERVED confounder (no indicator).
NODES = [
    "CatchmentLoading",    # 0  LATENT confounder (runoff-driven external loading)
    "WaterTemperature",    # 1
    "Nitrate",             # 2
    "Turbidity",           # 3
    "CDOM",                # 4
    "Phytoplankton",       # 5
    "DissolvedOxygen",     # 6
    "pH",                  # 7
    "Zooplankton",         # 8
]
IDX = {n: i for i, n in enumerate(NODES)}
D = len(NODES)
HIDDEN_IDX = 0

DEFINITIONS = {
    "CatchmentLoading":
        "Latent intensity of external material loading delivered to the lake "
        "from its watershed, driven by rainfall/runoff pulses. Bundles the "
        "washed-in nutrients, suspended sediment and terrestrial dissolved "
        "organic matter that arrive together after storms. No direct sensor.",
    "WaterTemperature":
        "Surface mixed-layer water temperature. Responds quickly to weather "
        "and insolation; sets metabolic and solubility rates for the rest of "
        "the system.",
    "Nitrate":
        "Dissolved nitrate-nitrogen concentration in surface water, the main "
        "bioavailable nitrogen pool fueling algal growth.",
    "Turbidity":
        "Optical cloudiness of the water from suspended inorganic particles "
        "(clay/silt). Rises fast with runoff and settles fast once inputs stop.",
    "CDOM":
        "Colored (chromophoric) dissolved organic matter -- the 'tea-stained' "
        "humic material leached from the catchment. Absorbs light and drifts "
        "slowly.",
    "Phytoplankton":
        "Standing algal biomass (phytoplankton) in the surface layer. Grows on "
        "nitrogen, warmth and light; self-limits at bloom densities.",
    "DissolvedOxygen":
        "Dissolved-oxygen saturation of the surface water, set by the balance "
        "of photosynthetic production, temperature-dependent solubility and "
        "respiration.",
    "pH":
        "Acidity/alkalinity of the surface water. Rises as photosynthesis draws "
        "down dissolved CO2 during productive periods.",
    "Zooplankton":
        "Abundance of grazing crustacean zooplankton (Daphnia-type). Tracks "
        "algal food and temperature with a slow, generational lag and a "
        "carrying-capacity ceiling.",
}

# --------------------------------------------------------------------------
# Latent SDE parameters
# --------------------------------------------------------------------------
# self-relaxation rate a_i (1/day); relaxation time tau_i = 1/a_i
A = {
    "CatchmentLoading":  0.50,   # tau 2.0 d   (storm pulses decay over ~2 d)
    "WaterTemperature":  0.833,  # tau 1.2 d
    "Nitrate":           0.25,   # tau 4.0 d
    "Turbidity":         2.50,   # tau 0.4 d   (fast settling)
    "CDOM":              0.1667,  # tau 6.0 d   (slow)
    "Phytoplankton":     0.40,   # tau 2.5 d
    "DissolvedOxygen":   1.25,   # tau 0.8 d   (fast gas exchange)
    "pH":                1.00,   # tau 1.0 d
    "Zooplankton":       0.1429,  # tau 7.0 d   (slow, generational)
}
# centers (relaxation target in latent space)
C = {n: 0.0 for n in NODES}
# quartic self-stiffening q_i >= 0 (nonlinearity)
Q = {n: 0.0 for n in NODES}
Q["Phytoplankton"] = 0.30   # bloom self-limitation
Q["Zooplankton"] = 0.40     # population carrying capacity
# diffusion sigma_i, roughly sqrt(2 a_i) so free (edge-less) sd ~ 1
SIGMA = {
    "CatchmentLoading":  1.00,
    "WaterTemperature":  1.29,
    "Nitrate":           0.55,
    "Turbidity":         2.05,
    "CDOM":              0.55,
    "Phytoplankton":     0.80,
    "DissolvedOxygen":   1.35,
    "pH":                1.41,
    "Zooplankton":       0.52,
}

# edges: (parent, child, weight, form)   form in {"linear","tanh"}
EDGES = [
    ("CatchmentLoading", "Nitrate",         0.50, "linear"),
    ("CatchmentLoading", "Turbidity",       0.70, "linear"),
    ("CatchmentLoading", "CDOM",            0.55, "linear"),
    ("WaterTemperature", "Nitrate",        -0.25, "linear"),  # warm -> more uptake/denitrification
    ("Nitrate",          "Phytoplankton",   0.45, "linear"),
    ("WaterTemperature", "Phytoplankton",   0.35, "linear"),
    ("Turbidity",        "Phytoplankton",  -0.50, "tanh"),    # light limitation, saturating
    ("CDOM",             "Phytoplankton",  -0.30, "linear"),  # shading
    ("Phytoplankton",    "DissolvedOxygen", 0.60, "linear"),  # photosynthetic O2
    ("WaterTemperature", "DissolvedOxygen", -0.40, "linear"),  # solubility
    ("Phytoplankton",    "pH",              0.55, "linear"),  # CO2 drawdown
    ("Phytoplankton",    "Zooplankton",     0.70, "tanh"),    # saturating functional response
    ("WaterTemperature", "Zooplankton",     0.30, "linear"),
]

# --------------------------------------------------------------------------
# Emissions (one indicator per observed construct)
# --------------------------------------------------------------------------
# node, indicator column, response desc, family, link, lam, b, sigma_e (None for poisson)
EMISSIONS = [
    ("WaterTemperature", "water_temp_C",  "continuous", "gaussian", "identity",   2.00, 22.0, 0.15),
    ("Nitrate",          "nitrate_mgL",   "continuous", "gaussian", "identity",   0.11,  0.55, 0.015),
    ("Turbidity",        "turbidity_NTU", "continuous", "gaussian", "identity",   4.20, 14.0, 0.5),
    ("CDOM",             "fdom_QSU",      "continuous", "gaussian", "identity",   6.00, 26.0, 0.6),
    ("Phytoplankton",    "chl_a_ugL",     "continuous", "gaussian", "identity",   5.50, 16.0, 0.8),
    ("DissolvedOxygen",  "do_sat_pct",    "bounded 0-100 index", "gaussian", "sigmoid100", 0.35, 1.30, 1.8),
    ("pH",               "ph",            "continuous", "gaussian", "identity",   0.28,  7.9, 0.03),
    ("Zooplankton",      "zoop_count",    "count",      "poisson",  "exp",        0.50,  3.70, None),
]


# --------------------------------------------------------------------------
# Drift / integrator
# --------------------------------------------------------------------------
def _build_arrays():
    a = np.array([A[n] for n in NODES])
    c = np.array([C[n] for n in NODES])
    q = np.array([Q[n] for n in NODES])
    sigma = np.array([SIGMA[n] for n in NODES])
    edges = [(IDX[p], IDX[ch], w, form) for (p, ch, w, form) in EDGES]
    return a, c, q, sigma, edges


def _drift(x, a, c, q, edges):
    dx = -(a * (x - c) + q * (x - c) ** 3)
    for pi, ci, w, form in edges:
        contrib = w * (np.tanh(x[pi]) if form == "tanh" else x[pi])
        dx[ci] += contrib
    return dx


def _simulate(x0, n_steps, sigma, a, c, q, edges, rng, store=True):
    x = x0.copy()
    sqrt_dt = np.sqrt(DT)
    traj = np.empty((n_steps + 1, D)) if store else None
    if store:
        traj[0] = x
    for k in range(n_steps):
        noise = rng.standard_normal(D)
        x = x + _drift(x, a, c, q, edges) * DT + sigma * sqrt_dt * noise
        if store:
            traj[k + 1] = x
    return traj if store else x


# --------------------------------------------------------------------------
# 1) Empirical stationary sd from a long free run
# --------------------------------------------------------------------------
def _stationary_sd(a, c, q, sigma, edges):
    rng = np.random.default_rng(_ss_freerun)
    burn_days = 200.0
    run_days = 4000.0
    n_burn = int(burn_days / DT)
    n_run = int(run_days / DT)
    x = _simulate(np.array([C[n] for n in NODES]), n_burn, sigma, a, c, q, edges,
                  rng, store=False)
    # collect every 10th step
    samples = []
    sqrt_dt = np.sqrt(DT)
    for k in range(n_run):
        noise = rng.standard_normal(D)
        x = x + _drift(x, a, c, q, edges) * DT + sigma * sqrt_dt * noise
        if k % 10 == 0:
            samples.append(x.copy())
    arr = np.asarray(samples)
    return arr.std(axis=0), arr.mean(axis=0)


# --------------------------------------------------------------------------
# 2) Irregular observation times
# --------------------------------------------------------------------------
def _observation_times():
    rng = np.random.default_rng(_ss_times)
    n_obs = 100
    # irregular sub-daily to ~2-day gaps: Gamma(shape=1.6, scale=0.42) -> mean ~0.67 d
    gaps = rng.gamma(shape=1.6, scale=0.42, size=n_obs - 1)
    gaps = np.clip(gaps, 0.05, None)
    t0 = float(rng.uniform(0.3, 0.9))
    times = np.concatenate([[t0], t0 + np.cumsum(gaps)])
    return np.round(times, 4)


# --------------------------------------------------------------------------
# 3) Main simulation over the observation window + emissions
# --------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    a, c, q, sigma, edges = _build_arrays()

    stat_sd, stat_mean = _stationary_sd(a, c, q, sigma, edges)

    obs_times = _observation_times()
    span = float(obs_times[-1])
    burn = 40.0  # days of pre-window burn-in to shed the initial transient

    grid_t0 = -burn
    total_days = span - grid_t0
    n_steps = int(np.ceil(total_days / DT))

    rng_sim = np.random.default_rng(_ss_sim)
    x0 = np.array([C[n] for n in NODES])
    traj = _simulate(x0, n_steps, sigma, a, c, q, edges, rng_sim, store=True)
    grid = grid_t0 + DT * np.arange(n_steps + 1)

    # nearest-grid latent state at each observation time
    obs_idx = np.round((obs_times - grid_t0) / DT).astype(int)
    obs_idx = np.clip(obs_idx, 0, n_steps)
    latent_obs = traj[obs_idx]  # (n_obs, D)

    # emissions
    rng_emit = np.random.default_rng(_ss_emit)
    columns = {}
    for node, name, _resp, family, link, lam, b, sigma_e in EMISSIONS:
        xi = latent_obs[:, IDX[node]]
        eta = lam * xi + b
        if family == "poisson":
            rate = np.exp(eta)
            y = rng_emit.poisson(rate).astype(int)
        elif link == "sigmoid100":
            y = 100.0 * _sigmoid(eta) + rng_emit.normal(0.0, sigma_e, size=xi.shape)
            y = np.round(y, 2)
        else:  # identity gaussian
            y = eta + rng_emit.normal(0.0, sigma_e, size=xi.shape)
            # round to a plausible instrument precision
            prec = {"water_temp_C": 2, "nitrate_mgL": 3, "turbidity_NTU": 1,
                    "fdom_QSU": 1, "chl_a_ugL": 1, "ph": 2}[name]
            y = np.round(y, prec)
        columns[name] = y

    # ---- write observations.csv ----
    csv_path = OUT_DIR / "observations.csv"
    order = [name for (_n, name, *_r) in EMISSIONS]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + order)
        for i, t in enumerate(obs_times):
            row = [f"{t:.4f}"] + [columns[name][i] for name in order]
            w.writerow(row)

    # ---- write hidden/truth.json ----
    nodes_block = {}
    for n in NODES:
        nodes_block[n] = {
            "a": A[n],
            "relaxation_time_days": 1.0 / A[n],
            "c": C[n],
            "q": Q[n],
            "sigma": SIGMA[n],
            "stationary_sd": float(stat_sd[IDX[n]]),
            "stationary_mean_freerun": float(stat_mean[IDX[n]]),
            "definition": DEFINITIONS[n],
        }
    edges_block = [
        {"parent": p, "child": ch, "weight": w, "form": form}
        for (p, ch, w, form) in EDGES
    ]
    emissions_block = []
    for node, name, resp, family, link, lam, b, sigma_e in EMISSIONS:
        e = {
            "indicator": name,
            "construct": node,
            "response": resp,
            "family": family,
            "link": link,
            "loading_lambda": lam,
            "intercept_b": b,
        }
        if sigma_e is not None:
            e["noise_sigma_e"] = sigma_e
        emissions_block.append(e)

    truth = {
        "seed": SEED,
        "dt": DT,
        "domain": "Single small mesotrophic freshwater lake; one monitored "
                  "station; summer stratification season.",
        "generative_form": (
            "dx_i = -(a_i(x_i-c_i)+q_i(x_i-c_i)^3)dt "
            "+ sum_j f_ij(x_j) dt + sigma_i dW_i ; "
            "f_ij linear (w*x_j) or tanh (w*tanh(x_j))."
        ),
        "node_names_topological": NODES,
        "unobserved_node": {"name": NODES[HIDDEN_IDX], "index": HIDDEN_IDX},
        "nodes": nodes_block,
        "edges": edges_block,
        "emissions": emissions_block,
        "observation_design": {
            "span_days": span,
            "n_observations": int(len(obs_times)),
            "burn_in_days": burn,
            "note": "Irregular Gamma-spaced visit times; all indicators logged "
                    "at each visit.",
        },
        "stationary_sd_note": (
            "stationary_sd / stationary_mean_freerun estimated empirically from "
            "a 4000-day free run (dt=0.01) after 200-day burn-in, sampled every "
            "10th step, on the full coupled system."
        ),
    }
    (HERE / "truth.json").write_text(json.dumps(truth, indent=2))

    # ---- console summary (safe: no hidden params) ----
    print(f"wrote {csv_path}  ({len(obs_times)} rows)")
    print(f"span = {span:.2f} d, "
          f"median gap = {np.median(np.diff(obs_times)):.3f} d")
    print("column ranges:")
    for name in order:
        col = np.asarray(columns[name], dtype=float)
        print(f"  {name:16s} min={col.min():8.3f} med={np.median(col):8.3f} "
              f"max={col.max():8.3f}")


if __name__ == "__main__":
    main()
