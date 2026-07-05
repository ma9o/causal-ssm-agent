"""Hidden ground-truth generator for the d10 case study.

Self-contained (numpy only). Fixed seed. Run from apps/data-pipeline:

    uv run python notebooks/data/d10_case_study/hidden/generator.py

Simulates a 10-node continuous-time nonlinear latent SSM (Euler-Maruyama),
emits 9 indicators (one per observed construct), and writes:
    - <case>/observations.csv   (visible to the modeler)
    - hidden/truth.json         (never read by the modeler)

The modeler must NOT read anything under hidden/.
"""

import json
from pathlib import Path

import numpy as np

SEED = 20260703

# ----------------------------------------------------------------------------
# Constructs (latent node order). Node 1 (AutonomicArousal) is UNOBSERVED.
# 0 CaffeineIntake        (fast root)
# 1 AutonomicArousal      (hidden, slowly drifting root; >=2 children)
# 2 PerceivedStress
# 3 SleepQuality
# 4 Fatigue
# 5 MusculoskeletalPain
# 6 PhysicalActivity
# 7 NegativeMood
# 8 CognitiveFocus
# 9 SocialEngagement
# ----------------------------------------------------------------------------
NODE_NAMES = [
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
HIDDEN_NODE = 1  # AutonomicArousal has no indicator

# self-dynamics: dx_i = -( a_i (x_i - c_i) + q_i (x_i - c_i)^3 ) dt + coupling + sigma_i dW
A = np.array([4.00, 0.08, 0.667, 1.00, 0.40, 0.25, 0.833, 0.333, 1.25, 0.333])
C = np.array([0.20, -0.10, 0.00, 0.30, -0.20, 0.00, 0.10, -0.10, 0.00, 0.20])
Q = np.array([0.00, 0.00, 0.15, 0.10, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00])
SIGMA = np.array([1.27, 0.32, 1.00, 1.20, 0.85, 0.22, 2.00, 0.45, 0.60, 0.35])

# x(0): person sits at their per-construct setpoint at study start.
X0 = C.copy()

# directed edges (parent, child, weight, form). form in {"linear", "tanh"}.
# f_ij(x_j) = w_ij * x_j            (linear)
# f_ij(x_j) = w_ij * tanh(x_j)      (saturating)
EDGES = [
    (0, 3, -0.40, "linear"),   # CaffeineIntake      -> SleepQuality
    (1, 2, 0.80, "tanh"),      # AutonomicArousal    -> PerceivedStress   (saturating)
    (1, 3, -0.50, "linear"),   # AutonomicArousal    -> SleepQuality
    (1, 5, 0.30, "linear"),    # AutonomicArousal    -> MusculoskeletalPain
    (2, 3, -0.35, "linear"),   # PerceivedStress     -> SleepQuality
    (2, 7, 0.50, "linear"),    # PerceivedStress     -> NegativeMood
    (2, 4, 0.30, "linear"),    # PerceivedStress     -> Fatigue
    (3, 4, -0.60, "linear"),   # SleepQuality        -> Fatigue
    (4, 5, 0.25, "linear"),    # Fatigue             -> MusculoskeletalPain
    (4, 6, -0.50, "linear"),   # Fatigue             -> PhysicalActivity
    (4, 8, -0.40, "linear"),   # Fatigue             -> CognitiveFocus
    (5, 6, -0.70, "tanh"),     # MusculoskeletalPain -> PhysicalActivity  (saturating)
    (6, 7, -0.25, "linear"),   # PhysicalActivity    -> NegativeMood
    (7, 9, -0.45, "linear"),   # NegativeMood        -> SocialEngagement
    (7, 8, -0.45, "linear"),   # NegativeMood        -> CognitiveFocus
]

# ----------------------------------------------------------------------------
# Emissions: exactly one indicator per OBSERVED construct (9 total).
#   gaussian + identity   : y = lam*x + b + N(0, sigma_e)
#   gaussian + sigmoid100 : y = 100*sigmoid(lam*x + b) + N(0, sigma_e)   (0-100 slider)
#   poisson  + exp        : y ~ Poisson( exp(lam*x + b) )                (daily count)
# ----------------------------------------------------------------------------
EMISSIONS = [
    {"node": 0, "name": "caffeine_servings", "resp": "daily count",
     "family": "poisson", "link": "exp", "lam": 0.60, "b": 0.975},
    {"node": 2, "name": "stress_vas", "resp": "0-100 slider",
     "family": "gaussian", "link": "sigmoid100", "lam": 0.90, "b": -0.20, "sigma_e": 4.0},
    {"node": 3, "name": "sleep_quality_vas", "resp": "0-100 slider",
     "family": "gaussian", "link": "sigmoid100", "lam": 0.90, "b": 0.30, "sigma_e": 5.0},
    {"node": 4, "name": "fatigue_score", "resp": "continuous",
     "family": "gaussian", "link": "identity", "lam": 1.30, "b": 5.50, "sigma_e": 0.63},
    {"node": 5, "name": "pain_nrs", "resp": "continuous",
     "family": "gaussian", "link": "identity", "lam": 1.60, "b": 3.50, "sigma_e": 0.75},
    {"node": 6, "name": "active_minutes", "resp": "continuous",
     "family": "gaussian", "link": "identity", "lam": 6.00, "b": 45.0, "sigma_e": 2.51},
    {"node": 7, "name": "irritability_index", "resp": "continuous",
     "family": "gaussian", "link": "identity", "lam": 0.50, "b": 0.30, "sigma_e": 0.60},
    {"node": 8, "name": "reaction_time_ms", "resp": "continuous",
     "family": "gaussian", "link": "identity", "lam": -30.0, "b": 340.0, "sigma_e": 4.96},
    {"node": 9, "name": "social_contacts", "resp": "daily count",
     "family": "poisson", "link": "exp", "lam": 0.25, "b": 1.168},
]

# observation design
T_DAYS = 120
DT = 0.01
JITTER = 0.3          # uniform +/- jitter (days) around nominal prompt
DROP_FRACTION = 0.18  # fraction of days randomly dropped


def drift(x):
    out = -(A * (x - C) + Q * (x - C) ** 3)
    for (p, ch, w, form) in EDGES:
        out[ch] += w * (np.tanh(x[p]) if form == "tanh" else x[p])
    return out


def simulate(rng):
    n_steps = round(T_DAYS / DT)
    d = len(A)
    traj = np.empty((n_steps + 1, d))
    x = X0.copy()
    traj[0] = x
    sq = np.sqrt(DT)
    for t in range(1, n_steps + 1):
        x = x + drift(x) * DT + SIGMA * sq * rng.standard_normal(d)
        traj[t] = x
    return traj  # grid time g*DT for g in [0, n_steps]


def observation_times(rng):
    nominal = np.arange(T_DAYS) + 0.5
    jittered = nominal + rng.uniform(-JITTER, JITTER, size=T_DAYS)
    n_drop = round(DROP_FRACTION * T_DAYS)
    dropped = np.sort(rng.choice(T_DAYS, size=n_drop, replace=False))
    keep_mask = np.ones(T_DAYS, dtype=bool)
    keep_mask[dropped] = False
    kept_days = np.arange(T_DAYS)[keep_mask]
    times = np.sort(jittered[keep_mask])
    return times, kept_days, dropped


def emit(emission, x, rng):
    z = emission["lam"] * x + emission["b"]
    link = emission["link"]
    if link == "exp":
        return int(rng.poisson(np.exp(z)))
    if link == "sigmoid100":
        return float(100.0 / (1.0 + np.exp(-z)) + rng.normal(0.0, emission["sigma_e"]))
    if link == "identity":
        return float(z + rng.normal(0.0, emission["sigma_e"]))
    raise ValueError(link)


def main():
    here = Path(__file__).resolve().parent          # hidden/
    case_dir = here.parent                            # d10_case_study/
    csv_path = case_dir / "observations.csv"
    truth_path = here / "truth.json"
    brief_path = case_dir / "brief.md"
    gen_path = here / "generator.py"

    rng = np.random.default_rng(SEED)
    traj = simulate(rng)
    times, kept_days, dropped = observation_times(rng)

    grid_idx = np.rint(times / DT).astype(int)
    grid_idx = np.clip(grid_idx, 0, traj.shape[0] - 1)

    header = ["t"] + [e["name"] for e in EMISSIONS]
    rows = []
    for t_obs, gidx in zip(times, grid_idx, strict=True):
        x = traj[gidx]
        cells = [f"{t_obs:.4f}"]
        for e in EMISSIONS:
            val = emit(e, x[e["node"]], rng)
            if e["link"] == "exp":
                cells.append(str(val))
            elif e["resp"] == "0-100 slider" or e["name"] in ("active_minutes", "reaction_time_ms"):
                cells.append(f"{val:.2f}")
            else:
                cells.append(f"{val:.3f}")
        rows.append(cells)

    # --- verification -------------------------------------------------------
    arr = np.array([[float(c) for c in r] for r in rows], dtype=float)
    assert np.all(np.isfinite(arr)), "non-finite values in observations"
    for j in range(1, arr.shape[1]):
        assert arr[:, j].var() > 0, f"zero variance in column {header[j]}"
    for e in EMISSIONS:
        if e["link"] == "exp":
            col = header.index(e["name"])
            v = arr[:, col]
            assert np.all(v >= 0), "counts must be nonnegative"
            assert np.all(v == np.floor(v)), "counts must be integers"
    assert abs(arr.shape[0] - 0.82 * T_DAYS) <= 3, "row count far from 0.82*T"

    # --- write CSV ----------------------------------------------------------
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    # --- write truth.json (never read by the modeler) -----------------------
    truth = {
        "seed": SEED,
        "story": "Single-subject N-of-1 study of stress, sleep and daily "
                 "functioning over ~4 months.",
        "node_names": NODE_NAMES,
        "hidden_node_index": HIDDEN_NODE,
        "hidden_node_name": NODE_NAMES[HIDDEN_NODE],
        "latent_sde": {
            "form": "dx_i = -(a_i(x_i-c_i)+q_i(x_i-c_i)^3)dt + sum_j f_ij(x_j) dt + sigma_i dW_i",
            "a": A.tolist(),
            "c": C.tolist(),
            "q": Q.tolist(),
            "sigma": SIGMA.tolist(),
            "x0": X0.tolist(),
            "relaxation_time_days": (1.0 / A).tolist(),
        },
        "edges": [
            {"parent": NODE_NAMES[p], "child": NODE_NAMES[ch],
             "weight": w, "form": form}
            for (p, ch, w, form) in EDGES
        ],
        "emissions": EMISSIONS,
        "observation_design": {
            "T_days": T_DAYS,
            "dt": DT,
            "nominal_prompt_time": "k + 0.5 for k in 0..119",
            "jitter_days": JITTER,
            "drop_fraction": DROP_FRACTION,
            "n_dropped": len(dropped),
            "n_kept": len(times),
            "dropped_day_indices": dropped.tolist(),
            "kept_day_indices": kept_days.tolist(),
        },
        "reference_stationary_sd_note":
            "Approx stationary sds (2500-day sim): Caffeine 0.46, Arousal 0.82, "
            "Stress 0.86, Sleep 0.90, Fatigue 1.54, Pain 0.93, Activity 2.05, "
            "Mood 1.83, Focus 1.09, Social 2.28.",
    }
    with truth_path.open("w") as f:
        json.dump(truth, f, indent=2)

    # --- print ONLY file paths and CSV shape --------------------------------
    print(str(gen_path))
    print(str(truth_path))
    print(str(csv_path))
    print(str(brief_path))
    print(f"CSV shape: {arr.shape[0]} rows x {len(header)} columns")


if __name__ == "__main__":
    main()
