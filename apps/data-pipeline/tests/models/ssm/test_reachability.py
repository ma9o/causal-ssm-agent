"""Unit tests for the prior-predictive reachability battery (pure, array-in)."""

from __future__ import annotations

import numpy as np

from nof1_causal_lab.models.ssm.reachability import (
    CHECK_MODES,
    CheckResult,
    check_confinement,
    check_coverage,
    check_data_availability,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
    check_transmission,
    stage_outcome,
)


def _by_id(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.check: r for r in results}


class TestConfinement:
    def test_bounded_paths_pass(self):
        x = np.random.default_rng(0).normal(0, 1, (200, 40))
        res = _by_id(check_confinement("A", x, np.arange(40) * 0.1))
        assert res["C1a finiteness"].passed
        assert res["C1b confinement"].passed

    def test_nonfinite_fails_c1a(self):
        x = np.random.default_rng(1).normal(0, 1, (200, 40))
        x[:20, 30:] = np.nan
        res = _by_id(check_confinement("A", x, np.arange(40) * 0.1))
        assert not res["C1a finiteness"].passed

    def test_growth_fails_c1b(self):
        t = np.arange(40)
        base = np.random.default_rng(2).normal(0, 0.1, (200, 1))
        x = base * np.exp(0.3 * t)[None, :]  # amplitude grows ~e^{0.3t}
        res = _by_id(check_confinement("A", x, t * 0.1))
        assert res["C1a finiteness"].passed
        assert not res["C1b confinement"].passed

    def test_c1b_calibration_is_designable(self):
        t = np.arange(40)
        x = np.random.default_rng(5).normal(0, 1, (400, 40))
        x[:5] = 0.1 * np.exp(0.2 * t)[None, :]  # 1.25% of draws grow ~e^{0.2t}
        strict = _by_id(check_confinement("A", x, t * 0.1))
        assert not strict["C1b confinement"].passed
        lenient = _by_id(check_confinement("A", x, t * 0.1, max_explosive_frac=0.05))
        assert lenient["C1b confinement"].passed
        immune = _by_id(check_confinement("A", x, t * 0.1, growth_ratio=1000.0))
        assert immune["C1b confinement"].passed


class TestScale:
    def test_scale_in_band_passes(self):
        x = np.random.default_rng(3).normal(0, 1.0, (200, 40))
        r = check_scale("A", x, scale_anchor=1.0, anchor_src="test", anchor_detail="d")
        assert r.passed

    def test_scale_far_below_anchor_fails(self):
        x = np.random.default_rng(4).normal(0, 1.0, (200, 40))
        r = check_scale("A", x, scale_anchor=10.0, anchor_src="test", anchor_detail="d")
        assert not r.passed  # median sd ~1 vs band [3.3, 30]

    def test_static_state_uses_across_draw_scale(self):
        values = np.linspace(-2.0, 2.0, 200)
        x = np.repeat(values[:, None], 40, axis=1)
        assert check_scale("static", x).passed


class TestResolvability:
    # cadence 0.5, span 60 -> window [0.167, 15]
    def test_tau_in_window_passes(self):
        tau = np.full(200, 4.0)
        assert check_resolvability("A", tau, np.arange(0.0, 60.5, 0.5)).passed

    def test_tau_below_floor_fails(self):
        tau = np.full(200, 0.05)
        r = check_resolvability("A", tau, np.arange(0.0, 60.5, 0.5))
        assert not r.passed
        assert "too fast" in " ".join(r.diagnosis)

    def test_tau_above_ceiling_fails(self):
        tau = np.full(200, 25.0)
        r = check_resolvability("A", tau, np.arange(0.0, 60.5, 0.5))
        assert not r.passed
        assert "too slow" in " ".join(r.diagnosis)


class TestEdgeShare:
    def test_edge_not_overwhelming_passes(self):
        on = np.random.default_rng(5).normal(0, 1, (200, 30))
        off = on * 0.98  # edge barely moves the child
        r = _by_id(check_edge_share("A", on, off))["C4b edge overwhelm"]
        assert r.passed

    def test_edge_overwhelm_fails(self):
        on = np.random.default_rng(6).normal(0, 1, (200, 30))
        off = np.zeros_like(on)  # child entirely edge-driven
        r = _by_id(check_edge_share("A", on, off))["C4b edge overwhelm"]
        assert not r.passed

    def test_constant_level_shift_is_not_temporal_overwhelm(self):
        on = np.full((200, 30), 2.0)
        off = np.full((200, 30), 1.0)
        result = _by_id(check_edge_share("P->C", on, off))["C4b edge overwhelm"]
        assert result.passed
        assert result.evidence is not None
        assert np.median(result.evidence["level_shift"]) == 1.0


class TestSaturation:
    def test_ec50_in_parent_range_passes(self):
        parent = np.random.default_rng(7).lognormal(0, 1.0, (200, 30))
        ec50 = np.full(200, 0.8)
        hill_n = np.full(200, 2.0)
        assert check_saturation("P->C", ec50, hill_n, parent).passed

    def test_ec50_above_range_is_dead_arm(self):
        parent = np.random.default_rng(8).normal(0, 1.0, (200, 30))
        ec50 = np.full(200, 6.0)
        r = check_saturation("P->C", ec50, np.full(200, 2.0), parent)
        assert not r.passed
        assert "dead-low" in " ".join(r.diagnosis)

    def test_draw_pairing_cannot_be_destroyed_by_pooling(self):
        ec50 = np.r_[np.full(100, 0.1), np.full(100, 10.0)]
        parent = np.r_[np.full((100, 30), 10.0), np.full((100, 30), 0.1)]
        result = check_saturation("P->C", ec50, np.full(200, 2.0), parent)
        assert not result.passed


class TestCoverage:
    def _obs(self, rng):
        return rng.normal(10.0, 2.0, 100)

    def test_well_specified_passes_all(self):
        rng = np.random.default_rng(9)
        y_obs = self._obs(rng)
        signal = rng.normal(10.0, 2.0, (200, 100))  # signal matches data spread
        pp = signal + rng.normal(0, 0.8, (200, 100))  # + modest noise
        res = _by_id(
            [
                *check_coverage("y", pp, y_obs, distribution="gaussian"),
                check_transmission("y", signal, np.full_like(signal, 0.8**2)),
            ]
        )
        assert res["C5a location reach"].passed
        assert res["C5b width"].passed
        assert res["C5c transmission"].passed

    def test_noise_dominated_flat_signal_fails_c5c_only(self):
        rng = np.random.default_rng(10)
        y_obs = self._obs(rng)
        signal = np.full((200, 100), 10.0) + rng.normal(0, 0.05, (200, 100))  # nearly flat signal
        pp = signal + rng.normal(0, 2.0, (200, 100))  # noise alone covers the data spread
        res = _by_id(
            [
                *check_coverage("y", pp, y_obs, distribution="gaussian"),
                check_transmission("y", signal, np.full_like(signal, 2.0**2)),
            ]
        )
        assert res["C5a location reach"].passed
        assert res["C5b width"].passed  # noise widens the band
        assert not res["C5c transmission"].passed  # but the signal explains almost nothing

    def test_location_miss_fails_c5a(self):
        rng = np.random.default_rng(11)
        y_obs = self._obs(rng)
        signal = rng.normal(50.0, 2.0, (200, 100))  # centered far from the data
        pp = signal + rng.normal(0, 0.8, (200, 100))
        res = _by_id(check_coverage("y", pp, y_obs, distribution="gaussian"))
        assert not res["C5a location reach"].passed

    def test_zero_iqr_count_data_uses_family_statistics(self):
        rng = np.random.default_rng(12)
        y_obs = np.zeros(100)
        pp = rng.poisson(0.1, (200, 100))
        results = _by_id(check_coverage("events", pp, y_obs, distribution="poisson"))
        assert "1000000000" not in results["C5b width"].value
        assert "zero fraction" in results["C5b width"].value

    def test_categorical_checks_are_label_permutation_invariant(self):
        rng = np.random.default_rng(13)
        observed = rng.integers(0, 3, 120)
        predictive = rng.integers(0, 3, (200, 120))
        probabilities = rng.dirichlet(np.ones(3), size=(200, 120))
        original = _by_id(
            [
                *check_coverage(
                    "category",
                    predictive,
                    observed,
                    distribution="categorical",
                    level_count=3,
                ),
                check_transmission("category", probabilities),
            ]
        )
        permutation = np.array([2, 0, 1])
        inverse = np.argsort(permutation)
        relabeled = _by_id(
            [
                *check_coverage(
                    "category",
                    permutation[predictive],
                    permutation[observed],
                    distribution="categorical",
                    level_count=3,
                ),
                check_transmission("category", probabilities[..., inverse]),
            ]
        )
        for check in original:
            assert original[check].passed == relabeled[check].passed

    def test_no_observed_values_is_explicit_soft_failure(self):
        result = check_data_availability("y")
        assert result.check == "C5d data availability"
        assert result.passed is False
        outcome, annotations = stage_outcome(
            [result],
            {("C5d data availability", "y"): "No measurements were available in this panel."},
        )
        assert outcome == "ADMITTED with accepted consequences"
        assert "prior-driven" in annotations[0]


class TestTransmission:
    def test_sparse_poisson_rate_movement_does_not_collapse_with_zero_iqr(self):
        signal = np.tile(np.r_[np.full(50, 0.01), np.full(50, 0.5)], (200, 1))
        predictive = np.random.default_rng(14).poisson(signal)
        predictive_iqr = np.subtract(*np.percentile(predictive, [75, 25], axis=1))
        assert np.median(predictive_iqr) == 0.0
        result = check_transmission("events", signal, signal)
        assert result.passed
        assert result.evidence is not None
        assert np.median(result.evidence["signal_fraction"]) > 0.04

    def test_sparse_bernoulli_probability_movement_does_not_collapse_with_zero_iqr(self):
        probability = np.tile(np.r_[np.full(50, 0.01), np.full(50, 0.4)], (200, 1))
        predictive = np.random.default_rng(15).binomial(1, probability)
        predictive_iqr = np.subtract(*np.percentile(predictive, [75, 25], axis=1))
        assert np.median(predictive_iqr) == 0.0
        conditional_variance = probability * (1.0 - probability)
        result = check_transmission("event", probability, conditional_variance)
        assert result.passed
        assert result.evidence is not None
        assert np.median(result.evidence["signal_fraction"]) > 0.04

    def test_flat_sparse_signal_fails(self):
        signal = np.full((200, 100), 0.1)
        result = check_transmission("events", signal, signal)
        assert not result.passed
        assert "conditional observation variance" in " ".join(result.diagnosis)


class TestStageOutcome:
    def _res(self, check: str, passed: bool) -> CheckResult:
        return CheckResult(check, "A", "", "", passed, "note")

    def test_all_pass_admits(self):
        out, ann = stage_outcome([self._res("C2 latent scale", True)], {})
        assert out == "ADMITTED"
        assert ann == ()

    def test_hard_failure_blocks(self):
        out, _ = stage_outcome([self._res("C1a finiteness", False)], {})
        assert out.startswith("BLOCKED")

    def test_unaccepted_soft_needs_decision(self):
        out, _ = stage_outcome([self._res("C3 resolvability", False)], {})
        assert out.startswith("NEEDS DECISION")
        assert "C3 resolvability" in out

    def test_accepted_soft_admits_with_annotation(self):
        out, ann = stage_outcome(
            [self._res("C3 resolvability", False)],
            {("C3 resolvability", "A"): "sub-daily settling is a design limit"},
        )
        assert out == "ADMITTED with accepted consequences"
        assert len(ann) == 1
        assert "design limit" in ann[0]

    def test_hard_beats_accepted_soft(self):
        out, _ = stage_outcome(
            [self._res("C1a finiteness", False), self._res("C3 resolvability", False)],
            {("C3 resolvability", "A"): "ok"},
        )
        assert out.startswith("BLOCKED")

    def test_acceptance_is_scoped_to_one_target(self):
        first = CheckResult("C5b width", "first", "", "", False, "note")
        second = CheckResult("C5b width", "second", "", "", False, "note")
        out, annotations = stage_outcome(
            [first, second],
            {("C5b width", "first"): "accepted only for the first indicator"},
        )
        assert out.startswith("NEEDS DECISION")
        assert "second" in out
        assert len(annotations) == 1


def test_every_check_id_has_a_mode():
    # Guard: every check id any function can emit must be classified in CHECK_MODES.
    emitted = {
        "C1a finiteness",
        "C1b confinement",
        "C2 latent scale",
        "C3 resolvability",
        "C4b edge overwhelm",
        "C4c saturation",
        "C5a location reach",
        "C5b width",
        "C5c transmission",
        "C5d data availability",
    }
    assert emitted == set(CHECK_MODES)
