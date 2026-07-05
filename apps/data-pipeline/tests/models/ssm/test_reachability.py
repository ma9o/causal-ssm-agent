"""Unit tests for the prior-predictive reachability battery (pure, array-in)."""

from __future__ import annotations

import numpy as np

from nof1_causal_lab.models.ssm.reachability import (
    CHECK_MODES,
    CheckResult,
    check_confinement,
    check_coverage,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
    stage_outcome,
)


def _by_id(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.check: r for r in results}


class TestConfinement:
    def test_bounded_paths_pass(self):
        x = np.random.default_rng(0).normal(0, 1, (200, 40))
        res = _by_id(check_confinement("A", x, dt=0.1))
        assert res["C1a finiteness"].passed
        assert res["C1b confinement"].passed

    def test_nonfinite_fails_c1a(self):
        x = np.random.default_rng(1).normal(0, 1, (200, 40))
        x[:20, 30:] = np.nan
        res = _by_id(check_confinement("A", x, dt=0.1))
        assert not res["C1a finiteness"].passed

    def test_growth_fails_c1b(self):
        t = np.arange(40)
        base = np.random.default_rng(2).normal(0, 0.1, (200, 1))
        x = base * np.exp(0.3 * t)[None, :]  # amplitude grows ~e^{0.3t}
        res = _by_id(check_confinement("A", x, dt=0.1))
        assert res["C1a finiteness"].passed
        assert not res["C1b confinement"].passed


class TestScale:
    def test_scale_in_band_passes(self):
        x = np.random.default_rng(3).normal(0, 1.0, (200, 40))
        r = check_scale("A", x, scale_anchor=1.0, anchor_src="test", anchor_detail="d")
        assert r.passed

    def test_scale_far_below_anchor_fails(self):
        x = np.random.default_rng(4).normal(0, 1.0, (200, 40))
        r = check_scale("A", x, scale_anchor=10.0, anchor_src="test", anchor_detail="d")
        assert not r.passed  # median sd ~1 vs band [3.3, 30]


class TestResolvability:
    # cadence 0.5, span 60 -> window [0.167, 15]
    def test_tau_in_window_passes(self):
        tau = np.full(200, 4.0)
        assert check_resolvability("A", tau, cadence=0.5, span=60.0).passed

    def test_tau_below_floor_fails(self):
        tau = np.full(200, 0.05)
        r = check_resolvability("A", tau, cadence=0.5, span=60.0)
        assert not r.passed
        assert "below the design floor" in " ".join(r.diagnosis)

    def test_tau_above_ceiling_fails(self):
        tau = np.full(200, 25.0)
        r = check_resolvability("A", tau, cadence=0.5, span=60.0)
        assert not r.passed
        assert "above the design ceiling" in " ".join(r.diagnosis)


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


class TestSaturation:
    def test_ec50_in_parent_range_passes(self):
        parent = np.random.default_rng(7).normal(0, 1.0, (200, 30))
        ec50 = np.full(200, 0.8)  # inside ~[-1.28, 1.28]
        assert check_saturation("P->C", ec50, parent).passed

    def test_ec50_above_range_is_dead_arm(self):
        parent = np.random.default_rng(8).normal(0, 1.0, (200, 30))
        ec50 = np.full(200, 6.0)
        r = check_saturation("P->C", ec50, parent)
        assert not r.passed
        assert "dead" in " ".join(r.diagnosis) or "never bends" in " ".join(r.diagnosis)


class TestCoverage:
    def _obs(self, rng):
        return rng.normal(10.0, 2.0, 100)

    def test_well_specified_passes_all(self):
        rng = np.random.default_rng(9)
        y_obs = self._obs(rng)
        signal = rng.normal(10.0, 2.0, (200, 100))  # signal matches data spread
        pp = signal + rng.normal(0, 0.8, (200, 100))  # + modest noise
        res = _by_id(check_coverage("y", pp, signal, y_obs))
        assert res["C5a location reach"].passed
        assert res["C5b width"].passed
        assert res["C5c transmission"].passed

    def test_saturated_link_fails_c5c_only(self):
        rng = np.random.default_rng(10)
        y_obs = self._obs(rng)
        signal = np.full((200, 100), 10.0) + rng.normal(0, 0.05, (200, 100))  # nearly flat signal
        pp = signal + rng.normal(0, 2.0, (200, 100))  # noise alone covers the data spread
        res = _by_id(check_coverage("y", pp, signal, y_obs))
        assert res["C5a location reach"].passed
        assert res["C5b width"].passed  # noise widens the band
        assert not res["C5c transmission"].passed  # but the signal transmits ~nothing

    def test_location_miss_fails_c5a(self):
        rng = np.random.default_rng(11)
        y_obs = self._obs(rng)
        signal = rng.normal(50.0, 2.0, (200, 100))  # centered far from the data
        pp = signal + rng.normal(0, 0.8, (200, 100))
        res = _by_id(check_coverage("y", pp, signal, y_obs))
        assert not res["C5a location reach"].passed


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
            {"C3 resolvability": "sub-daily settling is a design limit"},
        )
        assert out == "ADMITTED with accepted consequences"
        assert len(ann) == 1
        assert "design limit" in ann[0]

    def test_hard_beats_accepted_soft(self):
        out, _ = stage_outcome(
            [self._res("C1a finiteness", False), self._res("C3 resolvability", False)],
            {"C3 resolvability": "ok"},
        )
        assert out.startswith("BLOCKED")


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
    }
    assert emitted == set(CHECK_MODES)
