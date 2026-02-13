"""Tests for the LaTeX renderer utility."""

import json

# tools/utils is not a package on the normal path, so add it
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from utils.latex_renderer import (
    model_spec_to_latex,
    render_measurement,
    render_priors,
    render_structural,
)

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def model_spec1():
    with (DATA_DIR / "eval/questions/1_resolve-errors-faster/model_spec.json").open() as f:
        return json.load(f)


@pytest.fixture
def causal_spec1():
    with (DATA_DIR / "eval/questions/1_resolve-errors-faster/causal_spec.json").open() as f:
        return json.load(f)


class TestRenderMeasurement:
    def test_returns_one_per_likelihood(self, model_spec1, causal_spec1):
        eqs = render_measurement(model_spec1, causal_spec1)
        assert len(eqs) == len(model_spec1["likelihoods"])

    def test_normal_identity(self, model_spec1, causal_spec1):
        eqs = render_measurement(model_spec1, causal_spec1)
        # morning_start_hour is Normal/identity
        normal_eqs = [e for e in eqs if "morning" in e and r"\mathcal{N}" in e]
        assert len(normal_eqs) >= 1

    def test_bernoulli_logit(self, model_spec1, causal_spec1):
        eqs = render_measurement(model_spec1, causal_spec1)
        bern_eqs = [e for e in eqs if "Bern" in e]
        assert len(bern_eqs) >= 1

    def test_negbin_log(self, model_spec1, causal_spec1):
        eqs = render_measurement(model_spec1, causal_spec1)
        nb_eqs = [e for e in eqs if "NegBin" in e]
        assert len(nb_eqs) >= 1

    def test_ordered_logistic(self, model_spec1, causal_spec1):
        eqs = render_measurement(model_spec1, causal_spec1)
        ol_eqs = [e for e in eqs if "OrdLogistic" in e]
        assert len(ol_eqs) >= 1

    def test_poisson(self, model_spec1, causal_spec1):
        eqs = render_measurement(model_spec1, causal_spec1)
        pois_eqs = [e for e in eqs if "Pois" in e and "NegBin" not in e]
        assert len(pois_eqs) >= 1

    def test_reference_indicator_has_loading_1(self, model_spec1, causal_spec1):
        """First indicator per construct should have loading=1 (no free loading param)."""
        eqs = render_measurement(model_spec1, causal_spec1)
        # late_night_browsing is reference for Sleep Quality — should have 1 not lambda
        late_night_eq = next(e for e in eqs if "late" in e)
        assert r"\lambda" not in late_night_eq

    def test_free_loading_has_lambda(self, model_spec1, causal_spec1):
        """Non-reference indicators should show lambda."""
        eqs = render_measurement(model_spec1, causal_spec1)
        # morning_start_hour has a loading_morning_start_hour param
        morning_eq = next(e for e in eqs if "morning" in e)
        assert r"\lambda" in morning_eq


class TestRenderStructural:
    def test_returns_equations_for_endogenous(self, model_spec1, causal_spec1):
        eqs = render_structural(model_spec1, causal_spec1)
        # 9 endogenous time-varying constructs
        assert len(eqs) == 9

    def test_has_ar_term(self, model_spec1, causal_spec1):
        eqs = render_structural(model_spec1, causal_spec1)
        ar_eqs = [e for e in eqs if r"\rho" in e]
        assert len(ar_eqs) == 9

    def test_has_innovation(self, model_spec1, causal_spec1):
        eqs = render_structural(model_spec1, causal_spec1)
        for eq in eqs:
            assert r"\varepsilon" in eq

    def test_lagged_uses_t_minus_1(self, model_spec1, causal_spec1):
        eqs = render_structural(model_spec1, causal_spec1)
        # At least one equation should have a t-1 parent (lagged edge)
        lagged = [e for e in eqs if "t-1" in e]
        assert len(lagged) >= 1


class TestRenderPriors:
    def test_groups_by_role(self, model_spec1):
        grouped = render_priors(model_spec1)
        assert "fixed_effect" in grouped
        assert "ar_coefficient" in grouped
        assert "residual_sd" in grouped
        assert "loading" in grouped
        assert "random_intercept_sd" in grouped

    def test_fixed_effect_count(self, model_spec1):
        grouped = render_priors(model_spec1)
        n_beta = len([p for p in model_spec1["parameters"] if p["role"] == "fixed_effect"])
        assert len(grouped["fixed_effect"]) == n_beta

    def test_prior_uses_correct_distribution(self, model_spec1):
        grouped = render_priors(model_spec1)
        for eq in grouped["ar_coefficient"]:
            assert r"\text{Beta}" in eq
        for eq in grouped["residual_sd"]:
            assert r"\text{HalfNormal}" in eq


class TestFullRoundtrip:
    def test_model_spec_to_latex(self, model_spec1, causal_spec1):
        result = model_spec_to_latex(model_spec1, causal_spec1)
        assert "measurement" in result
        assert "structural" in result
        assert "priors" in result
        assert len(result["measurement"]) == 26
        assert len(result["structural"]) == 9

    def test_without_causal_spec(self, model_spec1):
        result = model_spec_to_latex(model_spec1)
        assert len(result["measurement"]) == 26
        assert result["structural"] == []
