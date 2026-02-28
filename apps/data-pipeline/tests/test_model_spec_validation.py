"""Tests for ModelSpec domain validation.

Covers: validate_model_spec (distribution-link compatibility, role-constraint
compatibility, dtype-distribution compatibility, duplicate detection).
"""

from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LikelihoodSpec,
    LinkFunction,
    ModelSpec,
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
    validate_model_spec,
)


def _make_likelihood(variable="x", distribution="gaussian", link="identity"):
    return LikelihoodSpec(
        variable=variable,
        distribution=DistributionFamily(distribution),
        link=LinkFunction(link),
        reasoning="test",
    )


def _make_param(name="beta_x", role="fixed_effect", constraint="none"):
    return ParameterSpec(
        name=name,
        role=ParameterRole(role),
        constraint=ParameterConstraint(constraint),
        description="test param",
        search_context="test context",
    )


def _make_spec(likelihoods=None, parameters=None):
    return ModelSpec(
        likelihoods=likelihoods or [_make_likelihood()],
        parameters=parameters or [_make_param()],
        reasoning="test",
    )


# =============================================================================
# validate_model_spec
# =============================================================================


class TestValidateModelSpec:
    def test_valid_spec_no_issues(self):
        spec = _make_spec()
        issues = validate_model_spec(spec)
        assert issues == []

    def test_duplicate_likelihood_variable(self):
        spec = _make_spec(likelihoods=[
            _make_likelihood("mood"),
            _make_likelihood("mood"),
        ])
        issues = validate_model_spec(spec)
        assert any(i["severity"] == "error" and "duplicate" in i["issue"] for i in issues)

    def test_duplicate_parameter_name(self):
        spec = _make_spec(parameters=[
            _make_param("beta_x"),
            _make_param("beta_x"),
        ])
        issues = validate_model_spec(spec)
        assert any(i["severity"] == "error" and "duplicate" in i["issue"] for i in issues)

    def test_invalid_link_for_distribution(self):
        """Gaussian requires identity link, not log."""
        spec = _make_spec(likelihoods=[
            _make_likelihood("x", "gaussian", "log"),
        ])
        issues = validate_model_spec(spec)
        assert any(i["severity"] == "error" and "link" in i["issue"] for i in issues)

    def test_valid_link_for_bernoulli(self):
        """Bernoulli accepts logit link."""
        spec = _make_spec(likelihoods=[
            _make_likelihood("x", "bernoulli", "logit"),
        ])
        issues = validate_model_spec(spec)
        error_issues = [i for i in issues if i["severity"] == "error"]
        assert error_issues == []

    def test_role_constraint_mismatch_warning(self):
        """residual_sd with none constraint should warn."""
        spec = _make_spec(parameters=[
            _make_param("sigma_x", "residual_sd", "none"),
        ])
        issues = validate_model_spec(spec)
        assert any(i["severity"] == "warning" and "constraint" in i["issue"] for i in issues)

    def test_role_constraint_correct_no_warning(self):
        """residual_sd with positive constraint should not warn."""
        spec = _make_spec(parameters=[
            _make_param("sigma_x", "residual_sd", "positive"),
        ])
        issues = validate_model_spec(spec)
        constraint_warnings = [
            i for i in issues
            if i["severity"] == "warning" and "constraint" in i["issue"]
        ]
        assert constraint_warnings == []

    def test_dtype_distribution_mismatch(self):
        """Binary dtype with gaussian distribution should error."""
        spec = _make_spec(likelihoods=[
            _make_likelihood("x", "gaussian", "identity"),
        ])
        indicators = [{"name": "x", "measurement_dtype": "binary"}]
        issues = validate_model_spec(spec, indicators=indicators)
        assert any(i["severity"] == "error" and "dtype" in i["issue"] for i in issues)

    def test_dtype_distribution_correct(self):
        """Binary dtype with bernoulli should not error."""
        spec = _make_spec(likelihoods=[
            _make_likelihood("x", "bernoulli", "logit"),
        ])
        indicators = [{"name": "x", "measurement_dtype": "binary"}]
        issues = validate_model_spec(spec, indicators=indicators)
        dtype_errors = [
            i for i in issues
            if i["severity"] == "error" and "dtype" in i["issue"]
        ]
        assert dtype_errors == []

    def test_missing_indicator_coverage_warning(self):
        """Indicator without likelihood should warn."""
        spec = _make_spec(likelihoods=[
            _make_likelihood("x"),
        ])
        indicators = [
            {"name": "x", "measurement_dtype": "continuous"},
            {"name": "y", "measurement_dtype": "continuous"},
        ]
        issues = validate_model_spec(spec, indicators=indicators)
        assert any(i["severity"] == "warning" and "y" in i["issue"] for i in issues)

    def test_poisson_with_log_link_valid(self):
        spec = _make_spec(likelihoods=[
            _make_likelihood("x", "poisson", "log"),
        ])
        issues = validate_model_spec(spec)
        error_issues = [i for i in issues if i["severity"] == "error"]
        assert error_issues == []

    def test_multiple_issues_collected(self):
        """Should collect all issues, not just the first."""
        spec = _make_spec(
            likelihoods=[
                _make_likelihood("x", "gaussian", "log"),
                _make_likelihood("y", "bernoulli", "identity"),
            ],
            parameters=[
                _make_param("sigma", "residual_sd", "none"),
            ],
        )
        issues = validate_model_spec(spec)
        assert len(issues) >= 2  # at least one per bad likelihood
