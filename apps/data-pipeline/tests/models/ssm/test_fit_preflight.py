"""Tests for the fit-time observation/prior preflight checks."""

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, StateDecaySpec
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.preflight import (
    ObservationPreflightError,
    validate_observations_for_fit,
)
from nof1_causal_lab.models.ssm.structure import SparseVectorBlockSpec
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.testing import block_ssm_spec

RNG = np.random.default_rng(7)


def _manifest_means_block(free_support, template):
    return SparseVectorBlockSpec(
        n=len(template),
        free_support=np.asarray(free_support, dtype=bool),
        template=jnp.asarray(template, dtype=jnp.float32),
        free_site_name="manifest_means_free",
        det_site_name="manifest_means",
        support=SupportClass.REAL,
        site_kind=SiteKind.MANIFEST_MEANS,
        assembly_group="manifest",
        fixed_spec_field="manifest_means",
        priors_field="manifest_means",
    )


def _model(
    *,
    free_means,
    means_template=(0.0, 0.0),
    dists=(DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN),
    links=(LinkFunction.IDENTITY, LinkFunction.IDENTITY),
    centered=None,
):
    spec = block_ssm_spec(
        n_latent=1,
        n_manifest=2,
        dynamics_spec=DynamicsSpec(n_latent=1, components=(StateDecaySpec(target=0),)),
        manifest_means_block=_manifest_means_block(free_means, means_template),
        manifest_dists=list(dists),
        manifest_links=list(links),
        manifest_centered=list(centered) if centered is not None else None,
        manifest_names=["raw_channel", "small_channel"],
    )
    return SSMModel(spec)


def _observations(mean_a, mean_b, n=200):
    return np.column_stack([RNG.normal(mean_a, 1.0, size=n), RNG.normal(mean_b, 1.0, size=n)])


def test_raises_on_unreachable_free_manifest_mean():
    model = _model(free_means=(True, True))
    with pytest.raises(ObservationPreflightError, match="raw_channel"):
        validate_observations_for_fit(model, _observations(87.0, 0.1))


def test_passes_when_free_mean_is_within_prior_reach():
    model = _model(free_means=(True, True))
    validate_observations_for_fit(model, _observations(0.5, -0.3))


def test_fixed_manifest_means_are_not_judged():
    model = _model(free_means=(False, False), means_template=(87.0, 0.0))
    validate_observations_for_fit(model, _observations(87.0, 0.1))


def test_passes_when_centered_flag_matches_centered_data():
    model = _model(free_means=(True, True), centered=(True, False))
    obs = _observations(87.0, 0.1)
    obs[:, 0] -= obs[:, 0].mean()
    validate_observations_for_fit(model, obs)


def test_raises_when_centered_flag_without_centered_data():
    model = _model(free_means=(True, True), centered=(True, False))
    with pytest.raises(ObservationPreflightError, match="marked centered"):
        validate_observations_for_fit(model, _observations(87.0, 0.1))


def test_non_identity_links_are_not_judged():
    model = _model(
        free_means=(True, True),
        dists=(DistributionFamily.NEGATIVE_BINOMIAL, DistributionFamily.GAUSSIAN),
        links=(LinkFunction.LOG, LinkFunction.IDENTITY),
    )
    obs = _observations(0.0, 0.1)
    obs[:, 0] = RNG.poisson(80.0, size=obs.shape[0]).astype(np.float64)
    validate_observations_for_fit(model, obs)


def test_nan_only_channels_are_skipped():
    model = _model(free_means=(True, True))
    obs = _observations(0.2, 0.1)
    obs[:, 0] = np.nan
    validate_observations_for_fit(model, obs)


def test_fit_runs_preflight_before_dispatch():
    model = _model(free_means=(True, True))
    obs = _observations(87.0, 0.1)
    with pytest.raises(ObservationPreflightError, match="raw_channel"):
        fit(model, jnp.asarray(obs), jnp.arange(obs.shape[0], dtype=jnp.float32))
