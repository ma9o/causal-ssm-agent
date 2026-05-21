"""Slow AutoReparam integration tests."""

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.models.ssm.dynamics.composite import linear_drift_spec
from tests.ssm_test_utils import block_ssm_spec

pytestmark = pytest.mark.slow


def _make_simple_ssm():
    from nof1_causal_lab.models.ssm.model import SSMModel

    return SSMModel(
        spec=block_ssm_spec(
            n_latent=2,
            n_manifest=2,
            drift_spec=linear_drift_spec(
                n_latent=2,
                drift_diag_mask=np.ones(2, dtype=bool),
                drift_offdiag_mask=np.array([[False, True], [True, False]]),
                drift_template=jnp.zeros((2, 2)),
                cint_mask=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
        )
    )


class TestAutoReparamSSM:
    def test_fit_map_filters_auxiliary_sites(self):
        from nof1_causal_lab.models.ssm.inference import fit

        model = _make_simple_ssm()
        observations = jnp.zeros((8, 2))
        times = jnp.linspace(0, 1, 8)

        result = fit(
            model,
            observations,
            times,
            method="map",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=0,
        )

        sample_names = set(result.get_samples())
        assert "vf_0_base_decay" in sample_names
        assert "diffusion_diag_free" in sample_names
        assert all("_decentered" not in name for name in sample_names)

        diag = result.get_mcmc_diagnostics()
        assert diag is None
