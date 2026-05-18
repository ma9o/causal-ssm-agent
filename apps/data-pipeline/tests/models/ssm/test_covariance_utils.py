import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter


def test_symmetrize_with_jitter_handles_batched_covariances():
    cov = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    result = symmetrize_with_jitter(cov, jitter=1e-3)

    expected = np.array(
        [
            [[1.001, 2.5], [2.5, 4.001]],
            [[5.001, 6.5], [6.5, 8.001]],
        ]
    )
    np.testing.assert_allclose(np.asarray(result), expected)


def test_symmetrize_handles_batched_covariances():
    cov = jnp.array(
        [
            [[0.0, 1.0], [3.0, 2.0]],
            [[4.0, 5.0], [7.0, 6.0]],
        ]
    )

    result = symmetrize(cov)

    expected = np.array(
        [
            [[0.0, 2.0], [2.0, 2.0]],
            [[4.0, 6.0], [6.0, 6.0]],
        ]
    )
    np.testing.assert_allclose(np.asarray(result), expected)
