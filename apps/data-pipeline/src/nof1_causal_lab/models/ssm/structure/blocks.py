"""Block-spec abstraction for every param-bearing concern of an SSM.

Each ``*BlockSpec`` is the canonical declarative representation of one
SSM concern (process-noise Cholesky, manifest mean, initial-state
covariance, …). ``SSMSpec`` stores these blocks as its only
param-bearing fields: there are no flat-field duplicates.

Each block is a frozen dataclass with:

- Its structural data (masks + templates) — direct fields
- Its priors — ``Distribution | dict | None`` resolved through the
  canonical prior materialization path
- A ``sample_params(prefix)`` method that emits the sampled values
  via ``numpyro.sample`` with bare site names so existing autoreparam
  / posterior-analysis tooling keeps working
- Assembly delegated to ``structure.assembly`` (single algorithmic
  source of truth shared with ``SSMParameterLayout``)

At sample time, ``SSMModel._sample_*`` clones the spec's block with
runtime priors attached (via ``dataclasses.replace``) and calls
``sample_params``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpyro

from nof1_causal_lab.models.ssm.priors import resolve_prior_distribution

if TYPE_CHECKING:
    import jax.numpy as jnp
    import numpy as np
    from jax import Array


# ---------------------------------------------------------------------------
# Helpers (single source of truth lives in ``structure.assembly``; the
# helpers here are thin wrappers that pre-compute positions from masks)
# ---------------------------------------------------------------------------


# Position extractors live in ``structure.assembly`` — the single
# canonical implementation also used by ``SSMParameterLayout``. The
# block specs below import them lazily to avoid an eager dependency at
# module-load time.


# ---------------------------------------------------------------------------
# Diffusion block: process-noise Cholesky factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DiffusionBlockSpec:
    """Process-noise Cholesky factor ``L_Q`` block.

    Free entries on the lower-Cholesky mask are sampled from
    ``diag_prior`` (diagonal entries) and ``lower_prior`` (strict-lower
    entries). Time-invariant latents have their diagonal forced toward
    a tiny epsilon so their diffusion is effectively zero.
    """

    n_latent: int
    diffusion_chol_mask: np.ndarray
    diffusion_chol_template: jnp.ndarray
    time_invariant_mask: np.ndarray | None = None
    diag_prior: Any = None
    lower_prior: Any = None
    bare_site_names: bool = True

    @property
    def diffusion_diag_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import chol_diag_positions

        return chol_diag_positions(self.diffusion_chol_mask, self.n_latent)

    @property
    def diffusion_lower_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import strict_lower_positions

        return strict_lower_positions(self.diffusion_chol_mask, self.n_latent)

    @property
    def n_diffusion_diag(self) -> int:
        return len(self.diffusion_diag_positions)

    @property
    def n_diffusion_lower(self) -> int:
        return len(self.diffusion_lower_positions)

    def assemble(
        self,
        diag_free: jnp.ndarray | None = None,
        lower_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_diffusion_chol

        return assemble_diffusion_chol(
            diffusion_chol_template=self.diffusion_chol_template,
            diag_positions=self.diffusion_diag_positions,
            lower_positions=self.diffusion_lower_positions,
            diag_free=diag_free,
            lower_free=lower_free,
            time_invariant_mask=self.time_invariant_mask,
        )

    def sample_params(self, prefix: str = "") -> dict[str, Array]:
        def _name(bare: str) -> str:
            return bare if self.bare_site_names else f"{prefix}_{bare}"

        diag_free = None
        if self.n_diffusion_diag > 0:
            dist = resolve_prior_distribution(self.diag_prior)
            if dist is None:
                raise ValueError(
                    "DiffusionBlockSpec requires diag_prior when n_diffusion_diag > 0"
                )
            diag_free = numpyro.sample(_name("diffusion_diag_free"), dist)

        lower_free = None
        if self.n_diffusion_lower > 0:
            dist = resolve_prior_distribution(self.lower_prior)
            if dist is None:
                raise ValueError(
                    "DiffusionBlockSpec requires lower_prior when n_diffusion_lower > 0"
                )
            lower_free = numpyro.sample(_name("diffusion_lower_free"), dist)

        diffusion = self.assemble(diag_free, lower_free)
        numpyro.deterministic(_name("diffusion"), diffusion)
        return {"diffusion": diffusion}


# ---------------------------------------------------------------------------
# Sparse-vector block: a vector-shape parameter (means, intercepts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SparseVectorBlockSpec:
    """Generic sparse-vector block: free entries on a 1-D mask substituted
    into a template. Used for ``t0_means``, ``manifest_means``, and
    any other length-``n`` parameter sampled element-wise.

    The ``free_site_name`` controls the bare NumPyro site name for the
    sparse free vector (e.g. ``"t0_means_free"``,
    ``"manifest_means_free"``). The ``det_site_name`` controls the bare
    NumPyro deterministic site name for the assembled vector (e.g.
    ``"t0_means"``, ``"manifest_means"``).
    """

    n: int
    mask: np.ndarray
    template: jnp.ndarray
    free_site_name: str
    det_site_name: str
    prior: Any = None
    bare_site_names: bool = True

    @property
    def free_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return dense_vector_positions(self.mask, self.n)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_sparse_vector

        return assemble_sparse_vector(
            template=self.template,
            free_positions=self.free_positions,
            free=free,
        )

    def sample_params(self, prefix: str = "") -> dict[str, Array]:
        def _name(bare: str) -> str:
            return bare if self.bare_site_names else f"{prefix}_{bare}"

        free = None
        if self.n_free > 0:
            dist = resolve_prior_distribution(self.prior)
            if dist is None:
                raise ValueError(
                    f"SparseVectorBlockSpec({self.free_site_name}) requires prior "
                    f"when n_free={self.n_free} > 0"
                )
            free = numpyro.sample(_name(self.free_site_name), dist)

        assembled = self.assemble(free)
        numpyro.deterministic(_name(self.det_site_name), assembled)
        return {self.det_site_name: assembled}


# ---------------------------------------------------------------------------
# Sparse-matrix block: a rectangular-shape parameter (loadings, input_effect)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SparseMatrixBlockSpec:
    """Generic sparse-matrix block: free entries on a 2-D rectangular mask
    substituted into a template. Used for ``lambda_mat`` (loading
    matrix) and ``input_effect``.
    """

    n_rows: int
    n_cols: int
    mask: np.ndarray
    template: jnp.ndarray
    free_site_name: str
    det_site_name: str
    prior: Any = None
    bare_site_names: bool = True

    @property
    def free_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import rect_matrix_positions

        return rect_matrix_positions(self.mask, self.n_rows, self.n_cols)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_sparse_matrix

        return assemble_sparse_matrix(
            template=self.template,
            free_positions=self.free_positions,
            free=free,
        )

    def sample_params(self, prefix: str = "") -> dict[str, Array]:
        def _name(bare: str) -> str:
            return bare if self.bare_site_names else f"{prefix}_{bare}"

        free = None
        if self.n_free > 0:
            dist = resolve_prior_distribution(self.prior)
            if dist is None:
                raise ValueError(
                    f"SparseMatrixBlockSpec({self.free_site_name}) requires prior "
                    f"when n_free={self.n_free} > 0"
                )
            free = numpyro.sample(_name(self.free_site_name), dist)

        assembled = self.assemble(free)
        numpyro.deterministic(_name(self.det_site_name), assembled)
        return {self.det_site_name: assembled}


# ---------------------------------------------------------------------------
# Manifest-Cholesky block: diagonal Cholesky factor for observation noise
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ManifestCholBlockSpec:
    """Manifest-noise Cholesky factor block. Diagonal Cholesky
    (per-channel variance); off-diagonal correlation is not modelled at
    this layer.
    """

    n_manifest: int
    diag_mask: np.ndarray
    template: jnp.ndarray
    diag_prior: Any = None
    bare_site_names: bool = True

    @property
    def free_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return dense_vector_positions(self.diag_mask, self.n_manifest)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_manifest_chol

        return assemble_manifest_chol(
            template=self.template,
            free_positions=self.free_positions,
            free=free,
        )

    def sample_params(self, prefix: str = "") -> dict[str, Array]:
        """Sample the diagonal free entries and assemble the Cholesky.

        Does NOT emit a ``manifest_cov`` deterministic — that's the
        caller's responsibility; ``SSMModel._sample_manifest_params``
        emits it once after combining the means and Cholesky blocks.
        """

        def _name(bare: str) -> str:
            return bare if self.bare_site_names else f"{prefix}_{bare}"

        free = None
        if self.n_free > 0:
            dist = resolve_prior_distribution(self.diag_prior)
            if dist is None:
                raise ValueError(
                    "ManifestCholBlockSpec requires diag_prior when n_free > 0"
                )
            free = numpyro.sample(_name("manifest_var_diag_free"), dist)

        chol = self.assemble(free)
        return {"manifest_chol": chol}


# ---------------------------------------------------------------------------
# Initial-state covariance block: diagonal SDs + off-diagonal correlations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class T0CholBlockSpec:
    """Initial-state covariance block.

    The template is a Cholesky factor ``L`` such that ``L L^T`` is the
    base covariance. The factor is internally decomposed into base SDs
    (``sqrt(diag(LL^T))``) and base correlations
    (``LL^T / (std outer std)``). Free entries on ``diag_mask`` replace
    base SDs; free entries on ``correlation_mask`` (strict lower) replace
    base correlations symmetrically.

    The block assembles the LATENT-only covariance. Any static-factor
    contribution is added by the caller (see
    ``SSMModel._sample_t0_params``).
    """

    n_latent: int
    diag_mask: np.ndarray  # (n_latent,) bool
    correlation_mask: np.ndarray  # (n_latent, n_latent) strict lower bool
    template: jnp.ndarray  # (n_latent, n_latent) lower-Cholesky factor
    diag_prior: Any = None
    correlation_prior: Any = None
    bare_site_names: bool = True

    @property
    def diag_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return dense_vector_positions(self.diag_mask, self.n_latent)

    @property
    def correlation_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import strict_lower_positions

        return strict_lower_positions(self.correlation_mask, self.n_latent)

    @property
    def n_diag_free(self) -> int:
        return len(self.diag_positions)

    @property
    def n_correlation_free(self) -> int:
        return len(self.correlation_positions)

    @property
    def base_cov(self) -> jnp.ndarray:
        import jax.numpy as jnp_local

        L = jnp_local.asarray(self.template)
        return L @ L.T

    @property
    def base_std(self) -> jnp.ndarray:
        import jax.numpy as jnp_local

        return jnp_local.sqrt(jnp_local.clip(jnp_local.diag(self.base_cov), min=0.0))

    @property
    def base_corr(self) -> jnp.ndarray:
        import jax.numpy as jnp_local

        std = self.base_std
        cov = self.base_cov
        denom = std[:, None] * std[None, :]
        corr = jnp_local.where(
            denom > 0,
            cov / denom,
            jnp_local.eye(self.n_latent, dtype=cov.dtype),
        )
        corr = 0.5 * (corr + corr.T)
        return corr.at[jnp_local.diag_indices(self.n_latent)].set(1.0)

    def assemble_cov(
        self,
        diag_free: jnp.ndarray | None = None,
        correlation_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        import jax.numpy as jnp_local

        std = self.base_std
        if diag_free is not None:
            diag_free = jnp_local.asarray(diag_free, dtype=std.dtype)
            for idx, latent_idx in enumerate(self.diag_positions):
                std = std.at[latent_idx].set(diag_free[idx])
        corr = self.base_corr
        if correlation_free is not None:
            correlation_free = jnp_local.asarray(correlation_free, dtype=corr.dtype)
            for idx, (row, col) in enumerate(self.correlation_positions):
                corr = corr.at[row, col].set(correlation_free[idx])
                corr = corr.at[col, row].set(correlation_free[idx])
        cov = corr * (std[:, None] * std[None, :])
        return 0.5 * (cov + cov.T)

    def sample_params(self, prefix: str = "") -> dict[str, Array]:
        """Sample free diagonal SDs and free off-diagonal correlations.

        Returns a dict with ``diag_free`` and ``correlation_free`` entries
        (may be ``None`` for empty masks). The caller assembles the final
        covariance (possibly composing with a static-factor contribution).
        """

        def _name(bare: str) -> str:
            return bare if self.bare_site_names else f"{prefix}_{bare}"

        diag_free = None
        if self.n_diag_free > 0:
            dist = resolve_prior_distribution(self.diag_prior)
            if dist is None:
                raise ValueError(
                    "T0CholBlockSpec requires diag_prior when n_diag_free > 0"
                )
            diag_free = numpyro.sample(_name("t0_var_diag_free"), dist)

        correlation_free = None
        if self.n_correlation_free > 0:
            dist = resolve_prior_distribution(self.correlation_prior)
            if dist is None:
                raise ValueError(
                    "T0CholBlockSpec requires correlation_prior when n_correlation_free > 0"
                )
            correlation_free = numpyro.sample(_name("t0_var_lower_free"), dist)

        return {
            "diag_free": diag_free,
            "correlation_free": correlation_free,
        }


# ---------------------------------------------------------------------------
# Default factories — single-purpose constructors for the "standard test
# default" shape of each block. Tests that don't care about a particular
# block's structure call these; tests that do care construct the block
# inline. Production code uses these only for trivial bootstrap paths
# (e.g. ``ssm.builder`` auto-detect).
# ---------------------------------------------------------------------------


def default_diffusion_block(n_latent: int) -> DiffusionBlockSpec:
    """Full lower-Cholesky free entries, identity template, no time-invariant mask."""
    import jax.numpy as _jnp
    import numpy as _np

    return DiffusionBlockSpec(
        n_latent=n_latent,
        diffusion_chol_mask=_np.tri(n_latent, dtype=bool),
        diffusion_chol_template=_jnp.eye(n_latent),
    )


def default_lambda_block(n_manifest: int, n_latent: int) -> SparseMatrixBlockSpec:
    """Zero free mask, identity loading template."""
    import jax.numpy as _jnp
    import numpy as _np

    return SparseMatrixBlockSpec(
        n_rows=n_manifest,
        n_cols=n_latent,
        mask=_np.zeros((n_manifest, n_latent), dtype=bool),
        template=_jnp.eye(n_manifest, n_latent),
        free_site_name="lambda_free",
        det_site_name="lambda",
    )


def default_manifest_means_block(n_manifest: int) -> SparseVectorBlockSpec:
    """Zero free mask, zeros template."""
    import jax.numpy as _jnp
    import numpy as _np

    return SparseVectorBlockSpec(
        n=n_manifest,
        mask=_np.zeros(n_manifest, dtype=bool),
        template=_jnp.zeros(n_manifest),
        free_site_name="manifest_means_free",
        det_site_name="manifest_means",
    )


def default_manifest_chol_block(n_manifest: int) -> ManifestCholBlockSpec:
    """Full diagonal free mask, zeros template."""
    import jax.numpy as _jnp
    import numpy as _np

    return ManifestCholBlockSpec(
        n_manifest=n_manifest,
        diag_mask=_np.ones(n_manifest, dtype=bool),
        template=_jnp.zeros((n_manifest, n_manifest)),
    )


def default_t0_means_block(n_latent: int) -> SparseVectorBlockSpec:
    """Full free mask, zeros template."""
    import jax.numpy as _jnp
    import numpy as _np

    return SparseVectorBlockSpec(
        n=n_latent,
        mask=_np.ones(n_latent, dtype=bool),
        template=_jnp.zeros(n_latent),
        free_site_name="t0_means_free",
        det_site_name="t0_means",
    )


def default_t0_chol_block(n_latent: int) -> T0CholBlockSpec:
    """Full diagonal free, full strict-lower correlation free, identity template."""
    import jax.numpy as _jnp
    import numpy as _np

    return T0CholBlockSpec(
        n_latent=n_latent,
        diag_mask=_np.ones(n_latent, dtype=bool),
        correlation_mask=_np.tri(n_latent, k=-1, dtype=bool),
        template=_jnp.eye(n_latent),
    )


def default_input_effect_block(n_latent: int) -> SparseMatrixBlockSpec:
    """Empty input-effect block (no inputs)."""
    import jax.numpy as _jnp
    import numpy as _np

    return SparseMatrixBlockSpec(
        n_rows=n_latent,
        n_cols=0,
        mask=_np.zeros((n_latent, 0), dtype=bool),
        template=_jnp.zeros((n_latent, 0)),
        free_site_name="input_effect_free",
        det_site_name="input_effect",
    )


def default_static_state_sd_block() -> SparseVectorBlockSpec:
    """Empty static-state-sd block (no static factors)."""
    import jax.numpy as _jnp
    import numpy as _np

    return SparseVectorBlockSpec(
        n=0,
        mask=_np.zeros(0, dtype=bool),
        template=_jnp.zeros(0),
        free_site_name="static_state_sd_free",
        det_site_name="static_state_sds",
    )
