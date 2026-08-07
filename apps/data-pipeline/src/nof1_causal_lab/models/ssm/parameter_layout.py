"""Parameter layout for block-based SSM specs.

``SSMParameterLayout`` is an index over the block-owned sample-site
descriptors produced by ``SSMSpec.iter_sample_sites()``. It does not
recompute positions from supports (the blocks already do that). It does
not own templates and it does not assemble matrices. Assembly belongs
to the block that owns the parameter.

The named position/index/count accessors are derived ``SiteKind``
lookups against ``by_name``; they exist to keep compile-time consumers'
call sites narrow, not to encode the topology a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SitePosition

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor


@dataclass(frozen=True)
class SSMParameterLayout:
    """Cached index over a spec's sample-site descriptors."""

    spec: SSMSpec
    sites: tuple[SiteDescriptor, ...]
    by_name: dict[str, SiteDescriptor]
    static_factor_name_index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: SSMSpec) -> SSMParameterLayout:
        sites = tuple(spec.iter_sample_sites())
        return cls(
            spec=spec,
            sites=sites,
            by_name={s.name: s for s in sites},
            static_factor_name_index={
                name: idx for idx, name in enumerate(spec.static_factor_names or [])
            },
        )

    # ------------------------------------------------------------------
    # SiteKind lookups (canonical)
    # ------------------------------------------------------------------

    def sites_by_kind(self, site_kind: SiteKind) -> tuple[SiteDescriptor, ...]:
        return tuple(site for site in self.sites if site.site_kind == site_kind)

    def site_by_kind(self, site_kind: SiteKind) -> SiteDescriptor | None:
        sites = self.sites_by_kind(site_kind)
        if not sites:
            return None
        if len(sites) > 1:
            raise ValueError(
                f"Expected one active site of kind {site_kind.value!r}, got "
                f"{[site.name for site in sites]}"
            )
        return sites[0]

    def _positions_for_kind(self, site_kind: SiteKind) -> list[SitePosition]:
        site = self.site_by_kind(site_kind)
        return list(site.positions) if site is not None else []

    def _count_for_kind(self, site_kind: SiteKind) -> int:
        site = self.site_by_kind(site_kind)
        return len(site.positions) if site is not None else 0

    def _index_for_kind(self, site_kind: SiteKind) -> dict[SitePosition, int]:
        site = self.site_by_kind(site_kind)
        if site is None:
            return {}
        return {position: flat_idx for flat_idx, position in enumerate(site.positions)}

    def _vector_positions_for_kind(self, site_kind: SiteKind) -> list[int]:
        return cast("list[int]", self._positions_for_kind(site_kind))

    def _vector_index_for_kind(self, site_kind: SiteKind) -> dict[int, int]:
        return cast("dict[int, int]", self._index_for_kind(site_kind))

    def _matrix_positions_for_kind(self, site_kind: SiteKind) -> list[tuple[int, int]]:
        return cast("list[tuple[int, int]]", self._positions_for_kind(site_kind))

    def _matrix_index_for_kind(self, site_kind: SiteKind) -> dict[tuple[int, int], int]:
        return cast("dict[tuple[int, int], int]", self._index_for_kind(site_kind))

    # ------------------------------------------------------------------
    # Named positions / index / count accessors (derived from SiteKind)
    # ------------------------------------------------------------------

    @property
    def static_state_sd_free_positions(self) -> list[int]:
        return self._vector_positions_for_kind(SiteKind.STATIC_STATE_SD)

    @property
    def static_state_sd_free_index(self) -> dict[int, int]:
        return self._vector_index_for_kind(SiteKind.STATIC_STATE_SD)

    @property
    def n_static_state_sd(self) -> int:
        return self._count_for_kind(SiteKind.STATIC_STATE_SD)

    @property
    def input_effect_positions(self) -> list[tuple[int, int]]:
        return self._matrix_positions_for_kind(SiteKind.INPUT_EFFECT)

    @property
    def input_effect_index(self) -> dict[tuple[int, int], int]:
        return self._matrix_index_for_kind(SiteKind.INPUT_EFFECT)

    @property
    def n_input_effect(self) -> int:
        return self._count_for_kind(SiteKind.INPUT_EFFECT)

    @property
    def diffusion_diag_positions(self) -> list[int]:
        return self._vector_positions_for_kind(SiteKind.DIFFUSION_DIAG)

    @property
    def diffusion_diag_index(self) -> dict[int, int]:
        return self._vector_index_for_kind(SiteKind.DIFFUSION_DIAG)

    @property
    def n_diffusion_diag(self) -> int:
        return self._count_for_kind(SiteKind.DIFFUSION_DIAG)

    @property
    def diffusion_lower_positions(self) -> list[tuple[int, int]]:
        return self._matrix_positions_for_kind(SiteKind.DIFFUSION_LOWER)

    @property
    def diffusion_lower_index(self) -> dict[tuple[int, int], int]:
        return self._matrix_index_for_kind(SiteKind.DIFFUSION_LOWER)

    @property
    def n_diffusion_lower(self) -> int:
        return self._count_for_kind(SiteKind.DIFFUSION_LOWER)

    @property
    def lambda_free_positions(self) -> list[tuple[int, int]]:
        return self._matrix_positions_for_kind(SiteKind.LOADING)

    @property
    def lambda_free_index(self) -> dict[tuple[int, int], int]:
        return self._matrix_index_for_kind(SiteKind.LOADING)

    @property
    def n_lambda_free(self) -> int:
        return self._count_for_kind(SiteKind.LOADING)

    @property
    def manifest_means_free_positions(self) -> list[int]:
        return self._vector_positions_for_kind(SiteKind.MANIFEST_MEANS)

    @property
    def manifest_means_free_index(self) -> dict[int, int]:
        return self._vector_index_for_kind(SiteKind.MANIFEST_MEANS)

    @property
    def n_manifest_means(self) -> int:
        return self._count_for_kind(SiteKind.MANIFEST_MEANS)

    @property
    def manifest_var_free_positions(self) -> list[int]:
        return self._vector_positions_for_kind(SiteKind.MANIFEST_VAR_DIAG)

    @property
    def manifest_var_free_index(self) -> dict[int, int]:
        return self._vector_index_for_kind(SiteKind.MANIFEST_VAR_DIAG)

    @property
    def n_manifest_var_diag(self) -> int:
        return self._count_for_kind(SiteKind.MANIFEST_VAR_DIAG)

    @property
    def t0_means_free_positions(self) -> list[int]:
        return self._vector_positions_for_kind(SiteKind.T0_MEANS)

    @property
    def t0_means_free_index(self) -> dict[int, int]:
        return self._vector_index_for_kind(SiteKind.T0_MEANS)

    @property
    def n_t0_means(self) -> int:
        return self._count_for_kind(SiteKind.T0_MEANS)

    @property
    def t0_diag_free_positions(self) -> list[int]:
        return self._vector_positions_for_kind(SiteKind.T0_VAR_DIAG)

    @property
    def t0_diag_free_index(self) -> dict[int, int]:
        return self._vector_index_for_kind(SiteKind.T0_VAR_DIAG)

    @property
    def n_t0_diag(self) -> int:
        return self._count_for_kind(SiteKind.T0_VAR_DIAG)

    @property
    def t0_correlation_positions(self) -> list[tuple[int, int]]:
        return self._matrix_positions_for_kind(SiteKind.T0_VAR_LOWER)

    @property
    def t0_correlation_index(self) -> dict[tuple[int, int], int]:
        return self._matrix_index_for_kind(SiteKind.T0_VAR_LOWER)

    @property
    def n_t0_correlation(self) -> int:
        return self._count_for_kind(SiteKind.T0_VAR_LOWER)
