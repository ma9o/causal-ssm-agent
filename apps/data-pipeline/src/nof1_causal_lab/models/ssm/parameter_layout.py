"""Parameter layout for block-based SSM specs.

``SSMParameterLayout`` is a thin index over the block-owned sample-site
descriptors produced by ``SSMSpec.iter_sample_sites()``. It does not
recompute positions from masks (the blocks already do that). It does
not own templates and it does not assemble matrices. Assembly belongs
to the block that owns the parameter.

The named position/index/count accessors are derived lookups against
``by_name``; they exist to keep compile-time consumers' call sites
narrow, not to encode the topology a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.structure.sites import SiteKind

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
    # Generic descriptor lookups
    # ------------------------------------------------------------------

    def _positions(self, site_name: str) -> list:
        site = self.by_name.get(site_name)
        return list(site.positions) if site is not None else []

    def _count(self, site_name: str) -> int:
        site = self.by_name.get(site_name)
        return len(site.positions) if site is not None else 0

    def _index(self, site_name: str) -> dict:
        site = self.by_name.get(site_name)
        if site is None:
            return {}
        return {position: flat_idx for flat_idx, position in enumerate(site.positions)}

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

    def _positions_for_kind(self, site_kind: SiteKind) -> list:
        site = self.site_by_kind(site_kind)
        return list(site.positions) if site is not None else []

    def _count_for_kind(self, site_kind: SiteKind) -> int:
        site = self.site_by_kind(site_kind)
        return len(site.positions) if site is not None else 0

    def _index_for_kind(self, site_kind: SiteKind) -> dict:
        site = self.site_by_kind(site_kind)
        if site is None:
            return {}
        return {position: flat_idx for flat_idx, position in enumerate(site.positions)}

    # ------------------------------------------------------------------
    # Named positions / index / count accessors (derived)
    # ------------------------------------------------------------------

    @property
    def drift_base_decay_positions(self) -> list[int]:
        return self._positions_for_kind(SiteKind.DRIFT_BASE_DECAY)

    @property
    def drift_base_decay_index(self) -> dict[int, int]:
        return self._index_for_kind(SiteKind.DRIFT_BASE_DECAY)

    @property
    def n_drift_base_decay(self) -> int:
        return self._count_for_kind(SiteKind.DRIFT_BASE_DECAY)

    @property
    def offdiag_positions(self) -> list[tuple[int, int]]:
        return self._positions_for_kind(SiteKind.DRIFT_OFFDIAG)

    @property
    def offdiag_index(self) -> dict[tuple[int, int], int]:
        return self._index_for_kind(SiteKind.DRIFT_OFFDIAG)

    @property
    def n_drift_offdiag(self) -> int:
        return self._count_for_kind(SiteKind.DRIFT_OFFDIAG)

    @property
    def cint_free_positions(self) -> list[int]:
        return self._positions_for_kind(SiteKind.CINT)

    @property
    def cint_free_index(self) -> dict[int, int]:
        return self._index_for_kind(SiteKind.CINT)

    @property
    def n_cint(self) -> int:
        return self._count_for_kind(SiteKind.CINT)

    @property
    def static_state_sd_free_positions(self) -> list[int]:
        return self._positions("static_state_sd_free")

    @property
    def static_state_sd_free_index(self) -> dict[int, int]:
        return self._index("static_state_sd_free")

    @property
    def n_static_state_sd(self) -> int:
        return self._count("static_state_sd_free")

    @property
    def input_effect_positions(self) -> list[tuple[int, int]]:
        return self._positions("input_effect_free")

    @property
    def input_effect_index(self) -> dict[tuple[int, int], int]:
        return self._index("input_effect_free")

    @property
    def n_input_effect(self) -> int:
        return self._count("input_effect_free")

    @property
    def diffusion_diag_positions(self) -> list[int]:
        return self._positions("diffusion_diag_free")

    @property
    def diffusion_diag_index(self) -> dict[int, int]:
        return self._index("diffusion_diag_free")

    @property
    def n_diffusion_diag(self) -> int:
        return self._count("diffusion_diag_free")

    @property
    def diffusion_lower_positions(self) -> list[tuple[int, int]]:
        return self._positions("diffusion_lower_free")

    @property
    def diffusion_lower_index(self) -> dict[tuple[int, int], int]:
        return self._index("diffusion_lower_free")

    @property
    def n_diffusion_lower(self) -> int:
        return self._count("diffusion_lower_free")

    @property
    def lambda_free_positions(self) -> list[tuple[int, int]]:
        return self._positions("lambda_free")

    @property
    def lambda_free_index(self) -> dict[tuple[int, int], int]:
        return self._index("lambda_free")

    @property
    def n_lambda_free(self) -> int:
        return self._count("lambda_free")

    @property
    def manifest_means_free_positions(self) -> list[int]:
        return self._positions("manifest_means_free")

    @property
    def manifest_means_free_index(self) -> dict[int, int]:
        return self._index("manifest_means_free")

    @property
    def n_manifest_means(self) -> int:
        return self._count("manifest_means_free")

    @property
    def manifest_var_free_positions(self) -> list[int]:
        return self._positions("manifest_var_diag_free")

    @property
    def manifest_var_free_index(self) -> dict[int, int]:
        return self._index("manifest_var_diag_free")

    @property
    def n_manifest_var_diag(self) -> int:
        return self._count("manifest_var_diag_free")

    @property
    def t0_means_free_positions(self) -> list[int]:
        return self._positions("t0_means_free")

    @property
    def t0_means_free_index(self) -> dict[int, int]:
        return self._index("t0_means_free")

    @property
    def n_t0_means(self) -> int:
        return self._count("t0_means_free")

    @property
    def t0_diag_free_positions(self) -> list[int]:
        return self._positions("t0_var_diag_free")

    @property
    def t0_diag_free_index(self) -> dict[int, int]:
        return self._index("t0_var_diag_free")

    @property
    def n_t0_diag(self) -> int:
        return self._count("t0_var_diag_free")

    @property
    def t0_correlation_positions(self) -> list[tuple[int, int]]:
        return self._positions("t0_var_lower_free")

    @property
    def t0_correlation_index(self) -> dict[tuple[int, int], int]:
        return self._index("t0_var_lower_free")

    @property
    def n_t0_correlation(self) -> int:
        return self._count("t0_var_lower_free")
