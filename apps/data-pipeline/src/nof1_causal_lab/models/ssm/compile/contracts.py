"""Strict persisted contracts for compiled state-space model artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.prior import PriorValidationResult  # noqa: TC001
from nof1_causal_lab.artifacts.statistical_model_spec import (  # noqa: TC001
    DistributionFamily,
    LinkFunction,
)
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.structure.sites import (  # noqa: TC001
    PriorAuthoringTransform,
    SiteKind,
    SupportClass,
    TransformKind,
)

type SerializedNumeric = int | float | list[SerializedNumeric]


class PersistedModel(BaseModel):
    """Base configuration for immutable, versioned persisted contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SerializedSSMSpec(PersistedModel):
    """JSON representation of the structural ``SSMSpec`` runtime contract."""

    n_latent: int = Field(ge=0)
    n_manifest: int = Field(ge=0)
    dynamics_spec: JsonObject
    diffusion_block: JsonObject
    lambda_block: JsonObject
    manifest_means_block: JsonObject
    manifest_chol_block: JsonObject
    t0_means_block: JsonObject
    t0_chol_block: JsonObject
    input_effect_block: JsonObject
    static_state_sd_block: JsonObject
    static_factor_loadings: list[list[float]]
    diffusion_dists: list[DistributionFamily]
    manifest_dists: list[DistributionFamily]
    manifest_level_counts: list[int] | None = None
    manifest_links: list[LinkFunction] | None = None
    manifest_standardized: list[bool] | None = None
    manifest_cat_anchor: list[bool] | None = None
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None
    input_names: list[str] | None = None
    input_source_indicators: list[str] | None = None
    input_scales: list[float] | None = None
    input_missing_policies: list[Literal["zero", "forward_fill"]] | None = None
    input_lagged: list[bool]
    static_factor_names: list[str] | None = None


class SerializedEdgeLag(PersistedModel):
    """One directed continuous-time lag attached to a compiled edge."""

    source_id: str
    effect_idx: int = Field(ge=0)
    cause_idx: int = Field(ge=0)
    lag_days: float = Field(gt=0)


class SerializedSiteDescriptor(PersistedModel):
    """Persisted topology for one runtime sample site."""

    name: str
    shape: list[int]
    support: SupportClass
    assembly_group: str
    site_kind: SiteKind
    transform_kind: TransformKind
    deterministic_name: str | None = None
    fixed_spec_field: str | None = None
    priors_field: str | None = None
    runtime_prior_key: str | None = None
    is_runtime_prior_controlled: bool


class CompiledPriorSite(PersistedModel):
    """Serialized runtime prior parameters for one compiled sample site."""

    family: SerializedNumeric
    loc: SerializedNumeric | None = None
    scale: SerializedNumeric | None = None
    low: SerializedNumeric | None = None
    high: SerializedNumeric | None = None
    concentration: SerializedNumeric | None = None
    rate: SerializedNumeric | None = None
    value: SerializedNumeric | None = None


class CompiledPriorSemantics(PersistedModel):
    """Versioned runtime site registry and its compiled prior state."""

    schema_version: Literal[5]
    site_registry: list[SerializedSiteDescriptor]
    prior_state: dict[str, CompiledPriorSite]


class CompiledParameterBinding(PersistedModel):
    """Semantic parameter-to-runtime-site binding."""

    parameter: str
    site_name: str
    prior_field: str | None
    flat_index: int = Field(ge=0)
    site_kind: SiteKind
    transform: PriorAuthoringTransform
    construct_names: list[str]
    indicator_names: list[str]
    component_index: int | None
    effect_idx: int | None
    cause_idx: int | None


class CompiledStructuralBinding(PersistedModel):
    """Stable structural-plan source identity bound to one runtime target."""

    source_id: str
    source_kind: Literal[
        "state",
        "manifest",
        "known_input",
        "edge",
        "induced_dependency",
    ]
    target_kind: Literal[
        "latent_state",
        "manifest_channel",
        "transition_input",
        "dynamics_edge",
        "input_effect",
        "diffusion_correlation",
        "static_factor",
    ]
    target_indices: tuple[int, ...]
    target_name: str


class AnchorCertificate(PersistedModel):
    """Compiler proof that one retained latent has location and scale anchors."""

    construct_id: str
    construct_name: str
    location_anchor: Literal[
        "standardized_manifest",
        "fixed_dynamics_center",
        "fixed_initial_mean",
    ]
    location_source_id: str | None = None
    scale_anchor: Literal["fixed_manifest_loading", "categorical_slope_pin"]
    scale_source_id: str


class CompiledStructure(PersistedModel):
    """Executable structure plus total provenance back to StructuralPlan."""

    spec: SerializedSSMSpec
    edge_lag_days: list[SerializedEdgeLag]
    bindings: list[CompiledStructuralBinding]
    anchor_certificates: list[AnchorCertificate]


class CompiledSSMArtifact(PersistedModel):
    """Complete versioned artifact required to restore an executable SSM."""

    schema_version: Literal[2]
    structure: CompiledStructure
    compiled_prior_semantics: CompiledPriorSemantics
    parameter_bindings: list[CompiledParameterBinding]
    compile_diagnostics: list[PriorValidationResult]

    @property
    def spec(self) -> SerializedSSMSpec:
        """Runtime-facing compiled spec."""
        return self.structure.spec

    @property
    def edge_lag_days(self) -> list[SerializedEdgeLag]:
        """Runtime-facing edge lag metadata."""
        return self.structure.edge_lag_days
