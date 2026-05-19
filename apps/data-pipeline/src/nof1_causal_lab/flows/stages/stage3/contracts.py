"""Stage 3 contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.flows.contracts_base import BaseStageContract

IS_INTERACTIVE_STAGE = False


class ValidationIssueContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str | None = None
    issue_type: str
    severity: Literal["error", "warning", "info"]
    message: str


class IndicatorEmpiricalProfileContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_dtype: str | None = None
    n_obs: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    variance: float | None
    time_coverage_ratio: float | None
    max_gap_ratio: float | None
    dtype_violations: int | None = None
    duplicate_pct: float | None = None
    arithmetic_sequence_detected: bool
    n_unparseable_timestamps: int | None = None
    zero_fraction: float | None = None
    is_nonnegative: bool | None = None
    is_unit_interval: bool | None = None
    looks_integer_valued: bool | None = None
    variance_to_mean_ratio: float | None = None


class IndicatorValidationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssueContract]
    checks: dict[str, Literal["ok", "warning", "error"]]


class IndicatorAuditContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: IndicatorEmpiricalProfileContract | None = None
    validation: IndicatorValidationContract


class Stage3Contract(BaseStageContract):
    is_valid: bool
    indicators: dict[str, IndicatorAuditContract]
    dataset_issues: list[ValidationIssueContract]

    def summary_message(self) -> str:
        indicator_issues = [
            issue for audit in self.indicators.values() for issue in audit.validation.issues
        ]
        all_issues = [*indicator_issues, *self.dataset_issues]
        errors = sum(1 for issue in all_issues if issue.severity == "error")
        warnings = sum(1 for issue in all_issues if issue.severity == "warning")
        return (
            f"Stage 3 summary: is_valid={self.is_valid} "
            f"issues={len(all_issues)} "
            f"errors={errors} warnings={warnings} outcome={self.outcome}"
        )
