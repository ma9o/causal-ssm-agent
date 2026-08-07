"""Application projection between executable priors and evidence-rich proposals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.prior import ExecutablePrior, PriorPlan
from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.models.ssm.compile.artifact import resolve_executable_priors
from nof1_causal_lab.workers.schemas_prior import PriorProposal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.compile.contracts import CompiledSSMArtifact

type PriorPayload = UncheckedJsonObject


def _executable_prior(proposal: PriorProposal) -> ExecutablePrior:
    return ExecutablePrior(
        parameter=proposal.parameter,
        distribution=proposal.distribution,
        params=proposal.params,
        reference_interval_days=proposal.reference_interval_days,
    )


def _compiler_reasoning(parameter: str) -> str:
    if parameter.startswith("t0_mean_"):
        latent = parameter.removeprefix("t0_mean_").replace("_", " ")
        return f"Default weakly informative prior for the initial state mean of {latent}."
    if parameter.startswith("t0_sd_"):
        latent = parameter.removeprefix("t0_sd_").replace("_", " ")
        return (
            "Default weakly informative prior for the initial state standard deviation "
            f"of {latent}."
        )
    return f"Compiler-resolved prior for {parameter}."


def resolve_prior_proposals(
    compiled_ssm: CompiledSSMArtifact,
    *,
    authored_priors: Mapping[str, PriorPayload],
) -> list[PriorPayload]:
    """Combine compiler-owned prior membership with authored evidence metadata."""
    authored_by_parameter = {
        parameter: PriorProposal.model_validate({**payload, "parameter": parameter})
        for parameter, payload in authored_priors.items()
    }
    authored_plan = PriorPlan(
        priors={
            parameter: _executable_prior(proposal)
            for parameter, proposal in authored_by_parameter.items()
        }
    )
    resolved = resolve_executable_priors(compiled_ssm, authored_plan=authored_plan)

    rows: list[PriorPayload] = []
    for prior in resolved:
        authored = authored_by_parameter.get(prior.parameter)
        if authored is not None:
            rows.append(authored.model_dump(mode="json"))
            continue
        rows.append(
            PriorProposal(
                parameter=prior.parameter,
                distribution=prior.distribution,
                params=prior.params,
                sources=[],
                reasoning=_compiler_reasoning(prior.parameter),
                reference_interval_days=prior.reference_interval_days,
            ).model_dump(mode="json")
        )
    return rows


__all__ = ["resolve_prior_proposals"]
