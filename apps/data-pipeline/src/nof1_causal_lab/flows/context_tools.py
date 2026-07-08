"""Aggregate context-owned tool metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.flows.transitions.analysis.contracts import (
    ANALYSIS_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.transitions.analysis.contracts import (
    IS_INTERACTIVE_CONTEXT as ANALYSIS_INTERACTIVE,
)
from nof1_causal_lab.flows.transitions.extraction.contracts import (
    EXTRACTION_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.transitions.extraction.contracts import (
    IS_INTERACTIVE_CONTEXT as EXTRACTION_INTERACTIVE,
)
from nof1_causal_lab.flows.transitions.ingestion.contracts import (
    INGESTION_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.transitions.ingestion.contracts import (
    IS_INTERACTIVE_CONTEXT as INGESTION_INTERACTIVE,
)
from nof1_causal_lab.flows.transitions.latent_structure.contracts import (
    IS_INTERACTIVE_CONTEXT as LATENT_STRUCTURE_INTERACTIVE,
)
from nof1_causal_lab.flows.transitions.latent_structure.contracts import (
    LATENT_STRUCTURE_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
    IS_INTERACTIVE_CONTEXT as MEASUREMENT_STRUCTURE_INTERACTIVE,
)
from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
    MEASUREMENT_STRUCTURE_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.transitions.model_spec.contracts import (
    IS_INTERACTIVE_CONTEXT as MODEL_SPEC_INTERACTIVE,
)
from nof1_causal_lab.flows.transitions.model_spec.contracts import (
    MODEL_SPEC_TOOL_CONTRACTS,
)

if TYPE_CHECKING:
    from nof1_causal_lab.flows.contracts_base import ToolContract

CONTEXT_TOOLS: dict[str, list[ToolContract]] = {
    "ingestion": INGESTION_TOOL_CONTRACTS,
    "latent-structure": LATENT_STRUCTURE_TOOL_CONTRACTS,
    "measurement-structure": MEASUREMENT_STRUCTURE_TOOL_CONTRACTS,
    "measurement": EXTRACTION_TOOL_CONTRACTS,
    "statistical-model-spec": MODEL_SPEC_TOOL_CONTRACTS,
    "ranking": ANALYSIS_TOOL_CONTRACTS,
}

INTERACTIVE_CONTEXTS: frozenset[str] = frozenset(
    context_id
    for context_id, is_interactive in (
        ("ingestion", INGESTION_INTERACTIVE),
        ("latent-structure", LATENT_STRUCTURE_INTERACTIVE),
        ("measurement-structure", MEASUREMENT_STRUCTURE_INTERACTIVE),
        ("measurement", EXTRACTION_INTERACTIVE),
        ("statistical-model-spec", MODEL_SPEC_INTERACTIVE),
        ("ranking", ANALYSIS_INTERACTIVE),
    )
    if is_interactive
)
