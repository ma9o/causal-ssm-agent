"""Context adapters for generic Temporal LLM subroutines."""

from __future__ import annotations

from nof1_causal_lab.machine.temporal.llm_subroutine_storage import read_subroutine_json
from nof1_causal_lab.machine.temporal.messages import LLMSubroutineContextKind, LLMToolSpec


def subroutine_context_messages(
    context_kind: LLMSubroutineContextKind,
    context_ref: str,
) -> tuple[str | None, list[str], list[LLMToolSpec]]:
    if context_kind == "measurement_extraction":
        from nof1_causal_lab.workers.messages import WorkerMessages

        spec = read_subroutine_json(context_ref)
        messages = WorkerMessages(
            question=spec["question"],
            measurement_structure=spec["measurement_structure"],
            window_text=spec["window_text"],
            n_windows=len(spec["window_starts"]),
        ).extraction_messages()
        system_prompt = None
        user_messages: list[str] = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_messages.append(message["content"])
        return (
            system_prompt,
            user_messages,
            [
                LLMToolSpec(
                    name="validate_extractions",
                    description="Validate worker extraction output JSON.",
                    param_name="output_json",
                    param_description="The JSON string containing the worker output.",
                )
            ],
        )

    if context_kind == "latent_structure":
        context = read_subroutine_json(context_ref)
        return (
            context["system_prompt"],
            list(context["user_messages"]),
            [
                LLMToolSpec(
                    name="validate_latent_structure",
                    description="Validate latent structure JSON.",
                    param_name="structure_json",
                    param_description="The JSON string containing the latent structure.",
                )
            ],
        )

    if context_kind == "measurement_structure":
        context = read_subroutine_json(context_ref)
        return (
            context["system_prompt"],
            list(context["user_messages"]),
            [
                LLMToolSpec(
                    name="validate_measurement_structure",
                    description=(
                        "Validate measurement structure, known-input declarations, "
                        "and compiler constraints."
                    ),
                    param_name="measurement_json",
                    param_description=(
                        "The JSON string containing the measurement structure and "
                        "known-input declarations."
                    ),
                )
            ],
        )

    if context_kind == "analysis_commentary":
        context = read_subroutine_json(context_ref)
        return context["system_prompt"], list(context["user_messages"]), []

    if context_kind == "raw_data_ingestion":
        from nof1_causal_lab.flows.transitions.ingestion.flow import SYSTEM_PROMPT, USER_PROMPT

        return (
            SYSTEM_PROMPT,
            [USER_PROMPT],
            [
                LLMToolSpec(
                    name="list_files",
                    description="List files in the prepared input directory.",
                    kind="read_only",
                    executor="raw_data_list_files",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within the input directory.",
                                "default": ".",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                ),
                LLMToolSpec(
                    name="read_file_sample",
                    description="Read a sample of lines from a file to understand its format.",
                    kind="read_only",
                    executor="raw_data_read_file_sample",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file within the input directory.",
                            },
                            "n_lines": {
                                "type": "integer",
                                "description": "Number of lines to read.",
                                "default": 50,
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                LLMToolSpec(
                    name="execute_python",
                    description="Execute Python code in the local pipeline process to parse files into a Polars DataFrame.",
                    kind="checkpoint",
                    executor="raw_data_execute_python",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to execute. Assign the final Polars DataFrame "
                                    "to result_df."
                                ),
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                ),
                LLMToolSpec(
                    name="submit_table",
                    description="Validate and finalize the ingested DataFrame with column descriptions.",
                    param_name="column_descriptions_json",
                    param_description="JSON object mapping column names to descriptions.",
                    kind="terminal",
                    executor="raw_data_submit_table",
                ),
            ],
        )

    if context_kind == "model_spec_construct":
        context = read_subroutine_json(context_ref)
        tools = [
            LLMToolSpec(
                name="submit_construct",
                description=(
                    "Submit one construct: its indicator emission choices and priors keyed "
                    "by canonical parameter name. The cumulative model is compiled and "
                    "gated by the exact prior-predictive reachability battery."
                ),
                kind="terminal",
                executor="model_spec_submit_construct",
                success_output=None,
                parameters_schema=context["submit_construct_schema"],
            )
        ]
        if context.get("enable_literature"):
            tools.append(
                LLMToolSpec(
                    name="search_literature",
                    description="Search for empirical literature about effect sizes for model parameters.",
                    kind="read_only",
                    executor="model_spec_search_literature",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Search query for empirical literature about effect sizes."
                                ),
                            },
                            "parameter_name": {
                                "type": "string",
                                "description": (
                                    "Name of the parameter this search is for "
                                    "(e.g. 'beta_stress_sleep')."
                                ),
                            },
                        },
                        "required": ["query", "parameter_name"],
                        "additionalProperties": False,
                    },
                )
            )
        return context["system_prompt"], list(context["user_messages"]), tools

    raise ValueError(f"unknown LLM subroutine context kind {context_kind!r}")
