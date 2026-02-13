"""Shared LLM utilities for multi-turn generation."""

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
)
from inspect_ai.tool import Tool, tool

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

    from causal_ssm_agent.orchestrator.schemas import LatentModel


# Type aliases for generate functions
OrchestratorGenerateFn = Callable[
    [list, list | None, list[str] | None],  # (messages, tools, follow_ups)
    Awaitable[str],
]
WorkerGenerateFn = Callable[
    [list, list | None],  # (messages, tools)
    Awaitable[str],
]


def get_generate_config() -> GenerateConfig:
    """Get standard GenerateConfig for all model calls.

    Reads settings from config.yaml llm section.
    """
    from causal_ssm_agent.utils.config import get_config

    llm = get_config().llm
    return GenerateConfig(
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=llm.reasoning_effort,
        reasoning_tokens=llm.reasoning_tokens,
        reasoning_history="all",  # Preserve reasoning across tool calls (required by Gemini)
    )


def dict_messages_to_chat(messages: list[dict]) -> list["ChatMessage"]:
    """Convert dict messages to ChatMessage objects.

    Args:
        messages: List of dicts with 'role' and 'content' keys

    Returns:
        List of ChatMessage objects (ChatMessageSystem or ChatMessageUser)
    """
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            chat_messages.append(ChatMessageSystem(content=msg["content"]))
        elif msg["role"] == "user":
            chat_messages.append(ChatMessageUser(content=msg["content"]))
    return chat_messages


def make_orchestrator_generate_fn(
    model: Model, config: GenerateConfig | None = None
) -> OrchestratorGenerateFn:
    """Create a generate function for orchestrator stages (1a, 1b).

    The returned function has signature: (messages, tools, follow_ups) -> str

    Args:
        model: The model to use for generation
        config: Optional generation config (uses get_generate_config() if None)

    Returns:
        An async function that handles multi-turn generation with tools and follow-ups
    """
    if config is None:
        config = get_generate_config()

    async def generate(messages: list, tools: list | None, follow_ups: list[str] | None) -> str:
        chat_messages = dict_messages_to_chat(messages)

        if follow_ups or tools:
            return await multi_turn_generate(
                messages=chat_messages,
                model=model,
                follow_ups=follow_ups,
                tools=tools or [],
                config=config,
            )
        else:
            response = await model.generate(chat_messages, config=config)
            return response.completion

    return generate


def make_worker_generate_fn(model: Model, config: GenerateConfig | None = None) -> WorkerGenerateFn:
    """Create a generate function for worker extraction.

    The returned function has signature: (messages, tools) -> str

    Args:
        model: The model to use for generation
        config: Optional generation config (uses get_generate_config() if None)

    Returns:
        An async function that handles multi-turn generation with tools
    """
    if config is None:
        config = get_generate_config()

    async def generate(messages: list, tools: list | None) -> str:
        chat_messages = dict_messages_to_chat(messages)
        return await multi_turn_generate(
            messages=chat_messages,
            model=model,
            tools=tools or [],
            config=config,
        )

    return generate


def parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse model response as JSON: {e}") from e


@tool
def validate_latent_model_tool():
    """Tool for validating latent model JSON (Stage 1a)."""

    async def execute(structure_json: str) -> str:
        """
        Validate a latent model and return all validation errors.

        Args:
            structure_json: The JSON string containing the latent model to validate.

        Returns:
            "VALID" if the structure passes validation, otherwise a list of all errors found.
        """
        from causal_ssm_agent.orchestrator.schemas import validate_latent_model

        try:
            data = json.loads(structure_json)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {e}"

        _structure, errors = validate_latent_model(data)

        if not errors:
            return "VALID"

        return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    return execute


def make_validate_measurement_model_tool(latent_model: "LatentModel") -> Tool:
    """Create a validation tool for measurement model, bound to a latent model.

    Args:
        latent_model: The latent model to validate against

    Returns:
        A tool function that validates measurement model JSON
    """

    @tool
    def validate_measurement_model_tool():
        """Tool for validating measurement model JSON (Stage 1b)."""

        async def execute(measurement_json: str) -> str:
            """
            Validate a measurement model and return all validation errors.

            Args:
                measurement_json: The JSON string containing the measurement model to validate.

            Returns:
                "VALID" if the model passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.orchestrator.schemas import validate_measurement_model

            try:
                data = json.loads(measurement_json)
            except json.JSONDecodeError as e:
                return f"JSON parse error: {e}"

            _model, errors = validate_measurement_model(data, latent_model)

            if not errors:
                return "VALID"

            return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

        return execute

    return validate_measurement_model_tool()


def make_validate_model_spec_tool(causal_spec: dict) -> tuple[Tool, dict]:
    """Create a validation tool for model spec, bound to a causal spec.

    The tool captures the last valid ModelSpec so the caller can retrieve it
    directly without needing the LLM to re-output the JSON.

    Args:
        causal_spec: The full CausalSpec dict (to extract indicators for dtype checking)

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["spec"] for the last validated ModelSpec (or None).
    """
    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    capture: dict = {}

    @tool
    def validate_model_spec_tool():
        """Tool for validating model specification JSON (Stage 4)."""

        async def execute(model_spec_json: str) -> str:
            """
            Validate a model specification and return all validation errors.

            Args:
                model_spec_json: The JSON string containing the model spec to validate.

            Returns:
                "VALID" if the spec passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict

            try:
                data = json.loads(model_spec_json)
            except json.JSONDecodeError as e:
                return f"JSON parse error: {e}"

            spec, errors = validate_model_spec_dict(data, indicators=indicators or None)

            if not errors:
                capture["spec"] = spec
                return "VALID"

            return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

        return execute

    return validate_model_spec_tool(), capture


def make_worker_tools(schema: dict) -> list[Tool]:
    """Create the standard toolset for worker agents.

    This is the single source of truth for worker tools.
    Used by both production workers and evals.

    Args:
        schema: The model schema dict to validate extractions against

    Returns:
        List of tools: [validate_extractions, parse_date, calculate]
    """
    return [
        make_validate_worker_output_tool(schema),
        parse_date(),
        calculate(),
    ]


def make_validate_worker_output_tool(schema: dict) -> Tool:
    """Create a validation tool for worker output, bound to a specific schema.

    Args:
        schema: The model schema dict to validate extractions against

    Returns:
        A tool function that validates worker output JSON
    """

    @tool
    def validate_extractions():
        """Tool for validating worker extraction output JSON."""

        async def execute(output_json: str) -> str:
            """
            Validate worker extractions and return all validation errors.

            Args:
                output_json: The JSON string containing the worker output to validate.

            Returns:
                "VALID" if the output passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.workers.schemas import validate_worker_output

            # Parse JSON first
            try:
                data = json.loads(output_json)
            except json.JSONDecodeError as e:
                return f"JSON parse error: {e}"

            # Validate and collect all errors
            _output, errors = validate_worker_output(data, schema)

            if not errors:
                return "VALID"

            return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

        return execute

    return validate_extractions()


@tool
def calculate():
    """Tool for evaluating simple arithmetic calculations."""

    async def execute(expression: str) -> str:
        """
        Evaluate a simple arithmetic expression.

        Args:
            expression: A Python arithmetic expression (e.g., "2 + 3 * 4", "100 / 5", "(10 + 5) * 2", "10 % 3", "2 ** 8")

        Returns:
            The result of the calculation, or an error message if evaluation fails.
        """
        # Whitelist of allowed characters for safe evaluation
        allowed_chars = set("0123456789+-*/%()._ ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and +-*/%()._ are allowed."

        try:
            result = eval(expression)
            return str(result)
        except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
            return f"Error evaluating expression: {e}"

    return execute


@tool
def parse_date():
    """Tool for parsing dates into a human-readable spelled out format."""

    async def execute(date_string: str) -> str:
        """
        Parse a date or timestamp into spelled out format.

        Args:
            date_string: A date or timestamp string (e.g., "2024-03-15", "2024-03-15T10:30:00")

        Returns:
            Spelled out date (e.g., "Friday, March 15, 2024") or an error message if parsing fails.
        """
        from datetime import datetime

        # Common formats to try
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_string.strip(), fmt)
                return dt.strftime("%A, %B %d, %Y")  # e.g., "Friday, March 15, 2024"
            except ValueError:
                continue

        return f"Could not parse date: {date_string}"

    return execute


async def multi_turn_generate(
    messages: list["ChatMessage"],
    model: Model,
    follow_ups: list[str] | None = None,
    tools: list[Tool] | None = None,
    follow_up_tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
) -> str:
    """
    Run a multi-turn conversation with optional tool use.

    Args:
        messages: Initial messages (typically system + user prompt)
        model: The model to use for generation
        follow_ups: List of follow-up user prompts to send after each response (default: none)
        tools: Optional list of tools the model can use on the first turn (enables generate_loop)
        follow_up_tools: Optional list of tools for follow-up (self-review) turns.
            Defaults to None, meaning follow-up turns use no tools (plain generation).
            This prevents the LLM from re-invoking validation tools during self-review
            and potentially overwriting a previously captured valid model.
        config: Optional generation config

    Returns:
        The final completion string
    """
    messages = list(messages)  # Don't mutate original
    follow_ups = follow_ups or []

    if tools:
        # Use generate_loop for tool-enabled generation
        final_messages, output = await model.generate_loop(
            messages,
            tools=tools,
            config=config,
        )
        messages = list(final_messages)

        # Follow-up turns: use follow_up_tools if provided, otherwise no tools
        for prompt in follow_ups:
            messages.append(ChatMessageUser(content=prompt))
            if follow_up_tools:
                final_messages, output = await model.generate_loop(
                    messages,
                    tools=follow_up_tools,
                    config=config,
                )
                messages = list(final_messages)
            else:
                response = await model.generate(messages, config=config)
                messages.append(ChatMessageAssistant(content=response.completion))
                output = response

        return output.completion
    else:
        # Simple generation without tools
        response = await model.generate(messages, config=config)
        messages.append(ChatMessageAssistant(content=response.completion))

        for prompt in follow_ups:
            messages.append(ChatMessageUser(content=prompt))
            response = await model.generate(messages, config=config)
            messages.append(ChatMessageAssistant(content=response.completion))

        return response.completion
