"""Shared LLM utilities for multi-turn generation."""

import json
from typing import TYPE_CHECKING

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
)
from inspect_ai.tool import Tool, tool

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage


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
def validate_dsem_structure():
    """Tool for validating DSEM structure JSON."""

    async def execute(structure_json: str) -> str:
        """
        Validate a DSEM structure and return all validation errors.

        Args:
            structure_json: The JSON string containing the DSEM structure to validate.

        Returns:
            "VALID" if the structure passes validation, otherwise a list of all errors found.
        """
        from causal_agent.orchestrator.schemas import validate_structure

        # Parse JSON first
        try:
            data = json.loads(structure_json)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {e}"

        # Validate and collect all errors
        structure, errors = validate_structure(data)

        if not errors:
            return "VALID"

        return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    return execute


def make_validate_worker_output_tool(schema: dict) -> Tool:
    """Create a validation tool for worker output, bound to a specific schema.

    Args:
        schema: The DSEM schema dict to validate extractions against

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
            from causal_agent.workers.schemas import validate_worker_output

            # Parse JSON first
            try:
                data = json.loads(output_json)
            except json.JSONDecodeError as e:
                return f"JSON parse error: {e}"

            # Validate and collect all errors
            output, errors = validate_worker_output(data, schema)

            if not errors:
                return "VALID"

            return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

        return execute

    return validate_extractions()


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
    config: GenerateConfig | None = None,
) -> str:
    """
    Run a multi-turn conversation with optional tool use.

    Args:
        messages: Initial messages (typically system + user prompt)
        model: The model to use for generation
        follow_ups: List of follow-up user prompts to send after each response (default: none)
        tools: Optional list of tools the model can use (enables generate_loop)
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

        # Follow-up turns with tools
        for prompt in follow_ups:
            messages.append(ChatMessageUser(content=prompt))
            final_messages, output = await model.generate_loop(
                messages,
                tools=tools,
                config=config,
            )
            messages = list(final_messages)

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
