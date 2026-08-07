"""Named types for JSON-safe values and explicitly unchecked boundaries."""

from __future__ import annotations

from typing import Any

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

# Deliberately unchecked JSON-shaped data at parsing and library boundaries.
# CUSTOM004 makes this the visible, searchable escape hatch instead of allowing
# anonymous ``dict[str, Any]`` annotations throughout the codebase.
type UncheckedJsonObject = dict[str, Any]
