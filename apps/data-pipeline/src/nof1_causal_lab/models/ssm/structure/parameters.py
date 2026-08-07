"""Declarative free-or-fixed parameter slots for compiled SSM structure."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Free:
    """A parameter slot inferred from its registered sample-site prior."""


@dataclass(frozen=True)
class Fixed:
    """A parameter slot held at one finite structural value."""

    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError(f"Fixed parameter value must be finite; got {self.value}.")


type ParameterSlot = Fixed | Free


__all__ = ["Fixed", "Free", "ParameterSlot"]
