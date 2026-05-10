"""Private validation helpers shared across ACML model layers."""

from __future__ import annotations


def validated_sequence(
    value,
    *,
    allowed_types: tuple[type, ...],
    field_name: str,
):
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field_name} must be a sequence")
    normalized = tuple(value)
    for item in normalized:
        if not isinstance(item, allowed_types):
            allowed = ", ".join(type_.__name__ for type_ in allowed_types)
            raise ValueError(f"{field_name} items must be {allowed}")
    return normalized
