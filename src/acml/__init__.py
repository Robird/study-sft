"""Stable root API for reading and writing ACML text.

Model and semantic dataclasses live in ``acml.model`` and
``acml.semantic_model`` so the package root can stay small and intentional.
"""

from acml.errors import ACMLError, ACMLParseError
from acml.parser import parse_document
from acml.serializer import serialize_document


__all__ = [
    "ACMLError",
    "ACMLParseError",
    "parse_document",
    "serialize_document",
]
