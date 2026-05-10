"""ACML error types."""

from __future__ import annotations


class ACMLError(Exception):
    """Base ACML error."""


class ACMLParseError(ACMLError):
    """Raised when ACML text cannot be parsed."""

    def __init__(self, message: str, *, line: int, column: int, index: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.index = index

    def __str__(self) -> str:
        return f"ParseError at line {self.line}, column {self.column}: {self.message}"
