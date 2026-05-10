"""Parser for ACML text."""

from __future__ import annotations

from acml.errors import ACMLParseError
from acml.model import ActionNode, Attribute, Document, EntryNode, PayloadNode, TextNode


_PRECISION_ESCAPE_CLOSE = "&lt;/acml"
_PRECISION_ESCAPE_OPEN = "&lt;acml"


def parse_document(source: str) -> Document:
    """Parse ACML source text into the in-memory ACML model."""

    if not isinstance(source, str):
        raise TypeError("parse_document expects a string source")
    return _Parser(source).parse_document()


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.index = 0

    def parse_document(self) -> Document:
        self._skip_whitespace()
        root_attrs = self._parse_start_tag("acml")
        version, extra_attrs = self._extract_required_attr(root_attrs, "version", "acml")
        entries: list[EntryNode] = []
        while True:
            self._skip_whitespace()
            if self._peek_end_tag("acml"):
                break
            if self._peek_start_tag("acml:entry"):
                entries.append(self._parse_entry())
                continue
            if self._eof():
                raise self._error("expected </acml> before end of input")
            raise self._error("root may only contain <acml:entry> children and interstitial whitespace")
        self._parse_end_tag("acml")
        self._skip_whitespace()
        if not self._eof():
            raise self._error("unexpected trailing content after </acml>")
        return Document(version=version, entries=tuple(entries), attrs=tuple(extra_attrs))

    def _parse_entry(self) -> EntryNode:
        attrs = self._parse_start_tag("acml:entry")
        kind, extra_attrs = self._extract_required_attr(attrs, "kind", "acml:entry")
        content = self._parse_mixed_content(
            end_tag="acml:entry",
            allowed_child_tags=("acml:payload", "acml:action"),
        )
        self._parse_end_tag("acml:entry")
        return EntryNode(kind=kind, content=content, attrs=tuple(extra_attrs))

    def _parse_payload(self) -> PayloadNode:
        attrs = self._parse_start_tag("acml:payload")
        content = self._parse_mixed_content(
            end_tag="acml:payload",
            allowed_child_tags=(),
        )
        self._parse_end_tag("acml:payload")
        return PayloadNode(text=_flatten_text_only(content, "acml:payload"), attrs=tuple(attrs))

    def _parse_action(self) -> ActionNode:
        attrs = self._parse_start_tag("acml:action")
        content = self._parse_mixed_content(
            end_tag="acml:action",
            allowed_child_tags=("acml:payload",),
        )
        self._parse_end_tag("acml:action")
        return ActionNode(content=content, attrs=tuple(attrs))

    def _parse_mixed_content(
        self,
        *,
        end_tag: str,
        allowed_child_tags: tuple[str, ...],
    ) -> tuple[TextNode | PayloadNode | ActionNode, ...]:
        items: list[TextNode | PayloadNode | ActionNode] = []
        while True:
            if self._eof():
                raise self._error(f"expected </{end_tag}> before end of input")
            if self._peek_end_tag(end_tag):
                break
            if "acml:payload" in allowed_child_tags and self._peek_start_tag("acml:payload"):
                items.append(self._parse_payload())
                continue
            if "acml:action" in allowed_child_tags and self._peek_start_tag("acml:action"):
                items.append(self._parse_action())
                continue
            if self._peek_reserved_prefix():
                raise self._error(f"unexpected ACML reserved tag while parsing <{end_tag}> content")
            text = self._consume_text_chunk(end_tag=end_tag, allowed_child_tags=allowed_child_tags)
            if text:
                _append_text(items, text)
        return tuple(items)

    def _consume_text_chunk(self, *, end_tag: str, allowed_child_tags: tuple[str, ...]) -> str:
        pieces: list[str] = []
        while not self._eof():
            if self._peek_end_tag(end_tag):
                break
            if "acml:payload" in allowed_child_tags and self._peek_start_tag("acml:payload"):
                break
            if "acml:action" in allowed_child_tags and self._peek_start_tag("acml:action"):
                break
            if self.source.startswith(_PRECISION_ESCAPE_CLOSE, self.index):
                pieces.append("</acml")
                self.index += len(_PRECISION_ESCAPE_CLOSE)
                continue
            if self.source.startswith(_PRECISION_ESCAPE_OPEN, self.index):
                pieces.append("<acml")
                self.index += len(_PRECISION_ESCAPE_OPEN)
                continue
            if self._peek_reserved_prefix():
                break
            pieces.append(self.source[self.index])
            self.index += 1
        return "".join(pieces)

    def _parse_start_tag(self, name: str) -> list[Attribute]:
        self._expect(f"<{name}")
        attrs: list[Attribute] = []
        if self._eof():
            raise self._error(f"unterminated <{name}> tag")
        next_char = self.source[self.index]
        if next_char not in (" ", "\t", "\n", "\r", ">"):
            raise self._error(f"malformed <{name}> tag")
        while True:
            self._skip_whitespace()
            if self._eof():
                raise self._error(f"unterminated <{name}> tag")
            if self.source[self.index] == ">":
                self.index += 1
                return attrs
            if self.source[self.index] == "/":
                raise self._error(f"self-closing <{name} /> is not supported")
            attrs.append(self._parse_attribute())

    def _parse_attribute(self) -> Attribute:
        name_start = self.index
        while not self._eof() and self.source[self.index] not in (" ", "\t", "\n", "\r", "=", ">", "/"):
            self.index += 1
        if self.index == name_start:
            raise self._error("expected attribute name")
        name = self.source[name_start:self.index]
        self._skip_whitespace()
        if self._eof() or self.source[self.index] != "=":
            raise self._error(f'attribute "{name}" must use = followed by a double-quoted value')
        self.index += 1
        self._skip_whitespace()
        if self._eof() or self.source[self.index] != '"':
            raise self._error(f'attribute "{name}" must use a double-quoted value')
        self.index += 1
        value_start = self.index
        while not self._eof():
            current = self.source[self.index]
            if current == '"':
                value = self.source[value_start:self.index]
                self.index += 1
                return Attribute(name=name, value=value)
            if current in ("\n", "\r"):
                raise self._error(f'attribute "{name}" value may not contain newlines')
            if current in ("<", ">"):
                raise self._error(f'attribute "{name}" value may not contain angle brackets')
            self.index += 1
        raise self._error(f'unterminated double-quoted value for attribute "{name}"')

    def _parse_end_tag(self, name: str) -> None:
        self._expect(f"</{name}>")

    def _extract_required_attr(
        self,
        attrs: list[Attribute],
        required_name: str,
        tag_name: str,
    ) -> tuple[str, list[Attribute]]:
        values = [attr.value for attr in attrs if attr.name == required_name]
        if not values:
            raise self._error(f"<{tag_name}> requires a double-quoted {required_name} attribute")
        if len(values) > 1:
            raise self._error(f"<{tag_name}> may not repeat attribute {required_name!r}")
        extra_attrs = [attr for attr in attrs if attr.name != required_name]
        return values[0], extra_attrs

    def _peek_start_tag(self, name: str) -> bool:
        prefix = f"<{name}"
        if not self.source.startswith(prefix, self.index):
            return False
        next_index = self.index + len(prefix)
        if next_index >= self.length:
            return False
        return self.source[next_index] in (" ", "\t", "\n", "\r", ">")

    def _peek_end_tag(self, name: str) -> bool:
        return self.source.startswith(f"</{name}>", self.index)

    def _peek_reserved_prefix(self) -> bool:
        return self.source.startswith("<acml", self.index) or self.source.startswith("</acml", self.index)

    def _expect(self, text: str) -> None:
        if not self.source.startswith(text, self.index):
            raise self._error(f"expected {text}")
        self.index += len(text)

    def _skip_whitespace(self) -> None:
        while not self._eof() and self.source[self.index].isspace():
            self.index += 1

    def _eof(self) -> bool:
        return self.index >= self.length

    def _error(self, message: str) -> ACMLParseError:
        line = self.source.count("\n", 0, self.index) + 1
        last_newline = self.source.rfind("\n", 0, self.index)
        column = self.index + 1 if last_newline < 0 else self.index - last_newline
        return ACMLParseError(message, line=line, column=column, index=self.index)


def _append_text(items: list[TextNode | PayloadNode | ActionNode], text: str) -> None:
    if items and isinstance(items[-1], TextNode):
        items[-1] = TextNode(items[-1].text + text)
    else:
        items.append(TextNode(text=text))


def _flatten_text_only(items: tuple[TextNode | PayloadNode | ActionNode, ...], tag_name: str) -> str:
    pieces: list[str] = []
    for item in items:
        if not isinstance(item, TextNode):
            raise ValueError(f"<{tag_name}> may only contain content text")
        pieces.append(item.text)
    return "".join(pieces)
