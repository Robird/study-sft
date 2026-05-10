from __future__ import annotations

import unittest

from acml import ACMLParseError, parse_document, serialize_document
from acml.model import ActionNode, Attribute, Document, EntryNode, PayloadNode, TextNode


class ACMLParserTests(unittest.TestCase):
    def test_parse_document_rejects_non_string_source(self) -> None:
        with self.assertRaisesRegex(TypeError, "string source"):
            parse_document(None)  # type: ignore[arg-type]

    def test_parse_document_preserves_text_payload_action_and_extra_attrs(self) -> None:
        source = """<acml version="0" project="demo">
<acml:entry kind="observation" source="console">hello <div>world</div> && more</acml:entry>
<acml:entry kind="me">thinking
<acml:action dialect="demo">Call(<acml:payload mime="text/plain">unsafe <T> && raw</acml:payload>)</acml:action></acml:entry>
</acml>"""

        document = parse_document(source)

        self.assertEqual(
            document,
            Document(
                version="0",
                attrs=(Attribute("project", "demo"),),
                entries=(
                    EntryNode(
                        kind="observation",
                        attrs=(Attribute("source", "console"),),
                        content=(TextNode("hello <div>world</div> && more"),),
                    ),
                    EntryNode(
                        kind="me",
                        content=(
                            TextNode("thinking\n"),
                            ActionNode(
                                attrs=(Attribute("dialect", "demo"),),
                                content=(
                                    TextNode("Call("),
                                    PayloadNode(
                                        text="unsafe <T> && raw",
                                        attrs=(Attribute("mime", "text/plain"),),
                                    ),
                                    TextNode(")"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def test_parse_document_unescapes_precise_acml_sequences_only(self) -> None:
        source = (
            '<acml version="0"><acml:entry kind="belief">'
            "literal &lt;acml:payload> and &lt;/acml:payload> plus &amp; untouched"
            "</acml:entry></acml>"
        )

        document = parse_document(source)

        self.assertEqual(
            document.entries[0].content,
            (TextNode("literal <acml:payload> and </acml:payload> plus &amp; untouched"),),
        )

    def test_serialize_document_round_trips_parser_model(self) -> None:
        document = Document(
            version="0",
            attrs=(Attribute("project", "demo"),),
            entries=(
                EntryNode(
                    kind="me",
                    content=(
                        TextNode("before "),
                        ActionNode(
                            content=(
                                TextNode("Call("),
                                PayloadNode(text="literal <acml> marker", attrs=(Attribute("kind", "opaque"),)),
                                TextNode(")"),
                            ),
                        ),
                        TextNode(" after"),
                    ),
                    attrs=(Attribute("sender", "agent"),),
                ),
            ),
        )

        serialized = serialize_document(document)
        reparsed = parse_document(serialized)

        self.assertEqual(reparsed, document)
        self.assertIn('&lt;acml> marker', serialized)
        self.assertIn("<acml:entry", serialized)
        self.assertIn(' kind="me"', serialized)
        self.assertIn("<acml:action>", serialized)

    def test_parse_document_requires_root_version(self) -> None:
        with self.assertRaisesRegex(ACMLParseError, r"<acml> requires a double-quoted version attribute"):
            parse_document('<acml><acml:entry kind="observation">x</acml:entry></acml>')

    def test_parse_document_rejects_mismatched_reserved_tags(self) -> None:
        with self.assertRaisesRegex(ACMLParseError, r"expected </acml:payload> before end of input|unexpected ACML reserved tag"):
            parse_document(
                '<acml version="0"><acml:entry kind="me"><acml:payload>x</acml:action></acml:entry></acml>'
            )

    def test_parse_document_requires_double_quoted_kind(self) -> None:
        with self.assertRaisesRegex(ACMLParseError, r'double-quoted value|double-quoted kind attribute'):
            parse_document('<acml version="0"><acml:entry kind=\'me\'>x</acml:entry></acml>')

    def test_parse_document_rejects_angle_brackets_inside_attribute_values(self) -> None:
        with self.assertRaisesRegex(ACMLParseError, r'may not contain angle brackets'):
            parse_document('<acml version="0"><acml:entry kind="me" source="a<b">x</acml:entry></acml>')

    def test_serialize_document_rejects_non_document_input(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects a Document"):
            serialize_document(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
