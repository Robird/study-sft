from __future__ import annotations

import unittest

from acml.model import ActionNode, Attribute, Document, EntryNode, PayloadNode, TextNode
from acml.semantic_model import (
    SemanticAction,
    SemanticContext,
    SemanticDocument,
    SemanticEntry,
    SemanticPayload,
    SemanticText,
    document_to_semantic_context,
    document_to_semantic_document,
    semantic_context_to_document,
    semantic_document_to_document,
)


class ACMLSemanticModelTests(unittest.TestCase):
    def test_document_model_normalizes_sequences_and_rejects_invalid_children(self) -> None:
        document = Document(
            version="0",
            entries=[EntryNode(kind="observation", content=[TextNode("hello")])],
            attrs=[Attribute("project", "demo")],
        )

        self.assertEqual(document.entries, (EntryNode(kind="observation", content=(TextNode("hello"),)),))
        self.assertEqual(document.attrs, (Attribute("project", "demo"),))

        with self.assertRaisesRegex(ValueError, "Document.entries items must be EntryNode"):
            Document(version="0", entries=["bad"])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "Document.attrs must be a sequence"):
            Document(version="0", entries=(), attrs="bad")  # type: ignore[arg-type]

    def test_document_to_semantic_document_preserves_extra_attrs_without_duplicating_promoted_fields(self) -> None:
        document = Document(
            version="0",
            attrs=(Attribute("project", "demo"),),
            entries=(
                EntryNode(
                    kind="me",
                    attrs=(Attribute("loss", "true"), Attribute("source", "demo")),
                    content=(
                        TextNode("thought "),
                        PayloadNode("payload", attrs=(Attribute("mime", "text/plain"),)),
                        ActionNode(
                            (TextNode("Call("), PayloadNode("x", attrs=(Attribute("kind", "opaque"),)), TextNode(")")),
                            attrs=(Attribute("dialect", "demo"),),
                        ),
                    ),
                ),
            ),
        )

        semantic_document = document_to_semantic_document(document)

        self.assertEqual(
            semantic_document,
            SemanticDocument(
                version="0",
                attrs=(Attribute("project", "demo"),),
                entries=(
                    SemanticEntry(
                        kind="me",
                        attrs=(Attribute("loss", "true"), Attribute("source", "demo")),
                        content=(
                            SemanticText("thought "),
                            SemanticPayload("payload", attrs=(Attribute("mime", "text/plain"),)),
                            SemanticAction(
                                (
                                    SemanticText("Call("),
                                    SemanticPayload("x", attrs=(Attribute("kind", "opaque"),)),
                                    SemanticText(")"),
                                ),
                                attrs=(Attribute("dialect", "demo"),),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def test_document_to_semantic_context_preserves_tag_level_attrs_but_drops_root_metadata(self) -> None:
        document = Document(
            version="0",
            attrs=(Attribute("project", "demo"),),
            entries=(
                EntryNode(
                    kind="me",
                    attrs=(Attribute("loss", "true"),),
                    content=(PayloadNode("answer", attrs=(Attribute("mime", "text/plain"),)),),
                ),
            ),
        )

        context = document_to_semantic_context(document)

        self.assertEqual(
            context,
            SemanticContext(
                entries=(
                    SemanticEntry(
                        kind="me",
                        attrs=(Attribute("loss", "true"),),
                        content=(SemanticPayload("answer", attrs=(Attribute("mime", "text/plain"),)),),
                    ),
                ),
            ),
        )

    def test_semantic_context_to_document_reconstructs_document_given_explicit_version(self) -> None:
        context = SemanticContext(
            entries=(
                SemanticEntry(
                    kind="observation",
                    attrs=(Attribute("source", "console"),),
                    content=(SemanticText("hello"),),
                ),
                SemanticEntry(
                    kind="me",
                    attrs=(Attribute("loss", "true"),),
                    content=(SemanticPayload("answer", attrs=(Attribute("mime", "text/plain"),)),),
                ),
            )
        )

        document = semantic_context_to_document(context, version="1")

        self.assertEqual(
            document,
            Document(
                version="1",
                entries=(
                    EntryNode(
                        kind="observation",
                        attrs=(Attribute("source", "console"),),
                        content=(TextNode("hello"),),
                    ),
                    EntryNode(
                        kind="me",
                        attrs=(Attribute("loss", "true"),),
                        content=(PayloadNode("answer", attrs=(Attribute("mime", "text/plain"),)),),
                    ),
                ),
            ),
        )

    def test_semantic_document_to_document_round_trips_root_and_nested_extra_attrs(self) -> None:
        semantic_document = SemanticDocument(
            version="0",
            attrs=(Attribute("project", "demo"),),
            entries=(
                SemanticEntry(
                    kind="me",
                    attrs=(Attribute("loss", "true"),),
                    content=(
                        SemanticAction(
                            (
                                SemanticText("Call("),
                                SemanticPayload("x", attrs=(Attribute("kind", "opaque"),)),
                                SemanticText(")"),
                            ),
                            attrs=(Attribute("dialect", "demo"),),
                        ),
                    ),
                ),
            ),
        )

        document = semantic_document_to_document(semantic_document)

        self.assertEqual(document_to_semantic_document(document), semantic_document)

    def test_semantic_context_model_normalizes_sequences_and_rejects_invalid_children(self) -> None:
        context = SemanticContext(
            entries=[SemanticEntry(kind="observation", content=[SemanticText("hello")])],
        )

        self.assertEqual(context.entries, (SemanticEntry(kind="observation", content=(SemanticText("hello"),)),))

        with self.assertRaisesRegex(ValueError, "SemanticContext.entries items must be SemanticEntry"):
            SemanticContext(entries=["bad"])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "SemanticPayload.attrs items must be Attribute"):
            SemanticPayload("hello", attrs=["bad"])  # type: ignore[list-item]

    def test_entry_and_semantic_entry_reject_duplicate_promoted_kind_attr(self) -> None:
        with self.assertRaisesRegex(ValueError, "promoted attribute 'kind'"):
            EntryNode(
                kind="me",
                attrs=(Attribute("kind", "observation"),),
                content=(),
            )
        with self.assertRaisesRegex(ValueError, "promoted attribute 'kind'"):
            SemanticEntry(
                kind="me",
                attrs=(Attribute("kind", "observation"),),
                content=(),
            )

    def test_semantic_document_rejects_duplicate_promoted_version_attr(self) -> None:
        with self.assertRaisesRegex(ValueError, "promoted attribute 'version'"):
            SemanticDocument(
                version="0",
                attrs=(Attribute("version", "1"),),
                entries=(),
            )


if __name__ == "__main__":
    unittest.main()
