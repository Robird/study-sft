from __future__ import annotations

import unittest

from acml.model import Attribute
from acml.semantic_model import SemanticContext, SemanticEntry, SemanticPayload
from study_sft.adapters.acml import (
    agentic_context_from_acml_record,
    agentic_context_from_acml_text,
    agentic_context_from_semantic_context,
    semantic_context_from_acml_text,
)
from study_sft.agentic_context_model import AgenticContext, AgenticEntry, AgenticOpaquePayload


class StudySFTACMLAdapterTests(unittest.TestCase):
    def test_semantic_context_from_acml_text_preserves_action_nodes(self) -> None:
        context = semantic_context_from_acml_text(
            '<acml version="0"><acml:entry kind="me" loss="true">x<acml:action>Call(<acml:payload>y</acml:payload>)</acml:action></acml:entry></acml>'
        )

        self.assertEqual(context.entries[0].kind, "me")
        self.assertEqual(context.entries[0].attrs, (Attribute("loss", "true"),))
        self.assertEqual(len(context.entries[0].content), 2)

    def test_agentic_context_from_acml_text_uses_explicit_entry_loss_hints(self) -> None:
        context = agentic_context_from_acml_text(
            '<acml version="0"><acml:entry kind="me" loss="true">thought<acml:action>Call(<acml:payload>y</acml:payload>)</acml:action></acml:entry></acml>'
        )

        self.assertEqual(
            context,
            AgenticContext(
                entries=(
                    AgenticEntry(
                        kind="me",
                        loss=True,
                        content=(
                            AgenticOpaquePayload("thought"),
                            AgenticOpaquePayload("Call(y)"),
                        ),
                    ),
                )
            ),
        )

    def test_agentic_context_from_semantic_context_can_apply_kind_based_loss_policy(self) -> None:
        semantic_context = SemanticContext(
            entries=(
                SemanticEntry(kind="belief", attrs=(Attribute("source", "ruleset"),), content=(SemanticPayload("rules"),)),
                SemanticEntry(kind="me", attrs=(Attribute("loss", "false"),), content=(SemanticPayload("answer"),)),
            )
        )

        context = agentic_context_from_semantic_context(semantic_context, loss_policy="all_me")

        self.assertEqual([entry.loss for entry in context.entries], [False, True])

    def test_agentic_context_from_acml_record_requires_acml_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "column named 'acml'"):
            agentic_context_from_acml_record({"text": "missing"})


if __name__ == "__main__":
    unittest.main()
