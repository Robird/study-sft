from __future__ import annotations

import unittest

from study_sft.adapters.acml import agentic_context_from_acml_record
from study_sft.agentic_context import AgenticContextEncoder, EncodedContext, QWEN3_AGENTIC_TOKEN_TABLE
from study_sft.agentic_context_model import AgenticContext, AgenticEntry, AgenticOpaquePayload
from study_sft.training_data import TrainingEncodingConfig, encode_training_context, encode_training_features_from_record


class FakeTokenizer:
    def __init__(self) -> None:
        self.specials = QWEN3_AGENTIC_TOKEN_TABLE.text_by_name()
        self.special_ids = QWEN3_AGENTIC_TOKEN_TABLE.id_by_name()
        self.id_to_text = {self.special_ids[name]: token_text for name, token_text in self.specials.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        input_ids: list[int] = []
        index = 0
        special_items = sorted(self.specials.items(), key=lambda item: len(item[1]), reverse=True)
        while index < len(text):
            for name, token_text in special_items:
                if text.startswith(token_text, index):
                    input_ids.append(self.special_ids[name])
                    index += len(token_text)
                    break
            else:
                input_ids.append(1000 + ord(text[index]))
                index += 1
        return input_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            special_text = self.id_to_text.get(token_id)
            if special_text is not None:
                if not skip_special_tokens:
                    pieces.append(special_text)
                continue
            pieces.append(chr(token_id - 1000))
        return "".join(pieces)


class TrainingDataTests(unittest.TestCase):
    def test_agentic_context_from_record_projects_acml_document(self) -> None:
        context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="observation">question</acml:entry>'
                '<acml:entry kind="me" loss="true">answer</acml:entry></acml>'
            }
        )

        self.assertEqual(
            context,
            AgenticContext(
                entries=(
                    AgenticEntry(kind="observation", content=(AgenticOpaquePayload("question"),)),
                    AgenticEntry(kind="me", loss=True, content=(AgenticOpaquePayload("answer"),)),
                )
            ),
        )

    def test_agentic_context_from_record_rejects_missing_acml_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "column named 'acml'"):
            agentic_context_from_acml_record({"instruction": "Explain SFT"})

    def test_encode_training_features_from_acml_record(self) -> None:
        features = encode_training_features_from_record(
            {
                "acml": '<acml version="0"><acml:entry kind="observation">question</acml:entry>'
                '<acml:entry kind="me" loss="true">answer</acml:entry></acml>'
            },
            encoder=AgenticContextEncoder(FakeTokenizer()),
            config=TrainingEncodingConfig(
                max_length=256,
                label_policy="entry",
            ),
        )

        self.assertIn(-100, features["labels"])
        self.assertTrue(any(label != -100 for label in features["labels"]))

    def test_encode_training_context_rejects_unsupervised_acml_context(self) -> None:
        context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="observation">question</acml:entry>'
                '<acml:entry kind="me">answer</acml:entry></acml>'
            }
        )

        with self.assertRaisesRegex(ValueError, "no supervised labels"):
            encode_training_context(context, AgenticContextEncoder(FakeTokenizer()))

    def test_encode_training_context_payload_only_masks_structure_tokens(self) -> None:
        context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="me" loss="true">payload</acml:entry></acml>',
            }
        )

        features = encode_training_context(
            context,
            AgenticContextEncoder(FakeTokenizer()),
            label_policy="payload_only",
        )

        supervised_ids = [
            token_id
            for token_id, label in zip(features["input_ids"], features["labels"], strict=True)
            if label != -100
        ]
        self.assertEqual(supervised_ids, [1000 + ord(char) for char in "payload"])
        self.assertNotIn(QWEN3_AGENTIC_TOKEN_TABLE.message_start, supervised_ids)
        self.assertNotIn(QWEN3_AGENTIC_TOKEN_TABLE.message_end, supervised_ids)

    def test_encode_training_context_truncates_to_supervised_suffix_start(self) -> None:
        encoder = AgenticContextEncoder(FakeTokenizer())
        context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="observation">Question 1</acml:entry>'
                '<acml:entry kind="me">Answer 1</acml:entry>'
                '<acml:entry kind="observation">Question 2</acml:entry>'
                '<acml:entry kind="me" loss="true">Final answer that should stay intact.</acml:entry></acml>'
            }
        )
        target_only_context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="me" loss="true">'
                "Final answer that should stay intact."
                "</acml:entry></acml>"
            }
        )

        target_only_length = len(encode_training_context(target_only_context, encoder)["input_ids"])
        features = encode_training_context(context, encoder, max_length=target_only_length)
        encoded = EncodedContext(
            input_ids=features["input_ids"],
            loss_mask=[1 if label != -100 else 0 for label in features["labels"]],
        )

        encoder.validate(encoded)
        entry_spans = encoder.describe_entries(encoded)
        self.assertEqual(len(entry_spans), 1)
        self.assertEqual(encoded.input_ids[0], QWEN3_AGENTIC_TOKEN_TABLE.message_start)
        self.assertTrue(entry_spans[-1].loss)

    def test_payload_only_labels_survive_truncation_for_acml_context(self) -> None:
        encoder = AgenticContextEncoder(FakeTokenizer())
        context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="observation">Question 1</acml:entry>'
                '<acml:entry kind="me">Answer 1</acml:entry>'
                '<acml:entry kind="observation">Question 2</acml:entry>'
                '<acml:entry kind="me" loss="true">payload</acml:entry></acml>'
            }
        )
        target_only_context = agentic_context_from_acml_record(
            {
                "acml": '<acml version="0"><acml:entry kind="me" loss="true">payload</acml:entry></acml>',
            }
        )

        target_only_length = len(
            encode_training_context(
                target_only_context,
                encoder,
                label_policy="payload_only",
            )["input_ids"]
        )
        features = encode_training_context(
            context,
            encoder,
            max_length=target_only_length,
            label_policy="payload_only",
        )

        supervised_ids = [
            token_id
            for token_id, label in zip(features["input_ids"], features["labels"], strict=True)
            if label != -100
        ]
        self.assertEqual(supervised_ids, [1000 + ord(char) for char in "payload"])


if __name__ == "__main__":
    unittest.main()
