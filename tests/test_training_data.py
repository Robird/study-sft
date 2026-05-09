from __future__ import annotations

import unittest

from study_sft.agentic_context import AgenticContextEncoder, EncodedContext, QWEN3_AGENTIC_TOKEN_TABLE
from study_sft.agentic_context_model import AgenticContext, AgenticMessage, AgenticOpaquePayload
from study_sft.samples import (
    DEFAULT_BELIEF_PROMPT,
    NormalizedConversation,
    NormalizedTurn,
    TrainingSample,
    agentic_context_from_sample,
    agentic_context_from_conversation,
    conversation_from_record,
    conversation_from_user_text,
    training_samples_from_conversation,
)
from study_sft.training_data import encode_training_sample


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
    def test_conversation_from_alpaca_record_normalizes_fields(self) -> None:
        record = {
            "instruction": "Explain SFT",
            "input": "Use one sentence",
            "output": "SFT teaches the model from supervised target continuations.",
        }

        conversation = conversation_from_record(record, dataset_format="alpaca")

        self.assertEqual([turn.role for turn in conversation.turns], ["system", "user", "assistant"])
        self.assertEqual(conversation.turns[0].content, DEFAULT_BELIEF_PROMPT)

    def test_agentic_context_from_conversation_keeps_context_generic(self) -> None:
        conversation = conversation_from_record(
            {
                "instruction": "Explain SFT",
                "output": "SFT teaches the model from supervised target continuations.",
            },
            dataset_format="alpaca",
        )

        typed_context = agentic_context_from_conversation(conversation)

        self.assertIsInstance(typed_context, AgenticContext)
        self.assertEqual(
            typed_context,
            AgenticContext(
                messages=(
                    AgenticMessage(role="belief", content=(AgenticOpaquePayload(text=DEFAULT_BELIEF_PROMPT),)),
                    AgenticMessage(role="observation", content=(AgenticOpaquePayload(text="Explain SFT"),)),
                    AgenticMessage(
                        role="me",
                        content=(
                            AgenticOpaquePayload(
                                text="SFT teaches the model from supervised target continuations."
                            ),
                        ),
                    ),
                )
            ),
        )
        self.assertFalse(typed_context.messages[-1].loss)

    def test_encode_training_sample_marks_only_the_selected_supervised_message(self) -> None:
        conversation = conversation_from_record(
            {
                "messages": [
                    {"role": "system", "content": "You are a tutor."},
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"},
                    {"role": "user", "content": "Dangling follow-up"},
                ]
            },
            dataset_format="messages",
        )
        sample = training_samples_from_conversation(conversation)[-1]
        context = agentic_context_from_sample(sample)
        labels = encode_training_sample(sample, AgenticContextEncoder(FakeTokenizer()))["labels"]

        self.assertEqual(
            [message.role for message in context.messages],
            ["belief", "observation", "me", "observation", "me"],
        )
        self.assertFalse(any(message.loss for message in context.messages))
        self.assertIn(-100, labels)
        self.assertTrue(any(label != -100 for label in labels))

    def test_agentic_context_from_sample_enables_loss_only_for_selected_message(self) -> None:
        tokenizer = FakeTokenizer()
        sample = TrainingSample(
            conversation=NormalizedConversation(
                turns=(
                    NormalizedTurn(role="user", content="question"),
                    NormalizedTurn(role="assistant", content="historical answer"),
                    NormalizedTurn(role="assistant", content="target answer"),
                )
            ),
            target_turn_index=2,
        )

        typed_training_context = agentic_context_from_sample(sample, mark_target_loss=True)
        encoded = AgenticContextEncoder(tokenizer).encode_context(typed_training_context)
        last_message_start = max(
            index for index, token_id in enumerate(encoded.input_ids) if token_id == QWEN3_AGENTIC_TOKEN_TABLE.message_start
        )

        self.assertFalse(typed_training_context.messages[0].loss)
        self.assertFalse(typed_training_context.messages[1].loss)
        self.assertTrue(typed_training_context.messages[2].loss)
        self.assertTrue(all(mask == 0 for mask in encoded.loss_mask[:last_message_start]))
        self.assertTrue(all(mask == 1 for mask in encoded.loss_mask[last_message_start:]))

    def test_agentic_context_from_sample_rejects_invalid_supervision_index(self) -> None:
        sample = TrainingSample(
            conversation=NormalizedConversation(turns=(NormalizedTurn(role="assistant", content="answer"),)),
            target_turn_index=2,
        )

        with self.assertRaisesRegex(ValueError, "out of range"):
            agentic_context_from_sample(sample, mark_target_loss=True)

    def test_agentic_context_from_sample_rejects_non_assistant_supervision(self) -> None:
        sample = TrainingSample(
            conversation=NormalizedConversation(
                turns=(
                    NormalizedTurn(role="user", content="question"),
                    NormalizedTurn(role="assistant", content="answer"),
                )
            ),
            target_turn_index=0,
        )

        with self.assertRaisesRegex(ValueError, "assistant turn"):
            agentic_context_from_sample(sample, mark_target_loss=True)

    def test_conversation_from_record_rejects_unsupported_dataset_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dataset_format"):
            conversation_from_record({"text": "hello"}, dataset_format="text")  # type: ignore[arg-type]

    def test_conversation_from_user_text_projects_to_agentic_context(self) -> None:
        conversation = conversation_from_user_text("hello", belief_prompt="You are a tester.")
        typed_context = agentic_context_from_conversation(conversation)

        self.assertEqual([message.role for message in typed_context.messages], ["belief", "observation"])
        self.assertEqual(typed_context, agentic_context_from_conversation(conversation))
        self.assertEqual(typed_context.messages[0].content[0].text, "You are a tester.")
        self.assertEqual(typed_context.messages[1].content[0].text, "hello")

    def test_training_samples_from_conversation_expands_each_assistant_turn(self) -> None:
        conversation = conversation_from_record(
            {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"},
                ]
            },
            dataset_format="messages",
        )

        samples = training_samples_from_conversation(conversation)

        self.assertEqual(len(samples), 2)
        self.assertEqual([turn.role for turn in samples[0].conversation.turns], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(samples[0].prefix_turn_count, 3)
        self.assertEqual(samples[1].prefix_turn_count, 5)
        self.assertEqual(samples[0].target_turn_index, 2)
        self.assertEqual(samples[1].target_turn_index, 4)
        self.assertEqual(
            [message.role for message in agentic_context_from_sample(samples[0], mark_target_loss=True).messages],
            ["belief", "observation", "me"],
        )

    def test_encode_training_sample_truncates_at_message_boundaries(self) -> None:
        encoder = AgenticContextEncoder(FakeTokenizer())
        conversation = conversation_from_record(
            {
                "messages": [
                    {"role": "system", "content": "You are a tutor."},
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Final answer that should stay intact."},
                ]
            },
            dataset_format="messages",
        )
        sample = training_samples_from_conversation(conversation)[-1]
        full_features = encode_training_sample(sample, encoder)
        full_message_count = len(
            encoder.describe_messages(
                EncodedContext(
                    input_ids=full_features["input_ids"],
                    loss_mask=[1 if label != -100 else 0 for label in full_features["labels"]],
                )
            )
        )
        target_only_length = len(
            encode_training_sample(
                training_samples_from_conversation(
                    conversation_from_record(
                        {"messages": [{"role": "assistant", "content": "Final answer that should stay intact."}]},
                        dataset_format="messages",
                    )
                )[0],
                encoder,
            )["input_ids"]
        )

        features = encode_training_sample(sample, encoder, max_length=target_only_length)
        encoded = EncodedContext(
            input_ids=features["input_ids"],
            loss_mask=[1 if label != -100 else 0 for label in features["labels"]],
        )

        encoder.validate(encoded)
        message_spans = encoder.describe_messages(encoded)
        self.assertEqual(encoded.input_ids[0], QWEN3_AGENTIC_TOKEN_TABLE.message_start)
        self.assertLess(len(message_spans), full_message_count)
        self.assertTrue(message_spans[-1].loss)

    def test_encode_training_sample_rejects_target_message_that_exceeds_max_length(self) -> None:
        encoder = AgenticContextEncoder(FakeTokenizer())
        sample = training_samples_from_conversation(
            conversation_from_record(
                {
                    "instruction": "Explain SFT",
                    "output": "SFT teaches a model with supervised continuations.",
                },
                dataset_format="alpaca",
            )
        )[0]

        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            encode_training_sample(sample, encoder=encoder, max_length=8)

    def test_agentic_context_from_sample_rejects_supervision_outside_prefix(self) -> None:
        sample = TrainingSample(
            conversation=NormalizedConversation(
                turns=(
                    NormalizedTurn(role="user", content="question"),
                    NormalizedTurn(role="assistant", content="answer"),
                )
            ),
            prefix_turn_count=1,
            target_turn_index=1,
        )

        with self.assertRaisesRegex(ValueError, "inside the prefix"):
            agentic_context_from_sample(sample, mark_target_loss=True)

    def test_encode_training_sample_payload_only_masks_structure_tokens(self) -> None:
        encoder = AgenticContextEncoder(FakeTokenizer())
        sample = training_samples_from_conversation(
            conversation_from_record(
                {
                    "instruction": "Explain SFT",
                    "output": "payload",
                },
                dataset_format="alpaca",
            )
        )[0]

        features = encode_training_sample(sample, encoder=encoder, label_policy="payload_only")

        supervised_ids = [token_id for token_id, label in zip(features["input_ids"], features["labels"], strict=True) if label != -100]
        self.assertEqual(supervised_ids, [1000 + ord(char) for char in "payload"])
        self.assertNotIn(QWEN3_AGENTIC_TOKEN_TABLE.message_start, supervised_ids)
        self.assertNotIn(QWEN3_AGENTIC_TOKEN_TABLE.message_end, supervised_ids)

    def test_encode_training_sample_payload_only_preserves_payload_labels_after_truncation(self) -> None:
        encoder = AgenticContextEncoder(FakeTokenizer())
        conversation = conversation_from_record(
            {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "payload"},
                ]
            },
            dataset_format="messages",
        )
        sample = training_samples_from_conversation(conversation)[-1]

        target_only_length = len(
            encode_training_sample(
                training_samples_from_conversation(
                    conversation_from_record(
                        {"messages": [{"role": "assistant", "content": "payload"}]},
                        dataset_format="messages",
                    )
                )[0],
                encoder,
                label_policy="payload_only",
            )["input_ids"]
        )
        features = encode_training_sample(
            sample,
            encoder,
            max_length=target_only_length,
            label_policy="payload_only",
        )

        supervised_ids = [token_id for token_id, label in zip(features["input_ids"], features["labels"], strict=True) if label != -100]
        self.assertEqual(supervised_ids, [1000 + ord(char) for char in "payload"])


if __name__ == "__main__":
    unittest.main()
