from __future__ import annotations

import unittest
from unittest.mock import patch

import study_sft.inference_runtime as inference_runtime
from study_sft.agentic_context import AgenticContextEncoder, QWEN3_AGENTIC_TOKEN_TABLE
from study_sft.inference_prompts import agentic_context_from_conversation, conversation_from_user_text
from study_sft.inference_runtime import (
    STOP_REASON_EOS_TOKEN,
    STOP_REASON_MESSAGE_END,
    STOP_REASON_PROTOCOL_VIOLATION,
    STOP_REASON_STRUCTURE_TOKEN,
    SingleTurnGenerationResult,
    format_generation_result,
    generate_single_turn_result,
    parse_single_turn_generation,
    prepare_single_turn_generation,
)

from tests.test_training_data import FakeTokenizer


class FakeModel:
    def __init__(self, output_ids: list[int]) -> None:
        import torch

        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.output_ids = output_ids

    def parameters(self):
        yield self._parameter

    def generate(self, **kwargs):
        import torch

        prefix = kwargs["input_ids"]
        continuation = torch.tensor([self.output_ids], dtype=torch.long, device=prefix.device)
        return torch.cat([prefix, continuation], dim=1)


class InferLoraTests(unittest.TestCase):
    def test_prepare_single_turn_generation_uses_me_role_and_filters_none_stop_ids(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = None
        tokenizer.pad_token_id = 7
        encoder = AgenticContextEncoder(tokenizer)

        inputs = prepare_single_turn_generation(
            "hello",
            encoder,
            tokenizer,
            belief_prompt="You are a tester.",
        )

        self.assertEqual(inputs.pad_token_id, 7)
        self.assertEqual(inputs.stop_token_ids, [QWEN3_AGENTIC_TOKEN_TABLE.message_end])
        self.assertEqual(
            inputs.prefix_ids,
            encoder.encode_generation_payload_prefix(
                agentic_context_from_conversation(
                    conversation_from_user_text("hello", belief_prompt="You are a tester.")
                )
            ),
        )
        self.assertEqual(
            inputs.prefix_ids[-5:],
            [
                QWEN3_AGENTIC_TOKEN_TABLE.message_start,
                1000 + ord("m"),
                1000 + ord("e"),
                1000 + ord("\n"),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
            ],
        )

    def test_parse_single_turn_generation_treats_closed_message_as_clean_stop(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("hello", add_special_tokens=False)
            + [
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                QWEN3_AGENTIC_TOKEN_TABLE.message_end,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.display_text, "hello")
        self.assertEqual(result.stop_reason, STOP_REASON_MESSAGE_END)
        self.assertTrue(result.clean_termination)
        self.assertEqual(result.stop_token_name, "message_end")
        self.assertEqual(result.parser_state_at_stop, inference_runtime.PARSER_STATE_AFTER_PAYLOAD_END)
        self.assertEqual(result.termination_detail, "closed_message")

    def test_parse_single_turn_generation_marks_unclosed_message_end_as_not_clean(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("hello", add_special_tokens=False) + [QWEN3_AGENTIC_TOKEN_TABLE.message_end],
            encoder=encoder,
            tokenizer=tokenizer,
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.stop_reason, STOP_REASON_MESSAGE_END)
        self.assertFalse(result.clean_termination)
        self.assertEqual(result.termination_detail, "message_end_before_payload_end")

    def test_parse_single_turn_generation_stops_before_unexpected_structure_token(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("hello", add_special_tokens=False) + [QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start],
            encoder=encoder,
            tokenizer=tokenizer,
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.display_text, "hello")
        self.assertEqual(result.stop_reason, STOP_REASON_STRUCTURE_TOKEN)
        self.assertFalse(result.clean_termination)
        self.assertEqual(result.stop_token_name, "opaque_payload_start")
        self.assertEqual(result.termination_detail, "unexpected_structure_in_payload")

    def test_parse_single_turn_generation_rejects_text_after_payload_end(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("hello", add_special_tokens=False)
            + [
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                1000 + ord("!"),
            ],
            encoder=encoder,
            tokenizer=tokenizer,
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.stop_reason, STOP_REASON_PROTOCOL_VIOLATION)
        self.assertFalse(result.clean_termination)
        self.assertEqual(result.termination_detail, "text_after_payload_end")

    def test_generate_single_turn_result_returns_structured_result(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")

        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        tokenizer.pad_token_id = 7
        encoder = AgenticContextEncoder(tokenizer)
        model = FakeModel(tokenizer.encode("world", add_special_tokens=False) + [tokenizer.eos_token_id])

        result = generate_single_turn_result(
            "hello",
            encoder=encoder,
            tokenizer=tokenizer,
            model=model,
            belief_prompt="You are a tester.",
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertIsInstance(result, SingleTurnGenerationResult)
        self.assertEqual(result.text, "world")
        self.assertEqual(result.display_text, "world")
        self.assertEqual(result.stop_reason, STOP_REASON_EOS_TOKEN)
        self.assertFalse(result.clean_termination)

    def test_load_lora_inference_model_uses_torch_dtype_keyword(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        captured_kwargs: dict[str, object] = {}

        class _FakeAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_name_or_path: str, **kwargs):
                del model_name_or_path
                captured_kwargs.update(kwargs)
                return object()

        class _FakePeftModel:
            @staticmethod
            def from_pretrained(model, adapter_path: str):
                del adapter_path

                class _WrappedModel:
                    def eval(self):
                        return None

                self.assertIsNotNone(model)
                return _WrappedModel()

        class _FakeBitsAndBytesConfig:
            def __init__(self, *, load_in_4bit: bool) -> None:
                self.load_in_4bit = load_in_4bit

        with patch.object(inference_runtime, "load_base_tokenizer", return_value=FakeTokenizer()):
            with patch.dict(
                "sys.modules",
                {
                    "peft": type("peft", (), {"PeftModel": _FakePeftModel}),
                    "transformers": type(
                        "transformers",
                        (),
                        {
                            "AutoModelForCausalLM": _FakeAutoModelForCausalLM,
                            "BitsAndBytesConfig": _FakeBitsAndBytesConfig,
                        },
                    ),
                },
            ):
                inference_runtime.load_lora_inference_model(
                    "fake-model",
                    "fake-adapter",
                    local_files_only=True,
                    load_in_4bit=False,
                )

        self.assertIn("torch_dtype", captured_kwargs)
        self.assertNotIn("dtype", captured_kwargs)
        self.assertEqual(captured_kwargs["torch_dtype"], torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    def test_format_generation_result_hides_structure_tokens_and_shows_status(self) -> None:
        rendered = format_generation_result(
            SingleTurnGenerationResult(
                text="answer",
                display_text="answer",
                output_ids=[1000 + ord("a"), QWEN3_AGENTIC_TOKEN_TABLE.message_start],
                content_ids=[1000 + ord("a")],
                stop_reason=STOP_REASON_STRUCTURE_TOKEN,
                clean_termination=False,
                stop_token_id=QWEN3_AGENTIC_TOKEN_TABLE.message_start,
                stop_token_name="message_start",
                termination_detail="unexpected_structure_in_payload",
            )
        )

        self.assertEqual(
            rendered,
            "answer\n[stop_reason=structure_token, clean_termination=false, detail=unexpected_structure_in_payload, token=message_start]",
        )
        self.assertNotIn(QWEN3_AGENTIC_TOKEN_TABLE.message_start_text, rendered)

    def test_format_generation_result_strips_only_at_display_boundary(self) -> None:
        rendered = format_generation_result(
            SingleTurnGenerationResult(
                text="  answer  ",
                display_text="answer",
                output_ids=[1000 + ord("a")],
                content_ids=[1000 + ord("a")],
                stop_reason=STOP_REASON_MESSAGE_END,
                clean_termination=True,
            )
        )

        self.assertEqual(rendered, "answer")


if __name__ == "__main__":
    unittest.main()
