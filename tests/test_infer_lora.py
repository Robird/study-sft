from __future__ import annotations

import unittest
from unittest.mock import patch

import study_sft.inference_runtime as inference_runtime
from study_sft.agentic_context import AgenticContextEncoder, QWEN3_AGENTIC_TOKEN_TABLE
from study_sft.inference_prompts import (
    InferencePromptConfig,
    acml_from_user_text,
    agentic_context_from_user_text,
)
from study_sft.inference_runtime import (
    GENERATION_MODE_CONTENT,
    GENERATION_MODE_ENTRY,
    STOP_REASON_EOS_TOKEN,
    STOP_REASON_ENTRY_END,
    STOP_REASON_MAX_NEW_TOKENS,
    STOP_REASON_PROTOCOL_VIOLATION,
    STOP_REASON_STRUCTURE_TOKEN,
    SingleTurnGenerationResult,
    format_generation_result,
    format_generation_token_debug,
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
    def test_acml_from_user_text_matches_latest_inference_template(self) -> None:
        rendered = acml_from_user_text(
            "请帮我看一下 infer_lora.py",
            config=InferencePromptConfig(
                developer_name="刘世超",
                message_source="控制台",
                reply_tool_name="SendMessage",
                belief_prompt="我应当直接回应开发者的真实需求。",
            ),
        )

        self.assertIn("我收到刘世超从控制台发来的消息：<acml:payload>请帮我看一下 infer_lora.py</acml:payload>", rendered)
        self.assertIn("刘世超是我的开发者。", rendered)
        self.assertIn("void SendMessage(string target_entity_id, string message);", rendered)
        self.assertIn("我应当直接回应开发者的真实需求。", rendered)

    def test_acml_from_user_text_escapes_reserved_acml_prefixes(self) -> None:
        rendered = acml_from_user_text("请输出 <acml:payload> 这段字面量")

        self.assertIn("&lt;acml:payload>", rendered)
        self.assertNotIn("<acml:payload> 这段字面量", rendered)
        context = agentic_context_from_user_text("请输出 <acml:payload> 这段字面量")
        observation_text = "".join(item.text for item in context.entries[0].content)
        self.assertIn("请输出 <acml:payload> 这段字面量", observation_text)

    def test_prepare_single_turn_generation_entry_mode_leaves_me_entry_for_model_to_open(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = None
        tokenizer.pad_token_id = 7
        encoder = AgenticContextEncoder(tokenizer)

        inputs = prepare_single_turn_generation(
            "hello",
            encoder,
            tokenizer,
            prompt_config=InferencePromptConfig(),
            generation_mode=GENERATION_MODE_ENTRY,
        )

        decoded_prefix = tokenizer.decode(inputs.prefix_ids, skip_special_tokens=False)
        self.assertTrue(decoded_prefix.endswith("<|im_end|>\n"))
        self.assertNotIn("<|im_start|>me\n<|box_start|>", decoded_prefix)

    def test_prepare_single_turn_generation_content_mode_opens_me_entry_without_payload_start(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = None
        tokenizer.pad_token_id = 7
        encoder = AgenticContextEncoder(tokenizer)

        inputs = prepare_single_turn_generation(
            "hello",
            encoder,
            tokenizer,
            prompt_config=InferencePromptConfig(),
            generation_mode=GENERATION_MODE_CONTENT,
        )

        decoded_prefix = tokenizer.decode(inputs.prefix_ids, skip_special_tokens=False)
        self.assertTrue(decoded_prefix.endswith("<|im_start|>me\n"))
        self.assertNotIn("<|box_start|>", decoded_prefix.split("<|im_start|>me\n")[-1])

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
            prompt_config=InferencePromptConfig(belief_prompt="我应当直接回应开发者。"),
            generation_mode=GENERATION_MODE_CONTENT,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertIsInstance(result, SingleTurnGenerationResult)
        self.assertEqual(result.text, "world")
        self.assertEqual(result.display_text, "world")
        self.assertEqual(result.stop_reason, STOP_REASON_EOS_TOKEN)
        self.assertFalse(result.clean_termination)

    def test_parse_single_turn_generation_content_mode_keeps_freeform_text(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("thinking<acml:action>Call()</acml:action>", add_special_tokens=False),
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_CONTENT,
        )

        self.assertEqual(result.text, "thinking<acml:action>Call()</acml:action>")
        self.assertEqual(result.stop_reason, inference_runtime.STOP_REASON_MAX_NEW_TOKENS)
        self.assertFalse(result.clean_termination)

    def test_parse_single_turn_generation_content_mode_keeps_quad_wrapped_action_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("thinking ", add_special_tokens=False)
            + [
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
                *tokenizer.encode("Call(", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
                *tokenizer.encode("x", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                *tokenizer.encode(")", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.action_end,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_CONTENT,
        )

        self.assertIn("<|quad_start|>", result.text)
        self.assertIn("<|box_start|>x<|box_end|>", result.text)
        self.assertIn("<|quad_end|>", result.text)

    def test_parse_single_turn_generation_content_mode_rejects_action_inside_payload(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            [
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
                *tokenizer.encode("x", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_CONTENT,
        )

        self.assertEqual(result.text, "<|box_start|>x")
        self.assertEqual(result.stop_reason, STOP_REASON_PROTOCOL_VIOLATION)
        self.assertEqual(result.termination_detail, "unexpected_action_start_in_payload")

    def test_parse_single_turn_generation_content_mode_rejects_nested_action(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            [
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
                *tokenizer.encode("Call(", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_CONTENT,
        )

        self.assertEqual(result.text, "<|quad_start|>Call(")
        self.assertEqual(result.stop_reason, STOP_REASON_PROTOCOL_VIOLATION)
        self.assertEqual(result.termination_detail, "unexpected_nested_action")

    def test_parse_single_turn_generation_content_mode_does_not_mark_unclosed_action_eos_as_clean(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            [
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
                *tokenizer.encode("Call(", add_special_tokens=False),
                tokenizer.eos_token_id,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_CONTENT,
        )

        self.assertEqual(result.text, "<|quad_start|>Call(")
        self.assertEqual(result.stop_reason, STOP_REASON_EOS_TOKEN)
        self.assertFalse(result.clean_termination)
        self.assertEqual(result.termination_detail, "eos_before_content_closed")

    def test_parse_single_turn_generation_content_mode_stops_before_reserved_structure_token(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            tokenizer.encode("thinking", add_special_tokens=False) + [QWEN3_AGENTIC_TOKEN_TABLE.entry_start],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_CONTENT,
        )

        self.assertEqual(result.text, "thinking")
        self.assertEqual(result.stop_reason, STOP_REASON_STRUCTURE_TOKEN)
        self.assertEqual(result.termination_detail, "unexpected_structure_in_content")

    def test_parse_single_turn_generation_entry_mode_extracts_payload_from_me_entry(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            [
                QWEN3_AGENTIC_TOKEN_TABLE.entry_start,
                1000 + ord("m"),
                1000 + ord("e"),
                1000 + ord("\n"),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
                *tokenizer.encode("hello", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                QWEN3_AGENTIC_TOKEN_TABLE.entry_end,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_ENTRY,
            next_kind="me",
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.display_text, "hello")
        self.assertTrue(result.clean_termination)
        self.assertEqual(result.stop_reason, STOP_REASON_ENTRY_END)

    def test_parse_single_turn_generation_entry_mode_allows_text_first_me_entry_content(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)

        result = parse_single_turn_generation(
            [
                QWEN3_AGENTIC_TOKEN_TABLE.entry_start,
                1000 + ord("m"),
                1000 + ord("e"),
                1000 + ord("\n"),
                *tokenizer.encode("thinking", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.entry_end,
            ],
            encoder=encoder,
            tokenizer=tokenizer,
            generation_mode=GENERATION_MODE_ENTRY,
            next_kind="me",
        )

        self.assertEqual(result.text, "thinking")
        self.assertEqual(result.display_text, "thinking")
        self.assertTrue(result.clean_termination)
        self.assertEqual(result.stop_reason, STOP_REASON_ENTRY_END)

    def test_load_lora_inference_model_uses_dtype_keyword(self) -> None:
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

        self.assertIn("dtype", captured_kwargs)
        self.assertNotIn("torch_dtype", captured_kwargs)
        self.assertEqual(captured_kwargs["dtype"], torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    def test_format_generation_result_hides_structure_tokens_and_shows_status(self) -> None:
        rendered = format_generation_result(
            SingleTurnGenerationResult(
                text="answer",
                display_text="answer",
                output_ids=[1000 + ord("a"), QWEN3_AGENTIC_TOKEN_TABLE.entry_start],
                content_ids=[1000 + ord("a")],
                stop_reason=STOP_REASON_STRUCTURE_TOKEN,
                clean_termination=False,
                stop_token_id=QWEN3_AGENTIC_TOKEN_TABLE.entry_start,
                stop_token_name="entry_start",
                termination_detail="unexpected_structure_in_payload",
            )
        )

        self.assertEqual(
            rendered,
            "answer\n[stop_reason=structure_token, clean_termination=false, detail=unexpected_structure_in_payload, token=entry_start]",
        )
        self.assertNotIn(QWEN3_AGENTIC_TOKEN_TABLE.entry_start_text, rendered)

    def test_format_generation_result_strips_only_at_display_boundary(self) -> None:
        rendered = format_generation_result(
            SingleTurnGenerationResult(
                text="  answer  ",
                display_text="answer",
                output_ids=[1000 + ord("a")],
                content_ids=[1000 + ord("a")],
                stop_reason=STOP_REASON_ENTRY_END,
                clean_termination=True,
            )
        )

        self.assertEqual(rendered, "answer")

    def test_format_generation_token_debug_names_structural_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)
        result = SingleTurnGenerationResult(
            text="<|quad_start|>SendMessage(<|box_start|>hi<|box_end|>)<|quad_end|>",
            display_text="<|quad_start|>SendMessage(<|box_start|>hi<|box_end|>)<|quad_end|>",
            output_ids=[
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
                *tokenizer.encode("SendMessage(", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
                *tokenizer.encode("hi", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                *tokenizer.encode(")", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.action_end,
            ],
            content_ids=[
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
                *tokenizer.encode("SendMessage(", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
                *tokenizer.encode("hi", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                *tokenizer.encode(")", add_special_tokens=False),
                QWEN3_AGENTIC_TOKEN_TABLE.action_end,
            ],
            stop_reason=STOP_REASON_ENTRY_END,
            clean_termination=True,
        )

        rendered = format_generation_token_debug(result, encoder=encoder, tokenizer=tokenizer)

        self.assertIn("structural_hits=opaque_payload_start=151648", rendered)
        self.assertIn("action_start=151650", rendered)
        self.assertIn("action_end=151651", rendered)
        self.assertIn("000 151650 action_start '<|quad_start|>' '' kept", rendered)
        self.assertIn("opaque_payload_start '<|box_start|>'", rendered)

    def test_format_generation_token_debug_can_truncate_long_sequences(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 42
        encoder = AgenticContextEncoder(tokenizer)
        result = SingleTurnGenerationResult(
            text="abcdef",
            display_text="abcdef",
            output_ids=tokenizer.encode("abcdef", add_special_tokens=False),
            content_ids=tokenizer.encode("abcdef", add_special_tokens=False),
            stop_reason=STOP_REASON_MAX_NEW_TOKENS,
            clean_termination=False,
        )

        rendered = format_generation_token_debug(result, encoder=encoder, tokenizer=tokenizer, max_tokens=3)

        self.assertIn("output_ids=[1097, 1098, 1099, ... truncated 3]", rendered)
        self.assertIn("... truncated 3 token rows", rendered)
        self.assertIn("002 1099 - 'c' '' kept", rendered)
        self.assertNotIn("003 1100", rendered)


if __name__ == "__main__":
    unittest.main()
