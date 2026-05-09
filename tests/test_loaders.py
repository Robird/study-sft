from __future__ import annotations

import unittest

from study_sft.loaders import get_effective_pad_token_id


class _DummyTokenizer:
    def __init__(self, *, pad_token_id=None, eos_token_id=None) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class LoaderTests(unittest.TestCase):
    def test_get_effective_pad_token_id_keeps_zero_pad_token_id(self) -> None:
        tokenizer = _DummyTokenizer(pad_token_id=0, eos_token_id=42)
        self.assertEqual(get_effective_pad_token_id(tokenizer), 0)

    def test_get_effective_pad_token_id_falls_back_to_eos(self) -> None:
        tokenizer = _DummyTokenizer(pad_token_id=None, eos_token_id=42)
        self.assertEqual(get_effective_pad_token_id(tokenizer), 42)


if __name__ == "__main__":
    unittest.main()
