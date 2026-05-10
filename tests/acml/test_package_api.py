from __future__ import annotations

import unittest

import acml


class ACMLPackageAPITests(unittest.TestCase):
    def test_root_api_exports_only_text_io_surface(self) -> None:
        self.assertEqual(
            set(acml.__all__),
            {"ACMLError", "ACMLParseError", "parse_document", "serialize_document"},
        )


if __name__ == "__main__":
    unittest.main()
