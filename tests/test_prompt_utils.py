from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prompt_utils import load_oracle_prompts_from_file, prompt_key


class PromptUtilsTests(unittest.TestCase):
    def test_prompt_key_hash_length(self) -> None:
        key = prompt_key("Hello world", preview_len=5)
        preview, digest = key.rsplit("_", 1)
        self.assertEqual(preview, "Hello")
        self.assertEqual(len(digest), 12)

    def test_load_oracle_prompts_from_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prompts.json"
            p.write_text(json.dumps(["a", " ", "b"]), encoding="utf-8")
            self.assertEqual(load_oracle_prompts_from_file(str(p)), ["a", "b"])

    def test_load_oracle_prompts_from_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prompts.json"
            p.write_text(json.dumps({"oracle_prompts": ["x", "", "y"]}), encoding="utf-8")
            self.assertEqual(load_oracle_prompts_from_file(str(p)), ["x", "y"])

    def test_load_oracle_prompts_from_text_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            txt = Path(td) / "prompts.txt"
            txt.write_text("a\n\nb\n", encoding="utf-8")
            self.assertEqual(load_oracle_prompts_from_file(str(txt)), ["a", "b"])

            jsonl = Path(td) / "prompts.jsonl"
            jsonl.write_text("l1\n\nl2\n", encoding="utf-8")
            self.assertEqual(load_oracle_prompts_from_file(str(jsonl)), ["l1", "l2"])

    def test_load_oracle_prompts_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                load_oracle_prompts_from_file(str(missing))

            bad_ext = Path(td) / "bad.csv"
            bad_ext.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_oracle_prompts_from_file(str(bad_ext))

            bad_json = Path(td) / "bad.json"
            bad_json.write_text(json.dumps({"wrong": ["a"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_oracle_prompts_from_file(str(bad_json))


if __name__ == "__main__":
    unittest.main()
