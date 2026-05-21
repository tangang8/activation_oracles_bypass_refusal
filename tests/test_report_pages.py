from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import report_pages
except Exception:
    report_pages = None


class _FakeTokenizer:
    class _Ids(list):
        def tolist(self):
            return list(self)

    def __call__(self, input_text, return_tensors, add_special_tokens, padding):
        del input_text, return_tensors, add_special_tokens, padding
        return {"input_ids": [self._Ids([1, 2, 3])]}

    def decode(self, token_ids):
        return f"tok{token_ids[0]}"


@unittest.skipIf(report_pages is None, "report_pages dependencies unavailable")
class ReportPagesTests(unittest.TestCase):
    def test_entry_rollout_label(self) -> None:
        self.assertEqual(report_pages._entry_rollout_label({"rollout_index": 1}, 9), "1")
        self.assertEqual(report_pages._entry_rollout_label({"oracle_rollout_index": 2}, 9), "2")
        self.assertEqual(report_pages._entry_rollout_label({}, 9), "9")

    def test_save_rollouts_html(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = report_pages.save_rollouts_html(
                rollout_entries=[
                    {"rollout_index": 0, "compliance": {"score": 1, "reason": "r"}, "target_format": {"response_only": "x", "thinking": ""}}
                ],
                compliance_results={"compliance_rate": 0.1, "partial_compliance_rate": 0.2, "total": 1},
                output_path=str(Path(td) / "rollouts.html"),
            )
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("Rollouts Report", text)
            self.assertIn("Rollout 0", text)

    def test_save_oracle_rollouts_html_prompt_only_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = report_pages.save_oracle_rollouts_html(
                oracle_results=[
                    {
                        "oracle_rollout_index": 0,
                        "formatted_target_prompt": "fp",
                        "oracle_response": {"full_seq": "x", "token_points": {"last_prompt_token": "y"}},
                        "oracle_format": {"full_seq": {"response_only": "x"}, "token_points": {"last_prompt_token": {"response_only": "y"}}},
                        "oracle_points": {"combined_text": "fp", "token_points": {"last_prompt_token": 1}},
                        "compliance": {},
                    }
                ],
                oracle_prompt="oracle",
                tokenizer=_FakeTokenizer(),
                output_path=str(Path(td) / "oracle.html"),
            )
            self.assertTrue(out.exists())
            html = out.read_text(encoding="utf-8")
            self.assertIn("Oracle Rollouts Report", html)
            self.assertIn("Formatted Target Prompt", html)
            self.assertIn("last_prompt_token", html)


if __name__ == "__main__":
    unittest.main()
