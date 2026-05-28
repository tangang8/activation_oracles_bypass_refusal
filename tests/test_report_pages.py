from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import report_pages


class ReportPagesTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_save_strongreject_website(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            compiled = root / "compiled"
            website = root / "website"

            self._write_csv(
                compiled / "strongreject_summary.csv",
                [
                    {
                        "condition": "target_rollout_oracle",
                        "probe_name": "rollout_segment",
                        "oracle_prompt_file": "prompts/oracle_prompts/default_oracle_prompts.json",
                        "n_prompts": 1,
                        "mean_score": 0.8,
                        "se_score": 0.0,
                        "asr_0": 1.0,
                        "asr_0_se": 0.0,
                        "asr_0_3": 1.0,
                        "asr_0_3_se": 0.0,
                        "asr_0_5": 1.0,
                        "asr_0_5_se": 0.0,
                        "asr_0_8": 1.0,
                        "asr_0_8_se": 0.0,
                        "asr_1": 0.0,
                        "asr_1_se": 0.0,
                    }
                ],
            )
            self._write_csv(
                compiled / "strongreject_reliability.csv",
                [
                    {
                        "condition": "target_rollout_oracle",
                        "probe_name": "rollout_segment",
                        "oracle_prompt_file": "prompts/oracle_prompts/default_oracle_prompts.json",
                        "n_prompts_with_sd": 1,
                        "mean_within_prompt_sd_oracle_rollouts": "",
                        "mean_within_prompt_sd_target_rollouts": 0.1,
                        "mean_within_prompt_n": 2,
                    }
                ],
            )
            self._write_csv(
                compiled / "strongreject_details.csv",
                [
                    {
                        "condition": "target_rollout_oracle",
                        "target_prompt_index": 0,
                        "probe_name": "rollout_segment",
                        "oracle_prompt_file": "prompts/oracle_prompts/default_oracle_prompts.json",
                        "target_rollout_index": 0,
                        "oracle_rollout_index": 0,
                        "score": 0.8,
                        "target_prompt": "harmful prompt",
                        "cache_path": "/tmp/cache/file.json",
                    }
                ],
            )
            (compiled / "manifest.json").write_text(
                json.dumps(
                    {
                        "detail_row_count": 1,
                        "prompt_level_row_count": 1,
                        "summary_row_count": 1,
                        "reliability_row_count": 1,
                        "missing_files": [],
                        "malformed_files": [],
                        "skipped_score_leaves": [],
                        "coverage_warnings": [],
                        "outputs": {
                            "summary_csv": str(compiled / "strongreject_summary.csv"),
                            "details_csv": str(compiled / "strongreject_details.csv"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            out = report_pages.save_strongreject_website(compiled_dir=compiled, output_dir=website)
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("StrongReject Results", text)
            self.assertIn("Target Rollout Oracle", text)
            self.assertIn("Oracle Prompt A", text)
            self.assertIn("80.0%", text)

    def test_missing_compiled_outputs_raise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                report_pages.save_strongreject_website(compiled_dir=Path(td) / "missing", output_dir=Path(td) / "site")


if __name__ == "__main__":
    unittest.main()
