from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from compile_results import compile_cache_results


class CompileResultsTests(unittest.TestCase):
    def test_source_separation_and_all_null_probe_omission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_root = root / "cache"
            output_dir = root / "compiled"

            prompt_only_file = (
                cache_root
                / "target_ModelA"
                / "judge_JudgeA_temp-0.0"
                / "instruction_x"
                / "oracle_rollouts_judged"
                / "oracle_prompt_rollouts_temp-1.0"
                / "oracle_OracleA"
                / "prompt_key_1"
                / "oracle_key_1.json"
            )
            prompt_only_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_only_file.write_text(
                json.dumps(
                    [
                        {
                            "oracle_rollout_index": 0,
                            "compliance": {
                                "full_seq": {"score": 1},
                                "segment": {"score": None, "judge_skipped": True},
                                "prompt_segment": {"score": 2},
                                "rollout_segment": {"score": None, "judge_skipped": True},
                                "token_points": {"last_prompt_token": {"score": 3}},
                                "tokens": {},
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            target_backed_file = (
                cache_root
                / "target_ModelA"
                / "judge_JudgeA_temp-0.0"
                / "instruction_x"
                / "oracle_rollouts_judged"
                / "oracle_rollouts_temp-1.0"
                / "oracle_OracleA"
                / "prompt_key_1"
                / "oracle_key_1.json"
            )
            target_backed_file.parent.mkdir(parents=True, exist_ok=True)
            target_backed_file.write_text(
                json.dumps(
                    [
                        {
                            "rollout_index": 0,
                            "target_rollout_index": 0,
                            "oracle_rollout_index": 0,
                            "compliance": {
                                "full_seq": {"score": 4},
                                "segment": {"score": None},
                                "prompt_segment": {"score": 4},
                                "rollout_segment": {"score": 5},
                                "token_points": {},
                                "tokens": {},
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            target_file = (
                cache_root
                / "target_ModelA"
                / "judge_JudgeA_temp-0.0"
                / "instruction_x"
                / "target_rollouts_judged"
                / "prompt_key_1.json"
            )
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(
                json.dumps(
                    [
                        {
                            "rollout_index": 0,
                            "compliance": {"score": 3},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            manifest = compile_cache_results(cache_root=cache_root, output_dir=output_dir)
            self.assertEqual(manifest["oracle_judged_files"], 2)
            self.assertEqual(manifest["target_judged_files"], 1)

            aggregate_csv = output_dir / "classification_aggregates.csv"
            with aggregate_csv.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertTrue(any(r["rollout_source"] == "prompt_only_oracle" for r in rows))
            self.assertTrue(any(r["rollout_source"] == "target_backed_oracle" for r in rows))
            self.assertTrue(any(r["rollout_source"] == "target_rollout" for r in rows))

            # Prompt-only "segment" is all-null in this fixture, so it should be omitted.
            self.assertFalse(
                any(r["rollout_source"] == "prompt_only_oracle" and r["probe_kind"] == "segment" for r in rows)
            )
            # But prompt-only "full_seq" and token point should remain.
            self.assertTrue(
                any(r["rollout_source"] == "prompt_only_oracle" and r["probe_kind"] == "full_seq" for r in rows)
            )
            self.assertTrue(
                any(
                    r["rollout_source"] == "prompt_only_oracle"
                    and r["probe_kind"] == "token_points"
                    and r["probe_name"] == "last_prompt_token"
                    for r in rows
                )
            )

    def test_float_scores_are_preserved_in_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_root = root / "cache"
            output_dir = root / "compiled"

            target_file = (
                cache_root
                / "target_ModelA"
                / "judge_JudgeA_temp-0.0"
                / "strongReject"
                / "target_rollouts_judged"
                / "prompt_key_1.json"
            )
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(
                json.dumps(
                    [
                        {"rollout_index": 0, "compliance": {"score": 0.7}},
                        {"rollout_index": 1, "compliance": {"score": 0.3}},
                    ]
                ),
                encoding="utf-8",
            )

            compile_cache_results(cache_root=cache_root, output_dir=output_dir)

            aggregate_csv = output_dir / "classification_aggregates.csv"
            with aggregate_csv.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            target_rows = [r for r in rows if r["rollout_source"] == "target_rollout"]
            self.assertEqual(len(target_rows), 1)
            self.assertEqual(target_rows[0]["n_scored"], "2")
            self.assertEqual(target_rows[0]["mean_score"], "0.5")


if __name__ == "__main__":
    unittest.main()
