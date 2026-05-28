from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cache_utils import deterministic_oracle_judge_cache_file_path, judge_cache_file_path
from compile_results import compile_cache_results
from compile_strongreject_results import ROLLOUT_POST_PROMPT_VARIANT


class CompileResultsTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _target_path(self, root: Path, prompt: str, *, target_lora_path: str, target_thinking_mode: str) -> Path:
        return judge_cache_file_path(
            cache_root=str(root / "cache"),
            target_model_name="ModelA",
            target_lora_path=target_lora_path,
            judge_model_name="JudgeA",
            judge_lora_path="default",
            generation_kwargs={"temperature": 0.0},
            judge_thinking_mode="off",
            target_thinking_mode=target_thinking_mode,
            judge_instruction_stem="strongReject_v5",
            user_prompt=prompt,
        )

    def _oracle_path(self, root: Path, prompt: str, oracle_prompt: str, *, dir_base: str, temp: float) -> Path:
        return deterministic_oracle_judge_cache_file_path(
            cache_root=str(root / "cache"),
            target_model_name="ModelA",
            target_lora_path="default",
            judge_model_name="JudgeA",
            judge_lora_path="default",
            judge_generation_kwargs={"temperature": 0.0},
            judge_thinking_mode="off",
            judge_instruction_stem="strongReject_v5",
            oracle_model_name="OracleA",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"temperature": temp},
            target_prompt=prompt,
            oracle_prompt=oracle_prompt,
            oracle_rollouts_dir_base=dir_base,
            cache_variant_key=ROLLOUT_POST_PROMPT_VARIANT if dir_base == "oracle_rollouts" else None,
        )

    def test_compile_results_delegates_to_strongreject_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target_prompt = "harmful prompt"
            oracle_prompt = "oracle prompt"

            self._write_json(
                self._target_path(root, target_prompt, target_lora_path="default", target_thinking_mode="default"),
                [
                    {"rollout_index": 0, "compliance": {"score": 0.0, "score_scale": "strongreject_0_1"}},
                    {"rollout_index": 1, "compliance": {"score": 0.2, "score_scale": "strongreject_0_1"}},
                ],
            )
            self._write_json(
                self._target_path(root, target_prompt, target_lora_path="oracle", target_thinking_mode="off"),
                [
                    {"rollout_index": 0, "compliance": {"score": 0.1, "score_scale": "strongreject_0_1"}},
                    {"rollout_index": 1, "compliance": {"score": 0.3, "score_scale": "strongreject_0_1"}},
                ],
            )
            self._write_json(
                self._oracle_path(root, target_prompt, oracle_prompt, dir_base="oracle_prompt_rollouts", temp=1.0),
                [
                    {
                        "oracle_rollout_index": 0,
                        "compliance": {"full_seq": {"score": 0.4, "score_scale": "strongreject_0_1"}},
                    },
                    {
                        "oracle_rollout_index": 1,
                        "compliance": {"full_seq": {"score": 0.6, "score_scale": "strongreject_0_1"}},
                    },
                ],
            )
            self._write_json(
                self._oracle_path(root, target_prompt, oracle_prompt, dir_base="oracle_rollouts", temp=0.0),
                [
                    {
                        "target_rollout_index": 0,
                        "oracle_rollout_index": 0,
                        "compliance": {"rollout_segment": {"score": 0.7, "score_scale": "strongreject_0_1"}},
                    },
                    {
                        "target_rollout_index": 1,
                        "oracle_rollout_index": 0,
                        "compliance": {"rollout_segment": {"score": 0.9, "score_scale": "strongreject_0_1"}},
                    },
                ],
            )

            manifest = compile_cache_results(
                cache_root=root / "cache",
                output_dir=root / "compiled",
                target_model_name="ModelA",
                judge_model_name="JudgeA",
                oracle_model_name="OracleA",
                expected_target_prompts=1,
                expected_target_rollouts=2,
                expected_oracle_rollouts=2,
                oracle_prompts_paths=("oracle_a.json",),
                target_prompts=[target_prompt],
                oracle_prompts_by_file={"oracle_a.json": [oracle_prompt]},
            )

            self.assertEqual(manifest["loaded_files"]["target_baseline"], 1)
            self.assertEqual(manifest["loaded_files"]["oracle_rollout_control"], 1)
            self.assertEqual(manifest["loaded_files"]["user_prompt_oracle"], 1)
            self.assertEqual(manifest["loaded_files"]["target_rollout_oracle"], 1)
            self.assertIn("summary_csv", manifest["outputs"])
            self.assertNotIn("aggregate_csv", manifest["outputs"])

            with (root / "compiled" / "strongreject_summary.csv").open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rollout = [
                r
                for r in rows
                if r["condition"] == "target_rollout_oracle" and r["probe_name"] == "rollout_segment"
            ][0]
            self.assertEqual(rollout["mean_score"], "0.8")


if __name__ == "__main__":
    unittest.main()
