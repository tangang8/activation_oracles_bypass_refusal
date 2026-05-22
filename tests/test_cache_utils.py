from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cache_utils import (
    deterministic_oracle_cache_file_path,
    deterministic_oracle_judge_cache_file_path,
    judge_cache_file_path,
    load_json,
    oracle_cache_file_path,
    oracle_prompt_rollout_cache_file_path,
    preview_hash_name,
    sanitize_for_path,
    target_rollout_cache_file_path,
    write_json,
)


class CacheUtilsTests(unittest.TestCase):
    def test_sanitize_for_path(self) -> None:
        self.assertEqual(sanitize_for_path("Qwen/Qwen3-8B"), "Qwen_Qwen3-8B")
        self.assertEqual(sanitize_for_path("  hello world!!  "), "hello_world")
        self.assertEqual(sanitize_for_path("!!!"), "unknown")

    def test_preview_hash_name_respects_lengths(self) -> None:
        key = preview_hash_name("abcdef", preview_len=3, hash_len=12)
        preview, digest = key.rsplit("_", 1)
        self.assertEqual(preview, "abc")
        self.assertEqual(len(digest), 12)

    def test_target_rollout_cache_path(self) -> None:
        path = target_rollout_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            generation_kwargs={"temperature": 1.0},
            user_prompt="hello",
            target_thinking_mode="default",
        )
        s = str(path)
        self.assertIn("target_Qwen_Qwen3-8B", s)
        self.assertIn("target_rollouts_temp-1.0", s)
        self.assertTrue(s.endswith(".json"))

        off_path = target_rollout_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            generation_kwargs={"temperature": 1.0},
            user_prompt="hello",
            target_thinking_mode="off",
        )
        self.assertIn("target-thinking-off", str(off_path))

    def test_oracle_paths(self) -> None:
        oracle_path = oracle_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            oracle_model_name="Qwen/Qwen3-8B",
            oracle_lora_path="oracle",
            generation_kwargs={"temperature": 1.0},
            oracle_prompt="oracle",
            user_prompt_preview_text="target",
            cache_key_text="cache-key",
        )
        deterministic = deterministic_oracle_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            oracle_model_name="Qwen/Qwen3-8B",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"temperature": 0.0},
            target_prompt="target",
            oracle_prompt="oracle",
        )
        prompt_only = oracle_prompt_rollout_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            oracle_model_name="Qwen/Qwen3-8B",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"temperature": 1.0},
            target_prompt="target",
            oracle_prompt="oracle",
        )
        self.assertIn("oracle_rollouts_temp-1.0", str(oracle_path))
        self.assertIn("oracle_rollouts_temp-0.0", str(deterministic))
        self.assertIn("oracle_prompt_rollouts_temp-1.0", str(prompt_only))

    def test_judge_paths(self) -> None:
        target_judge = judge_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            judge_model_name="Qwen/Qwen3-8B",
            judge_lora_path="default",
            generation_kwargs={"temperature": 1.0},
            target_thinking_mode="default",
            judge_thinking_mode="off",
            judge_instruction_stem="my/stem",
            user_prompt="prompt",
        )
        oracle_judge = deterministic_oracle_judge_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            judge_model_name="Qwen/Qwen3-8B",
            judge_lora_path="default",
            judge_generation_kwargs={"temperature": 0.0},
            judge_thinking_mode="off",
            judge_instruction_stem="my/stem",
            oracle_model_name="Qwen/Qwen3-8B",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"temperature": 1.0},
            target_prompt="target",
            oracle_prompt="oracle",
            oracle_rollouts_dir_base="oracle_prompt_rollouts",
        )
        self.assertIn("my_stem", str(target_judge))
        self.assertNotIn("thinking-off", str(target_judge))
        self.assertIn("oracle_prompt_rollouts_temp-1.0", str(oracle_judge))

        target_judge_default = judge_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            judge_model_name="Qwen/Qwen3-8B",
            judge_lora_path="default",
            generation_kwargs={"temperature": 1.0},
            target_thinking_mode="default",
            judge_thinking_mode="default",
            judge_instruction_stem="my/stem",
            user_prompt="prompt",
        )
        self.assertIn("thinking-default", str(target_judge_default))

        target_judge_target_thinking_off = judge_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            judge_model_name="Qwen/Qwen3-8B",
            judge_lora_path="default",
            generation_kwargs={"temperature": 1.0},
            target_thinking_mode="off",
            judge_thinking_mode="off",
            judge_instruction_stem="my/stem",
            user_prompt="prompt",
        )
        self.assertIn("target-thinking-off", str(target_judge_target_thinking_off))
        self.assertNotIn("thinking-default", str(target_judge_target_thinking_off))

    def test_load_and_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "nested" / "x.json"
            payload = [{"a": 1}]
            write_json(target, payload)
            self.assertEqual(load_json(target), payload)

            bad = Path(td) / "bad.json"
            bad.write_text("{bad", encoding="utf-8")
            self.assertIsNone(load_json(bad))
            self.assertIsNone(load_json(Path(td) / "missing.json"))


if __name__ == "__main__":
    unittest.main()
