from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cache_utils import (
    deterministic_oracle_cache_file_path,
    deterministic_oracle_judge_cache_file_path,
    effective_variant_k_rollouts,
    judge_cache_file_path,
    load_json,
    oracle_cache_file_path,
    oracle_cache_variant_key,
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

    def test_effective_variant_k_rollouts(self) -> None:
        # k belongs in the key only when it actually restricts the set
        self.assertEqual(effective_variant_k_rollouts(5, 50), 5)   # k < count -> keep
        self.assertIsNone(effective_variant_k_rollouts(50, 50))    # k == count -> no-op
        self.assertIsNone(effective_variant_k_rollouts(80, 50))    # k > count -> no-op
        self.assertIsNone(effective_variant_k_rollouts(None, 50))  # unset

    def test_oracle_cache_variant_key_single_source(self) -> None:
        # default axis -> None (default namespace)
        self.assertIsNone(oracle_cache_variant_key(None, "all", None))
        # any non-default component -> stable, sorted JSON
        key = oracle_cache_variant_key(["rollout_segment", "token_points"], "post_prompt", 5)
        self.assertEqual(
            key,
            '{"k_rollouts": 5, "oracle_input_types": ["rollout_segment", "token_points"], '
            '"oracle_token_point_filter": "post_prompt"}',
        )
        # k omitted from the dict when None even if other axes set
        self.assertNotIn("k_rollouts", oracle_cache_variant_key(None, "post_prompt", None))

    def test_oracle_cache_variant_key_max_new_tokens(self) -> None:
        # The 1000 baseline is omitted, so pre-existing on-disk namespaces are preserved.
        self.assertIsNone(oracle_cache_variant_key(None, "all", None, max_new_tokens=1000))
        self.assertIsNone(oracle_cache_variant_key(None, "all", None, max_new_tokens=None))
        legacy_key = oracle_cache_variant_key(["rollout_segment", "token_points"], "post_prompt", 5)
        self.assertEqual(
            legacy_key,
            oracle_cache_variant_key(
                ["rollout_segment", "token_points"], "post_prompt", 5, max_new_tokens=1000
            ),
        )
        # A non-baseline cap forks the namespace — alone or combined with other axes.
        solo = oracle_cache_variant_key(None, "all", None, max_new_tokens=128)
        self.assertIn('"max_new_tokens": 128', solo)
        combined = oracle_cache_variant_key(
            ["rollout_segment", "token_points"], "post_prompt", 5, max_new_tokens=128
        )
        self.assertNotEqual(legacy_key, combined)
        self.assertIn('"max_new_tokens": 128', combined)

    def test_prompt_only_path_variant_suffix(self) -> None:
        common = dict(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            oracle_model_name="Qwen/Qwen3-8B",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"temperature": 1.0},
            target_prompt="target",
            oracle_prompt="oracle",
        )
        default_path = oracle_prompt_rollout_cache_file_path(**common)
        # No variant -> unchanged legacy path (existing prompt-only caches stay reachable).
        self.assertEqual(default_path, oracle_prompt_rollout_cache_file_path(**common, cache_variant_key=None))
        variant_path = oracle_prompt_rollout_cache_file_path(
            **common, cache_variant_key='{"max_new_tokens": 128}'
        )
        self.assertNotEqual(default_path, variant_path)
        self.assertIn("__", variant_path.name)
        self.assertEqual(variant_path.parent, default_path.parent)

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
