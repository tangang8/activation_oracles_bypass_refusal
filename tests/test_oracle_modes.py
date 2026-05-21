from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cache_utils import oracle_prompt_rollout_cache_file_path
try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from oracle_rollout_utils import (
        ALL_TARGET_DETERMINISTIC,
        PROMPT_ONLY_REPEATS,
        SAMPLED_TARGET_REPEATS,
        generate_oracle_rollouts_for_mode,
        generate_prompt_only_oracle_rollouts,
    )
else:
    ALL_TARGET_DETERMINISTIC = "all_target_deterministic"
    PROMPT_ONLY_REPEATS = "prompt_only_repeats"
    SAMPLED_TARGET_REPEATS = "sampled_target_repeats"
    generate_oracle_rollouts_for_mode = None
    generate_prompt_only_oracle_rollouts = None

try:
    from oracle_token_points import extract_token_points_prompt_qwen
except Exception:
    extract_token_points_prompt_qwen = None


class _FakeTokenizer:
    def __init__(self) -> None:
        self.prompt_ids = [100, 101, 10, 102, 20, 30, 999]

    def __call__(self, text: str, return_tensors: str, add_special_tokens: bool) -> dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("torch is unavailable")
        del text, return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([self.prompt_ids], dtype=torch.long)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        mapping = {
            "<|im_end|>": [10],
            "<|im_start|>": [20],
            "assistant": [30],
        }
        return mapping.get(text, [0])


class OracleCachePathTests(unittest.TestCase):
    def test_prompt_only_cache_path_layout(self) -> None:
        path = oracle_prompt_rollout_cache_file_path(
            cache_root="cache",
            target_model_name="Qwen/Qwen3-8B",
            target_lora_path="default",
            oracle_model_name="Qwen/Qwen3-8B",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"temperature": 1.0},
            target_prompt="How do we test cache paths?",
            oracle_prompt="You are an oracle.",
        )
        path_str = str(path)
        self.assertIn("oracle_prompt_rollouts_temp-1.0", path_str)
        self.assertIn("oracle_Qwen_Qwen3-8B_lora-oracle", path_str)
        self.assertTrue(path_str.endswith(".json"))


@unittest.skipIf(torch is None or extract_token_points_prompt_qwen is None, "token extractor deps unavailable")
class TokenPointExtractorTests(unittest.TestCase):
    def test_qwen_prompt_extractor_points(self) -> None:
        tokenizer = _FakeTokenizer()
        spec = extract_token_points_prompt_qwen(tokenizer, "formatted prompt")
        self.assertEqual(spec["rollout_len"], 0)
        self.assertEqual(spec["prompt_segment"], (0, len(tokenizer.prompt_ids)))
        self.assertEqual(spec["rollout_segment"], (len(tokenizer.prompt_ids), len(tokenizer.prompt_ids)))
        self.assertEqual(
            set(spec["token_points"].keys()),
            {
                "im_end_token",
                "token_before_im_end",
                "token_after_im_end",
                "trailing_im_start_token",
                "trailing_assistant_token",
                "last_prompt_token",
            },
        )


@unittest.skipIf(torch is None or generate_oracle_rollouts_for_mode is None, "oracle mode deps unavailable")
class OracleModeRoutingTests(unittest.TestCase):
    def test_mode_router_dispatches_to_expected_generator(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-8B"))
        tokenizer = object()
        device = torch.device("cpu")

        with (
            patch("oracle_rollout_utils.generate_deterministic_oracle_rollouts", return_value=([], Path("det.json"), {})) as deterministic_mock,
            patch("oracle_rollout_utils.generate_sampled_target_oracle_rollouts", return_value=([], Path("sampled.json"), {})) as sampled_mock,
            patch("oracle_rollout_utils.generate_prompt_only_oracle_rollouts", return_value=([], Path("prompt.json"), {})) as prompt_mock,
        ):
            generate_oracle_rollouts_for_mode(
                mode=ALL_TARGET_DETERMINISTIC,
                model=model,
                tokenizer=tokenizer,
                device=device,
                oracle_prompt="o",
                target_prompt="t",
                target_rollout_entries=[],
                target_model_name="Qwen/Qwen3-8B",
                target_lora_path="default",
            )
            generate_oracle_rollouts_for_mode(
                mode=SAMPLED_TARGET_REPEATS,
                model=model,
                tokenizer=tokenizer,
                device=device,
                oracle_prompt="o",
                target_prompt="t",
                target_rollout_entries=[],
                target_model_name="Qwen/Qwen3-8B",
                target_lora_path="default",
            )
            generate_oracle_rollouts_for_mode(
                mode=PROMPT_ONLY_REPEATS,
                model=model,
                tokenizer=tokenizer,
                device=device,
                oracle_prompt="o",
                target_prompt="t",
                target_rollout_entries=[],
                target_model_name="Qwen/Qwen3-8B",
                target_lora_path="default",
            )

        deterministic_mock.assert_called_once()
        sampled_mock.assert_called_once()
        prompt_mock.assert_called_once()

    def test_prompt_only_mode_forces_temp_one_and_prompt_cache(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-8B"))
        tokenizer = object()
        oracle_result = {
            "oracle_repeats": 2,
            "combined_text": "FORMATTED_PROMPT",
            "points": {"combined_text": "FORMATTED_PROMPT", "token_points": {"last_prompt_token": 1}},
            "full_seq": ["full_0", "full_1"],
            "segment": ["segment_0", "segment_1"],
            "prompt_segment": ["prompt_0", "prompt_1"],
            "rollout_segment": [],
            "tokens": {},
            "token_points": {1: ["point_0", "point_1"]},
        }

        with (
            patch("oracle_rollout_utils.format_user_target_prompt", return_value="FORMATTED_PROMPT"),
            patch("oracle_rollout_utils.oracle_prompt_rollout_cache_file_path", return_value=Path("cache/prompt.json")) as cache_path_mock,
            patch("oracle_rollout_utils.load_json", return_value=[]),
            patch("oracle_rollout_utils.write_json"),
            patch("oracle_rollout_utils.run_oracle_batched", return_value=[oracle_result]) as run_oracle_mock,
        ):
            entries, cache_file, _ = generate_prompt_only_oracle_rollouts(
                model=model,
                tokenizer=tokenizer,
                device=torch.device("cpu"),
                oracle_prompt="oracle prompt",
                target_prompt="target prompt",
                target_model_name="Qwen/Qwen3-8B",
                target_lora_path="default",
                num_oracle_rollouts=2,
                oracle_generation_kwargs={"temperature": 0.25},
            )

        self.assertEqual(cache_file, Path("cache/prompt.json"))
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["oracle_rollout_index"], 0)
        self.assertEqual(entries[1]["oracle_rollout_index"], 1)
        cache_path_mock.assert_called_once()
        run_oracle_kwargs = run_oracle_mock.call_args.kwargs
        self.assertEqual(run_oracle_kwargs["generation_kwargs"]["temperature"], 1.0)
        self.assertTrue(run_oracle_kwargs["generation_kwargs"]["do_sample"])
        self.assertEqual(run_oracle_kwargs["oracle_input_source_type"], "prompt_only")

    def test_prompt_only_cache_with_rollout_index_only_raises(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-8B"))
        tokenizer = object()

        with (
            patch("oracle_rollout_utils.format_user_target_prompt", return_value="FORMATTED_PROMPT"),
            patch("oracle_rollout_utils.oracle_prompt_rollout_cache_file_path", return_value=Path("cache/prompt.json")),
            patch(
                "oracle_rollout_utils.load_json",
                return_value=[{"rollout_index": 0, "oracle_response": {}, "oracle_format": {}}],
            ),
        ):
            with self.assertRaises(ValueError):
                generate_prompt_only_oracle_rollouts(
                    model=model,
                    tokenizer=tokenizer,
                    device=torch.device("cpu"),
                    oracle_prompt="oracle prompt",
                    target_prompt="target prompt",
                    target_model_name="Qwen/Qwen3-8B",
                    target_lora_path="default",
                    num_oracle_rollouts=2,
                    oracle_generation_kwargs={"temperature": 0.25},
                )


if __name__ == "__main__":
    unittest.main()
