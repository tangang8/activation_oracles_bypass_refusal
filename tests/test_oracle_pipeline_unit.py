from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import oracle_pipeline
except Exception:
    oracle_pipeline = None


@unittest.skipIf(oracle_pipeline is None, "oracle_pipeline dependencies unavailable")
class OraclePipelineUnitTests(unittest.TestCase):
    def test_aggregate_oracle_repeat_entries(self) -> None:
        merged = oracle_pipeline._aggregate_oracle_repeat_entries(
            [
                {
                    "combined_text": "x",
                    "points": {"token_points": {"a": 1}},
                    "full_seq": ["f1"],
                    "segment": ["s1"],
                    "prompt_segment": ["p1"],
                    "rollout_segment": ["r1"],
                    "tokens": {1: ["t1"]},
                    "token_points": {1: ["tp1"]},
                },
                {
                    "combined_text": "x",
                    "points": {"token_points": {"a": 1}},
                    "full_seq": ["f2"],
                    "segment": ["s2"],
                    "prompt_segment": ["p2"],
                    "rollout_segment": ["r2"],
                    "tokens": {1: ["t2"], 2: ["t3"]},
                    "token_points": {1: ["tp2"]},
                },
            ]
        )
        self.assertEqual(merged["oracle_repeats"], 2)
        self.assertEqual(merged["full_seq"], ["f1", "f2"])
        self.assertEqual(merged["tokens"][1], ["t1", "t2"])
        self.assertEqual(merged["tokens"][2], ["t3"])
        self.assertEqual(merged["token_points"][1], ["tp1", "tp2"])

    def test_aggregate_empty(self) -> None:
        self.assertEqual(oracle_pipeline._aggregate_oracle_repeat_entries([]), {})

    def test_oracle_input_source_labels(self) -> None:
        self.assertEqual(
            oracle_pipeline._oracle_input_source(None),
            ("prompt_only", "prompt_input_index"),
        )
        self.assertEqual(
            oracle_pipeline._oracle_input_source(["response"]),
            ("target_rollout", "target_rollout_index"),
        )

    def test_run_oracle_batched_default_generation_kwargs_use_1000_tokens(self) -> None:
        model = type("Model", (), {})()
        model.config = type("Config", (), {"_name_or_path": "test-model"})()

        def cached_entry(text: str) -> dict:
            return {
                "combined_text": "formatted",
                "points": {"token_points": {}},
                "full_seq": [text],
                "segment": [],
                "prompt_segment": [],
                "rollout_segment": [],
                "tokens": {},
                "token_points": {},
            }

        with (
            patch(
                "oracle_pipeline.build_prompt_only_points_spec",
                return_value={
                    "combined_text": "formatted",
                    "prompt_segment": (0, 1),
                    "rollout_segment": (1, 1),
                    "token_point_indices": [],
                },
            ),
            patch("oracle_pipeline.oracle_cache_file_path", return_value="cache.json") as cache_path_mock,
            patch("oracle_pipeline.load_json", return_value=[cached_entry("cached_0"), cached_entry("cached_1")]),
        ):
            for oracle_repeats, expected_do_sample, expected_temperature in (
                (1, False, 0.0),
                (2, True, 1.0),
            ):
                oracle_pipeline.run_oracle_batched(
                    model=model,
                    tokenizer=object(),
                    device=object(),
                    formatted_target_prompts=["formatted"],
                    target_responses=None,
                    oracle_prompt="oracle",
                    oracle_input_types=["full_seq"],
                    oracle_repeats=oracle_repeats,
                )
                generation_kwargs = cache_path_mock.call_args.kwargs["generation_kwargs"]
                self.assertEqual(generation_kwargs["max_new_tokens"], 1000)
                self.assertEqual(generation_kwargs["do_sample"], expected_do_sample)
                self.assertEqual(generation_kwargs["temperature"], expected_temperature)

    def test_run_oracle_batched_source_type_validation(self) -> None:
        with self.assertRaises(ValueError):
            oracle_pipeline.run_oracle_batched(
                model=object(),
                tokenizer=object(),
                device=object(),
                formatted_target_prompts=["formatted"],
                target_responses=["response"],
                oracle_prompt="oracle",
                oracle_input_source_type="prompt_only",
            )


if __name__ == "__main__":
    unittest.main()
