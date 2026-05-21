from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
