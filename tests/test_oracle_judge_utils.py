from __future__ import annotations

import unittest

try:
    import oracle_judge_utils as oju
except Exception:
    oju = None


@unittest.skipIf(oju is None, "oracle_judge_utils dependencies unavailable")
class OracleJudgeUtilsTests(unittest.TestCase):
    def test_entry_index(self) -> None:
        self.assertEqual(oju._entry_index({"rollout_index": 2}), 2)
        self.assertEqual(oju._entry_index({"oracle_rollout_index": 3}), 3)
        with self.assertRaises(KeyError):
            oju._entry_index({})

    def test_flatten_oracle_responses(self) -> None:
        entry = {
            "oracle_rollout_index": 7,
            "target_prompt": "u",
            "oracle_response": {
                "full_seq": "a",
                "segment": "b",
                "prompt_segment": "c",
                "rollout_segment": "d",
                "tokens": {"1": "t1"},
                "token_points": {"p": "tp"},
            },
            "oracle_format": {},
        }
        flat = oju._flatten_oracle_responses(entry)
        kinds = [x["probe_kind"] for x in flat]
        self.assertIn("full_seq", kinds)
        self.assertIn("segment", kinds)
        self.assertIn("prompt_segment", kinds)
        self.assertIn("rollout_segment", kinds)
        self.assertIn("tokens", kinds)
        self.assertIn("token_points", kinds)
        self.assertTrue(all(x["rollout_index"] == 7 for x in flat))

    def test_compliance_shell(self) -> None:
        shell = oju._compliance_shell(
            {
                "oracle_response": {
                    "full_seq": "x",
                    "tokens": {"1": "a"},
                    "token_points": {"k": "b"},
                }
            }
        )
        self.assertIn("full_seq", shell)
        self.assertEqual(shell["tokens"], {"1": None})
        self.assertEqual(shell["token_points"], {"k": None})

    def test_oracle_judge_summary(self) -> None:
        summary = oju._oracle_judge_summary(
            [
                {"compliance": {"full_seq": {"score": 2}, "tokens": {"0": {"score": 4}}}},
                {"compliance": {"full_seq": {"score": 4}}},
            ]
        )
        self.assertEqual(summary["oracle_judge/total_scored"], 3.0)
        self.assertIn("oracle_judge/full_seq_avg_score", summary)

    def test_oracle_judge_item_id_prompt_only(self) -> None:
        item_id = oju._oracle_judge_item_id(
            {
                "rollout_index": 2,
                "source_index_label": "oracle_rollout_index",
                "probe_kind": "full_seq",
            }
        )
        self.assertEqual(item_id, "oracle_rollout_index=2 probe=full_seq")

    def test_oracle_judge_item_id_target_backed(self) -> None:
        item_id = oju._oracle_judge_item_id(
            {
                "rollout_index": 5,
                "target_rollout_index": 9,
                "oracle_rollout_index": 3,
                "probe_kind": "token_points",
                "token_point_name": "last_prompt_token",
            }
        )
        self.assertEqual(
            item_id,
            "target_rollout_index=9 oracle_rollout_index=3 probe=token_points:last_prompt_token",
        )


if __name__ == "__main__":
    unittest.main()
