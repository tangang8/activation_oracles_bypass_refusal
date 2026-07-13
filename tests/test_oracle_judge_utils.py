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

    def test_entry_key_is_stable_across_num_oracle_rollouts(self) -> None:
        # sampled entries key on the explicit (target, oracle) pair, so the identity does NOT move
        # when num_oracle_rollouts renumbers the flattened rollout_index.
        n2 = {"target_rollout_index": 1, "oracle_rollout_index": 0, "rollout_index": 2}
        n3 = {"target_rollout_index": 1, "oracle_rollout_index": 0, "rollout_index": 3}
        self.assertEqual(oju._entry_key(n2), oju._entry_key(n3))
        self.assertEqual(oju._entry_key(n2), "t1_o0")
        # deterministic (rollout_index only) and prompt-only (oracle_rollout_index only)
        self.assertEqual(oju._entry_key({"rollout_index": 5}), "r5")
        self.assertEqual(oju._entry_key({"oracle_rollout_index": 4}), "o4")

    def test_reusable_existing_leaf_verifies_response_and_provenance(self) -> None:
        text = "the model answer"
        sha = oju._response_sha(text)
        prov = "prov0123456789ab"
        existing = {
            "token_points": {
                "match": {"score": 1.0, "judged_response_sha": sha, "judge_provenance_sha": prov},
                "changed_text": {"score": 1.0, "judged_response_sha": "deadbeef00000000", "judge_provenance_sha": prov},
                "changed_rubric": {"score": 1.0, "judged_response_sha": sha, "judge_provenance_sha": "0000rubricchg000"},
                "legacy": {"score": 1.0},  # pre-fix leaf, no hashes
            }
        }
        # matching text AND provenance -> reuse
        self.assertEqual(
            oju._reusable_existing_leaf(existing, ("token_points", "match"), text, prov),
            existing["token_points"]["match"],
        )
        # different response text -> re-judge
        self.assertIsNone(oju._reusable_existing_leaf(existing, ("token_points", "changed_text"), text, prov))
        # same text but rubric/parser changed -> re-judge
        self.assertIsNone(oju._reusable_existing_leaf(existing, ("token_points", "changed_rubric"), text, prov))
        # legacy leaf without hashes is trusted (backward compatible)
        self.assertEqual(
            oju._reusable_existing_leaf(existing, ("token_points", "legacy"), text, prov),
            existing["token_points"]["legacy"],
        )
        # absent path -> nothing to reuse
        self.assertIsNone(oju._reusable_existing_leaf(existing, ("token_points", "missing"), text, prov))

    def test_judge_provenance_sha_changes_with_rubric_and_parser(self) -> None:
        a = oju._judge_provenance_sha("rubric text A")
        b = oju._judge_provenance_sha("rubric text B")
        self.assertNotEqual(a, b)                       # rubric edit invalidates
        self.assertEqual(a, oju._judge_provenance_sha("rubric text A"))  # stable

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
                {"compliance": {"full_seq": {"score": 0.7}, "tokens": {"0": {"score": 0.3}}}},
                {"compliance": {"full_seq": {"score": 1.0}}},
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
