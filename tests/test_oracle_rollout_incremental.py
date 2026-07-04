from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import oracle_rollout_utils as oru
    from cache_utils import load_json
    _IMPORT_OK = True
except Exception:
    oru = None
    load_json = None
    _IMPORT_OK = False


def _target_entries(n: int) -> list[dict]:
    return [
        {"rollout_index": i, "target_prompt": "P", "target_response": f"R{i}", "target_format": {}}
        for i in range(n)
    ]


def _fake_full_result(_prompt: str, response: str, points: dict[str, int]) -> dict:
    """A run_oracle_batched-style result for a full (all-probe) compute."""
    return {
        "full_seq": [f"FS::{response}"],
        "segment": [],
        "prompt_segment": [],
        "rollout_segment": [f"RS::{response}"],
        "tokens": {},
        "token_points": {idx: [f"TP{idx}::{response}"] for idx in points.values()},
        "points": {
            "token_points": dict(points),
            "token_point_indices": sorted(points.values()),
            "token_point_str": {name: f"str{idx}" for name, idx in points.items()},
            "combined_text": f"P{response}",
            "prompt_len": 1,
            "combined_len": 40,
        },
        "oracle_prompt": "OP",
    }


@unittest.skipIf(not _IMPORT_OK, "oracle_rollout_utils dependencies unavailable")
class MergeTokenPointsHelperTests(unittest.TestCase):
    def test_merge_adds_missing_only_and_refreshes_points(self) -> None:
        entry = {
            "oracle_response": {"full_seq": "FS", "token_points": {"a": "A", "b": "B"}},
            "oracle_format": {"token_points": {"a": {}, "b": {}}},
            "oracle_points": {"token_points": {"a": 1, "b": 2}, "token_point_indices": [1, 2]},
        }
        result = {
            "token_points": {3: ["C-DECODE"]},
            "points": {"token_points": {"a": 1, "b": 2, "c": 3}, "token_point_indices": [1, 2, 3]},
        }
        oru._merge_token_points_into_entry(entry, result, {"c": 3})

        tp = entry["oracle_response"]["token_points"]
        self.assertEqual(tp["c"], "C-DECODE")            # new point added
        self.assertEqual(tp["a"], "A")                   # existing untouched
        self.assertEqual(tp["b"], "B")
        self.assertEqual(entry["oracle_response"]["full_seq"], "FS")  # segment untouched
        self.assertIn("c", entry["oracle_format"]["token_points"])
        # oracle_points refreshed to include the new point
        self.assertEqual(entry["oracle_points"]["token_point_indices"], [1, 2, 3])


@unittest.skipIf(not _IMPORT_OK, "oracle_rollout_utils dependencies unavailable")
class DeterministicIncrementalTokenPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_root = self.tmp.name
        self.model = SimpleNamespace(config=SimpleNamespace(_name_or_path="test/model"))
        self.tokenizer = object()
        self.targets = _target_entries(2)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self):
        return oru.generate_deterministic_oracle_rollouts(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            oracle_prompt="OP",
            target_rollout_entries=self.targets,
            target_model_name="test/model",
            target_lora_path=None,
            oracle_lora_path="oracle",
            cache_root=self.cache_root,
        )

    def test_incremental_add_of_new_token_point(self) -> None:
        old_points = {"a": 1, "b": 2}
        new_points = {"a": 1, "b": 2, "c": 3}  # 'c' added later

        with mock.patch.object(oru, "format_user_target_prompt", return_value="FP"), \
             mock.patch.object(oru, "run_oracle_batched") as rob, \
             mock.patch.object(oru, "_required_combined_token_points") as req:

            # --- Phase 1: fresh run, both rollouts missing -> full compute of a,b ---
            req.return_value = dict(old_points)
            rob.side_effect = lambda **kw: [
                _fake_full_result("FP", r, old_points) for r in kw["target_responses"]
            ]
            entries1, cache_file, _ = self._run()
            self.assertEqual(len(entries1), 2)
            self.assertEqual(rob.call_args.kwargs["oracle_input_types"],
                             ["full_seq", "prompt_segment", "rollout_segment", "token_points"])

            # --- Phase 2: extractor now emits 'c' -> only 'c' should be computed & merged ---
            rob.reset_mock()
            req.return_value = dict(new_points)
            rob.side_effect = lambda **kw: [
                {"token_points": {3: [f"C::{r}"]},
                 "points": {"token_points": dict(new_points),
                            "token_point_indices": [1, 2, 3],
                            "token_point_str": {"a": "s1", "b": "s2", "c": "s3"}}}
                for r in kw["target_responses"]
            ]
            entries2, _, stats2 = self._run()

            # incremental call happened, token-points-only, just index 3
            self.assertEqual(rob.call_count, 1)
            call = rob.call_args.kwargs
            self.assertEqual(call["oracle_input_types"], ["token_points"])
            self.assertEqual(call["token_point_indices_by_target"], [[3], [3]])
            self.assertEqual(stats2["cache/oracle_incomplete"], 2.0)

            # new point merged, existing decodes preserved (segments NOT recomputed)
            for e in entries2:
                tp = e["oracle_response"]["token_points"]
                self.assertEqual(set(tp), {"a", "b", "c"})
                self.assertTrue(e["oracle_response"]["full_seq"].startswith("FS::"))
                self.assertTrue(tp["c"].startswith("C::"))

            # persisted to disk too
            on_disk = load_json(Path(cache_file))
            self.assertEqual(
                {p for e in on_disk for p in e["oracle_response"]["token_points"]},
                {"a", "b", "c"},
            )

            # --- Phase 3: everything present -> no compute at all ---
            rob.reset_mock()
            rob.side_effect = AssertionError("run_oracle_batched should not be called when complete")
            _, _, stats3 = self._run()
            self.assertEqual(rob.call_count, 0)
            self.assertEqual(stats3["cache/oracle_incomplete"], 0.0)


if __name__ == "__main__":
    unittest.main()
