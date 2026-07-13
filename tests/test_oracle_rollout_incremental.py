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
        "prompt_segment": [f"PS::{response}"],
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
    def test_merge_adds_response_and_metadata_from_required_spec(self) -> None:
        entry = {
            "oracle_response": {"full_seq": "FS", "token_points": {"a": "A", "b": "B"}},
            "oracle_format": {"token_points": {"a": {}, "b": {}}},
            "oracle_points": {"token_points": {"a": 1, "b": 2}, "token_point_indices": [1, 2],
                              "token_point_str": {"a": "sa", "b": "sb"}},
        }
        # The real backfill call resolves an EMPTY point spec (it is driven by explicit
        # indices), so oracle_result["points"] carries no token points — the regression.
        result = {
            "token_points": {3: ["C-DECODE"]},
            "points": {"token_points": {}, "token_point_indices": [], "token_point_str": {}},
        }
        required_spec = {
            "token_points": {"a": 1, "b": 2, "c": 3},
            "token_point_indices": [1, 2, 3],
            "token_point_str": {"a": "sa", "b": "sb", "c": "sc"},
        }
        oru._merge_token_points_into_entry(entry, result, {"c": 3}, required_spec)

        tp = entry["oracle_response"]["token_points"]
        self.assertEqual(tp["c"], "C-DECODE")            # new decode added
        self.assertEqual(tp["a"], "A")                   # existing untouched
        self.assertEqual(entry["oracle_response"]["full_seq"], "FS")  # segment untouched
        self.assertIn("c", entry["oracle_format"]["token_points"])
        # THE FIX: oracle_points metadata gets the new point (index + decoded str) even
        # though oracle_result["points"] was empty — taken from required_spec.
        op = entry["oracle_points"]
        self.assertEqual(op["token_points"]["c"], 3)
        self.assertEqual(op["token_points"]["a"], 1)     # existing preserved
        self.assertEqual(op["token_point_indices"], [1, 2, 3])
        self.assertEqual(op["token_point_str"]["c"], "sc")


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

        def _spec(points):
            return {"token_points": dict(points),
                    "token_point_indices": sorted(points.values()),
                    "token_point_str": {n: f"str_{n}" for n in points}}

        with mock.patch.object(oru, "format_user_target_prompt", return_value="FP"), \
             mock.patch.object(oru, "run_oracle_batched") as rob, \
             mock.patch.object(oru, "_required_combined_spec") as req:

            # --- Phase 1: fresh run, both rollouts missing -> full compute of a,b ---
            req.return_value = _spec(old_points)
            rob.side_effect = lambda **kw: [
                _fake_full_result("FP", r, old_points) for r in kw["target_responses"]
            ]
            entries1, cache_file, _ = self._run()
            self.assertEqual(len(entries1), 2)
            self.assertEqual(rob.call_args.kwargs["oracle_input_types"],
                             ["full_seq", "prompt_segment", "rollout_segment", "token_points"])

            # --- Phase 2: extractor now emits 'c' -> only 'c' computed & merged. The backfill's
            # run_oracle_batched returns an EMPTY point spec (real behavior), so metadata must
            # come from the required spec. ---
            rob.reset_mock()
            req.return_value = _spec(new_points)
            rob.side_effect = lambda **kw: [
                {"token_points": {3: [f"C::{r}"]},
                 "points": {"token_points": {}, "token_point_indices": [], "token_point_str": {}}}
                for r in kw["target_responses"]
            ]
            entries2, _, stats2 = self._run()

            # incremental call happened, token-points-only, just index 3
            self.assertEqual(rob.call_count, 1)
            call = rob.call_args.kwargs
            self.assertEqual(call["oracle_input_types"], ["token_points"])
            self.assertEqual(call["token_point_indices_by_target"], [[3], [3]])
            self.assertEqual(stats2["cache/oracle_incomplete"], 2.0)

            # new point merged into BOTH response and oracle_points metadata; existing
            # decodes/segments preserved (not recomputed).
            for e in entries2:
                tp = e["oracle_response"]["token_points"]
                self.assertEqual(set(tp), {"a", "b", "c"})
                self.assertTrue(e["oracle_response"]["full_seq"].startswith("FS::"))
                self.assertTrue(tp["c"].startswith("C::"))
                op = e["oracle_points"]
                self.assertEqual(op["token_points"]["c"], 3)                 # name -> index
                self.assertIn(3, op["token_point_indices"])                  # index list
                self.assertEqual(op["token_point_str"]["c"], "str_c")        # decoded token str
                self.assertEqual(op["token_points"]["a"], 1)                 # existing preserved

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

    def test_stale_scalar_probe_triggers_full_regen_of_that_entry(self) -> None:
        points = {"a": 1}

        def _spec():
            return {"token_points": dict(points),
                    "token_point_indices": sorted(points.values()),
                    "token_point_str": {n: f"str_{n}" for n in points}}

        with mock.patch.object(oru, "format_user_target_prompt", return_value="FP"), \
             mock.patch.object(oru, "run_oracle_batched") as rob, \
             mock.patch.object(oru, "_required_combined_spec") as req:
            req.return_value = _spec()
            rob.side_effect = lambda **kw: [
                _fake_full_result("FP", r, points) for r in kw["target_responses"]
            ]
            entries1, cache_file, _ = self._run()
            self.assertEqual(len(entries1), 2)

            # Simulate a legacy/incomplete cached entry: rollout 0 lost its rollout_segment
            # decode (old writer emitted "" for probes the run never generated).
            on_disk = load_json(Path(cache_file))
            on_disk[0]["oracle_response"]["rollout_segment"] = ""
            import json
            Path(cache_file).write_text(json.dumps(on_disk))

            rob.reset_mock()
            entries2, _, stats2 = self._run()
            # Only the stale entry is regenerated, with the FULL probe set.
            self.assertEqual(rob.call_count, 1)
            call = rob.call_args.kwargs
            self.assertEqual(call["target_responses"], ["R0"])
            self.assertEqual(
                call["oracle_input_types"],
                ["full_seq", "prompt_segment", "rollout_segment", "token_points"],
            )
            self.assertEqual(stats2["cache/oracle_missing"], 1.0)
            self.assertTrue(entries2[0]["oracle_response"]["rollout_segment"].startswith("RS::"))
            self.assertTrue(entries2[1]["oracle_response"]["rollout_segment"].startswith("RS::"))


@unittest.skipIf(not _IMPORT_OK, "oracle_rollout_utils dependencies unavailable")
class PromptOnlySparseReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_root = self.tmp.name
        self.model = SimpleNamespace(config=SimpleNamespace(_name_or_path="test/model"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, num_oracle_rollouts: int):
        return oru.generate_prompt_only_oracle_rollouts(
            model=self.model,
            tokenizer=object(),
            device="cpu",
            oracle_prompt="OP",
            target_prompt="P",
            target_model_name="test/model",
            target_lora_path=None,
            num_oracle_rollouts=num_oracle_rollouts,
            oracle_lora_path="oracle",
            cache_root=self.cache_root,
        )

    def test_sparse_cached_indices_do_not_short_circuit(self) -> None:
        """A cache holding indices {0, 2} must NOT satisfy a request for 2 rollouts: the old
        count-based check returned only entry 0 (fewer entries than requested, silently)."""
        from cache_utils import oracle_prompt_rollout_cache_file_path, write_json

        cache_file = oracle_prompt_rollout_cache_file_path(
            cache_root=self.cache_root,
            target_model_name="test/model",
            target_lora_path=None,
            oracle_model_name="test/model",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"do_sample": True, "temperature": 1.0, "max_new_tokens": 1000},
            target_prompt="P",
            oracle_prompt="OP",
        )
        stale = [
            {"oracle_rollout_index": i, "target_prompt": "P",
             "oracle_response": {"full_seq": f"OLD{i}", "tokens": {}, "token_points": {}}}
            for i in (0, 2)
        ]
        write_json(cache_file, stale)

        combined = {
            "full_seq": ["NEW0", "NEW1"],
            "tokens": {},
            "token_points": {},
            "points": {"token_points": {}},
            "oracle_repeats": 2,
        }
        with mock.patch.object(oru, "format_user_target_prompt", return_value="FP"), \
             mock.patch.object(oru, "run_oracle_batched", return_value=[combined]) as rob:
            entries, _, stats = self._run(num_oracle_rollouts=2)

        self.assertEqual(rob.call_count, 1)  # regenerated, not served sparsely
        self.assertEqual([e["oracle_rollout_index"] for e in entries], [0, 1])
        self.assertEqual([e["oracle_response"]["full_seq"] for e in entries], ["NEW0", "NEW1"])
        self.assertEqual(stats["cache/oracle_missing"], 1.0)  # index 1 was the hole

    def test_contiguous_complete_cache_short_circuits(self) -> None:
        from cache_utils import oracle_prompt_rollout_cache_file_path, write_json

        cache_file = oracle_prompt_rollout_cache_file_path(
            cache_root=self.cache_root,
            target_model_name="test/model",
            target_lora_path=None,
            oracle_model_name="test/model",
            oracle_lora_path="oracle",
            oracle_generation_kwargs={"do_sample": True, "temperature": 1.0, "max_new_tokens": 1000},
            target_prompt="P",
            oracle_prompt="OP",
        )
        cached = [
            {"oracle_rollout_index": i, "target_prompt": "P",
             "oracle_response": {"full_seq": f"C{i}", "tokens": {}, "token_points": {}}}
            for i in (0, 1)
        ]
        write_json(cache_file, cached)

        with mock.patch.object(oru, "format_user_target_prompt", return_value="FP"), \
             mock.patch.object(oru, "run_oracle_batched") as rob:
            rob.side_effect = AssertionError("must not regenerate a complete contiguous cache")
            entries, _, stats = self._run(num_oracle_rollouts=2)

        self.assertEqual([e["oracle_response"]["full_seq"] for e in entries], ["C0", "C1"])
        self.assertEqual(stats["cache/oracle_hits"], 2.0)


if __name__ == "__main__":
    unittest.main()
