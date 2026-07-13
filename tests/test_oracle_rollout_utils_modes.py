from __future__ import annotations

import unittest

try:
    from oracle_rollout_utils import (
        DEFAULT_ORACLE_ROLLOUT_MODE,
        PROMPT_ONLY_ORACLE_INPUT_TYPES,
        entry_is_complete_and_current,
        oracle_rollouts_dir_base_for_mode,
        parse_oracle_rollout_mode,
    )
except Exception:
    DEFAULT_ORACLE_ROLLOUT_MODE = None
    PROMPT_ONLY_ORACLE_INPUT_TYPES = None
    entry_is_complete_and_current = None
    oracle_rollouts_dir_base_for_mode = None
    parse_oracle_rollout_mode = None


@unittest.skipIf(parse_oracle_rollout_mode is None, "oracle_rollout_utils dependencies unavailable")
class OracleRolloutUtilsModeTests(unittest.TestCase):
    def test_parse_mode_defaults_and_values(self) -> None:
        self.assertEqual(parse_oracle_rollout_mode(None), DEFAULT_ORACLE_ROLLOUT_MODE)
        self.assertEqual(parse_oracle_rollout_mode("sampled_target_repeats"), "sampled_target_repeats")
        self.assertEqual(parse_oracle_rollout_mode("prompt_only_repeats"), "prompt_only_repeats")
        self.assertEqual(parse_oracle_rollout_mode("all_target_deterministic"), "all_target_deterministic")
        with self.assertRaises(ValueError):
            parse_oracle_rollout_mode("bad-mode")

    def test_oracle_rollouts_dir_base_for_mode(self) -> None:
        self.assertEqual(oracle_rollouts_dir_base_for_mode("all_target_deterministic"), "oracle_rollouts")
        self.assertEqual(oracle_rollouts_dir_base_for_mode("sampled_target_repeats"), "oracle_rollouts")
        self.assertEqual(oracle_rollouts_dir_base_for_mode("prompt_only_repeats"), "oracle_prompt_rollouts")

    def test_prompt_only_default_probes_skip_prompt_segment(self) -> None:
        self.assertEqual(PROMPT_ONLY_ORACLE_INPUT_TYPES, ["full_seq", "token_points"])


@unittest.skipIf(entry_is_complete_and_current is None, "oracle_rollout_utils dependencies unavailable")
class EntryReusePredicateTests(unittest.TestCase):
    def _entry(self, **scalars) -> dict:
        return {"rollout_index": 0, "oracle_response": {**scalars, "tokens": {}, "token_points": {}}}

    def test_complete_entry_is_reusable(self) -> None:
        entry = self._entry(full_seq="FS", rollout_segment="RS")
        self.assertTrue(
            entry_is_complete_and_current(
                entry, index_field="rollout_index", required_scalar_probes={"full_seq", "rollout_segment"}
            )
        )

    def test_missing_or_empty_required_probe_fails(self) -> None:
        # Absent probe and legacy ""-placeholder probe are equally not-current.
        for entry in (self._entry(full_seq="FS"), self._entry(full_seq="FS", rollout_segment="")):
            self.assertFalse(
                entry_is_complete_and_current(
                    entry, index_field="rollout_index", required_scalar_probes={"full_seq", "rollout_segment"}
                )
            )

    def test_non_dict_and_missing_index_fail(self) -> None:
        self.assertFalse(
            entry_is_complete_and_current(None, index_field="rollout_index", required_scalar_probes=set())
        )
        entry = {"oracle_response": {"full_seq": "FS"}}
        self.assertFalse(
            entry_is_complete_and_current(
                entry, index_field="oracle_rollout_index", required_scalar_probes={"full_seq"}
            )
        )

    def test_exact_scalar_probes_rejects_extras(self) -> None:
        entry = self._entry(full_seq="FS", prompt_segment="PS")
        entry["oracle_rollout_index"] = 0
        self.assertFalse(
            entry_is_complete_and_current(
                entry,
                index_field="oracle_rollout_index",
                required_scalar_probes={"full_seq"},
                exact_scalar_probes=True,
            )
        )
        self.assertTrue(
            entry_is_complete_and_current(
                entry,
                index_field="oracle_rollout_index",
                required_scalar_probes={"full_seq", "prompt_segment"},
                exact_scalar_probes=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
