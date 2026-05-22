from __future__ import annotations

import unittest

try:
    from oracle_rollout_utils import (
        DEFAULT_ORACLE_ROLLOUT_MODE,
        PROMPT_ONLY_ORACLE_INPUT_TYPES,
        oracle_rollouts_dir_base_for_mode,
        parse_oracle_rollout_mode,
    )
except Exception:
    DEFAULT_ORACLE_ROLLOUT_MODE = None
    PROMPT_ONLY_ORACLE_INPUT_TYPES = None
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


if __name__ == "__main__":
    unittest.main()
