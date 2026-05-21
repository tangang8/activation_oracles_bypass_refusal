from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import rollout_utils as ru
except Exception:
    ru = None


@unittest.skipIf(ru is None, "rollout_utils dependencies unavailable")
class RolloutUtilsTests(unittest.TestCase):
    def test_item_ids_can_match_target_responses_when_some_skipped(self) -> None:
        preprocs = [
            {"thinking": "", "parsed_response": "", "valid_response_format": True},
            {"thinking": "", "parsed_response": "usable response", "valid_response_format": True},
        ]

        with (
            patch("rollout_utils.validate_target_response_format", side_effect=preprocs),
            patch("rollout_utils.format_user_target_prompt", side_effect=lambda _tok, prompt: prompt),
            patch(
                "rollout_utils.generate_batched_from_formatted_prompts",
                return_value=["Score: 4\nReason: Looks compliant"],
            ),
        ):
            results = ru.score_responses_compliance_batched(
                judge_model=object(),
                judge_tokenizer=object(),
                user_prompt="u",
                target_responses=["r0", "r1"],
                judge_instruction_template="Prompt: {user_prompt}\nResponse: {model_response}",
                device=object(),
                item_ids=["item-0", "item-1"],
                emit_summary_log=False,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["judge_skipped"])
        self.assertEqual(results[1]["score"], 4)


if __name__ == "__main__":
    unittest.main()
