from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import wandb_utils


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.name = "fake-run"
        self.id = "abc123"

    def log(self, payload: dict) -> None:
        self.logged.append(payload)


class WandbUtilsTests(unittest.TestCase):
    def test_init_wandb_run_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            run = wandb_utils.init_wandb_run({"a": 1})
        self.assertIsNone(run)

    def test_init_wandb_run_with_fake_module(self) -> None:
        fake_run = _FakeRun()

        def fake_init(**kwargs):
            self.assertIn("config", kwargs)
            self.assertEqual(kwargs.get("group"), "group-a")
            self.assertEqual(kwargs.get("job_type"), "job-a")
            return fake_run

        fake_module = types.SimpleNamespace(init=fake_init)
        with patch.dict(
            os.environ,
            {"WANDB_API_KEY": "k", "WANDB_GROUP": "group-a", "WANDB_JOB_TYPE": "job-a"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"wandb": fake_module}):
                run = wandb_utils.init_wandb_run({"x": 1})
        self.assertIs(run, fake_run)

    def test_log_rollout_metrics(self) -> None:
        run = _FakeRun()
        entries = [
            {"compliance": {"score": 1, "judge_instruction_file": "f"}},
            {"compliance": {"score": 3, "judge_instruction_file": "f"}},
        ]
        wandb_utils.log_rollout_metrics(run, entries, {"compliance_rate": 0.5, "partial_compliance_rate": 0.7})
        self.assertEqual(len(run.logged), 1)
        payload = run.logged[0]
        self.assertEqual(payload["rollouts/valid_count"], 2)
        self.assertAlmostEqual(payload["rollouts/avg_score"], 2.0)

    def test_log_oracle_metrics(self) -> None:
        run = _FakeRun()
        deterministic = [
            {"oracle_response": {}, "target_rollout_index": 0},
            {"oracle_response": {}, "target_rollout_index": 0},
            {"oracle_response": {}, "target_rollout_index": 1},
        ]
        wandb_utils.log_oracle_metrics(run, deterministic, 3.2)
        self.assertEqual(len(run.logged), 1)
        payload = run.logged[0]
        self.assertEqual(payload["oracle/targets_evaluated"], 3)
        self.assertAlmostEqual(payload["oracle/avg_repeats_observed"], 1.5)

    def test_log_oracle_judge_metrics_filters_non_numeric(self) -> None:
        run = _FakeRun()
        wandb_utils.log_oracle_judge_metrics(run, {"a": 1, "b": "x", "c": 2.0})
        self.assertEqual(len(run.logged), 1)
        payload = run.logged[0]
        self.assertEqual(payload, {"a": 1.0, "c": 2.0})


if __name__ == "__main__":
    unittest.main()
