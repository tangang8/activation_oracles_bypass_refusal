from __future__ import annotations

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

try:
    import bypass_refusal as br
except Exception:
    br = None


class _FakePerf:
    def track(self, *_args, **_kwargs):
        return nullcontext()


@unittest.skipIf(br is None, "bypass_refusal dependencies unavailable")
class BypassRefusalPipelineTests(unittest.TestCase):
    def _base_config(self, **overrides):
        cfg = br.ExperimentConfig(
            model_name="Qwen/Qwen3-8B",
            oracle_adapter_path="myorg/adapter",
            oracle_adapter_name="oracle",
            oracle_prompts_path="oracle_prompts.json",
            judge_instruction_path="user_request_fulfillment.jinja2",
            num_rollouts=2,
            k_rollouts=1,
            k_rollouts_raw=1,
            num_oracle_rollouts=3,
            oracle_rollout_mode="sampled_target_repeats",
            max_new_tokens=5,
            oracle_max_new_tokens=5,
            oracle_eval_batch_size=2,
            oracle_judge_batch_size=2,
            target_prompt_limit=1,
            run_target_rollouts=True,
            run_target_judging=True,
            run_oracle_rollouts=True,
            run_oracle_judging=True,
            target_lora_path="default",
            judge_lora_path="default",
            oracle_lora_path="oracle",
            experiment_preset="",
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_experiment_config_from_env_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORACLE_ADAPTER_PATH": "myorg/adapter",
            },
            clear=True,
        ):
            cfg = br.ExperimentConfig.from_env()
        self.assertTrue(cfg.run_target_rollouts)
        self.assertTrue(cfg.run_target_judging)
        self.assertTrue(cfg.run_oracle_rollouts)
        self.assertTrue(cfg.run_oracle_judging)
        self.assertEqual(cfg.target_lora_path, "default")
        self.assertEqual(cfg.judge_lora_path, "default")
        self.assertEqual(cfg.oracle_lora_path, "oracle")
        self.assertEqual(cfg.oracle_adapter_path, "myorg/adapter")

    def test_experiment_config_uses_explicit_env_stage_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EXPERIMENT_PRESET": "oracle_target_control",
                "ORACLE_ADAPTER_PATH": "myorg/adapter",
                "RUN_TARGET_ROLLOUTS": "true",
                "RUN_TARGET_JUDGING": "true",
                "RUN_ORACLE_ROLLOUTS": "true",
                "RUN_ORACLE_JUDGING": "true",
                "TARGET_LORA_PATH": "oracle",
                "ORACLE_LORA_PATH": "oracle",
            },
            clear=True,
        ):
            cfg = br.ExperimentConfig.from_env()
        self.assertTrue(cfg.run_oracle_rollouts)
        self.assertTrue(cfg.run_oracle_judging)
        self.assertEqual(cfg.target_lora_path, "oracle")
        self.assertEqual(cfg.oracle_lora_path, "oracle")
        self.assertEqual(cfg.experiment_preset, "oracle_target_control")

    def test_experiment_config_invalid_dependencies_raise(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORACLE_ADAPTER_PATH": "myorg/adapter",
                "RUN_ORACLE_JUDGING": "true",
                "RUN_ORACLE_ROLLOUTS": "false",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                br.ExperimentConfig.from_env()

        with patch.dict(
            os.environ,
            {
                "ORACLE_ADAPTER_PATH": "myorg/adapter",
                "RUN_TARGET_ROLLOUTS": "false",
                "RUN_ORACLE_ROLLOUTS": "true",
                "ORACLE_ROLLOUT_MODE": "sampled_target_repeats",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                br.ExperimentConfig.from_env()

    def test_oracle_target_control_stages_skip_oracle(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-8B"))
        tokenizer = object()
        ctx = SimpleNamespace(is_main=False, rank=0, world_size=1, device="cpu", enabled=False)
        perf = _FakePerf()

        target_entries = [
            {"rollout_index": 0, "target_prompt": "tp", "target_response": "tr", "target_format": {}}
        ]

        with (
            patch("bypass_refusal.format_user_target_prompt", return_value="formatted"),
            patch("bypass_refusal.generate_target_rollouts", return_value=(target_entries, "target.json")),
            patch("bypass_refusal.judge_target_rollouts", return_value=(target_entries, "judge.json", {"compliance_rate": 0.0, "partial_compliance_rate": 0.0, "total": 1})),
            patch("bypass_refusal.generate_oracle_rollouts_for_mode", return_value=(target_entries, "oracle.json", {})) as mode_mock,
            patch("bypass_refusal.judge_oracle_rollouts", return_value=(target_entries, "oracle_judge.json", {})) as judge_oracle_mock,
        ):
            combos = br.run_pipeline_for_target_prompt(
                model=model,
                tokenizer=tokenizer,
                ctx=ctx,
                wandb_run=None,
                perf=perf,
                cfg=self._base_config(
                    run_oracle_rollouts=False,
                    run_oracle_judging=False,
                    target_lora_path="oracle",
                ),
                target_prompt_str="tp",
                target_prompt_index=0,
                oracle_prompts=["o1", "o2"],
                judge_instruction_file="f",
                judge_instruction_stem="s",
                judge_instruction_template="tmpl",
            )

        self.assertEqual(combos, 0)
        mode_mock.assert_not_called()
        judge_oracle_mock.assert_not_called()

    def test_target_lora_and_raw_entries_used_when_target_judging_disabled(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-8B"))
        tokenizer = object()
        ctx = SimpleNamespace(is_main=False, rank=0, world_size=1, device="cpu", enabled=False)
        perf = _FakePerf()
        target_entries = [{"rollout_index": 0, "target_prompt": "tp", "target_response": "tr", "target_format": {}}]

        with (
            patch("bypass_refusal.format_user_target_prompt", return_value="formatted"),
            patch("bypass_refusal.generate_target_rollouts", return_value=(target_entries, "target.json")) as target_rollouts_mock,
            patch("bypass_refusal.judge_target_rollouts") as target_judge_mock,
            patch("bypass_refusal.generate_oracle_rollouts_for_mode", return_value=(target_entries, "oracle.json", {})) as mode_mock,
            patch("bypass_refusal.judge_oracle_rollouts", return_value=(target_entries, "oracle_judge.json", {})),
        ):
            combos = br.run_pipeline_for_target_prompt(
                model=model,
                tokenizer=tokenizer,
                ctx=ctx,
                wandb_run=None,
                perf=perf,
                cfg=self._base_config(
                    run_target_judging=False,
                    target_lora_path="oracle",
                ),
                target_prompt_str="tp",
                target_prompt_index=0,
                oracle_prompts=["o1"],
                judge_instruction_file="f",
                judge_instruction_stem="s",
                judge_instruction_template="tmpl",
            )

        self.assertEqual(combos, 1)
        target_judge_mock.assert_not_called()
        self.assertEqual(target_rollouts_mock.call_args.kwargs["target_lora_path"], "oracle")
        self.assertEqual(mode_mock.call_args.kwargs["target_rollout_entries"], target_entries)
        self.assertEqual(mode_mock.call_args.kwargs["target_lora_path"], "oracle")

    def test_oracle_judging_can_be_disabled(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-8B"))
        tokenizer = object()
        ctx = SimpleNamespace(is_main=False, rank=0, world_size=1, device="cpu", enabled=False)
        perf = _FakePerf()
        target_entries = [{"rollout_index": 0, "target_prompt": "tp", "target_response": "tr", "target_format": {}}]

        with (
            patch("bypass_refusal.format_user_target_prompt", return_value="formatted"),
            patch("bypass_refusal.generate_target_rollouts", return_value=(target_entries, "target.json")),
            patch("bypass_refusal.judge_target_rollouts", return_value=(target_entries, "judge.json", {"compliance_rate": 0.0, "partial_compliance_rate": 0.0, "total": 1})),
            patch("bypass_refusal.generate_oracle_rollouts_for_mode", return_value=(target_entries, "oracle.json", {})),
            patch("bypass_refusal.judge_oracle_rollouts", return_value=(target_entries, "oracle_judge.json", {})) as judge_oracle_mock,
        ):
            combos = br.run_pipeline_for_target_prompt(
                model=model,
                tokenizer=tokenizer,
                ctx=ctx,
                wandb_run=None,
                perf=perf,
                cfg=self._base_config(run_oracle_judging=False),
                target_prompt_str="tp",
                target_prompt_index=0,
                oracle_prompts=["o1"],
                judge_instruction_file="f",
                judge_instruction_stem="s",
                judge_instruction_template="tmpl",
            )

        self.assertEqual(combos, 1)
        judge_oracle_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
