from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "run_parallel_h200_strongreject_v5.sh"


class RunParallelH200ScriptTests(unittest.TestCase):
    def _write_fake_runner(self, tmpdir: str, mode: str) -> Path:
        fake = Path(tmpdir) / "fake_oracle_runner.sh"
        if mode == "success":
            body = (
                "#!/usr/bin/env bash\n"
                "echo \"FAKE_OK\"\n"
                "echo \"WANDB_RUN_NAME=${WANDB_RUN_NAME:-}\"\n"
                "echo \"WANDB_GROUP=${WANDB_GROUP:-}\"\n"
                "echo \"WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-}\"\n"
                "echo \"TARGET_THINKING=${TARGET_THINKING:-}\"\n"
                "exit 0\n"
            )
        elif mode == "fail_non_oom":
            body = (
                "#!/usr/bin/env bash\n"
                "echo \"regular failure\" >&2\n"
                "exit 7\n"
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        fake.write_text(body, encoding="utf-8")
        fake.chmod(0o755)
        return fake

    def test_dry_run_uses_run_label_in_log_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["RUN_LABEL"] = "unit_parallel_label"
            env["LOG_ROOT"] = str(Path(tmpdir) / "logs")
            env["DRY_RUN"] = "1"
            proc = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, env=env)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("run_label=unit_parallel_label", proc.stdout)

    def test_non_oom_failure_fails_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_runner = self._write_fake_runner(tmpdir, "fail_non_oom")
            env = os.environ.copy()
            env["RUN_LABEL"] = "unit_parallel_fail"
            env["LOG_ROOT"] = str(Path(tmpdir) / "logs")
            env["TARGET_PROMPT_TOTAL"] = "1"
            env["TARGET_PROMPT_SPLIT"] = "1"
            env["NUM_ROLLOUTS"] = "1"
            env["NUM_ORACLE_ROLLOUTS"] = "1"
            env["RUN_ORACLE_EXPERIMENT"] = str(fake_runner)
            proc = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, env=env)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Sequence failed", proc.stdout)
            self.assertIn("failed with exit code 7", proc.stdout)


if __name__ == "__main__":
    unittest.main()
