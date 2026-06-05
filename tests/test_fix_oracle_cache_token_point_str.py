from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fix_oracle_cache_token_point_str import (
    _infer_model_name_from_path,
    build_token_point_str,
    migrate_cache_files,
    update_payload,
)


class FakeTokenizer:
    def __call__(self, text: str, **kwargs):
        del kwargs
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(int(token_id)) for token_id in token_ids)


class FixOracleCacheTokenPointStrTests(unittest.TestCase):
    def test_build_token_point_str_uses_combined_text_indices(self) -> None:
        points = {
            "combined_text": "abc",
            "token_points": {"first": 0, "last": 2},
            "token_point_indices": [0, 1, 2],
        }

        self.assertEqual(
            build_token_point_str(FakeTokenizer(), points),
            {"first": "a", "last": "c", "1": "b"},
        )

    def test_update_payload_handles_points_and_oracle_points(self) -> None:
        payload = [
            {
                "points": {
                    "combined_text": "abc",
                    "token_points": {"middle": 1},
                    "token_point_indices": [1],
                }
            },
            {
                "oracle_points": {
                    "combined_text": "xyz",
                    "token_points": {"last": 2},
                    "token_point_indices": [2],
                }
            },
        ]

        changed, updated = update_payload(FakeTokenizer(), payload, overwrite=False)

        self.assertTrue(changed)
        self.assertEqual(updated, 2)
        self.assertEqual(payload[0]["points"]["token_point_str"], {"middle": "b"})
        self.assertEqual(payload[1]["oracle_points"]["token_point_str"], {"last": "z"})

    def test_infer_qwen_model_name_from_cache_path(self) -> None:
        cache_root = Path("/tmp/cache")
        path = (
            cache_root
            / "target_Qwen_Qwen3-8B"
            / "oracle_prompt_rollouts_temp-1.0"
            / "oracle_Qwen_Qwen3-8B_lora-oracle"
            / "target.json"
        )

        self.assertEqual(_infer_model_name_from_path(path, cache_root), "Qwen/Qwen3-8B")

    def test_migrate_cache_files_dry_run_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            cache_file = (
                cache_root
                / "target_Qwen_Qwen3-8B"
                / "oracle_prompt_rollouts_temp-1.0"
                / "oracle_Qwen_Qwen3-8B_lora-oracle"
                / "target"
                / "oracle.json"
            )
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text(
                json.dumps(
                    [
                        {
                            "oracle_points": {
                                "combined_text": "abc",
                                "token_points": {"last": 2},
                                "token_point_indices": [2],
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch("fix_oracle_cache_token_point_str._load_tokenizer", return_value=FakeTokenizer()):
                stats = migrate_cache_files(
                    cache_root=cache_root,
                    model_name=None,
                    write=False,
                    include_judged=False,
                    overwrite=False,
                    trust_remote_code=False,
                    limit=None,
                )

            self.assertEqual(stats["changed_files"], 1)
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            self.assertNotIn("token_point_str", payload[0]["oracle_points"])

    def test_migrate_cache_files_write_updates_raw_oracle_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            cache_file = (
                cache_root
                / "target_Qwen_Qwen3-8B"
                / "oracle_rollouts_temp-0.0"
                / "oracle_Qwen_Qwen3-8B_lora-oracle"
                / "target"
                / "oracle.json"
            )
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text(
                json.dumps(
                    [
                        {
                            "oracle_points": {
                                "combined_text": "abc",
                                "token_points": {"first": 0},
                                "token_point_indices": [0],
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch("fix_oracle_cache_token_point_str._load_tokenizer", return_value=FakeTokenizer()):
                stats = migrate_cache_files(
                    cache_root=cache_root,
                    model_name=None,
                    write=True,
                    include_judged=False,
                    overwrite=False,
                    trust_remote_code=False,
                    limit=None,
                )

            self.assertEqual(stats["changed_files"], 1)
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["oracle_points"]["token_point_str"], {"first": "a"})


if __name__ == "__main__":
    unittest.main()
