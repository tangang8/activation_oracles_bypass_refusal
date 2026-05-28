from __future__ import annotations

"""StrongReject-only compilation entrypoint.

The old version of this file scanned the cache tree and aggregated any
classification-looking judge output it found. That was too permissive for the
current experiment suite, where old exploratory caches and non-StrongReject
judge outputs may coexist with the real run artifacts.

This wrapper keeps the familiar ``compile_results.py`` entrypoint, but delegates
to the workflow-traced StrongReject compiler in ``results/``.
"""

import json
import sys
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parent / "results"
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))

from compile_strongreject_results import (  # noqa: E402
    StrongRejectCompileConfig,
    compile_strongreject_results,
    parse_args as parse_strongreject_args,
)


def compile_cache_results(
    cache_root: Path | str = Path("cache"),
    output_dir: Path | str = Path("results/compiled_strongreject_results"),
    *,
    judge_instruction_path: str = "strongReject_v5.jinja2",
    target_model_name: str = "Qwen/Qwen3-8B",
    judge_model_name: str = "Qwen/Qwen3-8B",
    oracle_model_name: str = "Qwen/Qwen3-8B",
    oracle_lora_path: str = "oracle",
    target_prompt_offset: int = 0,
    expected_target_prompts: int = 100,
    expected_target_rollouts: int = 50,
    expected_oracle_rollouts: int = 50,
    oracle_prompts_paths: tuple[str, ...] = (
        "prompts/oracle_prompts/default_oracle_prompts.json",
        "prompts/oracle_prompts/model_answer_min_200_words.json",
    ),
    thresholds: tuple[float, ...] = (0.0, 0.3, 0.5, 0.8, 1.0),
    strict: bool = False,
    target_prompts: list[str] | None = None,
    oracle_prompts_by_file: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compile only workflow-traced StrongReject results.

    ``target_prompts`` and ``oracle_prompts_by_file`` are test hooks; normal
    command-line usage loads the configured prompt datasets/files.
    """

    cfg = StrongRejectCompileConfig(
        cache_root=Path(cache_root),
        output_dir=Path(output_dir),
        judge_instruction_path=judge_instruction_path,
        target_model_name=target_model_name,
        judge_model_name=judge_model_name,
        oracle_model_name=oracle_model_name,
        oracle_lora_path=oracle_lora_path,
        target_prompt_offset=target_prompt_offset,
        expected_target_prompts=expected_target_prompts,
        expected_target_rollouts=expected_target_rollouts,
        expected_oracle_rollouts=expected_oracle_rollouts,
        oracle_prompts_paths=oracle_prompts_paths,
        thresholds=thresholds,
        strict=strict,
    )
    return compile_strongreject_results(
        cfg,
        target_prompts=target_prompts,
        oracle_prompts_by_file=oracle_prompts_by_file,
    )


def main() -> None:
    cfg = parse_strongreject_args()
    manifest = compile_strongreject_results(cfg)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
