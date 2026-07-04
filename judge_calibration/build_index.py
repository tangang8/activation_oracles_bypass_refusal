"""Step 0 -- Build a flat AO-response index from the existing judged-oracle cache.

Each judged-oracle cache file is a list of entries; each entry carries the harmful
prompt, the oracle prompt, an `oracle_response` tree (the text the judge saw) and a
mirrored `compliance` tree (the incumbent judge's StrongReject leaf). We walk both
trees together so every emitted row pairs the exact `response_text` with its
`qwen_score`. This is the single source the sampler draws from.

We deliberately read the cache directly (rather than the compiled details CSV) because
only the cache carries the response *text*, which the human labeler and the GPT-4o
judge both need.

Usage:
    PYTHONPATH=. python judge_calibration/build_index.py [--rubric-stem strongReject_v5]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge_calibration import config

# Scalar probe kinds store a single response string; token_points / tokens are dicts.
_SCALAR_PROBES = ("full_seq", "segment", "prompt_segment", "rollout_segment")
_DICT_PROBES = ("tokens", "token_points")


def _response_id(*, experiment: str, target_prompt: str, oracle_prompt: str,
                 probe_kind: str, probe_name: str, rollout_index: Any) -> str:
    key = "\x1f".join([
        experiment, target_prompt, oracle_prompt, probe_kind, probe_name, str(rollout_index),
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _experiment_from_path(path: Path) -> str | None:
    parts = path.as_posix()
    if "/oracle_prompt_rollouts_temp-" in parts:
        return "user_prompt_oracle"
    if "/oracle_rollouts_temp-" in parts:
        return "target_rollout_oracle"
    return None


def _valid_qwen_score(leaf: Any, rubric_stem: str) -> float | None:
    """Mirror results.compile_strongreject_results._valid_strongreject_leaf."""
    if not isinstance(leaf, dict):
        return None
    score = leaf.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    score = float(score)
    if leaf.get("score_scale") not in (None, "strongreject_0_1"):
        return None
    if score < 0.0 or score > 1.0:
        return None
    instruction_file = str(leaf.get("judge_instruction_file", ""))
    if instruction_file and Path(instruction_file).stem != rubric_stem:
        return None
    return score


def _iter_leaves(entry: dict) -> Iterator[tuple[str, str, str, Any]]:
    """Yield (probe_kind, probe_name, response_text, compliance_leaf) for one entry."""
    oracle_response = entry.get("oracle_response", {})
    compliance = entry.get("compliance", {})
    if not isinstance(oracle_response, dict) or not isinstance(compliance, dict):
        return
    for probe_kind in _SCALAR_PROBES:
        text = oracle_response.get(probe_kind)
        if isinstance(text, str):
            yield probe_kind, probe_kind, text, compliance.get(probe_kind)
    for probe_kind in _DICT_PROBES:
        text_container = oracle_response.get(probe_kind, {})
        leaf_container = compliance.get(probe_kind, {})
        if not isinstance(text_container, dict) or not isinstance(leaf_container, dict):
            continue
        for probe_name, text in text_container.items():
            if isinstance(text, str):
                yield probe_kind, str(probe_name), text, leaf_container.get(probe_name)


def _load_oracle_prompt_ab_map() -> dict[str, str]:
    """Map each oracle-prompt *text* to its A/B label."""
    from prompt_utils import load_oracle_prompts_from_file

    mapping: dict[str, str] = {}
    for label, rel_path in config.ORACLE_PROMPT_FILES.items():
        for prompt in load_oracle_prompts_from_file(str(config.REPO_ROOT / rel_path)):
            mapping[prompt] = label
    return mapping


def _load_prompt_id_map(target_prompts: list[str]) -> dict[str, int]:
    """Canonical harmful-prompt index. Prefer the dataset order; else stable fallback."""
    try:
        from prompt_utils import load_target_prompts_from_dataset

        dataset_prompts = load_target_prompts_from_dataset(limit=100, offset=0)
        if dataset_prompts:
            return {p: i for i, p in enumerate(dataset_prompts)}
    except Exception as exc:  # dataset lib / network unavailable
        print(f"[build_index] dataset unavailable ({exc}); using stable fallback prompt ids.")
    return {p: i for i, p in enumerate(sorted(target_prompts))}


def build_index(cache_root: Path, rubric_stem: str) -> list[dict[str, Any]]:
    pattern = f"**/{rubric_stem}/oracle_rollouts_judged/**/*.json"
    files = sorted(cache_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No judged-oracle cache files found under {cache_root} for rubric '{rubric_stem}'. "
            f"Run the oracle-judging stage first (see AGENTS.md)."
        )

    ab_map = _load_oracle_prompt_ab_map()

    raw_rows: list[dict[str, Any]] = []
    target_prompts_seen: set[str] = set()
    skipped_no_text = skipped_bad_score = 0

    for path in files:
        experiment = _experiment_from_path(path)
        if experiment is None:
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            target_prompt = str(entry.get("target_prompt", "")).strip()
            oracle_prompt = str(entry.get("oracle_prompt", "")).strip()
            if not target_prompt or not oracle_prompt:
                continue
            target_prompts_seen.add(target_prompt)
            # Prompt-only entries key on oracle_rollout_index; deterministic target-backed
            # entries key on the plain `rollout_index` field (they carry neither
            # oracle_rollout_index nor target_rollout_index). Without the final fallback all
            # 50 deterministic rollouts per (prompt, oracle, slice) collapse to rollout=None
            # and dedup drops 49 of them.
            rollout_index = entry.get("oracle_rollout_index")
            if rollout_index is None:
                rollout_index = entry.get("target_rollout_index")
            if rollout_index is None:
                rollout_index = entry.get("rollout_index")
            for probe_kind, probe_name, text, leaf in _iter_leaves(entry):
                response_text = text.strip()
                if not response_text:
                    skipped_no_text += 1
                    continue
                qwen_score = _valid_qwen_score(leaf, rubric_stem)
                if qwen_score is None:
                    skipped_bad_score += 1
                    continue
                raw_rows.append({
                    "experiment": experiment,
                    "target_prompt": target_prompt,
                    "oracle_prompt_text": oracle_prompt,
                    "oracle_prompt": ab_map.get(oracle_prompt, "?"),
                    "slice": probe_name,
                    "probe_kind": probe_kind,
                    "rollout_index": rollout_index,
                    "response_text": response_text,
                    "qwen_score": qwen_score,
                    "cache_path": str(path.relative_to(config.REPO_ROOT)),
                })

    prompt_id_map = _load_prompt_id_map(sorted(target_prompts_seen))

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for r in raw_rows:
        rid = _response_id(
            experiment=r["experiment"], target_prompt=r["target_prompt"],
            oracle_prompt=r["oracle_prompt_text"], probe_kind=r["probe_kind"],
            probe_name=r["slice"], rollout_index=r["rollout_index"],
        )
        if rid in seen_ids:  # identical (prompt, oracle, slice, rollout) -> keep first
            continue
        seen_ids.add(rid)
        rows.append({
            "response_id": rid,
            "experiment": r["experiment"],
            "prompt_id": prompt_id_map.get(r["target_prompt"], -1),
            "harmful_prompt": r["target_prompt"],
            "slice": r["slice"],
            "probe_kind": r["probe_kind"],
            "oracle_prompt": r["oracle_prompt"],
            "oracle_prompt_text": r["oracle_prompt_text"],
            "rollout_index": r["rollout_index"],
            "response_text": r["response_text"],
            "qwen_score": r["qwen_score"],
            "cache_path": r["cache_path"],
        })

    print(
        f"[build_index] files={len(files)} rows={len(rows)} "
        f"(skipped: no_text={skipped_no_text}, bad_score={skipped_bad_score})"
    )
    return rows


_FIELDNAMES = [
    "response_id", "experiment", "prompt_id", "harmful_prompt", "slice", "probe_kind",
    "oracle_prompt", "oracle_prompt_text", "rollout_index", "response_text",
    "qwen_score", "cache_path",
]


def write_index(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the flat AO-response index.")
    parser.add_argument("--cache-root", default=str(config.CACHE_ROOT))
    parser.add_argument("--rubric-stem", default=Path(config.RUBRIC_PATH).stem)
    parser.add_argument("--out", default=str(config.INDEX_CSV))
    args = parser.parse_args()

    rows = build_index(Path(args.cache_root), args.rubric_stem)
    write_index(rows, Path(args.out))
    print(f"[build_index] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
