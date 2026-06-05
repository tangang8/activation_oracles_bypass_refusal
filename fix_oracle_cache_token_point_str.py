from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

KNOWN_SANITIZED_MODEL_NAMES = {
    "Qwen_Qwen3-8B": "Qwen/Qwen3-8B",
}


def _infer_model_name_from_path(path: Path, cache_root: Path) -> str | None:
    try:
        rel_parts = path.resolve().relative_to(cache_root.resolve()).parts
    except ValueError:
        rel_parts = path.parts

    target_part = next((part for part in rel_parts if part.startswith("target_")), None)
    if target_part is None:
        return None

    model_part = target_part[len("target_") :]
    model_part = model_part.split("_lora-", 1)[0]
    if model_part in KNOWN_SANITIZED_MODEL_NAMES:
        return KNOWN_SANITIZED_MODEL_NAMES[model_part]

    pieces = model_part.split("_", 1)
    if len(pieces) == 2:
        return f"{pieces[0]}/{pieces[1]}"
    return model_part or None


def _is_oracle_rollout_cache(path: Path, *, include_judged: bool) -> bool:
    parts = path.parts
    if not include_judged and "oracle_rollouts_judged" in parts:
        return False
    return any(
        part.startswith("oracle_rollouts_temp-") or part.startswith("oracle_prompt_rollouts_temp-")
        for part in parts
    )


def _iter_cache_files(cache_root: Path, *, include_judged: bool) -> list[Path]:
    if not cache_root.exists():
        raise FileNotFoundError(f"cache root does not exist: {cache_root}")
    return sorted(
        path
        for path in cache_root.rglob("*.json")
        if _is_oracle_rollout_cache(path, include_judged=include_judged)
    )


def _tokenize_combined_text(tokenizer: Any, combined_text: str) -> list[int]:
    tokenized = tokenizer(
        combined_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    input_ids = tokenized["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_token_point_str(tokenizer: Any, oracle_points: dict[str, Any]) -> dict[str, str]:
    combined_text = oracle_points.get("combined_text")
    if not isinstance(combined_text, str) or not combined_text:
        raise ValueError("oracle_points.combined_text is missing or empty")

    token_ids = _tokenize_combined_text(tokenizer, combined_text)

    named_points = oracle_points.get("token_points", {})
    if not isinstance(named_points, dict):
        named_points = {}

    indices_raw = oracle_points.get("token_point_indices", [])
    if not isinstance(indices_raw, list):
        indices_raw = []

    token_point_str: dict[str, str] = {}
    seen_indices: set[int] = set()

    for name, raw_idx in named_points.items():
        idx = int(raw_idx)
        if idx < 0 or idx >= len(token_ids):
            raise IndexError(
                f"token point {name!r} index {idx} is out of bounds for "
                f"combined_text token length {len(token_ids)}"
            )
        token_point_str[str(name)] = _decode_token(tokenizer, token_ids[idx])
        seen_indices.add(idx)

    for raw_idx in indices_raw:
        idx = int(raw_idx)
        if idx in seen_indices:
            continue
        if idx < 0 or idx >= len(token_ids):
            raise IndexError(
                f"token_point_indices entry {idx} is out of bounds for "
                f"combined_text token length {len(token_ids)}"
            )
        token_point_str[str(idx)] = _decode_token(tokenizer, token_ids[idx])
        seen_indices.add(idx)

    return token_point_str


def _entry_oracle_points(entry: Any) -> list[dict[str, Any]]:
    if not isinstance(entry, dict):
        return []
    out: list[dict[str, Any]] = []
    for key in ("oracle_points", "points"):
        value = entry.get(key)
        if isinstance(value, dict):
            out.append(value)
    return out


def update_payload(tokenizer: Any, payload: Any, *, overwrite: bool) -> tuple[bool, int]:
    entries = payload if isinstance(payload, list) else [payload]
    changed = False
    updated_points = 0

    for entry in entries:
        for oracle_points in _entry_oracle_points(entry):
            if "token_point_str" in oracle_points and not overwrite:
                continue
            token_point_str = build_token_point_str(tokenizer, oracle_points)
            if oracle_points.get("token_point_str") != token_point_str:
                oracle_points["token_point_str"] = token_point_str
                changed = True
            updated_points += 1

    return changed, updated_points


def _load_tokenizer(model_name: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    token = os.getenv("HF_TOKEN")
    return AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def migrate_cache_files(
    *,
    cache_root: Path,
    model_name: str | None,
    write: bool,
    include_judged: bool,
    overwrite: bool,
    trust_remote_code: bool,
    limit: int | None,
) -> dict[str, int]:
    candidate_paths = _iter_cache_files(cache_root, include_judged=include_judged)
    if limit is not None:
        candidate_paths = candidate_paths[:limit]

    tokenizers: dict[str, Any] = {}
    stats = {
        "candidate_files": len(candidate_paths),
        "read_files": 0,
        "changed_files": 0,
        "updated_oracle_points": 0,
        "skipped_files": 0,
        "error_files": 0,
    }

    for path in candidate_paths:
        effective_model_name = model_name or _infer_model_name_from_path(path, cache_root)
        if not effective_model_name:
            print(f"[skip] could not infer model name: {path}", file=sys.stderr)
            stats["skipped_files"] += 1
            continue

        if effective_model_name not in tokenizers:
            tokenizers[effective_model_name] = _load_tokenizer(
                effective_model_name,
                trust_remote_code=trust_remote_code,
            )
        tokenizer = tokenizers[effective_model_name]

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stats["read_files"] += 1
            changed, updated_points = update_payload(tokenizer, payload, overwrite=overwrite)
        except Exception as exc:
            print(f"[error] {path}: {exc}", file=sys.stderr)
            stats["error_files"] += 1
            continue

        stats["updated_oracle_points"] += updated_points
        if not changed:
            continue

        stats["changed_files"] += 1
        action = "write" if write else "dry-run"
        print(f"[{action}] {path} ({updated_points} oracle_points)")
        if write:
            _write_json_atomic(path, payload)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill oracle_points.token_point_str in oracle rollout cache files by "
            "tokenizing oracle_points.combined_text and decoding the saved token point indices."
        )
    )
    parser.add_argument("--cache-root", default="cache", type=Path)
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Tokenizer model name to use for every file. If omitted, the script "
            "infers from cache path target_<model>; Qwen_Qwen3-8B maps to Qwen/Qwen3-8B."
        ),
    )
    parser.add_argument("--write", action="store_true", help="Actually rewrite cache files. Default is dry-run.")
    parser.add_argument(
        "--include-judged",
        action="store_true",
        help="Also patch oracle_rollouts_judged files. Default patches only raw oracle rollout caches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute token_point_str even when it is already present.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N matching files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = migrate_cache_files(
        cache_root=args.cache_root,
        model_name=args.model_name,
        write=args.write,
        include_judged=args.include_judged,
        overwrite=args.overwrite,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
    )
    mode = "WRITE" if args.write else "DRY RUN"
    print(f"\n{mode} summary")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if not args.write and stats["changed_files"]:
        print("\nNo files were changed. Re-run with --write to apply the migration.")


if __name__ == "__main__":
    main()
