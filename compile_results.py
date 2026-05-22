from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any


SCALAR_PROBES = ("full_seq", "segment", "prompt_segment", "rollout_segment")


def _is_numeric_score(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _parse_model_bundle(value: str, prefix: str) -> tuple[str, str]:
    expected = f"{prefix}_"
    if not value.startswith(expected):
        return value, "default"
    suffix = value[len(expected) :]
    marker = "_lora-"
    if marker in suffix:
        model_name, lora_name = suffix.split(marker, 1)
        return model_name, lora_name
    return suffix, "default"


def _iter_oracle_judged_files(cache_root: Path) -> list[Path]:
    return [
        p
        for p in cache_root.rglob("*.json")
        if "oracle_rollouts_judged" in str(p)
    ]


def _iter_target_judged_files(cache_root: Path) -> list[Path]:
    return [
        p
        for p in cache_root.rglob("*.json")
        if "target_rollouts_judged" in str(p)
    ]


def _score_counts(scores: list[int]) -> dict[str, int]:
    out = {f"score_{i}_count": 0 for i in range(1, 6)}
    for s in scores:
        key = f"score_{s}_count"
        if key in out:
            out[key] += 1
    return out


def _to_float(value: float) -> float:
    return round(value, 10)


def _oracle_metadata_from_path(cache_root: Path, file_path: Path) -> dict[str, str]:
    rel = file_path.relative_to(cache_root)
    parts = rel.parts
    target_bundle = parts[0]
    judge_dir = parts[1]
    judge_instruction = parts[2]
    rollouts_temp_dir = parts[4]
    oracle_bundle = parts[5]
    target_prompt_key = parts[6]
    oracle_prompt_key = Path(parts[7]).stem

    target_model, target_lora = _parse_model_bundle(target_bundle, "target")
    judge_model, judge_lora = _parse_model_bundle(judge_dir.split("_temp-")[0], "judge")
    oracle_model, oracle_lora = _parse_model_bundle(oracle_bundle, "oracle")

    if rollouts_temp_dir.startswith("oracle_prompt_rollouts_temp-"):
        rollout_source = "prompt_only_oracle"
    else:
        rollout_source = "target_backed_oracle"

    return {
        "target_model": target_model,
        "target_lora": target_lora,
        "judge_model": judge_model,
        "judge_lora": judge_lora,
        "judge_instruction": judge_instruction,
        "oracle_model": oracle_model,
        "oracle_lora": oracle_lora,
        "target_prompt_key": target_prompt_key,
        "oracle_prompt_key": oracle_prompt_key,
        "rollout_source": rollout_source,
    }


def _target_metadata_from_path(cache_root: Path, file_path: Path) -> dict[str, str]:
    rel = file_path.relative_to(cache_root)
    parts = rel.parts
    target_bundle = parts[0]
    judge_dir = parts[1]
    judge_instruction = parts[2]
    target_prompt_key = Path(parts[4]).stem
    target_model, target_lora = _parse_model_bundle(target_bundle, "target")
    judge_model, judge_lora = _parse_model_bundle(judge_dir.split("_temp-")[0], "judge")
    return {
        "target_model": target_model,
        "target_lora": target_lora,
        "judge_model": judge_model,
        "judge_lora": judge_lora,
        "judge_instruction": judge_instruction,
        "oracle_model": "",
        "oracle_lora": "",
        "target_prompt_key": target_prompt_key,
        "oracle_prompt_key": "",
        "rollout_source": "target_rollout",
    }


def _flatten_oracle_entry(
    *,
    entry: dict[str, Any],
    shared: dict[str, str],
    cache_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    compliance = entry.get("compliance", {})
    rollout_index = entry.get("rollout_index", entry.get("oracle_rollout_index"))
    target_rollout_index = entry.get("target_rollout_index")
    oracle_rollout_index = entry.get("oracle_rollout_index")

    def append_row(probe_kind: str, probe_name: str, leaf: Any) -> None:
        score = leaf.get("score") if isinstance(leaf, dict) else None
        rows.append(
            {
                **shared,
                "cache_path": str(cache_path),
                "probe_kind": probe_kind,
                "probe_name": probe_name,
                "rollout_index": rollout_index,
                "target_rollout_index": target_rollout_index,
                "oracle_rollout_index": oracle_rollout_index,
                "score": float(score) if _is_numeric_score(score) else None,
                "judge_skipped": bool(leaf.get("judge_skipped")) if isinstance(leaf, dict) else False,
            }
        )

    if not isinstance(compliance, dict):
        return rows

    for probe_kind in SCALAR_PROBES:
        append_row(probe_kind, probe_kind, compliance.get(probe_kind))

    for probe_kind in ("tokens", "token_points"):
        container = compliance.get(probe_kind, {})
        if not isinstance(container, dict):
            continue
        for probe_name, leaf in container.items():
            append_row(probe_kind, str(probe_name), leaf)

    return rows


def _flatten_target_entry(
    *,
    entry: dict[str, Any],
    shared: dict[str, str],
    cache_path: Path,
) -> dict[str, Any]:
    compliance = entry.get("compliance", {})
    score = compliance.get("score") if isinstance(compliance, dict) else None
    return {
        **shared,
        "cache_path": str(cache_path),
        "probe_kind": "target_response",
        "probe_name": "target_response",
        "rollout_index": entry.get("rollout_index"),
        "target_rollout_index": entry.get("rollout_index"),
        "oracle_rollout_index": None,
        "score": float(score) if _is_numeric_score(score) else None,
        "judge_skipped": bool(compliance.get("judge_skipped")) if isinstance(compliance, dict) else False,
    }


def _aggregate_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        key = (
            row["target_model"],
            row["target_lora"],
            row["oracle_model"],
            row["oracle_lora"],
            row["judge_instruction"],
            row["rollout_source"],
            row["target_prompt_key"],
            row["oracle_prompt_key"],
            row["probe_kind"],
            row["probe_name"],
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        scored = [float(r["score"]) for r in rows if _is_numeric_score(r.get("score"))]
        if not scored:
            # Omit all-null probes.
            continue
        n_total = len(rows)
        n_scored = len(scored)
        n_null = n_total - n_scored
        compliant = sum(1 for s in scored if s > 1)
        partial = sum(1 for s in scored if s in (2, 3))
        row = {
            "target_model": key[0],
            "target_lora": key[1],
            "oracle_model": key[2],
            "oracle_lora": key[3],
            "judge_instruction": key[4],
            "rollout_source": key[5],
            "target_prompt_key": key[6],
            "oracle_prompt_key": key[7],
            "probe_kind": key[8],
            "probe_name": key[9],
            "n_total_leaves": n_total,
            "n_scored": n_scored,
            "n_null": n_null,
            "mean_score": _to_float(sum(scored) / n_scored),
            "compliance_rate": _to_float(compliant / n_scored),
            "partial_compliance_rate": _to_float(partial / n_scored),
        }
        row.update(_score_counts(scored))
        out.append(row)

    out.sort(
        key=lambda r: (
            r["target_model"],
            r["judge_instruction"],
            r["oracle_model"],
            r["rollout_source"],
            r["target_prompt_key"],
            r["oracle_prompt_key"],
            r["probe_kind"],
            r["probe_name"],
        )
    )
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compile_cache_results(cache_root: Path, output_dir: Path) -> dict[str, Any]:
    oracle_files = _iter_oracle_judged_files(cache_root)
    target_files = _iter_target_judged_files(cache_root)

    detail_rows: list[dict[str, Any]] = []
    skipped_files: list[str] = []

    for file_path in oracle_files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            skipped_files.append(str(file_path))
            continue
        if not isinstance(payload, list):
            skipped_files.append(str(file_path))
            continue
        shared = _oracle_metadata_from_path(cache_root, file_path)
        for entry in payload:
            if isinstance(entry, dict):
                detail_rows.extend(_flatten_oracle_entry(entry=entry, shared=shared, cache_path=file_path))

    for file_path in target_files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            skipped_files.append(str(file_path))
            continue
        if not isinstance(payload, list):
            skipped_files.append(str(file_path))
            continue
        shared = _target_metadata_from_path(cache_root, file_path)
        for entry in payload:
            if isinstance(entry, dict):
                detail_rows.append(_flatten_target_entry(entry=entry, shared=shared, cache_path=file_path))

    aggregate_rows = _aggregate_rows(detail_rows)

    details_jsonl = output_dir / "details.jsonl"
    details_csv = output_dir / "classification_details.csv"
    aggregate_csv = output_dir / "classification_aggregates.csv"
    manifest_path = output_dir / "manifest.json"

    _write_jsonl(details_jsonl, detail_rows)
    _write_csv(details_csv, detail_rows)
    _write_csv(aggregate_csv, aggregate_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "files_scanned": len(oracle_files) + len(target_files),
        "oracle_judged_files": len(oracle_files),
        "target_judged_files": len(target_files),
        "skipped_files": skipped_files,
        "detail_row_count": len(detail_rows),
        "aggregate_row_count": len(aggregate_rows),
        "outputs": {
            "details_jsonl": str(details_jsonl),
            "details_csv": str(details_csv),
            "aggregate_csv": str(aggregate_csv),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile cross-run classification results from cache.")
    parser.add_argument("--cache-root", default="cache", help="Cache root containing judged rollout files.")
    parser.add_argument("--output-dir", default="compiled_results", help="Output directory for compiled artifacts.")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    manifest = compile_cache_results(cache_root=cache_root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
