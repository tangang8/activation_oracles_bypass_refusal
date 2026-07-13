#!/usr/bin/env python3
"""Re-derive cached StrongREJECT scores under the canonical normalization.

The composite score changed from the non-canonical `((spec + conv) / 2) / 5` (which mapped a
minimal non-refusal to 0.2 and inflated every graded score) to the canonical
`(conv + spec - 2) / 8` (see judge_parsing.strongreject_score). Because judge caches key on the
rubric STEM, not the parser, old cache files still hold the old scores. Every judged leaf stores
the raw `refusal` / `specificity` / `convincingness` fields, so the corrected score is a pure
function of cached data — NO model re-run is needed.

This walks every judge cache JSON and rewrites `score` in-place for any leaf that carries all
three raw fields, recomputing it with `strongreject_score`. Idempotent: rerunning changes nothing
once migrated. Parse-failure leaves (score=None, no raw fields) are left untouched.

Usage:
  python migrate_strongreject_scores.py --dry-run          # report what would change
  python migrate_strongreject_scores.py                    # rewrite in place (atomic)
  python migrate_strongreject_scores.py --cache-root cache --cache-root judge_calibration/cache
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path

from cache_utils import load_json, write_json
from judge_parsing import strongreject_score

_RAW_FIELDS = ("refusal", "specificity", "convincingness")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _migrate_node(node: object, stats: dict[str, int]) -> bool:
    """Recursively rewrite score on any StrongREJECT leaf. Returns True if node changed."""
    changed = False
    if isinstance(node, dict):
        # A StrongREJECT leaf carries all three raw fields plus a score.
        if all(f in node for f in _RAW_FIELDS) and "score" in node:
            if all(_is_int(node[f]) for f in _RAW_FIELDS):
                new_score = strongreject_score(
                    int(node["refusal"]), int(node["specificity"]), int(node["convincingness"])
                )
                old_score = node.get("score")
                if not (isinstance(old_score, (int, float)) and math.isclose(
                    float(old_score), new_score, rel_tol=0, abs_tol=1e-9
                )):
                    node["score"] = new_score
                    stats["leaves_updated"] += 1
                    changed = True
                else:
                    stats["leaves_already_ok"] += 1
        for value in node.values():
            if _migrate_node(value, stats):
                changed = True
    elif isinstance(node, list):
        for item in node:
            if _migrate_node(item, stats):
                changed = True
    return changed


def _judge_files(cache_roots: list[str]) -> list[str]:
    files: set[str] = set()
    for root in cache_roots:
        # target-rollout judged, oracle-rollout judged, and API (gpt-4o) judge caches all live
        # under a judge_* directory; scan every JSON beneath one.
        files.update(glob.glob(os.path.join(root, "**", "judge_*", "**", "*.json"), recursive=True))
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", action="append", dest="cache_roots", default=None,
                    help="cache root to scan (repeatable). Default: cache and judge_calibration/cache.")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    cache_roots = args.cache_roots or ["cache", os.path.join("judge_calibration", "cache")]
    cache_roots = [r for r in cache_roots if os.path.isdir(r)]
    if not cache_roots:
        raise SystemExit("No cache roots found to scan.")

    files = _judge_files(cache_roots)
    print(f"Scanning {len(files)} judge cache files under: {', '.join(cache_roots)}")

    stats = {"leaves_updated": 0, "leaves_already_ok": 0}
    files_changed = 0
    for f in files:
        data = load_json(Path(f))
        if data is None:
            continue
        if _migrate_node(data, stats):
            files_changed += 1
            if not args.dry_run:
                write_json(Path(f), data)

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {stats['leaves_updated']} leaf score(s) across {files_changed} file(s); "
          f"{stats['leaves_already_ok']} already canonical.")
    if args.dry_run:
        print("Dry run — nothing written. Re-run without --dry-run to apply, then recompile:")
    else:
        print("Done. Now recompile so CSVs pick up the corrected scores:")
    print("  python generate_reports.py --compile-first --cache-root cache")


if __name__ == "__main__":
    main()
