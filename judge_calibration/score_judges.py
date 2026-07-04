"""Step 3 (orchestration) -- Collect both judges' scores on the 250 gold rows.

The incumbent Qwen3-8B StrongReject score already exists for every AO response (it is
carried through the index as `qwen_score`); re-running the local judge would require the
GPU model stack and would reproduce those exact cached numbers, so we use them directly
as the incumbent judge's score. The challenger GPT-4o judge is queried here through
`openai_judge.OpenAIStrongRejectJudge` (same rubric + parser + cache style).

Output `judge_scores.csv` has one row per response_id with both judges' continuous scores
and the GPT-4o parse-status flags (parse failures are reported and left as NaN so the
analysis can drop them and report the count).

Usage:
    PYTHONPATH=. python judge_calibration/score_judges.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge_calibration import config
from judge_calibration.openai_judge import OpenAIJudgeConfig, OpenAIStrongRejectJudge, _summarize


def run(gold_sample_csv: Path, out_csv: Path, cfg: OpenAIJudgeConfig, use_cache: bool) -> None:
    with gold_sample_csv.open(encoding="utf-8") as f:
        gold = list(csv.DictReader(f))
    items = [(r["response_id"], r["harmful_prompt"], r["response_text"]) for r in gold]

    judge = OpenAIStrongRejectJudge(cfg)
    start = time.perf_counter()
    leaves = asyncio.run(judge.score_many(items, use_cache=use_cache))
    print(f"[score_judges] gpt4o: {_summarize(leaves)} elapsed={time.perf_counter()-start:.1f}s")

    by_id = {leaf["response_id"]: leaf for leaf in leaves}
    rows: list[dict[str, Any]] = []
    n_gpt_failures = 0
    for r in gold:
        leaf = by_id.get(r["response_id"], {})
        valid = leaf.get("valid_judge_format")
        gpt_score = leaf.get("score")
        gpt_usable = valid is True and isinstance(gpt_score, (int, float)) and not isinstance(gpt_score, bool)
        if not gpt_usable and not leaf.get("judge_skipped"):
            n_gpt_failures += 1
        rows.append({
            "response_id": r["response_id"],
            "experiment": r["experiment"],
            "slice": r["slice"],
            "oracle_prompt": r["oracle_prompt"],
            "sampling_band": r.get("sampling_band", ""),
            "sampling_weight": r.get("sampling_weight", ""),
            "qwen_score": r["qwen_score"],
            "gpt4o_score": gpt_score if gpt_usable else "",
            "gpt4o_valid": bool(gpt_usable),
            "gpt4o_skipped": bool(leaf.get("judge_skipped")),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[score_judges] gpt4o parse failures (excluded downstream): {n_gpt_failures}")
    print(f"[score_judges] wrote {len(rows)} rows -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the gold set with both judges.")
    parser.add_argument("--gold-sample", default=str(config.GOLD_SAMPLE_CSV))
    parser.add_argument("--out", default=str(config.JUDGE_SCORES_CSV))
    parser.add_argument("--model", default=config.GPT4O_JUDGE_MODEL)
    parser.add_argument("--temperature", type=float, default=config.JUDGE_TEMPERATURE)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = OpenAIJudgeConfig(
        model=args.model, temperature=args.temperature, max_concurrency=args.concurrency
    )
    run(Path(args.gold_sample), Path(args.out), cfg, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
