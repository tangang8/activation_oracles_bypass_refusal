"""Step 1 -- Draw the frozen 250-row gold set from the AO-response index.

Stratifies on the incumbent (`qwen_score`) band plus a uniform slice, spreads the H/M
draws across (experiment, oracle_prompt, slice), and guarantees the headline slices and
both experiments appear. The output `gold_sample.csv` is the frozen source of truth:
regenerating it is refused unless --force is passed.

Usage:
    PYTHONPATH=. python judge_calibration/sample_gold.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge_calibration import config


def _load_index(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["qwen_score"] = float(r["qwen_score"])
    return rows


def _spread_sample(
    pool: list[dict[str, Any]], k: int, rng: random.Random, key_fn: Callable[[dict], Any]
) -> list[dict[str, Any]]:
    """Round-robin across key groups so the draw spreads across strata rather than
    piling onto whichever group happens to be largest."""
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in pool:
        groups[key_fn(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    order = list(groups.keys())
    rng.shuffle(order)

    picked: list[dict] = []
    idx = 0
    while len(picked) < k and any(groups.values()):
        key = order[idx % len(order)]
        if groups[key]:
            picked.append(groups[key].pop())
        idx += 1
    return picked


def sample_gold(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    by_band: dict[str, list[dict]] = {"H": [], "M": [], "Z": []}
    for r in rows:
        by_band[config.band_for_score(r["qwen_score"])].append(r)
    band_population = {b: len(v) for b, v in by_band.items()}
    band_population["U"] = len(rows)

    selected: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], band: str, guaranteed: bool = False) -> bool:
        rid = row["response_id"]
        if rid in selected:
            return False
        new = dict(row)
        new["sampling_band"] = band
        new["guaranteed_coverage"] = guaranteed
        selected[rid] = new
        return True

    # H / M: spread across (experiment, oracle_prompt, slice). Z: plain uniform.
    spread_key = lambda r: (r["experiment"], r["oracle_prompt"], r["slice"])  # noqa: E731
    for band in ("H", "M"):
        draw = min(config.BAND_DRAWS[band], len(by_band[band]))
        for r in _spread_sample(by_band[band], draw, rng, spread_key):
            add(r, band)
    z_draw = min(config.BAND_DRAWS["Z"], len(by_band["Z"]))
    for r in rng.sample(by_band["Z"], z_draw):
        add(r, "Z")

    # U: uniform over all not-yet-selected responses.
    remaining = [r for r in rows if r["response_id"] not in selected]
    u_draw = min(config.BAND_DRAWS["U"], len(remaining))
    for r in rng.sample(remaining, u_draw):
        add(r, "U")

    # Coverage guarantees: headline slices + both experiments must appear.
    def ensure(field: str, value: str) -> None:
        if any(row[field] == value for row in selected.values()):
            return
        candidates = [r for r in rows if r[field] == value and r["response_id"] not in selected]
        if not candidates:
            print(f"[sample_gold] WARNING: no index rows with {field}={value!r} to guarantee.")
            return
        band_priority = {"H": 0, "M": 1, "Z": 2}
        candidates.sort(key=lambda r: band_priority[config.band_for_score(r["qwen_score"])])
        chosen = candidates[0]
        add(chosen, config.band_for_score(chosen["qwen_score"]), guaranteed=True)

    for exp in config.EXPERIMENTS:
        ensure("experiment", exp)
    for slice_name in config.HEADLINE_SLICES:
        ensure("slice", slice_name)

    # Reconcile to exactly GOLD_N.
    target = config.GOLD_N
    if len(selected) > target:
        removable = [
            rid for rid, row in selected.items()
            if not row["guaranteed_coverage"] and row["sampling_band"] in ("U", "H")
        ]
        rng.shuffle(removable)
        for rid in removable[: len(selected) - target]:
            del selected[rid]
    elif len(selected) < target:
        topup_pool = [r for r in by_band["H"] if r["response_id"] not in selected]
        for r in _spread_sample(topup_pool, target - len(selected), rng, spread_key):
            add(r, "H")

    final = list(selected.values())
    rng.shuffle(final)

    # Sampling metadata (weights let a later step reweight to real-world rates).
    band_sample_n = Counter(row["sampling_band"] for row in final)
    for row in final:
        band = row["sampling_band"]
        pop = band_population[band]
        n = band_sample_n[band]
        row["band_population_size"] = pop
        row["band_sample_n"] = n
        row["sampling_weight"] = round(pop / n, 6) if n else 0.0

    print(f"[sample_gold] n={len(final)} band_sample_n={dict(band_sample_n)}")
    print(f"[sample_gold] experiments={dict(Counter(r['experiment'] for r in final))}")
    print(f"[sample_gold] guaranteed={sum(1 for r in final if r['guaranteed_coverage'])}")
    return final


_EXTRA_FIELDS = [
    "sampling_band", "band_population_size", "band_sample_n", "sampling_weight",
    "guaranteed_coverage",
]


def write_sample(rows: list[dict[str, Any]], out_csv: Path) -> None:
    from judge_calibration.build_index import _FIELDNAMES  # reuse the index column order

    fieldnames = _FIELDNAMES + _EXTRA_FIELDS
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the frozen 250-row gold set.")
    parser.add_argument("--index", default=str(config.INDEX_CSV))
    parser.add_argument("--out", default=str(config.GOLD_SAMPLE_CSV))
    parser.add_argument("--seed", type=int, default=config.SAMPLING_SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing frozen sample.")
    args = parser.parse_args()

    out_csv = Path(args.out)
    if out_csv.exists() and not args.force:
        raise SystemExit(
            f"{out_csv} already exists and is the frozen gold set. Pass --force to regenerate "
            f"(this invalidates any labels/scores keyed to the old sample)."
        )

    rows = _load_index(Path(args.index))
    sample = sample_gold(rows, args.seed)
    write_sample(sample, out_csv)
    print(f"[sample_gold] wrote {len(sample)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
