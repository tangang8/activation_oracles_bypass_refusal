#!/usr/bin/env python3
"""Tokenize every cached oracle response and report token-count stats per oracle
prompt and per probe segment / token point.

Purpose: decide empirically whether any segment is hitting a HARD LENGTH CAP
(a large spike of responses piled at a single exact token count == the group max)
versus ending naturally on EOS (a spread of counts that tails off well below the
configured max_new_tokens).

Scans the *assembled* rollout caches only (skips hash-named leaf files and judge
files so nothing is double counted).

Usage:
  python count_oracle_tokens.py                        # prompt-only experiment, both oracle prompts
  python count_oracle_tokens.py --include-combined     # also the combined/deterministic caches
  python count_oracle_tokens.py --cache-root cache --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

HASH_RE = re.compile(r"^[0-9a-f]{16}$")
SEGMENT_KINDS = ("full_seq", "segment", "prompt_segment", "rollout_segment")
DICT_KINDS = ("tokens", "token_points")


def _as_texts(v) -> list[str]:
    """Normalize a stored response value (str | list | None) to a list of strings."""
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str) and x.strip()]
    return []


def iter_responses(entry: dict):
    """Yield (probe_kind, subkey, text) for every generated string in an entry."""
    resp = entry.get("oracle_response") or {}
    for kind in SEGMENT_KINDS:
        for t in _as_texts(resp.get(kind)):
            yield kind, None, t
    for kind in DICT_KINDS:
        d = resp.get(kind)
        if isinstance(d, dict):
            for sub, v in d.items():
                for t in _as_texts(v):
                    yield kind, str(sub), t


def oracle_label(entry: dict, path: str) -> str:
    p = entry.get("oracle_prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()[:55]
    # fall back to the oracle-prompt directory name in the path
    parts = path.split(os.sep)
    for i, seg in enumerate(parts):
        if seg.endswith("_temp-1.0") or seg.endswith("_temp-0.0"):
            if i + 1 < len(parts):
                return parts[i + 1][:55]
    return "<unknown>"


def find_files(cache_root: str, include_combined: bool) -> list[str]:
    dirs = ["oracle_prompt_rollouts_temp-*"]
    if include_combined:
        dirs.append("oracle_rollouts_temp-*")
    files: list[str] = []
    for d in dirs:
        files += glob.glob(os.path.join(cache_root, "**", d, "**", "*.json"), recursive=True)
    out = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if HASH_RE.match(stem):      # leaf compute-cache file
            continue
        if "judged" in f:            # judge output (would double count)
            continue
        out.append(f)
    return sorted(set(out))


def summarize(counts: list[int]) -> dict:
    counts = sorted(counts)
    n = len(counts)

    def pct(p: float) -> int:
        return counts[min(n - 1, int(p * n))]

    mx = counts[-1]
    at_max = sum(1 for c in counts if c >= mx - 1)          # within 1 token of the max
    mode_val, mode_ct = Counter(counts).most_common(1)[0]   # most common exact count
    return {
        "n": n,
        "min": counts[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": mx,
        "mean": round(statistics.mean(counts), 1),
        "at_max_pm1_pct": round(100 * at_max / n, 1),
        "mode_val": mode_val,
        "mode_pct": round(100 * mode_ct / n, 1),
    }


def cap_verdict(s: dict, configured_max: int | None) -> str:
    """Heuristic: a hard cap looks like a big pile-up at one exact value that is also the max."""
    piled = s["mode_pct"] >= 20.0 and abs(s["mode_val"] - s["max"]) <= 1
    near_cfg = configured_max is not None and s["max"] >= configured_max - 2
    if piled and near_cfg:
        return "HARD CAP at max_new_tokens"
    if piled:
        return f"HARD CAP-like ({s['mode_pct']}% at {s['mode_val']} tok)"
    if s["at_max_pm1_pct"] >= 20.0:
        return f"cap-like ({s['at_max_pm1_pct']}% at ceiling)"
    return "natural (spread / EOS)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", default="cache")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="tokenizer to load")
    ap.add_argument("--include-combined", action="store_true", help="also scan oracle_rollouts_temp-* (combined/deterministic)")
    ap.add_argument("--configured-max", type=int, default=1000, help="max_new_tokens the run used (for cap verdict)")
    ap.add_argument("--limit-files", type=int, default=0, help="cap number of files (debug)")
    ap.add_argument("--batch", type=int, default=512, help="tokenizer batch size")
    ap.add_argument("--token-points", action="store_true", help="also break token_points down by name")
    args = ap.parse_args()

    files = find_files(args.cache_root, args.include_combined)
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No assembled oracle files found under {args.cache_root!r}")
    print(f"Scanning {len(files)} assembled cache files under {args.cache_root!r} ...")

    # Collect records: (oracle_label, kind, subkey, text)
    records: list[tuple[str, str, str | None, str]] = []
    for f in files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            lbl = oracle_label(entry, f)
            for kind, sub, text in iter_responses(entry):
                records.append((lbl, kind, sub, text))
    if not records:
        raise SystemExit("Found files but no oracle_response strings inside them.")
    print(f"Collected {len(records)} response strings. Loading tokenizer {args.model!r} ...")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    # Batch-tokenize all texts.
    texts = [r[3] for r in records]
    tok_counts: list[int] = []
    for i in range(0, len(texts), args.batch):
        enc = tok(texts[i : i + args.batch], add_special_tokens=False)["input_ids"]
        tok_counts.extend(len(x) for x in enc)
        print(f"  tokenized {min(i + args.batch, len(texts))}/{len(texts)}", end="\r")
    print()

    # Aggregate.
    by_prompt_kind: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_prompt_tp: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (lbl, kind, sub, _), ntok in zip(records, tok_counts):
        by_prompt_kind[(lbl, kind)].append(ntok)
        if args.token_points and kind == "token_points" and sub is not None:
            by_prompt_tp[(lbl, sub)].append(ntok)

    hdr = f"{'oracle prompt':<57} {'segment':<15} {'n':>6} {'min':>4} {'p50':>4} {'p90':>4} {'p99':>4} {'max':>5} {'mean':>6} {'%@max':>6} {'mode':>10}  verdict"
    print("\n" + "=" * len(hdr))
    print("PER ORACLE PROMPT x SEGMENT (token counts)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for (lbl, kind) in sorted(by_prompt_kind):
        s = summarize(by_prompt_kind[(lbl, kind)])
        print(
            f"{lbl:<57.57} {kind:<15} {s['n']:>6} {s['min']:>4} {s['p50']:>4} {s['p90']:>4} {s['p99']:>4} "
            f"{s['max']:>5} {s['mean']:>6} {s['at_max_pm1_pct']:>5}% {str(s['mode_val'])+'('+str(s['mode_pct'])+'%)':>10}  "
            f"{cap_verdict(s, args.configured_max)}"
        )

    if args.token_points and by_prompt_tp:
        print("\n" + "=" * len(hdr))
        print("PER ORACLE PROMPT x TOKEN POINT")
        print("=" * len(hdr))
        for (lbl, sub) in sorted(by_prompt_tp):
            s = summarize(by_prompt_tp[(lbl, sub)])
            print(
                f"{lbl:<57.57} {sub:<15.15} {s['n']:>6} {s['min']:>4} {s['p50']:>4} {s['p90']:>4} {s['p99']:>4} "
                f"{s['max']:>5} {s['mean']:>6} {s['at_max_pm1_pct']:>5}% {str(s['mode_val'])+'('+str(s['mode_pct'])+'%)':>10}  "
                f"{cap_verdict(s, args.configured_max)}"
            )
    print("\nverdict legend: 'HARD CAP' = big pile-up at one exact token count == max (length-limited);")
    print("                'natural'  = counts spread out and tail off below the configured max (EOS-terminated).")


if __name__ == "__main__":
    main()
