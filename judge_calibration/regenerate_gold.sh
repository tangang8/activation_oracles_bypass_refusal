#!/usr/bin/env bash
# Regenerate the gold-set artifacts (Steps 0-2) from the CURRENT judged-oracle cache.
#
# Run this after re-generating / re-judging the oracle rollouts, to rebuild the index and
# re-draw the gold set. Overwrites:
#   gold/ao_response_index.csv   (Step 0 - build_index)
#   gold/gold_sample.csv         (Step 1 - sample_gold --force; otherwise write-once)
#   gold/labeling_sheet.csv      (Step 2 - make_labeling_sheet)
#   gold/row_index_map.csv       (Step 2)
#
# It does NOT run the judges or analysis (Steps 3-4), and does NOT delete gold_labels.csv.
# WARNING: re-sampling changes which responses are in the gold set, so any human labels from
# a previous sample (gold_labels.csv) will no longer line up and must be redone.
#
# Usage (from anywhere):
#   ./judge_calibration/regenerate_gold.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."          # repo root
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[regenerate_gold] Step 0: build_index"
python -m judge_calibration.build_index

echo "[regenerate_gold] Step 1: sample_gold --force"
python -m judge_calibration.sample_gold --force

echo "[regenerate_gold] Step 2: make_labeling_sheet"
python -m judge_calibration.make_labeling_sheet

echo "[regenerate_gold] done. Next (separate steps): label gold/labeling_sheet.csv -> gold/gold_labels.csv,"
echo "                  then: python -m judge_calibration.score_judges && python -m judge_calibration.analyze"
