"""Shared configuration and small helpers for the judge-calibration pipeline.

Everything the calibration scripts need to agree on lives here: the repo root (so the
existing `cache_utils` / `rollout_utils` / `judge_instruction_utils` modules import
cleanly), the frozen output directory, the rubric identity, and the oracle-prompt A/B
mapping. Keeping it in one place mirrors the repo's "cache keys are the contract"
philosophy: if the rubric or judge model changes, that change is visible here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# --- repo wiring -----------------------------------------------------------------
# This package lives at <repo>/judge_calibration/. The existing framework modules
# (cache_utils, rollout_utils, judge_instruction_utils, prompt_utils, results/*) are
# imported by path, exactly like the repo runs with PYTHONPATH=".".
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_RESULTS_DIR = REPO_ROOT / "results"
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))

# --- frozen artifacts -------------------------------------------------------------
GOLD_DIR = Path(__file__).resolve().parent / "gold"
INDEX_CSV = GOLD_DIR / "ao_response_index.csv"
GOLD_SAMPLE_CSV = GOLD_DIR / "gold_sample.csv"
LABELING_SHEET_CSV = GOLD_DIR / "labeling_sheet.csv"
ROW_INDEX_MAP_CSV = GOLD_DIR / "row_index_map.csv"  # hidden: row_index -> response_id
GOLD_LABELS_CSV = GOLD_DIR / "gold_labels.csv"       # human returns this (response_id, compliant)
JUDGE_SCORES_CSV = GOLD_DIR / "judge_scores.csv"     # response_id, qwen_score, gpt4o_score
RESULTS_MD = Path(__file__).resolve().parent / "judge_calibration_results.md"
ROC_PLOT_PNG = GOLD_DIR / "roc_curves.png"

# --- judge identity ---------------------------------------------------------------
# The incumbent judge already scored every AO response; those scores live in the cache
# and are carried through the index as `qwen_score`. The challenger is a GPT-4o judge
# that shares the exact same rubric prompt and parser (see openai_judge.py).
CACHE_ROOT = REPO_ROOT / "cache"
RUBRIC_PATH = "strongReject_v5.jinja2"  # passed to load_judge_instruction()
QWEN_JUDGE_MODEL = "Qwen/Qwen3-8B"
GPT4O_JUDGE_MODEL = "gpt-4o"
JUDGE_TEMPERATURE = 0.0

# --- oracle prompt A/B mapping ----------------------------------------------------
# The two oracle-prompt files, in the same order the compile pipeline uses them.
ORACLE_PROMPT_FILES = {
    "A": "prompts/oracle_prompts/default_oracle_prompts.json",
    "B": "prompts/oracle_prompts/model_answer_min_200_words.json",
}

# --- sampling (Step 1) ------------------------------------------------------------
GOLD_N = 250
BAND_DRAWS = {"H": 100, "M": 50, "Z": 60, "U": 40}
SAMPLING_SEED = 20240501
# Slices the headline results lean on; sampling guarantees each appears in the gold set.
HEADLINE_SLICES = (
    "trailing_im_start_token",
    "token_before_im_end",
    "first_token_after_think_close",
)
EXPERIMENTS = ("user_prompt_oracle", "target_rollout_oracle")

LABELING_SEED = 20240502


def rubric_version(rubric_template_text: str) -> str:
    """Stable short hash of the rubric text, recorded alongside every judge score."""
    return hashlib.sha256(rubric_template_text.encode("utf-8")).hexdigest()[:12]


def band_for_score(score: float) -> str:
    """Stratification band from a StrongReject score in {0} union (0, 1]."""
    if score >= 0.5:
        return "H"
    if score > 0.0:
        return "M"
    return "Z"
