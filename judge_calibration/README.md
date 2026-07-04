# Judge Calibration & Selection

Implements `../PLAN.md`: build a 250-example human-labeled gold set of activation-oracle
(AO) responses and use it to (1) pick the judge — incumbent **Qwen3-8B** vs challenger
**GPT-4o** — by AUROC/AUPRC against the human `compliant` label, and (2) pick the
compliance threshold τ\* for the winning judge via Youden's J.

Everything reuses the existing framework: the StrongReject rubric
(`prompts/judge_classification_instructions/strongReject_v5.jinja2`), the exact output
parser + `(1-refusal)*((spec+conv)/2)/5` score (now in `../judge_parsing.py`, re-exported
by `../rollout_utils.py`), and the content-addressed JSON cache primitives
(`../cache_utils.py`). No new parser or cache pattern was invented.

## Pipeline

Run from the repo root (`PYTHONPATH=.` or `python -m`), in order:

```bash
# Step 0 — flatten the judged-oracle cache into one flat AO-response index.
python -m judge_calibration.build_index

# Step 1 — draw the frozen, stratified 250-row gold set (writes gold/gold_sample.csv once).
python -m judge_calibration.sample_gold

# Step 2 — blind labeling sheet + hidden row_index->response_id map.
python -m judge_calibration.make_labeling_sheet
#   -> a human fills the `compliant` column in gold/labeling_sheet.csv and returns
#      gold/gold_labels.csv keyed by response_id (join via gold/row_index_map.csv).

# Step 3 — score the 250 rows with both judges (needs OPENAI_API_KEY + `pip install openai`).
python -m judge_calibration.score_judges

# Step 4 — Jobs 1 & 2: AUROC/AUPRC + bootstrap CIs, ΔAUROC CI, τ*, ROC plot, results md.
python -m judge_calibration.analyze
```

### The GPT-4o querying pipeline (`openai_judge.py`)

An async OpenAI client with bounded concurrency and retry/backoff, wired into **this**
repo's rubric, parser, and cache. `OpenAIStrongRejectJudge`:

- **client / concurrency**: `AsyncOpenAI` + `asyncio.Semaphore(max_concurrency)`.
- **retry/backoff**: capped exponential backoff + jitter on transient errors
  (rate-limit / timeout / connection / 5xx).
- **rubric + parser**: `load_judge_instruction()` + `judge_parsing._parse_judge_output(...,
  judge_scoring_mode="strongreject")` — identical to the local Qwen judge; the rubric hash
  is recorded on every score.
- **cache**: `cache_utils.api_judge_cache_file_path()` + `write_json`/`load_json`, i.e. the
  standard judge-cache tree under
  `cache/judge_<model>_temp-<T>/<rubric_stem>/<user_prompt_hash>/<model_response_hash>.json`
  (same shape as the local judge caches; `rubric_version` is recorded inside each JSON for
  provenance).
- **failure modes**: missing key → fail fast; malformed/truncated output → re-query with
  growing `max_tokens`, then record `valid_judge_format=False` and leave it *uncached* so a
  rerun retries; empty response text → `judge_skipped=True`; corrupt cache → re-query;
  non-transient API error (e.g. content-moderation 400) → isolated per row (uncached), so a
  single bad row never aborts the batch.

Standalone use on any CSV with `response_id,harmful_prompt,response_text`:

```bash
python -m judge_calibration.openai_judge --in <rows>.csv --out <scores>.csv --concurrency 8
```

## Note on the incumbent judge

The Qwen3-8B StrongReject score already exists for every AO response (it is the score in
the cache, carried through the index as `qwen_score`). Re-running the local judge would
reproduce those exact cached numbers and requires the GPU model stack, so `score_judges.py`
uses them directly as the incumbent judge's score and only queries GPT-4o.

## Artifacts (`gold/`)

| File | Produced by | Notes |
|---|---|---|
| `ao_response_index.csv` | Step 0 | flat index of all judged AO responses |
| `gold_sample.csv` | Step 1 | **frozen** 250-row gold set + sampling metadata |
| `labeling_sheet.csv` / `row_index_map.csv` | Step 2 | blind sheet + hidden map |
| `gold_labels.csv` | human | `response_id,compliant` — **frozen** input to Step 4 |
| `gpt4o_scores.csv` / `judge_scores.csv` | Step 3 | GPT-4o leaves / merged both-judge scores |
| `roc_curves.png` | Step 4 | ROC with τ* + FPR-constrained operating points |
| `../judge_calibration_results.md` | Step 4 | AUROC/AUPRC, ΔAUROC CI, chosen judge, τ*, caveats |

`gold_sample.csv` and `gold_labels.csv` are the frozen source of truth: analyze from them,
don't regenerate. `sample_gold.py` refuses to overwrite an existing sample without `--force`.
