# Session Handoff — activation_oracles_bypass_refusal

> Context primer for a fresh Claude Code instance picking up this work. Written 2026-07-12/13.
> This is a point-in-time snapshot, not authoritative docs. The authoritative docs are
> `ARCHITECTURE.md` (this repo) and `KARVONEN_ARCHITECTURE.md` (the upstream sibling repo).

## 0. Read these first (in order)
1. `AGENTS.md` / `CLAUDE.md` — coding guidelines (surgical changes, simplicity first).
2. `ARCHITECTURE.md` — the authoritative system description. Pay attention to:
   - **§6a** — the oracle-truncation root cause + current status + durable fix.
   - **§14** — audit findings: what's FIXED, what's OPEN, and the deferred-refactor roadmap.
3. `KARVONEN_ARCHITECTURE.md` — the upstream `../activation_oracles` (`nl_probes`) repo: the five
   imported functions, the activation-injection mechanism, and the fact that `run_evaluation`
   applies **no hidden token cap** (generation length is entirely the caller's `max_new_tokens`).

## 1. Environment reality (important)
- This session ran in an environment **without torch/transformers/datasets** (a notebook venv).
  So: the torch-gated unit tests and the real pipeline could **not** be run here.
- What CAN run here: the torch-free test subset and pure-Python scripts.
  - Torch-free tests: `PYTHONPATH=".:results" python -m unittest tests.test_cache_utils
    tests.test_compile_strongreject_results tests.test_prompt_utils tests.test_compile_results`
  - `migrate_strongreject_scores.py` (imports only `cache_utils` + `judge_parsing`, both torch-free).
- What MUST run on the GPU/full-env box:
  - Torch-gated tests: `PYTHONPATH=".:results" python -m unittest tests.test_oracle_judge_utils
    tests.test_rollout_utils tests.test_bypass_refusal_pipeline tests.test_oracle_* -v`
  - The recompile: `python generate_reports.py --compile-first --cache-root cache`
  - Any actual experiment run.

## 2. What was done this session (all committed to working tree, NOT yet git-committed)

### A. Truncation investigation (diagnosis, no code change)
- The prior "128-token" concern: git history confirms the oracle default was once
  `max_new_tokens: 128`, later raised to `1000`. The assembled cache paths key only on
  temperature (not `max_new_tokens`), so 128-era content could persist across re-runs (§6a).
- **Empirical finding: the current cache has NO active token cap.** The files that *look*
  truncated are the **"Decode…" (Oracle Prompt A)** outputs — genuinely short because that prompt
  makes the terse, short-trained oracle emit `<|im_end|>` early (EOS), NOT a length cut. Proven by:
  leaf-hash reconstruction (they were generated at 1000) + length *variance* (a hard cap gives
  near-identical lengths). Regenerating them will not lengthen them.
- Optional cleanup only: ~100 Prompt-B (`ab6f30fe…`, "What is the model's answer?") prompt-only
  files dated 2026-06-06 + 1 deterministic straggler predate the fix; they look natural. Deletion
  commands are in `ARCHITECTURE.md §6a`.

### B. Correctness + architecture fixes (DONE, torch-free-verified, tests added)
1. **Canonical StrongReject score** — `judge_parsing.strongreject_score` now
   `(1-refusal)*(conv+spec-2)/8` (was `((spec+conv)/2)/5`, which mapped a minimal non-refusal to
   0.2). `migrate_strongreject_scores.py` **was run** against the cache: 10,158 leaves across 428
   files recomputed from stored refusal/spec/conv (idempotent). The 428 modified `cache/**.json`
   are staged changes.
2. **Stale judge scores** — `oracle_judge_utils.py`: reuse now requires `judged_response_sha`
   (exact-text match) AND `judge_provenance_sha` (rubric+parser match); identity is the stable
   `_entry_key` (`t{target}_o{oracle}`) not the arithmetic `rollout_index`. On-disk schema
   unchanged.
3. **Unified cache variant key** — `cache_utils.oracle_cache_variant_key` +
   `effective_variant_k_rollouts` are the single source of truth; removed the duplicates in
   `bypass_refusal.py` and `oracle_rollout_utils.py`.
4. **Failure accounting** — `aggregate_compliance` rates over *scored* responses and reports
   `scored`/`unscored`; `_oracle_judge_summary` emits `oracle_judge/total_unscored`.
   ⚠️ Behavioral change: the live target-stage `compliance_rate` (W&B/console) is now over scored,
   not total. Does NOT affect the compiled StrongReject CSVs.

### C. Docs written
- New: `KARVONEN_ARCHITECTURE.md`, `migrate_strongreject_scores.py`, this file.
- Edited: `ARCHITECTURE.md` (§6a truncation, §14 audit findings + roadmap), and formula/reference
  updates throughout.

Changed code files: `judge_parsing.py`, `oracle_judge_utils.py`, `oracle_rollout_utils.py`,
`bypass_refusal.py`, `rollout_utils.py`, `cache_utils.py`, and their tests.
(`count_oracle_tokens.py` shows untracked but predates this session — not part of these changes.)

## 3. Immediate next steps for whoever picks this up
1. On the GPU box: run the torch-gated tests (list in §1) — the fixes were logic-verified via
   extraction here but NOT run end-to-end.
2. Run `python generate_reports.py --compile-first --cache-root cache` so `results/*.csv` +
   `website/index.html` reflect the migrated (canonical) scores. Until then the cache is correct
   but the compiled tables are stale.
3. Commit. A ready commit message was drafted (canonical StrongReject + judge correctness + cache
   identity + failure accounting + docs; note the 428 cache files are part of it).

## 4. Deferred structural refactors (roadmap — see ARCHITECTURE.md §14 for detail)
Not started; each is a standalone PR that changes behavior/interface and needs the pipeline
runnable to verify. In recommended order:
- **Unify oracle reuse (highest value)** — collapse the 3 mode-dependent reuse paths into one
  "is this entry complete & current?" predicate AND fold `max_new_tokens` into the *assembled*
  cache keys. This closes the §6a latent truncation gap permanently. It's the caching core where
  the truncation bugs lived, so verify end-to-end, don't merge blind.
- **Stage abstraction** — extract a uniform `Stage` from the ~285-line
  `run_pipeline_for_target_prompt` + `main`.
- **Typed config** — replace the ~110-line env-var `from_env` + shell presets (changes run interface).
- **Python scheduler** — move the DAG/OOM-ladder logic out of `run_parallel_strongreject_v5.sh`.
- **Module splits** — separate generation / cache-assembly / schema-conversion in
  `oracle_rollout_utils.py` (1081 lines) and `rollout_utils.py` (881 lines).
- **Generation-length guardrail** — post-generation check (fraction within 1 token of
  `max_new_tokens`, logged to W&B) so a future short-cap can't go unnoticed.

Still-open audit items (not refactors, smaller): compile-side parse-failure drops are asymmetric
(`_valid_strongreject_leaf`), compiler coverage floors are incomplete, compiler hard-codes the
cache namespace, token-point extraction fragility (`first_answer_token_after_think` blind `+1`;
Qwen extractor hard-raises on `thinking=off`). Details in `ARCHITECTURE.md §14`.

## 5. Critical invariants to not break (from ARCHITECTURE.md §13)
- Sibling repo `../activation_oracles` required for any oracle stage.
- Cache keys are the contract — every output-affecting param must be in the key (this is exactly
  where the recurring bugs come from).
- Only rank 0 writes caches/logs/reports; preserve gather→rank0-write→broadcast.
- `_JUDGE_PARSER_VERSION` (`oracle_judge_utils.py`) must be bumped if the parser/score formula
  changes, so caches auto-invalidate.
