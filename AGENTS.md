# AGENTS.md

This file provides guidance to agents (Claude Code, Cursor, etc.) when working in this repository. `CLAUDE.md` is a symlink to this file (`ln -s AGENTS.md CLAUDE.md`).

## How the codebase works → ARCHITECTURE.md

**For anything about how the code actually works — data flow, function names, the four-stage
pipeline, oracle rollout modes, the activation-probe engine, cache schema, env-var contracts,
results compilation, and per-file responsibilities — read [ARCHITECTURE.md](ARCHITECTURE.md).**
It is the authoritative, agent-facing description of the system.

This file (AGENTS.md) intentionally holds only **coding guidelines** and **operational
commands**. It does not duplicate the architecture, so that mechanics live in exactly one
place and can't drift out of sync. The ground truth for configuration is
`ExperimentConfig.from_env()` in `bypass_refusal.py`; the ground truth for result aggregation
is `results/compile_strongreject_results.py`.

## Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. These bias toward caution over speed; for trivial tasks, use judgment.

### Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it — do not delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.
- If you make a change, make sure you understand how it impacts other code and update all dependencies to be compatible. 

Every changed line should trace directly to the user's request.

### Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → write tests for invalid inputs, then make them pass
- "Fix the bug" → write a test that reproduces it, then make it pass
- "Refactor X" → ensure tests pass before and after

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria allow independent verification. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** diffs have fewer unnecessary changes, fewer rewrites from overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Directory Structure & Sibling Dependency

This repo requires a specific parent-folder layout:

```
<parent-folder>/
├── activation_oracles_bypass_refusal/  (this repo)
└── activation_oracles/                 (upstream repo)
```

Both repos must be cloned with these exact names. The upstream `activation_oracles` repo is
imported via `sys.path` manipulation in `oracle_pipeline.py` (`nl_probes.utils.*`) to access
the activation probing utilities. If oracle stages fail to import, this sibling is almost
always missing or renamed.

## Key Commands

### Run Tests
```bash
# From repo root, run all tests
PYTHONPATH=".:results" python -m unittest discover -v -s tests

# Run a specific test file
PYTHONPATH=".:results" python -m unittest tests.test_prompt_utils -v

# Run a specific test class or method
PYTHONPATH=".:results" python -m unittest tests.test_prompt_utils.PromptUtilsTests.test_prompt_key_hash_length -v
```

### Bash Script Syntax Check
```bash
bash -n run_oracle_experiment.sh
bash -n run_parallel_strongreject_v5.sh
```

### Run Experiments

**Easy preset-based runs** (recommended):
```bash
./run_oracle_experiment.sh --preset full_deterministic_oracle
./run_oracle_experiment.sh --preset sampled_target_repeats --k-rollouts 5 --num-oracle-rollouts 2
./run_oracle_experiment.sh --preset prompt_only_oracle --num-oracle-rollouts 4
```

**Small validation run** (one prompt per mode):
```bash
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=3 NUM_ORACLE_ROLLOUTS=1 ORACLE_ROLLOUT_MODE=all_target_deterministic python bypass_refusal.py
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=5 K_ROLLOUTS=2 NUM_ORACLE_ROLLOUTS=2 ORACLE_ROLLOUT_MODE=sampled_target_repeats python bypass_refusal.py
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=3 NUM_ORACLE_ROLLOUTS=3 ORACLE_ROLLOUT_MODE=prompt_only_repeats python bypass_refusal.py
```

**Direct Python entry** (if not using the bash wrapper):
```bash
ORACLE_ROLLOUT_MODE=prompt_only_repeats NUM_ORACLE_ROLLOUTS=3 TARGET_PROMPT_LIMIT=1 python bypass_refusal.py
```

**Parallel StrongReject v5** (multi-GPU scheduler; default preset for deterministic shards is `rollout_post_prompt_oracle`):
```bash
./run_parallel_strongreject_v5.sh
# Optional: DRY_RUN=1 GPU_IDS=0,1 ./run_parallel_strongreject_v5.sh
# Logs: logs/parallel_<timestamp>/parallel_driver.log
# Override label: RUN_LABEL=my_run LOG_ROOT=logs/my_run ./run_parallel_strongreject_v5.sh
```

Older runs may still have logs under `logs/parallel_h200_<timestamp>/` from before the script rename; new runs use `logs/parallel_<timestamp>/`.

> Preset behavior, the full env-var contract, and the scheduler's job graph / OOM retry
> ladders are documented in [ARCHITECTURE.md](ARCHITECTURE.md) (§3, §9). Do not rely on any
> env-var defaults quoted from memory — check `ExperimentConfig.from_env()`.

### Compile results & build the report
```bash
# Compile cached judge outputs → CSVs + manifest, then generate website/index.html
python generate_reports.py --compile-first --cache-root cache
```
See [ARCHITECTURE.md](ARCHITECTURE.md) §10 for the aggregation levels and outputs.

## Environment & Dependencies

- **Python version**: 3.10+ (depends on upstream activation_oracles)
- **Shared venv**: Create a single `.venv` at the parent folder and apply upstream lock:
  ```bash
  cd <parent-folder>
  python3 -m venv .venv
  source .venv/bin/activate
  uv sync --project activation_oracles --active
  ```
- **HuggingFace login**: Required for model access:
  ```bash
  huggingface-cli login --token <your_token>
  ```
- **Runtime deps** (from activation_oracles): torch, transformers, peft, tqdm, dotenv, wandb, bitsandbytes (GPU-only)

## Testing Patterns

Tests use Python's `unittest` framework and mock dependencies when necessary. Key patterns:

- Skip tests if dependencies unavailable: `@unittest.skipIf(condition, reason)`
- Use `patch()` / `SimpleNamespace` to mock external calls (transformers, torch, real models)
- Test isolation: temporary directories for file I/O, mocked models for pipeline tests
- Integration tests validate cache schema, stage output, and environment variable parsing

The `tests/` directory has one test module per source module plus the shell-script and
results-compilation tests. See [ARCHITECTURE.md](ARCHITECTURE.md) §11 for the full per-file
coverage map. (`test_oracle_chat.py` at the repo root is a standalone GPU smoke test, not part
of the unittest suite.)
