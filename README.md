# activation_oracles_extensions

Helpers and experiment utilities for activation-oracle extensions.

## Required local folder names

Use these exact sibling directory names:

```text
<parent-folder>/
├── activation_oracles_extensions
└── activation_oracles
```

- `activation_oracles_extensions` = this repo
- `activation_oracles` = upstream authors repo

## Clone commands (exact names)

```bash
cd <parent-folder>
git clone <YOUR_ACTIVATION_ORACLES_EXTENSIONS_REPO_URL> activation_oracles_extensions
git clone <UPSTREAM_AUTHORS_ACTIVATION_ORACLES_REPO_URL> activation_oracles
```

## Shared `.venv` installation at `<parent-folder>` root

To use one environment for both repos, create and use a single shared `.venv` at
`<parent-folder>` (not inside either repo):

```bash
cd <parent-folder>
python3 -m venv .venv
source .venv/bin/activate
uv sync --project activation_oracles --active
```

Alternative venv creation (equivalent):

```bash
cd <parent-folder>
uv venv .venv
source .venv/bin/activate
uv sync --project activation_oracles --active
```

This applies the upstream `activation_oracles/uv.lock` dependency set to the
active shared environment.

Optional cleanup if you previously created a repo-local venv:

```bash
rm -rf activation_oracles/.venv
```

If needed for model access:

```bash
huggingface-cli login --token <your_token>
```

Note: some upstream dependencies (for example `bitsandbytes`) may be platform-limited.
On unsupported platforms, use a Linux GPU environment.

## Notes

- Runtime deps come from pipeline/runtime code (`tqdm`, `torch`, `transformers`, `peft`, etc.).

## Oracle rollout modes

`bypass_refusal.py` supports three oracle rollout modes through `ORACLE_ROLLOUT_MODE`:

- `all_target_deterministic`:
  - uses all judged target rollouts
  - runs one oracle rollout per target at `temperature=0.0`
- `sampled_target_repeats`:
  - uses up to `K_ROLLOUTS` judged target rollouts
  - runs `NUM_ORACLE_ROLLOUTS` per selected target at `temperature=1.0`
- `prompt_only_repeats`:
  - ignores target responses and uses only formatted target prompt
  - runs `NUM_ORACLE_ROLLOUTS` oracle rollouts at `temperature=1.0`
  - caches under `oracle_prompt_rollouts_temp-1.0`

Relevant environment variables:

- `ORACLE_ROLLOUT_MODE` (default: `all_target_deterministic`)
- `NUM_ROLLOUTS` (target rollout count)
- `K_ROLLOUTS` (used by `sampled_target_repeats`)
- `NUM_ORACLE_ROLLOUTS` (used by sampled and prompt-only modes)
- `ORACLE_MAX_NEW_TOKENS`

## Easy run script

Use `run_oracle_experiment.sh` for a readable entrypoint with editable defaults and CLI overrides:

```bash
./run_oracle_experiment.sh --mode prompt_only_repeats --num-oracle-rollouts 3 --target-prompt-limit 1
```

List all presets and examples:

```bash
./run_oracle_experiment.sh --help
```

Example presets:

```bash
./run_oracle_experiment.sh --preset full_deterministic_oracle
./run_oracle_experiment.sh --preset sampled_target_repeats --k-rollouts 8 --num-oracle-rollouts 3
./run_oracle_experiment.sh --preset prompt_only_oracle --num-oracle-rollouts 4
./run_oracle_experiment.sh --preset oracle_target_control
./run_oracle_experiment.sh --preset target_judging_only
```

The script supports both:

- top-of-file defaults (easy to edit once), and
- command-line overrides (`--mode`, `--num-rollouts`, `--k-rollouts`, etc.)

You can also override with inline environment variables:

```bash
ORACLE_ROLLOUT_MODE=sampled_target_repeats K_ROLLOUTS=5 NUM_ORACLE_ROLLOUTS=2 ./run_oracle_experiment.sh
```

## Small GPU validation run

Run one small prompt/oracle combination per mode to validate cache layout and schema:

```bash
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=3 NUM_ORACLE_ROLLOUTS=1 ORACLE_ROLLOUT_MODE=all_target_deterministic python bypass_refusal.py
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=5 K_ROLLOUTS=2 NUM_ORACLE_ROLLOUTS=2 ORACLE_ROLLOUT_MODE=sampled_target_repeats python bypass_refusal.py
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=3 NUM_ORACLE_ROLLOUTS=3 ORACLE_ROLLOUT_MODE=prompt_only_repeats python bypass_refusal.py
```

## Run all tests

From repo home, run:

```bash
PYTHONPATH="activation_oracles_extensions" \
python -m unittest discover -v -s "activation_oracles_extensions/tests"
```

Optional shell script syntax check:

```bash
bash -n "activation_oracles_extensions/run_oracle_experiment.sh"
```
