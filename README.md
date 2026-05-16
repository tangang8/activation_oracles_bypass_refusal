# activation_oracles_extensions

Helpers for importing notebook-defined oracle functions as regular Python code.

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

## Regenerate materialized modules

Run from `<parent-folder>`:

```bash
python activation_oracles_extensions/materialize_notebook_functions.py
```

This reads:

- `activation_oracles/experiments/activation_oracle_demo.ipynb`
- `activation_oracles/experiments/multilayer_activation_oracle_demo.ipynb`

and writes:

- `activation_oracles_extensions/materialized/activation_oracle_demo_lib.py`
- `activation_oracles_extensions/materialized/multilayer_activation_oracle_demo_lib.py`

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

## Import usage

From code run inside `activation_oracles_extensions`:

```python
from oracle_imports import run_oracle_single_layer, run_oracle_multi_layer
```

If running from elsewhere, add `activation_oracles_extensions` to `PYTHONPATH` or `sys.path`.

## Notes

- `materialized/*.py` are auto-generated; do not edit manually.
- Re-run materialization whenever source notebooks change.
- Runtime deps come from notebook code (`tqdm`, `torch`, `transformers`, `peft`, etc.).
