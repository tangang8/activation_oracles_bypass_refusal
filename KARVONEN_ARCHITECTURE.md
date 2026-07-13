# KARVONEN_ARCHITECTURE.md — the upstream `activation_oracles` (`nl_probes`) repo

> Reference for the **sibling** upstream repo (`../activation_oracles`) by Adam Karvonen et al.
> (arXiv:2512.15674). This repo imports five functions from it via `sys.path` in
> `oracle_pipeline.py`. This document explains what those functions do and the assumptions
> our harness must respect. Paths below are relative to `../activation_oracles/`.
>
> Companion to `ARCHITECTURE.md` (which describes *this* harness). Written from a read of the
> upstream source; trust function/variable names and grep, line numbers drift.

---

## 1. What an activation oracle is

An **activation oracle** (a.k.a. natural-language probe / "verbalizer") is an LLM fine-tuned
with a LoRA adapter to accept *another model's internal residual-stream activations as input*
and answer natural-language questions about them. Instead of interpreting activations with
SAEs or linear probes, you **inject** the activation vectors into an "oracle" model at an early
layer and let it verbalize what they encode.

Mechanism:

1. Run a **target** model on a context prompt, capturing residual-stream activations at one or
   more middle layers (25%/50%/75% depth).
2. Build an oracle prompt of the form `Layer: <L>\n ? ? ? … \n<question>`, where each `" ?"`
   token is a placeholder slot — **one per captured activation vector**.
3. During the oracle's forward pass, a forward hook **modifies the residual stream at those
   placeholder positions** using the (normalized, rescaled) target activations.
4. The oracle generates a natural-language answer conditioned on the injected activations.

The same machinery trains the LoRA (teacher-forced next-token loss on target answers) and
evaluates it (free generation). Demonstrated uses: extracting secret words from "taboo"
models, detecting goals/emotions, classification, PersonaQA knowledge extraction.

Package: `nl_probes`, version `0.0.1`. Pinned to `transformers==4.55.2`, `peft==0.17.1`,
`torch==2.7.1`. **Upstream house style (their AGENTS.md): fail loudly — no defensive
`try/except`, `.get()`, or empty-length guards.** Expect hard asserts, not graceful
degradation, from anything you call in `nl_probes`.

---

## 2. The five functions we import (`oracle_pipeline.py`)

```python
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
```

### 2a. `collect_activations_multiple_layers` — `utils/activation_utils.py:64`
```python
def collect_activations_multiple_layers(model, submodules, inputs_BL,
                                        min_offset, max_offset) -> dict[int, torch.Tensor]:
```
Registers a forward hook on each module in `submodules` (`{layer_index: module}`, from
`get_hf_submodule`), runs **one** `torch.no_grad()` forward, and raises an internal
`EarlyStopException` once the **max** layer index is reached (skips the rest of the network).
Returns `{layer_index: [B, L, D]}` — the residual-stream *output* of that decoder block.

- `min_offset`/`max_offset` optionally slice the sequence dim as `acts[:, max_offset:min_offset, :]`.
  If `min_offset` is given, `max_offset` must be too; both must be `< 0` with `max_offset < min_offset`.
  **For all positions pass `min_offset=None, max_offset=None`** — this is what our harness does.
- Gotcha: the early-stop fires on `max(submodules.keys())`; a submodule dict missing your
  deepest layer short-circuits before it.

### 2b. `get_hf_submodule` — `utils/activation_utils.py:137`
```python
def get_hf_submodule(model, layer: int, use_lora: bool = False):
```
Returns the decoder-block module (the residual hook point) for a 0-based `layer`, dispatching
on `model.config._name_or_path` substring: `model.model.layers[layer]` for
gemma-2/mistral/Llama/Qwen, `model.language_model.layers[layer]` for gemma-3,
`model.gpt_neox.layers[layer]` for pythia. `use_lora=True` walks
`model.base_model.model.model.layers[layer]`.
- Layer index is the **decoder block index**; the hook point is the **block output**
  (post-MLP residual), not an internal submodule.
- Model-family detection is a substring match — renamed checkpoints raise `ValueError`.

### 2c. `layer_percent_to_layer` — `utils/common.py:132`
```python
def layer_percent_to_layer(model_name, layer_percent) -> int:
    return int(get_layer_count(model_name) * (layer_percent / 100))
```
`floor(num_hidden_layers * percent/100)`. Qwen3-8B (36 layers) at 50% → **layer 18**. Downloads
the HF config to read `num_hidden_layers`. Our harness passes `layer_percent=50`.

### 2d. `create_training_datapoint` — `utils/dataset_utils.py:288`
```python
def create_training_datapoint(datapoint_type, prompt, target_response, layer, num_positions,
        tokenizer, acts_BD, feature_idx, context_input_ids=None, context_positions=None,
        ds_label=None, meta_info=None) -> TrainingDataPoint:
```
Builds one oracle training/eval datapoint:
1. Prepends `get_introspection_prefix(layer, num_positions)` = `Layer: {layer}\n{" ?"*num_positions} \n`.
   The placeholder `SPECIAL_TOKEN = " ?"` (space-question-mark) **must tokenize to exactly one
   token** (asserted). Count of `" ?"` placeholders must equal `num_positions`, which must equal
   `acts_BD.shape[0]`.
2. Renders the prompt with `apply_chat_template(..., add_generation_prompt=True,
   enable_thinking=False)`, then again with `target_response` appended for the full sequence.
3. `labels` = full ids with the prompt region set to `-100` (loss only over the answer).
4. Locates the placeholder run with `find_pattern_in_tokens` (must be `num_positions`
   **consecutive** matches with a trailing newline).
5. Stores `acts_BD` (CPU, cloned, detached, must be 2-D `[num_positions, D]`) as
   `steering_vectors`. If `acts_BD is None`, `context_input_ids` + `context_positions` **must** be
   given so vectors are materialized lazily.

`TrainingDataPoint` (pydantic, `extra="forbid"`): `datapoint_type, input_ids, labels, layer,
steering_vectors, positions, feature_idx, target_output, context_input_ids, context_positions,
ds_label, meta_info`. Validator: `len(positions) == steering_vectors.shape[0]`.

**How our harness uses it (`oracle_pipeline.run_oracle_batched::add_probe`)**: one datapoint per
probe (`full_seq`, `segment`, `prompt_segment`, `rollout_segment`, per-`tokens`, per-`token_point`),
passing that probe's activation rows as `acts_BD` and `meta_info={target_idx, probe_kind,
repeat_idx, token_index}` so results can be regrouped.

### 2e. `run_evaluation` + steered generation — `utils/eval.py:100`
```python
def run_evaluation(eval_data, model, tokenizer, submodule, device, dtype, global_step,
        lora_path, eval_batch_size, steering_coefficient, generation_kwargs,
        verbose=False) -> list[FeatureResult]:
```
Flow: if `lora_path` is set, load/activate it (**the adapter name IS the path string**); then per
batch: strip response tokens with `get_prompt_tokens_only` (generation starts from the prompt
only), materialize any deferred steering vectors, left-pad into a batch, and run
`eval_features_batch`. Asserts `len(results) == len(eval_data)` and copies each datapoint's
`meta_info` onto its `FeatureResult` (`feature_idx`, **`api_response`** = decoded generation with
prompt stripped, `prompt`, `meta_info`).

The steered generation — `eval_features_batch`, `eval.py:22`:
```python
hook_fn = get_hf_activation_steering_hook(vectors=..., positions=...,
              steering_coefficient=steering_coefficient, device=device, dtype=dtype)
with add_hook(submodule, hook_fn):
    output_ids = model.generate(**tokenized_input, **generation_kwargs)   # eval.py:55
generated_tokens = output_ids[:, eval_batch.input_ids.shape[1]:]          # strips prompt only
decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
```
- **`submodule`** is the *injection* layer's module — our harness passes
  `get_hf_submodule(model, injection_layer)`, `injection_layer` default **1**. Activations
  captured at target layer 18 are injected into oracle layer 1.
- **Injection semantics** (`steering_hooks.py:129`): fires only on the prefill pass (`L > 1`).
  For each placeholder position it computes `norms = orig.norm()`, then
  `steered = normalize(vector) * norms * steering_coefficient`, and sets
  `resid[b, pos] = steered.detach() + orig`. **Note: the injected vector is ADDED on top of the
  original residual** (despite the docstring saying "replace"); it is unit-normalized then
  rescaled to the original token's residual norm.
- **`steering_coefficient`** default `1.0` (our harness passes `1.0`).

---

## 3. Generation length — where the cap lives (CRITICAL for our truncation issue)

**There is no hardcoded new-token cap anywhere in the `run_evaluation` / `eval_features_batch`
path.** `generation_kwargs` is forwarded **verbatim** to `model.generate` (`eval.py:55`) — no
`max_length`, no clamping, no post-hoc `[:N]` slice of the decoded string. The slice at
`eval.py:58` (`output_ids[:, prompt_len:]`) strips the *prompt* tokens only; it does not bound
generated length.

Therefore **length is entirely controlled by the caller's `generation_kwargs["max_new_tokens"]`.**
Defaults you inherit if you omit it:
- `configs/sft_config.py:22` → `{"do_sample": False, "max_new_tokens": 20}`
- `base_experiment.py` verbalizer eval → `{"do_sample": True, "temperature": 0.7,
  "max_new_tokens": 40, "top_p": 0.9}`
- per-experiment overrides in `experiments/*` range 10–40.

The only `max_length` / `truncation=True` in the whole repo are in
`dataset_classes/past_lens_dataset.py:202` (`max_length=512`, **dataset construction**) and
`trl_training/config.py:49` (`max_length=1024`, unrelated TRL target training) — **neither is on
the oracle generation path.**

**Consequence for us:** our harness always passes an explicit `generation_kwargs` with
`max_new_tokens` (1000 in the current regime), so upstream will honor it. Any short oracle output
we see is **either** (a) our own `max_new_tokens` being small in the dict we passed at generation
time, or (b) the oracle genuinely emitting EOS early — **not** an upstream hidden cap. See
`ARCHITECTURE.md §6a` (truncation analysis).

**Training relevance:** the oracle LoRA was trained with short targets (`max_new_tokens=20`
default, ≤40 in experiments). It is *optimized to answer tersely and emit `<|im_end|>` quickly*.
That is why, at inference, some probes stop after only tens of tokens even when we allow 1000 —
the model was never trained to sustain long generations. This is a **model/prompt behavior**, not
a length limit.

---

## 4. Training the oracle LoRA (context)

Entry: `torchrun nl_probes/sft.py`, config `SelfInterpTrainingConfig`
(`configs/sft_config.py`): `model_name="Qwen/Qwen3-8B"`, `hook_onto_layer=1`,
`layer_percents=[25,50,75]`, LoRA `r=64/alpha=128/dropout=0.05`, `target_modules="all-linear"`,
`lr=1e-5`, 1 epoch, `use_decoder_vectors=True`. Loss = cross-entropy on the answer tokens with
frozen base + injected activations (same hook as eval). Training mixture (`build_loader_groups`):
LatentQA system-prompt QA, binary classification (geometry-of-truth, sst2, snli, ner, tense,
md_gender, ag_news, language-id, relations), past-lens self-supervised token prediction, SAE
feature explain/yes-no/activating-sequence. Datasets cached to disk as `.pt` keyed by config hash.

---

## 5. Assumptions our harness must respect

- **Padding side = LEFT** (`common.load_tokenizer`). Positions are shifted by pad length in
  `construct_batch` / `materialize_missing_steering_vectors`. Right padding points injection at
  the wrong tokens. Our `run_oracle_batched` mirrors this (`padding_side="left"`, shifts
  positions by `left_pad`).
- **Placeholder token `" ?"` must be one token id** for the tokenizer (asserted).
- **Chat template with `enable_thinking`** (Qwen3) is required by `create_training_datapoint`.
- **Layer index is 0-based decoder-block index**; block-output hook point. Target-capture layer
  need not equal the oracle `injection_layer` (default 1).
- **Model family must substring-match** (pythia/gemma-2/gemma-3/mistral/Llama/Qwen) or it raises.
- **PeftModel required**: `materialize_missing_steering_vectors` asserts `isinstance(model,
  PeftModel)` and uses `disable_adapter()` to collect base-model activations. Even base-only runs
  add a dummy LoRA so `peft_config` exists.
- **Per-datapoint shapes**: `steering_vectors` is `[num_positions, D]`, `positions` a
  `list[int]` of equal length; batched they become ragged `list[Tensor]`/`list[list[int]]`.

---

## 6. Quick file map

```
nl_probes/
├── sft.py                     # training entry (torchrun DDP)
├── base_experiment.py         # VerbalizerEvalConfig + run_verbalizer high-level driver
├── configs/sft_config.py      # SelfInterpTrainingConfig
├── utils/
│   ├── activation_utils.py    # collect_activations_multiple_layers, get_hf_submodule  [we import]
│   ├── steering_hooks.py      # get_hf_activation_steering_hook, add_hook  (injection)
│   ├── dataset_utils.py       # create_training_datapoint, TrainingDataPoint, construct_batch  [we import]
│   ├── common.py              # load_model/tokenizer, layer_percent_to_layer  [we import]
│   └── eval.py                # run_evaluation, eval_features_batch  [we import]
├── dataset_classes/           # dataset loaders (LatentQA, classification, past-lens, SAE)
├── experiments/               # paper reproductions (taboo, gender, ssc, personaqa, patchscopes)
└── datasets/                  # prompt lists + classification/factual data
```
