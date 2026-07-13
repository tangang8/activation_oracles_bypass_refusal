from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Any
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_utils import deterministic_oracle_judge_cache_file_path, load_json, write_json
from distributed_utils import DistributedContext, all_gather_objects, broadcast_object
from perf_utils import PerfLogger
from judge_parsing import (
    judge_provenance_sha as _judge_provenance_sha,
    judge_scoring_mode_for_stem,
    response_sha as _response_sha,
)
from rollout_utils import (
    THINKING_TAG_PATTERNS_BY_MODEL,
    resolve_judge_enable_thinking,
    score_responses_compliance_batched,
)


def _entry_index(entry: dict[str, Any]) -> int:
    if "rollout_index" in entry:
        return int(entry["rollout_index"])
    if "oracle_rollout_index" in entry:
        return int(entry["oracle_rollout_index"])
    raise KeyError("Entry is missing both rollout_index and oracle_rollout_index.")


def _entry_key(entry: dict[str, Any]) -> str:
    """Stable identity for judge-score reuse.

    Prefer the explicit `(target_rollout_index, oracle_rollout_index)` pair so the key does NOT
    shift when `num_oracle_rollouts` changes the flattened `rollout_index` (the arithmetic index
    `target*N + oracle` was the root of scores attaching to the wrong response). Deterministic and
    prompt-only entries are already 1:1 and stable, so their single index is used directly."""
    target_idx = entry.get("target_rollout_index")
    oracle_idx = entry.get("oracle_rollout_index")
    if target_idx is not None and oracle_idx is not None:
        return f"t{int(target_idx)}_o{int(oracle_idx)}"
    if "rollout_index" in entry:
        return f"r{int(entry['rollout_index'])}"
    if oracle_idx is not None:
        return f"o{int(oracle_idx)}"
    raise KeyError("Entry is missing rollout identity fields.")


def _flatten_oracle_responses(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rollout_index = _entry_index(entry)
    entry_key = _entry_key(entry)
    source_index_label = "rollout_index" if "rollout_index" in entry else "oracle_rollout_index"
    user_prompt = str(entry.get("target_prompt", ""))
    oracle_response = entry.get("oracle_response", {})
    oracle_format = entry.get("oracle_format", {})
    flattened: list[dict[str, Any]] = []

    for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
        if not isinstance(oracle_response, dict) or probe_kind not in oracle_response:
            continue
        leaf_format = oracle_format.get(probe_kind, {}) if isinstance(oracle_format, dict) else {}
        if isinstance(leaf_format, dict):
            response_text = str(leaf_format.get("response_only", "")).strip()
        else:
            response_text = str(oracle_response.get(probe_kind, "")).strip()
        flattened.append(
            {
                "rollout_index": rollout_index,
                "entry_key": entry_key,
                "source_index_label": source_index_label,
                "path": (probe_kind,),
                "probe_kind": probe_kind,
                "user_prompt": user_prompt,
                "response_text": response_text,
                "target_rollout_index": entry.get("target_rollout_index"),
                "oracle_rollout_index": entry.get("oracle_rollout_index"),
            }
        )

    tokens_response = oracle_response.get("tokens", {})
    tokens_format = oracle_format.get("tokens", {}) if isinstance(oracle_format, dict) else {}
    if isinstance(tokens_response, dict):
        for token_index, raw_response in tokens_response.items():
            token_key = str(token_index)
            leaf_format = tokens_format.get(token_key, {}) if isinstance(tokens_format, dict) else {}
            if isinstance(leaf_format, dict):
                response_text = str(leaf_format.get("response_only", raw_response)).strip()
            else:
                response_text = str(raw_response).strip()
            flattened.append(
                {
                    "rollout_index": rollout_index,
                    "entry_key": entry_key,
                    "source_index_label": source_index_label,
                    "path": ("tokens", token_key),
                    "probe_kind": "tokens",
                    "token_index": token_key,
                    "user_prompt": user_prompt,
                    "response_text": response_text,
                    "target_rollout_index": entry.get("target_rollout_index"),
                    "oracle_rollout_index": entry.get("oracle_rollout_index"),
                }
            )

    token_point_response = oracle_response.get("token_points", {})
    token_point_format = oracle_format.get("token_points", {}) if isinstance(oracle_format, dict) else {}
    if isinstance(token_point_response, dict):
        for token_point_name, raw_response in token_point_response.items():
            point_key = str(token_point_name)
            leaf_format = token_point_format.get(point_key, {}) if isinstance(token_point_format, dict) else {}
            if isinstance(leaf_format, dict):
                response_text = str(leaf_format.get("response_only", raw_response)).strip()
            else:
                response_text = str(raw_response).strip()
            flattened.append(
                {
                    "rollout_index": rollout_index,
                    "entry_key": entry_key,
                    "source_index_label": source_index_label,
                    "path": ("token_points", point_key),
                    "probe_kind": "token_points",
                    "token_point_name": point_key,
                    "user_prompt": user_prompt,
                    "response_text": response_text,
                    "target_rollout_index": entry.get("target_rollout_index"),
                    "oracle_rollout_index": entry.get("oracle_rollout_index"),
                }
            )
    return flattened


def _oracle_judge_item_id(item: dict[str, Any]) -> str:
    probe_suffix = ""
    if "token_index" in item:
        probe_suffix = f":{item['token_index']}"
    elif "token_point_name" in item:
        probe_suffix = f":{item['token_point_name']}"
    probe = f"probe={item['probe_kind']}{probe_suffix}"
    target_rollout_index = item.get("target_rollout_index")
    oracle_rollout_index = item.get("oracle_rollout_index")
    if target_rollout_index is not None and oracle_rollout_index is not None:
        return (
            f"target_rollout_index={int(target_rollout_index)} "
            f"oracle_rollout_index={int(oracle_rollout_index)} {probe}"
        )
    source_index_label = str(item.get("source_index_label", "rollout_index"))
    return f"{source_index_label}={int(item['rollout_index'])} {probe}"


def _get_path_leaf(root: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = root
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_path_leaf(root: dict[str, Any], path: tuple[str, ...], value: dict[str, Any]) -> None:
    node = root
    for key in path[:-1]:
        next_node = node.get(key)
        if not isinstance(next_node, dict):
            next_node = {}
            node[key] = next_node
        node = next_node
    node[path[-1]] = value


def _compliance_shell(entry: dict[str, Any]) -> dict[str, Any]:
    oracle_response = entry.get("oracle_response", {})
    shell: dict[str, Any] = {}
    for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
        if probe_kind in oracle_response:
            shell[probe_kind] = None
    if isinstance(oracle_response.get("tokens"), dict):
        shell["tokens"] = {str(key): None for key in oracle_response["tokens"].keys()}
    if isinstance(oracle_response.get("token_points"), dict):
        shell["token_points"] = {str(key): None for key in oracle_response["token_points"].keys()}
    return shell


def _reusable_existing_leaf(
    existing_compliance: Any,
    path: tuple[str, ...],
    response_text: str,
    provenance_sha: str,
) -> dict[str, Any] | None:
    """Return a prior compliance leaf ONLY if it is safe to reuse.

    Two guards, both required when present on the leaf:
      * `judged_response_sha` must match this response's hash — closes the bug where a leaf keyed
        only by `(rollout_index, path)` was overlaid onto a *different* response (e.g. after
        `num_oracle_rollouts` changed the flattened index, or the probe cache regenerated
        different sampled text).
      * `judge_provenance_sha` must match the current rubric+parser — so a rubric/formula change
        re-judges rather than reusing a score computed under a different rule.
    Legacy leaves written before these fields existed lack them and are trusted, so pre-fix caches
    don't force a full re-judge (the score migration already re-derived their values under the
    current formula); delete the judge cache to force a clean re-score if a run needs it."""
    if not isinstance(existing_compliance, dict):
        return None
    leaf = _get_path_leaf(existing_compliance, path)
    if not isinstance(leaf, dict):
        return None
    stored_response = leaf.get("judged_response_sha")
    if stored_response is not None and stored_response != _response_sha(response_text):
        return None
    stored_provenance = leaf.get("judge_provenance_sha")
    if stored_provenance is not None and stored_provenance != provenance_sha:
        return None
    return leaf


def _oracle_judge_summary(judged_entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_probe: dict[str, list[float]] = {}
    total_scored = 0
    total_unscored = 0
    for entry in judged_entries:
        compliance = entry.get("compliance", {})
        flattened = []
        if isinstance(compliance, dict):
            for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
                leaf = compliance.get(probe_kind)
                if isinstance(leaf, dict):
                    flattened.append((probe_kind, leaf))
            for probe_kind in ("tokens", "token_points"):
                container = compliance.get(probe_kind, {})
                if isinstance(container, dict):
                    for leaf in container.values():
                        if isinstance(leaf, dict):
                            flattened.append((probe_kind, leaf))
        for probe_kind, leaf in flattened:
            score = leaf.get("score")
            if isinstance(score, Real) and not isinstance(score, bool):
                by_probe.setdefault(probe_kind, []).append(float(score))
                total_scored += 1
            else:
                # Leaf present but no numeric score (judge-skipped/malformed): surface it as a
                # count instead of letting the probe silently vanish from the averages.
                total_unscored += 1

    summary: dict[str, Any] = {
        "oracle_judge/total_scored": float(total_scored),
        "oracle_judge/total_unscored": float(total_unscored),
    }
    for probe_kind, scores in by_probe.items():
        if scores:
            summary[f"oracle_judge/{probe_kind}_avg_score"] = float(sum(scores)) / float(len(scores))
            summary[f"oracle_judge/{probe_kind}_count"] = float(len(scores))
    return summary


def _apply_oracle_judge_updates(
    *,
    merged_by_key: dict[str, dict[str, Any]],
    updates: list[dict[str, Any]],
) -> None:
    for update in updates:
        key = str(update["entry_key"])
        if key not in merged_by_key:
            continue
        compliance_root = merged_by_key[key].setdefault("compliance", {})
        if not isinstance(compliance_root, dict):
            compliance_root = {}
            merged_by_key[key]["compliance"] = compliance_root
        _set_path_leaf(compliance_root, tuple(update["path"]), update["compliance"])


def _materialize_oracle_judge_entries(
    *,
    oracle_rollout_entries: list[dict[str, Any]],
    merged_by_key: dict[str, dict[str, Any]],
    existing_by_key: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    current = [
        merged_by_key[_entry_key(entry)]
        for entry in sorted(oracle_rollout_entries, key=_entry_index)
        if _entry_key(entry) in merged_by_key
    ]
    if not existing_by_key:
        return current
    # Preserve judged entries outside the current request (e.g. a smaller num_oracle_rollouts
    # re-run) instead of truncating them out of the cache file on write.
    current_keys = {_entry_key(entry) for entry in oracle_rollout_entries}
    extras = [entry for key, entry in existing_by_key.items() if key not in current_keys]
    extras.sort(key=_entry_index)
    return current + extras


def judge_oracle_rollouts(
    judge_model: AutoModelForCausalLM,
    judge_tokenizer: AutoTokenizer,
    oracle_rollout_entries: list[dict[str, Any]],
    oracle_prompt: str,
    judge_instruction_template: str,
    judge_instruction_file: str,
    judge_instruction_stem: str,
    device: torch.device,
    target_model_name: str,
    target_lora_path: str | None,
    oracle_model_name: str,
    oracle_lora_path: str | None,
    oracle_generation_kwargs: dict[str, Any],
    oracle_rollouts_dir_base: str = "oracle_rollouts",
    oracle_cache_variant_key: str | None = None,
    judge_generation_kwargs: dict[str, Any] | None = None,
    judge_thinking_mode: str = "default",
    judge_batch_size: int = 8,
    judge_lora_path: str | None = "default",
    cache_root: str = "cache",
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    if judge_generation_kwargs is None:
        judge_generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 1000}
    if judge_batch_size <= 0:
        raise ValueError(f"judge_batch_size must be > 0, got {judge_batch_size}")

    target_prompt = str(oracle_rollout_entries[0].get("target_prompt", "")) if oracle_rollout_entries else ""
    cache_file = deterministic_oracle_judge_cache_file_path(
        cache_root=cache_root,
        target_model_name=target_model_name,
        target_lora_path=target_lora_path,
        judge_model_name=judge_model.config._name_or_path,
        judge_lora_path=judge_lora_path,
        judge_generation_kwargs=judge_generation_kwargs,
        judge_thinking_mode=judge_thinking_mode,
        judge_instruction_stem=judge_instruction_stem,
        oracle_model_name=oracle_model_name,
        oracle_lora_path=oracle_lora_path,
        oracle_generation_kwargs=oracle_generation_kwargs,
        target_prompt=target_prompt,
        oracle_prompt=oracle_prompt,
        oracle_rollouts_dir_base=oracle_rollouts_dir_base,
        cache_variant_key=oracle_cache_variant_key,
    )

    provenance_sha = _judge_provenance_sha(judge_instruction_template)

    loaded = load_json(cache_file)
    existing_entries = loaded if isinstance(loaded, list) else []
    # Keyed by stable entry identity (_entry_key), NOT the flattened numeric rollout_index, so a
    # cached score can only be reused for the same (target, oracle) rollout even if a changed
    # num_oracle_rollouts renumbered rollout_index.
    existing_by_key: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            continue
        try:
            key = _entry_key(entry)
        except Exception:
            continue
        existing_by_key[key] = entry

    merged_by_key: dict[str, dict[str, Any]] = {}
    pending_items: list[dict[str, Any]] = []
    for oracle_entry in oracle_rollout_entries:
        entry_key = _entry_key(oracle_entry)
        # Base the merged entry on the CURRENT oracle entry, so oracle_response / oracle_points /
        # oracle_format always reflect the latest oracle output (which may have gained token
        # points via backfill). Carry over already-computed compliance scores from any existing
        # judge entry so unchanged probes are not re-judged; genuinely new probes fall through to
        # pending. (Basing on the stale existing entry left oracle_response with fewer points than
        # compliance had scores for.)
        base_entry = deepcopy(oracle_entry)
        existing_entry = existing_by_key.get(entry_key)
        existing_compliance = existing_entry.get("compliance") if isinstance(existing_entry, dict) else None
        # Build compliance mirroring the CURRENT oracle_response so its probe set always matches.
        # For each probe, reuse a prior score only if it was computed against this exact response
        # text AND the current rubric+parser (verified by hash); otherwise the probe is (re-)judged.
        # New probes stay None → pending, stale probes are dropped (not in the current shell).
        compliance = _compliance_shell(oracle_entry)
        base_entry["compliance"] = compliance
        merged_by_key[entry_key] = base_entry
        for item in _flatten_oracle_responses(oracle_entry):
            reusable = _reusable_existing_leaf(
                existing_compliance, item["path"], item["response_text"], provenance_sha
            )
            if reusable is not None:
                _set_path_leaf(compliance, item["path"], deepcopy(reusable))
                continue
            pending_items.append(item)

    rank = dist_ctx.rank if dist_ctx is not None else 0
    world_size = dist_ctx.world_size if dist_ctx is not None else 1
    total_probe_items = sum(len(_flatten_oracle_responses(entry)) for entry in oracle_rollout_entries)
    if perf is not None:
        perf.log_event(
            "cache/oracle_judge_status",
            {
                "cache/oracle_judge_hits": float(max(0, total_probe_items - len(pending_items))),
                "cache/oracle_judge_missing": float(len(pending_items)),
                "cache/oracle_judge_total_rollouts": float(len(oracle_rollout_entries)),
            },
            metadata={"rank": rank, "world_size": world_size},
        )

    if dist_ctx is None or not dist_ctx.enabled:
        local_items = pending_items
    else:
        local_items = [item for i, item in enumerate(pending_items) if i % dist_ctx.world_size == dist_ctx.rank]

    judge_thinking_tag = THINKING_TAG_PATTERNS_BY_MODEL.get(judge_model.config._name_or_path)
    judge_enable_thinking = resolve_judge_enable_thinking(judge_thinking_mode)
    judge_scoring_mode = judge_scoring_mode_for_stem(judge_instruction_stem)
    can_checkpoint_locally = dist_ctx is None or not dist_ctx.enabled
    local_updates: list[dict[str, Any]] = []
    if local_items:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in local_items:
            grouped.setdefault(item["user_prompt"], []).append(item)

        with (
            perf.track(
                "oracle/judge_generate",
                {
                    "pending_items": len(local_items),
                    "judge_batch_size": judge_batch_size,
                    "judge_max_new_tokens": judge_generation_kwargs.get("max_new_tokens"),
                    "rank": rank,
                    "world_size": world_size,
                },
            )
            if perf
            else nullcontext()
        ) as judge_metrics:
            judge_t0 = perf_counter()
            remaining_items_by_rollout = Counter(int(item["rollout_index"]) for item in local_items)
            pbar = tqdm(
                total=len(remaining_items_by_rollout),
                desc="Oracle judging rollouts",
                disable=not (dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main),
            )
            try:
                for user_prompt, group_items in grouped.items():
                    for offset in range(0, len(group_items), judge_batch_size):
                        chunk_items = group_items[offset : offset + judge_batch_size]
                        responses = [item["response_text"] for item in chunk_items]
                        judged = score_responses_compliance_batched(
                            judge_model=judge_model,
                            judge_tokenizer=judge_tokenizer,
                            user_prompt=user_prompt,
                            target_responses=responses,
                            judge_instruction_template=judge_instruction_template,
                            device=device,
                            judge_lora_path=judge_lora_path,
                            generation_kwargs=judge_generation_kwargs,
                            target_thinking_tag=None,
                            judge_thinking_tag=judge_thinking_tag,
                            judge_enable_thinking=judge_enable_thinking,
                            emit_summary_log=False,
                            stage_label="oracle judging",
                            item_ids=[_oracle_judge_item_id(item) for item in chunk_items],
                            malformed_retry_attempts=4,
                            judge_scoring_mode=judge_scoring_mode,
                        )
                        chunk_updates: list[dict[str, Any]] = []
                        for item, compliance in zip(chunk_items, judged, strict=True):
                            payload = {
                                **compliance,
                                "judge_instruction_file": judge_instruction_file,
                                "probe_kind": item["probe_kind"],
                                "judged_response_sha": _response_sha(item["response_text"]),
                                "judge_provenance_sha": provenance_sha,
                            }
                            if "token_index" in item:
                                payload["token_index"] = item["token_index"]
                            if "token_point_name" in item:
                                payload["token_point_name"] = item["token_point_name"]
                            chunk_updates.append(
                                {
                                    "entry_key": item["entry_key"],
                                    "rollout_index": item["rollout_index"],
                                    "path": list(item["path"]),
                                    "compliance": payload,
                                }
                            )
                        local_updates.extend(chunk_updates)
                        if can_checkpoint_locally and chunk_updates:
                            _apply_oracle_judge_updates(
                                merged_by_key=merged_by_key,
                                updates=chunk_updates,
                            )
                            checkpoint_entries = _materialize_oracle_judge_entries(
                                oracle_rollout_entries=oracle_rollout_entries,
                                merged_by_key=merged_by_key,
                                existing_by_key=existing_by_key,
                            )
                            write_json(cache_file, checkpoint_entries)
                        completed_rollouts = 0
                        for item in chunk_items:
                            rollout_index = int(item["rollout_index"])
                            remaining_items_by_rollout[rollout_index] -= 1
                            if remaining_items_by_rollout[rollout_index] == 0:
                                completed_rollouts += 1
                        if completed_rollouts > 0:
                            pbar.update(completed_rollouts)
            finally:
                pbar.close()
            if perf:
                elapsed = max(perf_counter() - judge_t0, 1e-12)
                judge_metrics["throughput/oracle_judgments_per_second"] = float(len(local_items)) / elapsed

    if dist_ctx is not None and dist_ctx.enabled:
        gathered_updates = all_gather_objects(dist_ctx, local_updates)
    else:
        gathered_updates = [local_updates]

    is_main = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    final_entries: list[dict[str, Any]] | None = None
    if is_main:
        for updates in gathered_updates:
            if not isinstance(updates, list):
                continue
            _apply_oracle_judge_updates(
                merged_by_key=merged_by_key,
                updates=updates,
            )
        write_json(
            cache_file,
            _materialize_oracle_judge_entries(
                oracle_rollout_entries=oracle_rollout_entries,
                merged_by_key=merged_by_key,
                existing_by_key=existing_by_key,
            ),
        )
        final_entries = _materialize_oracle_judge_entries(
            oracle_rollout_entries=oracle_rollout_entries,
            merged_by_key=merged_by_key,
        )

    if dist_ctx is not None and dist_ctx.enabled:
        final_entries = broadcast_object(dist_ctx, final_entries, src=0)
    if final_entries is None:
        final_entries = []

    summary = _oracle_judge_summary(final_entries)
    summary.update(
        {
            "cache/oracle_judge_missing": float(len(pending_items)),
            "cache/oracle_judge_total_rollouts": float(len(oracle_rollout_entries)),
        }
    )
    return final_entries, cache_file, summary
