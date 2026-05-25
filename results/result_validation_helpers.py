"""Result validation helpers for inspecting and peeking into StrongReject cache files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display

from viz_helpers import PathAliaser, apply_display_transforms, clip_text, rename_display_columns
from prompt_utils import load_target_prompts_from_dataset


def build_coverage_df(manifest: dict) -> pd.DataFrame:
    """Return a per-condition coverage summary DataFrame from a compiled manifest."""
    all_conditions = sorted(
        set(manifest.get('expected_files', {})) | set(manifest.get('loaded_files', {}))
    )
    df = pd.DataFrame([
        {
            'condition': condition,
            'expected_files': manifest.get('expected_files', {}).get(condition, 0),
            'loaded_files': manifest.get('loaded_files', {}).get(condition, 0),
        }
        for condition in all_conditions
    ])
    df['missing_files'] = df['expected_files'] - df['loaded_files']
    df['coverage_pct'] = df.apply(
        lambda row: (row['loaded_files'] / row['expected_files']) if row['expected_files'] else 1.0,
        axis=1,
    )
    return df


def display_coverage_report(manifest: dict, cfg) -> PathAliaser:
    """Print and display the full coverage validation report for a compiled manifest.

    Covers: warning counts, coverage warnings table, missing files with path aliases,
    parent-directory probing for missing files, and skipped score leaves.

    Returns the PathAliaser built from the manifest's missing files so callers can
    reuse it before the full details DataFrame is available.
    """
    print(f"Missing expected files: {len(manifest.get('missing_files', []))}")
    print(f"Malformed files: {len(manifest.get('malformed_files', []))}")
    print(
        f"Skipped score leaves (probe entries with no accepted numeric StrongReject score): "
        f"{len(manifest.get('skipped_score_leaves', []))}"
    )

    target_prompts = load_target_prompts_from_dataset(
        limit=cfg.expected_target_prompts, offset=cfg.target_prompt_offset
    )
    prompt_by_index = {cfg.target_prompt_offset + i: p for i, p in enumerate(target_prompts)}

    warnings_df = pd.DataFrame(manifest.get('coverage_warnings', []))
    if not warnings_df.empty:
        if 'target_prompt_index' in warnings_df.columns:
            warnings_df.insert(
                warnings_df.columns.get_loc('target_prompt_index') + 1,
                'target_prompt_preview',
                warnings_df['target_prompt_index'].map(
                    lambda idx: clip_text(prompt_by_index.get(idx, ''), 120)
                ),
            )
            warnings_df = warnings_df.drop(columns=['target_prompt_index'])
        warnings_df = apply_display_transforms(warnings_df)
        if 'probe_kind' in warnings_df.columns and 'probe_name' in warnings_df.columns:
            warnings_df = warnings_df.drop(columns=['probe_kind'])
        display(warnings_df.head(50))
    else:
        print('No coverage warnings.')

    coverage_aliaser = PathAliaser(
        cfg.target_model_name, cfg.cache_root, cfg.output_dir,
        [str(x) for x in manifest.get('missing_files', [])],
    )
    display(coverage_aliaser.legend_df())

    missing_df = pd.DataFrame({'missing_cache_path': manifest.get('missing_files', [])})
    if not missing_df.empty:
        missing_df['missing_cache_path_alias'] = missing_df['missing_cache_path'].map(coverage_aliaser.alias)
        display(rename_display_columns(missing_df[['missing_cache_path_alias', 'missing_cache_path']].head(50)))

        examples = []
        for raw_path in missing_df['missing_cache_path'].head(10):
            path = Path(raw_path)
            parent = path.parent
            siblings = sorted([c.name for c in parent.glob('*.json')])[:3] if parent.exists() else []
            examples.append({
                'missing_cache_path_alias': coverage_aliaser.alias(raw_path),
                'parent_exists': parent.exists(),
                'example_files_in_parent': '\n'.join(siblings) if siblings else '<none>',
            })
        display(pd.DataFrame(examples))
    else:
        print('No missing cache files in current compile pass.')

    skipped_df = pd.DataFrame(manifest.get('skipped_score_leaves', []))
    if not skipped_df.empty:
        skipped_df['path_alias'] = skipped_df['path'].map(coverage_aliaser.alias)
        skipped_reasons = (
            skipped_df.groupby('reason', dropna=False)
            .size().reset_index(name='n_leaves')
            .sort_values('n_leaves', ascending=False)
        )
        print('Skipped score leaves by reason:')
        display(skipped_reasons)
        display(skipped_df[['path_alias', 'reason']].head(50))
    else:
        print('No skipped score leaves.')

    return coverage_aliaser


def apply_filter(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = df.copy()
    for key in [
        'condition', 'oracle_prompt_file', 'probe_kind', 'probe_name',
        'target_prompt_index', 'target_rollout_index', 'oracle_rollout_index',
    ]:
        value = spec.get(key)
        if value is None:
            continue
        out = out[out[key] == value]
    return out


def load_cache_entries(cache_path: str) -> list:
    with open(cache_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload.get('entries', []) if isinstance(payload, dict) else payload


def extract_leaf(container, probe_kind: str, probe_name: str | None):
    if not isinstance(container, dict):
        return None
    node = container.get(probe_kind)
    if probe_name is None:
        return node
    if isinstance(node, dict):
        return node.get(probe_name)
    return None


def match_entry(entries: list, row: dict) -> dict | None:
    for entry in entries:
        if row.get('rollout_index') == entry.get('rollout_index'):
            return entry
    for entry in entries:
        if (row.get('oracle_rollout_index') is not None
                and row.get('oracle_rollout_index') == entry.get('oracle_rollout_index')):
            return entry
    return None


def build_peek_table(filtered: pd.DataFrame, path_aliaser) -> pd.DataFrame:
    """Load cache entries for each row in *filtered* and return a preview DataFrame.

    Args:
        filtered: Subset of the details DataFrame to inspect.
        path_aliaser: A PathAliaser instance used to compact cache paths for display.
    """
    rows = []
    for _, row in filtered.iterrows():
        cache_path = row['cache_path']
        try:
            entries = load_cache_entries(cache_path)
            entry = match_entry(entries, row)
        except Exception as exc:
            rows.append({
                'cache_path': cache_path,
                'probe_name': row.get('probe_name'),
                'score': row.get('score'),
                'error': str(exc),
            })
            continue

        oracle_leaf = (
            extract_leaf(entry.get('oracle_response', {}), row.get('probe_kind'), row.get('probe_name'))
            if isinstance(entry, dict) else None
        )
        compliance_leaf = (
            extract_leaf(entry.get('compliance', {}), row.get('probe_kind'), row.get('probe_name'))
            if isinstance(entry, dict) else None
        )

        rows.append({
            'condition': row.get('condition'),
            'target_prompt_index': row.get('target_prompt_index'),
            'rollout_index': row.get('rollout_index'),
            'target_rollout_index': row.get('target_rollout_index'),
            'oracle_rollout_index': row.get('oracle_rollout_index'),
            'probe_kind': row.get('probe_kind'),
            'probe_name': row.get('probe_name'),
            'score': row.get('score'),
            'target_prompt': clip_text(row.get('target_prompt'), 160),
            'oracle_prompt': clip_text(row.get('oracle_prompt'), 160),
            'oracle_response_preview': clip_text(oracle_leaf, 220),
            'compliance_leaf_preview': clip_text(compliance_leaf, 220),
            'cache_path': cache_path,
            'cache_path_alias': path_aliaser.alias(cache_path),
        })

    out = pd.DataFrame(rows)
    out = apply_display_transforms(out)
    return out
