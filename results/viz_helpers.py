"""Display and visualization helpers for StrongReject result notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_LABELS = {
    'target_baseline': 'Target Baseline',
    'oracle_rollout_control': 'Oracle Control Baseline',
    'user_prompt_oracle': 'User Prompt Oracle',
    'target_rollout_oracle': 'Target Rollout Oracle',
}

SCORE_COLS = {
    'mean_score', 'se_score', 'asr_0_2', 'asr_0_2_se', 'asr_0_5', 'asr_0_5_se',
    'asr_0_8', 'asr_0_8_se', 'asr_1', 'asr_1_se',
    'sd_score_within_prompt', 'mean_within_prompt_sd', 'median_within_prompt_sd',
    'score',
}


def display_condition(value: str) -> str:
    return CONDITION_LABELS.get(value, value)


def display_oracle_prompt_file(value) -> str | None:
    if pd.isna(value):
        return None
    name = Path(str(value)).name
    return name[:-5] if name.endswith('.json') else name


class PathAliaser:
    """Compacts long cache paths for display by aliasing shared subdirectory prefixes."""

    def __init__(
        self,
        target_model_name: str,
        cache_root: Path | str,
        output_dir: Path | str,
        paths_for_aliasing: list[str] | None = None,
        max_aliases: int = 8,
    ):
        self.target_model_dir = f"target_{target_model_name.replace('/', '_')}"
        self.target_marker = f"/{self.target_model_dir}/"
        self.cache_root = Path(cache_root)
        self.output_dir = Path(output_dir)
        self._aliases: dict[str, str] = {}
        if paths_for_aliasing:
            self._aliases = self._build_aliases(paths_for_aliasing, max_aliases)

    def _path_tail(self, path_text: str) -> str:
        if self.target_marker in path_text:
            return path_text.split(self.target_marker, 1)[1]
        cache_prefix = str(self.cache_root.resolve()).rstrip('/') + '/'
        if path_text.startswith(cache_prefix):
            return path_text[len(cache_prefix):]
        out_prefix = str(self.output_dir.resolve()).rstrip('/') + '/'
        if path_text.startswith(out_prefix):
            return path_text[len(out_prefix):]
        return path_text

    def _build_aliases(self, paths: list[str], max_aliases: int = 8) -> dict[str, str]:
        counts: dict[str, int] = {}
        for p in paths:
            tail = self._path_tail(str(p))
            parts = [x for x in tail.split('/') if x]
            if len(parts) >= 3:
                prefix = '/'.join(parts[:3])
                counts[prefix] = counts.get(prefix, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return {f"@P{i}": prefix for i, (prefix, _) in enumerate(ranked[:max_aliases], start=1)}

    def alias(self, path) -> str:
        if pd.isna(path):
            return path
        tail = self._path_tail(str(path))
        for alias, prefix in self._aliases.items():
            marker = prefix + '/'
            if tail.startswith(marker):
                return f"{alias}/{tail[len(marker):]}"
            if tail == prefix:
                return alias
        return tail

    def add_alias_column(self, df: pd.DataFrame, source_col: str, alias_col: str) -> pd.DataFrame:
        out = df.copy()
        out[alias_col] = out[source_col].map(self.alias)
        return out

    def legend_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {'alias': alias, 'shared_subdir_prefix': prefix}
            for alias, prefix in self._aliases.items()
        ])


_SCORE_LIKE_COLS = {
    'mean_score', 'score', 'asr_0_2', 'asr_0_5', 'asr_0_8', 'asr_1',
    'Mean Score', 'Score', 'ASR >= 0.2', 'ASR >= 0.5', 'ASR >= 0.8', 'ASR = 1.0',
}
_UNCERTAINTY_LIKE_COLS = {
    'se_score', 'asr_0_2_se', 'asr_0_5_se', 'asr_0_8_se', 'asr_1_se',
    'mean_within_prompt_sd', 'median_within_prompt_sd',
    'SE Across Prompts', 'Mean Within-Prompt SD', 'Median Within-Prompt SD',
    'ASR >= 0.2 SE Across Prompts', 'ASR >= 0.5 SE Across Prompts',
    'ASR >= 0.8 SE Across Prompts', 'ASR = 1.0 SE Across Prompts',
}

_HEATMAP_ALPHA = 0.88


def _contrasting_text(r8: int, g8: int, b8: int, alpha: float = _HEATMAP_ALPHA) -> str:
    """Return #ffffff or #1a1a1a based on WCAG relative luminance of the alpha-blended cell."""
    re = alpha * r8 + (1.0 - alpha) * 255.0
    ge = alpha * g8 + (1.0 - alpha) * 255.0
    be = alpha * b8 + (1.0 - alpha) * 255.0

    def _lin(c: float) -> float:
        c = min(max(c / 255.0, 0.0), 1.0)
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(re) + 0.7152 * _lin(ge) + 0.0722 * _lin(be)
    return '#ffffff' if lum < 0.45 else '#1a1a1a'


def _heatmap_cell_styles(
    df: pd.DataFrame,
    score_cols: list[str],
    uncertainty_cols: list[str],
    alpha: float = _HEATMAP_ALPHA,
    relative_score_norm: bool = False,
) -> pd.DataFrame:
    """Return a same-shape DataFrame of CSS strings with per-cell background + contrasting text."""
    score_cmap = plt.get_cmap('YlGn')
    unc_cmap = plt.get_cmap('YlOrRd')
    if relative_score_norm:
        all_vals = np.concatenate([
            df[c].dropna().to_numpy(dtype=float) for c in score_cols if c in df.columns
        ]) if score_cols else np.array([])
        finite = all_vals[np.isfinite(all_vals)] if all_vals.size else np.array([])
        score_norm = mcolors.Normalize(
            vmin=float(finite.min()) if finite.size else 0.0,
            vmax=float(finite.max()) if finite.size else 1.0,
        )
    else:
        score_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    unc_vals = np.concatenate([
        df[c].dropna().to_numpy(dtype=float) for c in uncertainty_cols if c in df.columns
    ]) if uncertainty_cols else np.array([])
    finite_unc = unc_vals[np.isfinite(unc_vals)] if unc_vals.size else np.array([])
    if finite_unc.size >= 2:
        unc_norm = mcolors.Normalize(
            vmin=float(np.percentile(finite_unc, 5)),
            vmax=float(np.percentile(finite_unc, 95)),
        )
    else:
        unc_norm = mcolors.Normalize(vmin=0.0, vmax=0.3)

    base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'
    out = pd.DataFrame(base, index=df.index, columns=df.columns)

    def _color_cell(val, cmap, norm) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return base
        if not np.isfinite(v):
            return base
        rgba = cmap(norm(v))
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        fg = _contrasting_text(r, g, b, alpha)
        return f'background-color: rgba({r},{g},{b},{alpha}); color: {fg}; {base}'

    for col in score_cols:
        if col in df.columns:
            out[col] = df[col].map(lambda v: _color_cell(v, score_cmap, score_norm))
    for col in uncertainty_cols:
        if col in df.columns:
            out[col] = df[col].map(lambda v: _color_cell(v, unc_cmap, unc_norm))

    return out


def percent_style(df: pd.DataFrame, extra_pct_cols=None, relative_score_norm: bool = False):
    pct_cols = [c for c in df.columns if c in SCORE_COLS or c.startswith('asr_')]
    pretty_pct_cols = [
        c for c in df.columns
        if c in _SCORE_LIKE_COLS or c in _UNCERTAINTY_LIKE_COLS or c.startswith('ASR')
    ]
    if extra_pct_cols:
        pct_cols = sorted(set(pct_cols) | set(extra_pct_cols))
    pct_cols = sorted(set(pct_cols) | set(pretty_pct_cols))

    fmt = {c: '{:.1%}' for c in pct_cols if c in df.columns}
    styler = df.style.format(fmt, na_rep='—')

    score_cols = [c for c in _SCORE_LIKE_COLS if c in df.columns]
    uncertainty_cols = [c for c in _UNCERTAINTY_LIKE_COLS if c in df.columns]
    if score_cols or uncertainty_cols:
        styler = styler.apply(
            lambda d: _heatmap_cell_styles(d, score_cols, uncertainty_cols, relative_score_norm=relative_score_norm),
            axis=None,
        )

    styler = styler.set_table_styles([
        {
            'selector': 'table',
            'props': [('border-collapse', 'collapse'), ('margin', '0 auto')],
        },
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#1e293b'), ('color', 'white'),
                ('font-weight', '600'), ('text-align', 'center'),
                ('vertical-align', 'middle'), ('padding', '5px 8px'),
                ('border-bottom', '2px solid #475569'),
            ],
        },
        {
            'selector': 'tbody td',
            'props': [
                ('border-bottom', '1px solid #e2e8f0'),
                ('padding', '5px 8px'),
            ],
        },
        {
            'selector': 'tbody tr:hover td',
            'props': [('filter', 'brightness(0.93)')],
        },
        {
            'selector': 'caption',
            'props': [('caption-side', 'top'), ('font-weight', '600'), ('color', '#1e293b')],
        },
    ], overwrite=True)

    try:
        styler = styler.hide(axis='index')
    except Exception:
        pass
    return styler


def clip_text(value, n: int = 180):
    if pd.isna(value):
        return value
    text = str(value)
    return text if len(text) <= n else text[:n] + '...'


def apply_display_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'condition' in out.columns:
        out['condition'] = out['condition'].map(display_condition)
    if 'oracle_prompt_file' in out.columns:
        out['oracle_prompt_file'] = out['oracle_prompt_file'].map(display_oracle_prompt_file)
    return out


def rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'probe_kind' in df.columns and 'probe_name' in df.columns:
        df = df.drop(columns=['probe_kind'])
    pretty = {
        'condition': 'Condition',
        'preset_source': 'Preset Source',
        'probe_kind': 'Probe Kind',
        'probe_name': 'Probe Name',
        'oracle_prompt_file': 'Oracle Prompt File',
        'target_prompt_index': 'Target Prompt Index',
        'rollout_index': 'Rollout Index',
        'target_rollout_index': 'Target Rollout Index',
        'oracle_rollout_index': 'Oracle Rollout Index',
        'n_prompts': 'Prompt Count',
        'n_rows': 'Scored Rows',
        'n_target_prompts': 'Unique Target Prompts',
        'n_cache_files': 'Cache Files',
        'mean_score': 'Mean Score',
        'se_score': 'SE Across Prompts',
        'score': 'Score',
        'asr_0_2': 'ASR >= 0.2',
        'asr_0_2_se': 'ASR >= 0.2 SE Across Prompts',
        'asr_0_5': 'ASR >= 0.5',
        'asr_0_5_se': 'ASR >= 0.5 SE Across Prompts',
        'asr_0_8': 'ASR >= 0.8',
        'asr_0_8_se': 'ASR >= 0.8 SE Across Prompts',
        'asr_1': 'ASR = 1.0',
        'asr_1_se': 'ASR = 1.0 SE Across Prompts',
        'n_prompts_with_sd': 'Prompts With Within-Prompt SD',
        'mean_within_prompt_sd': 'Mean Within-Prompt SD',
        'median_within_prompt_sd': 'Median Within-Prompt SD',
        'mean_within_prompt_n': 'Mean Scored Rollouts Per Prompt',
        'cache_path_alias': 'Cache Path',
        'missing_cache_path_alias': 'Missing Cache Path',
        'compliance_leaf_preview': 'Compliance Leaf Preview',
        'oracle_response_preview': 'Oracle Response Preview',
        'target_prompt': 'Target Prompt Preview',
        'oracle_prompt': 'Oracle Prompt Preview',
    }
    return df.rename(columns={k: v for k, v in pretty.items() if k in df.columns})


def probe_order_map(details_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    order_map: dict[tuple[str, str], float] = {}
    sample = (
        details_df[details_df['probe_kind'].isin(['token_points', 'tokens'])]
        [['cache_path', 'probe_kind', 'probe_name']].dropna().drop_duplicates()
    )
    for _, row in sample.iterrows():
        probe_kind = row['probe_kind']
        probe_name = row['probe_name']
        key = (probe_kind, probe_name)
        if key in order_map:
            continue
        try:
            with open(row['cache_path'], 'r', encoding='utf-8') as f:
                payload = json.load(f)
            entries = payload.get('entries', []) if isinstance(payload, dict) else payload
            if not entries:
                continue
            first = entries[0]
            points = first.get('oracle_points', {}).get('token_points', {})
            if isinstance(points, dict) and probe_name in points:
                order_map[key] = float(points[probe_name])
        except Exception:
            continue
    return order_map


def apply_probe_sort(df: pd.DataFrame, probe_order: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    if 'probe_kind' not in out.columns or 'probe_name' not in out.columns:
        return out
    if probe_order is None:
        probe_order = {}
    out['_probe_rank'] = out.apply(
        lambda r: probe_order.get((r.get('probe_kind'), r.get('probe_name')), 1e9), axis=1
    )
    out['_probe_name_sort'] = out['probe_name'].astype(str)
    sort_cols = [c for c in ['condition', 'oracle_prompt_file', 'probe_kind'] if c in out.columns]
    sort_cols.extend(['_probe_rank', '_probe_name_sort'])
    out = out.sort_values(sort_cols)
    return out.drop(columns=['_probe_rank', '_probe_name_sort'])


def build_provenance(details: pd.DataFrame, path_aliaser: PathAliaser, probe_order: dict) -> pd.DataFrame:
    """Aggregate details into a per-condition/probe provenance table, ready for display."""
    prov = (
        details
        .groupby(['condition', 'preset_source', 'oracle_prompt_file', 'probe_kind', 'probe_name'], dropna=False)
        .agg(
            n_rows=('score', 'size'),
            n_target_prompts=('target_prompt_key', 'nunique'),
            n_cache_files=('cache_path', 'nunique'),
            mean_score=('score', 'mean'),
        )
        .reset_index()
    )
    prov = path_aliaser.add_alias_column(prov, 'oracle_prompt_file', 'oracle_prompt_file_alias')
    prov = apply_probe_sort(prov, probe_order)
    return apply_display_transforms(prov)


def plot_summary(summary: pd.DataFrame, manifest: dict, top_n: int = 30) -> None:
    plot_df = summary.dropna(subset=['mean_score']).copy()
    plot_df['label'] = (
        plot_df['condition'] + ' | '
        + plot_df['probe_name'].astype(str) + ' | '
        + plot_df['oracle_prompt_file'].fillna('')
    )
    plot_df = plot_df.sort_values('mean_score', ascending=False).head(top_n)

    ax = plot_df.plot.barh(x='label', y='mean_score', xerr='se_score', figsize=(11, 10), legend=False)
    sample_size = f"actual prompts={manifest.get('actual_target_prompts')} / expected={manifest.get('expected_target_prompts')}"
    ax.set_title(f'StrongReject summary ({sample_size})')
    ax.set_xlabel('StrongReject score (%)')
    ax.set_ylabel('')
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v * 100:.0f}%"))
    ax.invert_yaxis()
    plt.tight_layout()
