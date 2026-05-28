from __future__ import annotations

"""Static HTML reporting for compiled StrongReject results."""

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable


CONDITION_LABELS = {
    "target_baseline": "Target Baseline",
    "oracle_rollout_control": "Oracle Control Baseline",
    "user_prompt_oracle": "User Prompt Oracle",
    "target_rollout_oracle": "Target Rollout Oracle",
}

ORACLE_PROMPT_FILE_LABELS = {
    "default_oracle_prompts": "Oracle Prompt A",
    "model_answer_min_200_words": "Oracle Prompt B",
}

PERCENT_COLUMNS = {
    "mean_score",
    "se_score",
    "score",
    "asr_0_2",
    "asr_0_2_se",
    "asr_0_5",
    "asr_0_5_se",
    "asr_0_8",
    "asr_0_8_se",
    "asr_1",
    "asr_1_se",
    "sd_within_prompt_oracle_rollouts",
    "sd_within_prompt_target_rollouts",
    "mean_within_prompt_sd_oracle_rollouts",
    "mean_within_prompt_sd_target_rollouts",
}

COLUMN_LABELS = {
    "condition": "Condition",
    "preset_source": "Preset",
    "oracle_prompt_file": "Oracle Prompt",
    "probe_kind": "Probe Kind",
    "probe_name": "Probe",
    "n_prompts": "Prompts",
    "n_scored": "Scored",
    "n_prompts_with_sd": "Prompts with SD",
    "mean_score": "Mean Score",
    "se_score": "SE Across Prompts",
    "score": "Score",
    "asr_0_2": "ASR >= 0.2",
    "asr_0_2_se": "ASR >= 0.2 SE",
    "asr_0_5": "ASR >= 0.5",
    "asr_0_5_se": "ASR >= 0.5 SE",
    "asr_0_8": "ASR >= 0.8",
    "asr_0_8_se": "ASR >= 0.8 SE",
    "asr_1": "ASR = 1.0",
    "asr_1_se": "ASR = 1.0 SE",
    "mean_within_prompt_sd_oracle_rollouts": "Std across Oracle Rollouts",
    "mean_within_prompt_sd_target_rollouts": "Std across Target Rollouts",
    "mean_within_prompt_n": "Mean Scored Per Prompt",
    "target_prompt_index": "Target Prompt Index",
    "target_rollout_index": "Target Rollout Index",
    "oracle_rollout_index": "Oracle Rollout Index",
    "target_prompt": "Target Prompt",
    "oracle_prompt": "Oracle Prompt Text",
    "cache_path": "Cache Path",
    "reason": "Reason",
    "path": "Path",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _display_condition(value: str) -> str:
    return CONDITION_LABELS.get(value, value)


def _display_oracle_prompt_file(value: str) -> str:
    if not value:
        return ""
    stem = Path(value).name
    if stem.endswith(".json"):
        stem = stem[:-5]
    return ORACLE_PROMPT_FILE_LABELS.get(stem, stem)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_percent(value: Any) -> str:
    parsed = _float_or_none(value)
    return "—" if parsed is None else f"{parsed * 100:.1f}%"


def _fmt_value(column: str, value: Any) -> str:
    if value in (None, ""):
        return "—"
    if column in PERCENT_COLUMNS:
        return _fmt_percent(value)
    if column == "condition":
        return _display_condition(str(value))
    if column == "oracle_prompt_file":
        return _display_oracle_prompt_file(str(value))
    text = str(value)
    if column in {"target_prompt", "oracle_prompt"} and len(text) > 260:
        text = text[:260] + "..."
    return html.escape(text)


def _score_cell_class(column: str, value: Any) -> str:
    if column not in PERCENT_COLUMNS:
        return ""
    parsed = _float_or_none(value)
    if parsed is None:
        return ""
    if parsed >= 0.8:
        return "score-high"
    if parsed >= 0.5:
        return "score-mid"
    if parsed >= 0.2:
        return "score-low"
    return "score-zero"


def _label(column: str) -> str:
    return COLUMN_LABELS.get(column, column.replace("_", " ").title())


def _table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    caption: str,
    limit: int | None = None,
    empty_text: str = "No rows.",
) -> str:
    shown = rows[:limit] if limit is not None else rows
    header = "".join(f"<th>{html.escape(_label(col))}</th>" for col in columns)
    if not shown:
        return (
            f"<section class='panel'><h2>{html.escape(caption)}</h2>"
            f"<p class='muted'>{html.escape(empty_text)}</p></section>"
        )
    body_rows = []
    for row in shown:
        cells = []
        for col in columns:
            css_class = _score_cell_class(col, row.get(col))
            cells.append(f"<td class='{css_class}'>{_fmt_value(col, row.get(col))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    note = ""
    if limit is not None and len(rows) > limit:
        note = f"<p class='muted'>Showing {limit} of {len(rows)} rows.</p>"
    return (
        f"<section class='panel'><h2>{html.escape(caption)}</h2>{note}"
        "<div class='table-wrap'><table>"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div></section>"
    )


def _card(label: str, value: Any) -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(str(value))}</div>"
        "</div>"
    )


def _file_links(compiled_dir: Path, manifest: dict[str, Any]) -> str:
    outputs = manifest.get("outputs", {})
    rows = []
    for label, path_text in outputs.items():
        path = Path(str(path_text))
        display = path.name
        href = path.name if path.parent.resolve() == compiled_dir.resolve() else str(path)
        rows.append(f"<li><a href='{html.escape(href)}'>{html.escape(label)}: {html.escape(display)}</a></li>")
    if not rows:
        return ""
    return "<section class='panel'><h2>Compiled Artifacts</h2><ul>" + "".join(rows) + "</ul></section>"


def _manifest_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            rows.append(item)
        else:
            rows.append({"path": str(item)})
    return rows


def save_strongreject_website(
    *,
    compiled_dir: Path | str = Path("results/compiled_strongreject_results"),
    output_dir: Path | str = Path("website"),
    max_detail_rows: int = 200,
    max_warning_rows: int = 100,
) -> Path:
    """Write a StrongReject-only static report and return ``index.html``."""

    compiled_dir = Path(compiled_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(compiled_dir / "manifest.json")
    summary_rows = _read_csv(compiled_dir / "strongreject_summary.csv")
    reliability_rows = _read_csv(compiled_dir / "strongreject_reliability.csv")
    detail_rows = _read_csv(compiled_dir / "strongreject_details.csv")

    if not manifest and not summary_rows and not detail_rows:
        raise FileNotFoundError(
            "No compiled StrongReject results found. Run "
            "`python compile_results.py` or `python results/compile_strongreject_results.py` first."
        )

    overview = "".join(
        [
            _card("Detail Rows", manifest.get("detail_row_count", len(detail_rows))),
            _card("Prompt-Level Rows", manifest.get("prompt_level_row_count", "—")),
            _card("Summary Rows", manifest.get("summary_row_count", len(summary_rows))),
            _card("Reliability Rows", manifest.get("reliability_row_count", len(reliability_rows))),
            _card("Missing Files", len(manifest.get("missing_files", []))),
            _card("Malformed Files", len(manifest.get("malformed_files", []))),
            _card("Skipped Score Leaves", len(manifest.get("skipped_score_leaves", []))),
            _card("Coverage Warnings", len(manifest.get("coverage_warnings", []))),
        ]
    )

    summary_columns = [
        "condition",
        "probe_name",
        "oracle_prompt_file",
        "n_prompts",
        "mean_score",
        "se_score",
        "asr_0_2",
        "asr_0_2_se",
        "asr_0_5",
        "asr_0_5_se",
        "asr_0_8",
        "asr_0_8_se",
        "asr_1",
        "asr_1_se",
    ]
    reliability_columns = [
        "condition",
        "probe_name",
        "oracle_prompt_file",
        "n_prompts_with_sd",
        "mean_within_prompt_sd_oracle_rollouts",
        "mean_within_prompt_sd_target_rollouts",
        "mean_within_prompt_n",
    ]
    detail_columns = [
        "condition",
        "target_prompt_index",
        "probe_name",
        "oracle_prompt_file",
        "target_rollout_index",
        "oracle_rollout_index",
        "score",
        "target_prompt",
        "cache_path",
    ]

    warning_rows = _manifest_rows(manifest.get("coverage_warnings", []))
    missing_rows = _manifest_rows(manifest.get("missing_files", []))
    malformed_rows = _manifest_rows(manifest.get("malformed_files", []))
    skipped_rows = _manifest_rows(manifest.get("skipped_score_leaves", []))

    html_page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>StrongReject Results</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #64748b;
      --panel: #ffffff;
      --line: #d8e0ea;
      --bg: #f7f9fc;
      --head: #1e293b;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 32px 20px;
      background: #0f172a;
      color: white;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; }}
    header p {{ margin: 0; color: #cbd5e1; }}
    main {{ padding: 24px 32px 48px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .metric-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
    }}
    .panel h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; min-width: 100%; font-size: 13px; }}
    th {{
      background: var(--head);
      color: white;
      text-align: left;
      padding: 8px;
      white-space: nowrap;
    }}
    td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 8px;
      vertical-align: top;
      max-width: 520px;
      overflow-wrap: anywhere;
    }}
    tr:hover td {{ filter: brightness(0.96); }}
    .score-zero {{ background: #eef2f7; }}
    .score-low {{ background: #fff1b8; }}
    .score-mid {{ background: #b9e6c9; }}
    .score-high {{ background: #42b883; color: #092016; font-weight: 600; }}
    a {{ color: #0f766e; }}
  </style>
</head>
<body>
  <header>
    <h1>StrongReject Results</h1>
    <p>Generated from workflow-traced StrongReject compiler outputs in {html.escape(str(compiled_dir))}</p>
  </header>
  <main>
    <section class="metrics">{overview}</section>
    {_file_links(compiled_dir, manifest)}
    {_table(summary_rows, summary_columns, caption="Summary")}
    {_table(reliability_rows, reliability_columns, caption="Within-Prompt Variability")}
    {_table(warning_rows, sorted({k for row in warning_rows for k in row.keys()}), caption="Coverage Warnings", limit=max_warning_rows, empty_text="No coverage warnings.")}
    {_table(missing_rows, ["path"], caption="Missing Files", limit=max_warning_rows, empty_text="No missing files.")}
    {_table(malformed_rows, sorted({k for row in malformed_rows for k in row.keys()}), caption="Malformed Files", limit=max_warning_rows, empty_text="No malformed files.")}
    {_table(skipped_rows, sorted({k for row in skipped_rows for k in row.keys()}), caption="Skipped Score Leaves", limit=max_warning_rows, empty_text="No skipped score leaves.")}
    {_table(detail_rows, detail_columns, caption="Detail Row Sample", limit=max_detail_rows)}
  </main>
</body>
</html>
"""

    out_path = output_dir / "index.html"
    out_path.write_text(html_page, encoding="utf-8")
    return out_path
