from __future__ import annotations

import json
from pathlib import Path


def _parse_qwen_thinking(text: str) -> tuple[str, str, bool]:
    open_tag = "<think>"
    close_tag = "</think>"
    opens = text.count(open_tag)
    closes = text.count(close_tag)
    if opens > closes:
        return text.strip(), "", False
    if closes == 0:
        return "", text.strip(), True

    start = text.find(open_tag)
    end = text.rfind(close_tag)
    thinking = ""
    if start != -1 and end != -1 and end >= start:
        thinking = text[start + len(open_tag) : end].strip()
    response_only = text[end + len(close_tag) :].strip()
    return thinking, response_only, True


def migrate() -> None:
    cache_root = Path(__file__).resolve().parent / "cache" / "target_Qwen_Qwen3-8B" / "target_rollouts_temp-1.0"
    legacy_dir = cache_root / "judge_Qwen_Qwen3-8B"
    if not legacy_dir.exists():
        print(f"No legacy directory found at {legacy_dir}; skipping migration.")
        return

    target_out_dir = cache_root
    judge_out_dir = (
        Path(__file__).resolve().parent
        / "cache"
        / "target_Qwen_Qwen3-8B"
        / "judge_Qwen_Qwen3-8B_temp-1.0"
        / "user_request_fulfillment"
        / "target_rollouts_judged"
    )
    judge_out_dir.mkdir(parents=True, exist_ok=True)

    legacy_files = sorted(legacy_dir.glob("*.json"))
    for legacy_file in legacy_files:
        entries = json.loads(legacy_file.read_text())
        if not isinstance(entries, list):
            continue

        target_entries = []
        judged_entries = []
        for entry in entries:
            compliance = entry.get("compliance", {}) if isinstance(entry, dict) else {}
            raw_judgment = str(compliance.get("raw_judgment", ""))
            judge_thinking, judge_response_only, _ = _parse_qwen_thinking(raw_judgment)

            target_format = {
                "response_only": str(compliance.get("response_only", "")),
                "thinking": str(compliance.get("thinking", "")),
                "valid_response_format": bool(compliance.get("valid_response_format", True)),
            }
            target_base = {
                "rollout_index": entry.get("rollout_index"),
                "target_prompt": entry.get("target_prompt", ""),
                "target_response": entry.get("target_response", ""),
                "target_format": target_format,
            }
            target_entries.append(target_base)

            judge_compliance = {
                "judge_instruction_file": "user_request_fulfillment.jinja2",
                "score": compliance.get("score"),
                "reason": compliance.get("reason", ""),
                "raw_judgment": raw_judgment,
                "response_only": judge_response_only,
                "thinking": judge_thinking,
                "judge_skipped": bool(compliance.get("judge_skipped", False)),
                "valid_judge_format": compliance.get("valid_judge_format"),
            }
            judged_entries.append({**target_base, "compliance": judge_compliance})

        target_file = target_out_dir / legacy_file.name
        judge_file = judge_out_dir / legacy_file.name
        target_file.write_text(json.dumps(target_entries, indent=2, ensure_ascii=True))
        judge_file.write_text(json.dumps(judged_entries, indent=2, ensure_ascii=True))
        legacy_file.unlink()
        print(f"Migrated {legacy_file.name}")

    try:
        legacy_dir.rmdir()
    except OSError:
        pass
    print(f"Migration complete. Legacy directory removed: {legacy_dir}")


if __name__ == "__main__":
    migrate()
