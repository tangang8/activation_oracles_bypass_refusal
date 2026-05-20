from __future__ import annotations

from pathlib import Path


INSTRUCTIONS_DIR = Path(__file__).resolve().parent / "judge_classification_instructions"


def load_judge_instruction(path_or_name: str) -> tuple[str, str, str]:
    candidate = Path(path_or_name)
    if not candidate.is_absolute():
        if candidate.parent == Path("."):
            candidate = INSTRUCTIONS_DIR / candidate
        else:
            candidate = (Path(__file__).resolve().parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Judge instruction file not found: {candidate}")

    template_text = candidate.read_text(encoding="utf-8")
    if "{user_prompt}" not in template_text or "{model_response}" not in template_text:
        raise ValueError(
            f"Judge instruction template must include '{{user_prompt}}' and '{{model_response}}': {candidate}"
        )

    filename = candidate.name
    stem = candidate.stem
    return filename, stem, template_text
