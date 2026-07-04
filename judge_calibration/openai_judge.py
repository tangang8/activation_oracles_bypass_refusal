"""Step 3 -- Async GPT-4o StrongReject judge for the gold set.

This is the challenger judge. It wraps the OpenAI API in an async client with bounded
concurrency and retry/backoff, but deliberately reuses this repo's own machinery for the
things that must stay identical to the incumbent judge:

  * the rubric prompt      -> `judge_instruction_utils.load_judge_instruction`
  * the output parser+score -> `judge_parsing._parse_judge_output` (strongreject mode)
  * the cache path + I/O    -> `cache_utils.{api_judge_cache_file_path,write_json,load_json}`

so a GPT-4o score is produced by exactly the same rubric + `(1-refusal)*((spec+conv)/2)/5`
normalization and stored in the same content-addressed cache tree as the local judge.

Failure modes handled explicitly (see inline notes):
  1. missing API key                         -> fail fast with a clear message
  2. transient API errors (429/5xx/timeout)  -> capped exponential backoff + jitter
                                                 (retryable set is transient-only)
  3. malformed / truncated judge output       -> re-query with growing max_tokens, then
                                                 record valid_judge_format=False (excluded
                                                 downstream) and DO NOT cache, so a rerun retries
  4. empty response text                       -> skip the judge (judge_skipped=True)
  5. cache corruption                          -> load_json returns None -> re-query
  6. non-transient API errors (400/401/403,    -> isolated per row as valid_judge_format=False
     e.g. content-moderation rejections)         (uncached, so a rerun retries); one bad row
                                                 never aborts the 250-row batch

Usage:
    PYTHONPATH=. python judge_calibration/openai_judge.py --in judge_calibration/gold/gold_sample.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge_calibration import config

# Reused, unchanged, from the existing framework.
from cache_utils import api_judge_cache_file_path, load_json, write_json
from judge_instruction_utils import load_judge_instruction


def _load_dotenv_files() -> None:
    """Best-effort .env loading (no hard dependency on python-dotenv), override=False."""
    candidates = [
        config.REPO_ROOT.parent / ".env",
        config.REPO_ROOT / ".env",
        Path("/workspace/.env"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _import_parser():
    """Import the shared strongreject parser.

    `judge_parsing` holds the exact parser used by the local Qwen judge (rollout_utils
    re-exports it) but carries no torch/transformers dependency, so the API judge reuses
    identical scoring without pulling in the model stack.
    """
    from judge_parsing import _parse_judge_output

    return _parse_judge_output


@dataclass
class OpenAIJudgeConfig:
    model: str = config.GPT4O_JUDGE_MODEL
    temperature: float = config.JUDGE_TEMPERATURE
    max_tokens: int = 1000
    max_concurrency: int = 8
    max_api_attempts: int = 6            # transient-error retries per call
    backoff_base_seconds: float = 1.5
    backoff_cap_seconds: float = 30.0
    malformed_retry_attempts: int = 4    # parse-failure re-queries (grow max_tokens)
    cache_root: Path = config.CACHE_ROOT
    rubric_path: str = config.RUBRIC_PATH
    api_key_env: str = "OPENAI_API_KEY"
    _jitter: random.Random = field(default_factory=lambda: random.Random(0), repr=False)


class OpenAIStrongRejectJudge:
    def __init__(self, cfg: OpenAIJudgeConfig | None = None, client: Any | None = None):
        self.cfg = cfg or OpenAIJudgeConfig()
        self.judge_instruction_file, self.rubric_stem, self.rubric_template = load_judge_instruction(
            self.cfg.rubric_path
        )
        self.rubric_version = config.rubric_version(self.rubric_template)
        self._parse_judge_output = _import_parser()
        self._client = client  # allow injection for tests
        self._retryable: tuple[type[BaseException], ...] | None = None

    # --- client / errors ---------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        _load_dotenv_files()
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The 'openai' package is required: pip install openai") from exc

        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No OpenAI API key found. Set {self.cfg.api_key_env} in the environment "
                f"or a .env file."
            )
        self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    def _retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        if self._retryable is not None:
            return self._retryable
        excs: list[type[BaseException]] = [TimeoutError, ConnectionError]
        try:
            import openai

            # Transient-only. Deliberately excludes the base ``openai.APIError`` because
            # permanent errors (BadRequestError/AuthenticationError/PermissionDeniedError,
            # e.g. content-moderation 400s) subclass it; retrying those wastes calls and
            # they are instead isolated per row in ``score_many``.
            for name in (
                "RateLimitError", "APITimeoutError", "APIConnectionError",
                "InternalServerError",
            ):
                exc = getattr(openai, name, None)
                if isinstance(exc, type):
                    excs.append(exc)
        except Exception:  # pragma: no cover
            pass
        self._retryable = tuple(dict.fromkeys(excs))
        return self._retryable

    # --- cache -------------------------------------------------------------------
    def _cache_path(self, user_prompt: str, model_response: str) -> Path:
        # Standard cache tree under cache/, built by cache_utils (the one place cache
        # paths live). Content-addressed on judge model + temperature + rubric version +
        # prompt + response.
        return api_judge_cache_file_path(
            cache_root=str(self.cfg.cache_root),
            judge_model_name=self.cfg.model,
            judge_generation_kwargs={"temperature": self.cfg.temperature},
            judge_instruction_stem=self.rubric_stem,
            user_prompt=user_prompt,
            model_response=model_response,
        )

    # --- prompt ------------------------------------------------------------------
    def _build_messages(self, user_prompt: str, model_response: str) -> list[dict[str, str]]:
        # The full rubric -- which embeds the harmful user prompt and the model response --
        # goes in a single user message, identical to what the local judge sends via the
        # chat template. There is no system message.
        content = self.rubric_template.format(user_prompt=user_prompt, model_response=model_response)
        return [{"role": "user", "content": content}]

    async def _call_api(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        client = self._get_client()
        retryable = self._retryable_exceptions()
        last_exc: BaseException | None = None
        for attempt in range(self.cfg.max_api_attempts):
            try:
                resp = await client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except retryable as exc:  # transient -> backoff and retry
                last_exc = exc
                if attempt == self.cfg.max_api_attempts - 1:
                    break
                delay = min(
                    self.cfg.backoff_cap_seconds,
                    self.cfg.backoff_base_seconds * (2 ** attempt),
                )
                delay += self.cfg._jitter.uniform(0, self.cfg.backoff_base_seconds)
                await asyncio.sleep(delay)
        raise RuntimeError(f"OpenAI call failed after {self.cfg.max_api_attempts} attempts: {last_exc}")

    # --- scoring -----------------------------------------------------------------
    def _skip_leaf(self, response_id: str, user_prompt: str, model_response: str) -> dict[str, Any]:
        return self._decorate(
            {
                "score": None,
                "reason": "No response text; skipped judge.",
                "raw_judgment": "",
                "response_only": "",
                "thinking": "",
                "judge_skipped": True,
                "valid_judge_format": None,
            },
            response_id, user_prompt, model_response,
        )

    def _error_leaf(self, response_id: str, user_prompt: str, model_response: str,
                    exc: BaseException) -> dict[str, Any]:
        """Per-row failure leaf for an API error that survived retries.

        Returned (not raised) so one bad row -- e.g. a content-moderation 400 -- cannot
        abort the batch. Not written to cache, so a rerun retries it. ``valid_judge_format``
        is False, so ``score_judges``/``analyze`` count and drop it like a parse failure.
        """
        return self._decorate(
            {
                "score": None,
                "reason": f"API call failed: {type(exc).__name__}: {exc}",
                "raw_judgment": "",
                "response_only": "",
                "thinking": "",
                "judge_skipped": False,
                "valid_judge_format": False,
            },
            response_id, user_prompt, (model_response or "").strip(),
        )

    def _decorate(self, leaf: dict[str, Any], response_id: str, user_prompt: str,
                  model_response: str) -> dict[str, Any]:
        return {
            **leaf,
            "response_id": response_id,
            "judge_model": self.cfg.model,
            "judge_instruction_file": self.judge_instruction_file,
            "rubric_version": self.rubric_version,
            "temperature": self.cfg.temperature,
            "user_prompt": user_prompt,
            "model_response": model_response,
        }

    async def score_one(
        self, response_id: str, user_prompt: str, model_response: str, *, use_cache: bool = True
    ) -> dict[str, Any]:
        model_response = (model_response or "").strip()
        if not model_response:
            return self._skip_leaf(response_id, user_prompt, model_response)

        cache_file = self._cache_path(user_prompt, model_response)
        if use_cache:
            cached = load_json(cache_file)
            if isinstance(cached, dict):
                return cached

        messages = self._build_messages(user_prompt, model_response)
        max_tokens = self.cfg.max_tokens
        leaf: dict[str, Any] = {}
        for attempt in range(self.cfg.malformed_retry_attempts + 1):
            raw = await self._call_api(messages, max_tokens)
            leaf = self._parse_judge_output(raw, None, judge_scoring_mode="strongreject")
            if leaf.get("valid_judge_format"):
                break
            # Truncation is the usual culprit; grow the budget before retrying.
            max_tokens = max(self.cfg.max_tokens + 1, self.cfg.max_tokens * (2 ** (attempt + 1)))

        result = self._decorate(leaf, response_id, user_prompt, model_response)
        # Only persist committed results; leave parse failures uncached so a rerun retries.
        if result.get("valid_judge_format") is True:
            write_json(cache_file, result)
        return result

    async def score_many(
        self, items: Sequence[tuple[str, str, str]], *, use_cache: bool = True, show_progress: bool = True
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self.cfg.max_concurrency)

        async def _worker(item: tuple[str, str, str]) -> dict[str, Any]:
            response_id, user_prompt, model_response = item
            async with sem:
                try:
                    return await self.score_one(*item, use_cache=use_cache)
                except Exception as exc:  # isolate per row; never abort the batch
                    return self._error_leaf(response_id, user_prompt, model_response, exc)

        tasks = [asyncio.ensure_future(_worker(item)) for item in items]
        results: list[dict[str, Any]] = []
        iterator = asyncio.as_completed(tasks)
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(tasks), desc=f"{self.cfg.model} judge")
        except Exception:
            pass
        # Preserve input order: collect by response_id then reindex.
        for fut in iterator:
            results.append(await fut)
        by_id: dict[str, dict[str, Any]] = {r["response_id"]: r for r in results}
        return [by_id[item[0]] for item in items]


def _summarize(leaves: list[dict[str, Any]]) -> str:
    total = len(leaves)
    skipped = sum(1 for l in leaves if l.get("judge_skipped"))
    invalid = sum(1 for l in leaves if l.get("valid_judge_format") is False)
    scored = [float(l["score"]) for l in leaves if isinstance(l.get("score"), (int, float))
              and not isinstance(l.get("score"), bool) and not l.get("judge_skipped")]
    mean = f"{sum(scored)/len(scored):.3f}" if scored else "n/a"
    return f"total={total} scored={len(scored)} skipped={skipped} invalid_format={invalid} mean_score={mean}"


def score_csv(
    in_csv: Path, out_csv: Path, cfg: OpenAIJudgeConfig, *,
    id_col: str = "response_id", prompt_col: str = "harmful_prompt",
    response_col: str = "response_text", use_cache: bool = True,
) -> list[dict[str, Any]]:
    with in_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = [(r[id_col], r[prompt_col], r[response_col]) for r in rows]

    judge = OpenAIStrongRejectJudge(cfg)
    start = time.perf_counter()
    leaves = asyncio.run(judge.score_many(items, use_cache=use_cache))
    elapsed = time.perf_counter() - start
    print(f"[openai_judge] {_summarize(leaves)} elapsed={elapsed:.1f}s")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "response_id", "judge_model", "judge_instruction_file", "rubric_version",
        "score", "score_scale", "refusal", "specificity", "convincingness",
        "valid_judge_format", "judge_skipped", "reason",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(leaves)
    print(f"[openai_judge] wrote {len(leaves)} scores -> {out_csv}")
    return leaves


def main() -> None:
    parser = argparse.ArgumentParser(description="Score responses with the GPT-4o StrongReject judge.")
    parser.add_argument("--in", dest="in_csv", default=str(config.GOLD_SAMPLE_CSV))
    parser.add_argument("--out", default=str(config.GOLD_DIR / "gpt4o_scores.csv"))
    parser.add_argument("--model", default=config.GPT4O_JUDGE_MODEL)
    parser.add_argument("--temperature", type=float, default=config.JUDGE_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = OpenAIJudgeConfig(
        model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
        max_concurrency=args.concurrency,
    )
    score_csv(Path(args.in_csv), Path(args.out), cfg, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
