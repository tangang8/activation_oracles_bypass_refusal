"""Judge calibration & selection for activation-oracle StrongReject scoring.

See PLAN.md at the repo root for the full spec. This package builds a human-labeled
gold set of AO responses and uses it to (1) pick the judge (Qwen3-8B vs GPT-4o) and
(2) pick the compliance threshold for the winning judge.
"""
