#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-logs/overnight_${timestamp}}"
mkdir -p "$LOG_ROOT"

DRY_RUN="${DRY_RUN:-0}"
JUDGE_INSTRUCTION_PATH="${JUDGE_INSTRUCTION_PATH:-strongReject_v5.jinja2}"
TARGET_PROMPT_LIMIT="${TARGET_PROMPT_LIMIT:-100}"
NUM_ORACLE_ROLLOUTS="${NUM_ORACLE_ROLLOUTS:-50}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"

echo "Writing overnight driver log to: $LOG_ROOT/overnight_driver.log"
exec >> "$LOG_ROOT/overnight_driver.log" 2>&1

is_oom_log() {
  local log_file="$1"
  grep -Eiq "CUDA out of memory|OutOfMemoryError|out of memory|CUBLAS_STATUS_ALLOC_FAILED" "$log_file"
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_attempt() {
  local log_file="$1"
  shift
  echo
  echo "[$(date)] running: $(print_command "$@")"
  echo "[$(date)] log: $log_file"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@" > "$log_file" 2>&1
}

run_prompt_only_oracle() {
  local eval_batches=(32 32 16 8 4 2 1 1 1 1)
  local judge_batches=(16 8 8 8 8 8 8 4 2 1)
  local attempt=1

  for i in "${!eval_batches[@]}"; do
    local eval_batch="${eval_batches[$i]}"
    local judge_batch="${judge_batches[$i]}"
    local log_file="$LOG_ROOT/prompt_only_oracle_attempt_${attempt}_eval-${eval_batch}_judge-${judge_batch}.log"
    local cmd=(
      ./run_oracle_experiment.sh
      --preset prompt_only_oracle
      --target-prompt-limit "$TARGET_PROMPT_LIMIT"
      --num-oracle-rollouts "$NUM_ORACLE_ROLLOUTS"
      --oracle-eval-batch-size "$eval_batch"
      --oracle-judge-batch-size "$judge_batch"
      --judge-instruction-path "$JUDGE_INSTRUCTION_PATH"
    )

    run_attempt "$log_file" "${cmd[@]}"
    local status=$?
    if [[ "$status" -eq 0 ]]; then
      echo "[$(date)] prompt-only oracle succeeded on attempt $attempt"
      return 0
    fi
    if ! is_oom_log "$log_file"; then
      echo "[$(date)] prompt-only oracle failed with non-OOM error on attempt $attempt"
      echo "See: $log_file"
      return "$status"
    fi
    echo "[$(date)] prompt-only oracle hit OOM on attempt $attempt; retrying with smaller batch settings"
    attempt=$((attempt + 1))
  done

  echo "[$(date)] prompt-only oracle exhausted OOM retry ladder"
  return 1
}

run_target_judging_only() {
  local target_judge_batches=(16 8 4 2 1)
  local attempt=1

  for target_judge_batch in "${target_judge_batches[@]}"; do
    local log_file="$LOG_ROOT/target_judging_only_attempt_${attempt}_target-judge-${target_judge_batch}.log"
    local cmd=(
      ./run_oracle_experiment.sh
      --preset target_judging_only
      --target-prompt-limit "$TARGET_PROMPT_LIMIT"
      --num-rollouts "$NUM_ROLLOUTS"
      --target-judge-batch-size "$target_judge_batch"
      --judge-instruction-path "$JUDGE_INSTRUCTION_PATH"
    )

    run_attempt "$log_file" "${cmd[@]}"
    local status=$?
    if [[ "$status" -eq 0 ]]; then
      echo "[$(date)] target judging succeeded on attempt $attempt"
      return 0
    fi
    if ! is_oom_log "$log_file"; then
      echo "[$(date)] target judging failed with non-OOM error on attempt $attempt"
      echo "See: $log_file"
      return "$status"
    fi
    echo "[$(date)] target judging hit OOM on attempt $attempt; retrying with smaller target judge batch"
    attempt=$((attempt + 1))
  done

  echo "[$(date)] target judging exhausted OOM retry ladder"
  return 1
}

echo "Overnight run directory: $LOG_ROOT"
echo "JUDGE_INSTRUCTION_PATH=$JUDGE_INSTRUCTION_PATH"
echo "TARGET_PROMPT_LIMIT=$TARGET_PROMPT_LIMIT"
echo "NUM_ORACLE_ROLLOUTS=$NUM_ORACLE_ROLLOUTS"
echo "NUM_ROLLOUTS=$NUM_ROLLOUTS"
echo "DRY_RUN=$DRY_RUN"

run_prompt_only_oracle || exit $?
run_target_judging_only || exit $?

echo
echo "[$(date)] overnight run completed successfully"
