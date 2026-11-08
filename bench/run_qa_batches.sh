#!/bin/bash
# Unattended LoCoMo QA batches. Idempotent: finished batches are skipped, so it is safe to start again.
#
#   DATA=/path/to/locomo10.json bench/run_qa_batches.sh
#
# Per batch: run `python -m bench.qa` (resumes from the fsynced cache), retry with growing backoff on any
# failure (memory pressure, Ollama hiccup, another run holding the lock), then commit the results.
# Env knobs: TOTAL=300 BATCH=100 THREADS=8 (ONNX) OLLAMA_THREADS=4 (more made Ollama's GPU runner thrash here) MIN_FREE_MB=3000 MAX_ATTEMPTS=6 BACKOFF=300 RUNNER=<cmd for tests>
set -u
cd "$(dirname "$0")/.."
: "${DATA:?set DATA to the path of locomo10.json}"
TOTAL=${TOTAL:-300}; BATCH=${BATCH:-100}; THREADS=${THREADS:-8}; MIN_FREE_MB=${MIN_FREE_MB:-3000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}; BACKOFF=${BACKOFF:-300}
RES=${RESULTS_DIR:-bench/results}; CACHE=${CACHE:-$RES/qa_hermes3.jsonl}; LOG=$RES/qa_run.log
LLM=${LLM:-ollama:hermes3:3b}; MODEL_NAME=${LLM#ollama:}
mkdir -p "$RES"
exec 9>"$RES/.qa_supervisor.lock"
flock -n 9 || { echo "another supervisor is already running"; exit 0; }

log() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
mem() { free -m | awk 'NR==2{printf "used %d MB, avail %d MB",$3,$7} NR==3{printf ", swap %d MB",$3}'; }
rows() { [ -f "$CACHE" ] && wc -l < "$CACHE" || echo 0; }

export PLAYMEMORY_ONNX_THREADS=$THREADS OMP_NUM_THREADS=$THREADS
default_runner() {
  nice -n 10 .venv/bin/python -m bench.qa --data "$DATA" --start "$1" --count "$BATCH" --llm "$LLM" \
    --ollama-opt num_batch=64 --ollama-opt num_ctx=2048 --ollama-opt num_thread="${OLLAMA_THREADS:-4}" \
    --judge f1 --budget 2000 --cache "$CACHE" --min-free-mb "$MIN_FREE_MB"
}
run_batch() { if [ -n "${RUNNER:-}" ]; then $RUNNER "$1"; else default_runner "$1"; fi; }

log "supervisor start: total=$TOTAL batch=$BATCH threads=$THREADS min_free=${MIN_FREE_MB}MB | rows=$(rows) | $(mem)"
failed=0
for start in $(seq 0 "$BATCH" $((TOTAL - BATCH))); do
  marker="$RES/.qa_done_${start}_${BATCH}"
  if [ -f "$marker" ]; then log "batch $start: already done, skipping"; continue; fi
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    log "batch $start: attempt $attempt | rows=$(rows) | $(mem)"
    run_batch "$start" 2>&1 | grep -v Warn > "$RES/qa_batch_${start}.txt"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
      touch "$marker"
      log "batch $start: DONE (rows=$(rows))"
      sed -n '/LoCoMo answer accuracy/,$p' "$RES/qa_batch_${start}.txt" | tail -45 | tee -a "$LOG" >/dev/null
      break
    fi
    log "batch $start: exit code $rc ($(tail -1 "$RES/qa_batch_${start}.txt" | cut -c1-150))"
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then log "batch $start: giving up after $attempt attempts"; failed=1; break; fi
    wait_s=$((BACKOFF * attempt)); log "batch $start: waiting ${wait_s}s before retry"; sleep "$wait_s"
  done
  ollama stop "$MODEL_NAME" >/dev/null 2>&1   # free the model's memory between batches
  git add "$CACHE" "$RES/qa_batch_${start}.txt" "$LOG" 2>/dev/null
  git commit -q -m "QA benchmark results after batch $start ($MODEL_NAME, F1 scoring)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" 2>/dev/null && log "batch $start: results committed to git"
  if [ "${PUSH:-1}" = "1" ]; then  # plain fast-forward push; a failure (offline, auth) never stops the run
    if timeout 120 git push -q origin HEAD 2>>"$LOG"; then log "batch $start: pushed to origin"; else log "batch $start: push failed (will retry after the next batch)"; fi
  fi
done
[ "${PUSH:-1}" = "1" ] && timeout 120 git push -q origin HEAD 2>>"$LOG" && log "final push ok"
log "supervisor finished (failed=$failed) | rows=$(rows) | $(mem)"
exit $failed
