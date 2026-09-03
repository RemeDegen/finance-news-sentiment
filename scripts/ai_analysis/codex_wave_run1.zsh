#!/bin/zsh
# Codex judge wave driver for run1.
# Usage: codex_wave_run1.zsh <max_packets>
# Skips packets already judged (non-empty .raw), processes the next
# <max_packets> in batches of 6 parallel lanes, stops early when a batch
# comes back mostly failed (quota exhausted).
set -u
MAX=${1:?usage: codex_wave_run1.zsh <max_packets>}
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="run1"
VERD="$PROJ/data/ai_analysis/runs/$RUN/verdicts"
LOG="$PROJ/data/ai_analysis/runs/$RUN/logs"
mkdir -p "$LOG"

# Collect pending packet ids (no non-empty verdict yet), up to MAX.
pending=()
for f in "$PROJ/data/ai_analysis/runs/$RUN/packets"/p*.json; do
  pid=$(basename "$f" .json)
  raw="$VERD/codex_${pid}.raw"
  if [[ -s "$raw" ]]; then continue; fi
  pending+=("$pid")
  (( ${#pending[@]} >= MAX )) && break
done
echo "WAVE_START pending=${#pending[@]} max=$MAX"
(( ${#pending[@]} == 0 )) && { echo "NOTHING_TO_DO"; exit 0; }

run_lane() {
  local pid=$1
  sleep $((RANDOM % 4 + 1))
  codex exec --skip-git-repo-check --sandbox read-only \
    -C "$PROJ" \
    -m gpt-5.6-sol -c model_reasoning_effort=medium --color never \
    -o "$VERD/codex_${pid}.raw" \
    "You are judge B in a blind labeling pipeline. Read the file scripts/ai_analysis/judge_spec.md and the packet file data/ai_analysis/runs/$RUN/packets/${pid}.json. Follow the spec exactly and answer with ONLY the strict JSON array of verdicts - one object per packet item, same order, every item covered. Do not echo the file contents to your output; read them quietly, then reply with the array only." \
    > "$LOG/${pid}.log" 2>&1
}

done_n=0; fail_n=0
for ((i = 1; i <= ${#pending[@]}; i += 6)); do
  batch=("${pending[@]:$((i-1)):6}")
  for pid in "${batch[@]}"; do run_lane "$pid" & done
  wait
  bfail=0
  for pid in "${batch[@]}"; do
    if [[ -s "$VERD/codex_${pid}.raw" ]]; then
      (( done_n++ ))
    else
      (( bfail++ )); (( fail_n++ ))
      echo "LANE_FAILED $pid (log tail):"; tail -3 "$LOG/${pid}.log"
    fi
  done
  echo "BATCH done=$done_n fail=$fail_n"
  if (( bfail >= 4 )); then
    echo "QUOTA_SUSPECTED: batch mostly failed, stopping wave."
    break
  fi
done
echo "WAVE_END done=$done_n fail=$fail_n"
ls "$VERD" | grep -c '^codex_.*\.raw$' | xargs echo TOTAL_CODEX_VERDICTS
grep -A1 -h "tokens used" "$LOG"/*.log 2>/dev/null \
  | grep -E "^[0-9][0-9,]*$" | tr -d ',' \
  | awk '{s+=$1; n+=1} END {print "WAVE_TOKENS", s, "lanes", n}'
