#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${RUNPOD_AUTO_STOP_PID_FILE:-/workspace/runpod_auto_stop.pid}"
LOG_FILE="${RUNPOD_AUTO_STOP_LOG_FILE:-/workspace/runpod_auto_stop.log}"

usage() {
  cat <<'USAGE'
Usage:
  ./runpod_auto_stop.sh <hours>
  ./runpod_auto_stop.sh status
  ./runpod_auto_stop.sh cancel

Examples:
  ./runpod_auto_stop.sh 5       # stop this RunPod pod after 5 hours
  ./runpod_auto_stop.sh 0.5     # stop after 30 minutes
  ./runpod_auto_stop.sh status  # show scheduled stop job
  ./runpod_auto_stop.sh cancel  # cancel scheduled stop job

Notes:
  - Uses: runpodctl pod stop $RUNPOD_POD_ID
  - This stops the RunPod pod instead of only shutting down Linux.
  - Requires runpodctl and RUNPOD_POD_ID to be available in the pod.
USAGE
}

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

status_job() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running_pid "$pid"; then
      echo "[OK] Auto-stop is scheduled. PID=$pid"
      echo "PID file: $PID_FILE"
      echo "Log file: $LOG_FILE"
      echo "--- recent log ---"
      tail -n 20 "$LOG_FILE" 2>/dev/null || true
      exit 0
    else
      echo "[INFO] PID file exists, but scheduled job is not running. PID=$pid"
      echo "Run './runpod_auto_stop.sh <hours>' to schedule again."
      exit 1
    fi
  else
    echo "[INFO] No auto-stop job scheduled."
    exit 1
  fi
}

cancel_job() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running_pid "$pid"; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      if is_running_pid "$pid"; then
        kill -9 "$pid" 2>/dev/null || true
      fi
      echo "[OK] Cancelled auto-stop job. PID=$pid"
    else
      echo "[INFO] No running auto-stop job found. Stale PID=$pid"
    fi
    rm -f "$PID_FILE"
  else
    echo "[INFO] No auto-stop job to cancel."
  fi
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  -h|--help|help)
    usage
    exit 0
    ;;
  status)
    status_job
    ;;
  cancel)
    cancel_job
    exit 0
    ;;
esac

HOURS="$1"

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "[ERROR] runpodctl not found. This script must run inside a RunPod pod with runpodctl installed." >&2
  exit 1
fi

POD_ID="${RUNPOD_POD_ID:-}"
if [[ -z "$POD_ID" ]]; then
  echo "[ERROR] RUNPOD_POD_ID is not set. Cannot stop the RunPod pod safely." >&2
  echo "        Check: echo \$RUNPOD_POD_ID" >&2
  exit 1
fi

SECONDS_TO_SLEEP="$(python - "$HOURS" <<'PY'
import sys, math
raw = sys.argv[1]
try:
    hours = float(raw)
except ValueError:
    raise SystemExit("[ERROR] hours must be a number, e.g. 5 or 0.5")
if not math.isfinite(hours) or hours <= 0:
    raise SystemExit("[ERROR] hours must be positive")
seconds = int(round(hours * 3600))
if seconds < 1:
    raise SystemExit("[ERROR] delay is too short")
print(seconds)
PY
)"

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if is_running_pid "$OLD_PID"; then
    echo "[WARN] Existing auto-stop job found. Cancelling old PID=$OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
  fi
fi

TARGET_TIME="$(date -d "+${SECONDS_TO_SLEEP} seconds" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date)"

nohup bash -c '
  set -euo pipefail
  seconds="$1"
  pod_id="$2"
  echo "[$(date)] Auto-stop scheduled: sleep ${seconds}s, pod=${pod_id}"
  sleep "$seconds"
  echo "[$(date)] Stopping RunPod pod: ${pod_id}"
  runpodctl pod stop "$pod_id"
  echo "[$(date)] runpodctl stop command finished"
' _ "$SECONDS_TO_SLEEP" "$POD_ID" >> "$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"

echo "[OK] RunPod auto-stop scheduled."
echo "Pod ID:   $POD_ID"
echo "Delay:    $HOURS hour(s) = ${SECONDS_TO_SLEEP}s"
echo "ETA:      $TARGET_TIME"
echo "PID:      $PID"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo
echo "Check:  ./runpod_auto_stop.sh status"
echo "Cancel: ./runpod_auto_stop.sh cancel"