#!/usr/bin/env bash
# Restart the WebUI in the background using the saved PID file.
set -uo pipefail
cd "$(dirname "$0")"
PORT="${1:-8077}"
if [[ -f app.pid ]]; then
  OLD="$(cat app.pid)"
  if kill -0 "$OLD" 2>/dev/null; then kill "$OLD" 2>/dev/null && sleep 1; fi
fi
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
. .venv/bin/activate
nohup python app.py --host 127.0.0.1 --port "$PORT" > server.log 2>&1 &
echo $! > app.pid
echo "restarted pid $(cat app.pid) on port $PORT"
