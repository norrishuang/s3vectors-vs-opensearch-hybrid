#!/usr/bin/env bash
# Launch the comparison WebUI in the foreground. Binds to localhost only.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8077}"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
. .venv/bin/activate
exec python app.py --host 127.0.0.1 --port "$PORT"
