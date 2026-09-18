#!/usr/bin/env bash
# Serve the annotated-frame dashboard from the last recorded run.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${1:-$ROOT/validation/dashboard}"
PORT="${PORT:-8099}"
[ -f "$DIR/index.json" ] || { echo "No run recorded in $DIR. Run with --dashboard $DIR first." >&2; exit 1; }
cp "$ROOT/docs/dashboard.html" "$DIR/index.html"
echo "Dashboard: http://localhost:$PORT/  (serving $DIR)"
cd "$DIR" && exec python3 -m http.server "$PORT"
