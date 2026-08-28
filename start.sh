#!/bin/sh
set -e
PORT="${1:-8080}"

echo
echo "My 5-Layer Chess Neural Network"
echo "Starting local server on port $PORT..."
echo
echo "Open Safari and visit:"
echo "  http://localhost:$PORT"
echo
echo "Press Ctrl+C here to stop the server."
echo

python3 -m http.server "$PORT"
