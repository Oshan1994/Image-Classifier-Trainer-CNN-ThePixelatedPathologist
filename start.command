#!/usr/bin/env bash
set -euo pipefail
# change into the script's folder (the app root)
cd "$(dirname "$0")"
# Prefer python3 if available; fall back to python
PY="$(command -v python3 || command -v python)"
exec "$PY" launch.py
