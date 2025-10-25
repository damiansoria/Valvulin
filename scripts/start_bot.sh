#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="config.yaml"
if [[ $# -gt 0 && $1 != --* ]]; then
  CONFIG_FILE="$1"
  shift
fi

export VALVULIN_CONFIG="$CONFIG_FILE"

exec python -m core.engine --config "$CONFIG_FILE" "$@"
