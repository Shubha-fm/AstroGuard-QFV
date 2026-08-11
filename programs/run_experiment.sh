#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASTROGUARD_OUTPUT="${ASTROGUARD_OUTPUT:-$ROOT/output}"
python "$ROOT/src/astroguard_qfv.py"
