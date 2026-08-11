#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAR="${TLA2TOOLS_JAR:-$ROOT/tools/tla2tools.jar}"
if [[ ! -f "$JAR" ]]; then
  echo "Missing tla2tools.jar. Set TLA2TOOLS_JAR=/path/to/tla2tools.jar" >&2
  exit 2
fi
mkdir -p "$ROOT/output/tlc"
java -cp "$JAR" tlc2.TLC   -config "$ROOT/formal/AstroGuardQFV.cfg"   "$ROOT/formal/AstroGuardQFV.tla"   | tee "$ROOT/output/tlc/corrected_model.log"
