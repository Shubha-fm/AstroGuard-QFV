#!/usr/bin/env bash
# AstroGuard-QFV: clean macOS 18K experiment, starting with an EMPTY input folder.
set -euo pipefail
cd "$(dirname "$0")"
ACTION="${1:-help}"
case "$ACTION" in
  setup)
    P="${PYTHON_BIN:-python3.11}"
    command -v "$P" >/dev/null 2>&1 || {
      echo "Cannot find $P. Install Python 3.11 (e.g. brew install python@3.11)" >&2
      echo "Or use: PYTHON_BIN=python3.12 bash mac_run.sh setup" >&2
      exit 1
    }
    "$P" - <<'PYCHECK'
import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit('Use Python 3.11 or 3.12. Detected: ' + sys.version)
print('Using Python', sys.version.split()[0])
PYCHECK
    "$P" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip setuptools wheel
    .venv/bin/python -m pip install -r requirements.txt -r requirements-alerce.txt
    .venv/bin/python -m pytest tests/test_18k.py tests/test_mac_fresh.py -q
    echo "Setup passed. Run: bash mac_run.sh check"
    exit 0
    ;;
  help)
    cat <<'HELP_TEXT'
Fresh macOS 18K experiment (no Windows files required)
  bash mac_run.sh setup                       create Python 3.11 environment
  bash mac_run.sh check                       inspect empty/fresh query frame
  bash mac_run.sh query 100 40                query up to 100 pages, max 40/subclass
  bash mac_run.sh collect 1000                download at most 1,000 new light curves
  bash mac_run.sh prepare                     freeze and split 18K balanced cohort
  bash mac_run.sh smoke                       classical software validation only
  bash mac_run.sh classical                   train six baselines, 5 seeds
  bash mac_run.sh analyze-classical           evaluate trained baselines
  bash mac_run.sh setup-quantum               install optional quantum dependencies
  bash mac_run.sh quantum-sanity              small quantum test (not manuscript result)
  bash mac_run.sh matched                     matched quantum+classical study
  bash mac_run.sh analyze-matched             paired comparison and plots
  bash mac_run.sh verify                      TLA+/TLC via ./tla2tools.jar
  bash mac_run.sh tests                       offline software tests

The collector saves input/collection_progress.sqlite3 and input/lightcurves/.
Query must finish before collection. Use 'query 100 80' to expand the candidate frame.
HELP_TEXT
    exit 0
    ;;
esac
[[ -x .venv/bin/python ]] || { echo 'Run: bash mac_run.sh setup' >&2; exit 1; }
PY=.venv/bin/python
case "$ACTION" in
  check) "$PY" mac_query_status.py --input input --pages "${2:-40}" ;;
  query)
    mkdir -p input RESULTS_18K
    "$PY" run_large.py --stage collect --input input/objects.csv --output RESULTS_18K/query_logs --target-objects 18000 --pages "${3:-40}" --page-size 500 --max-queries "${2:-100}" --query-only --request-delay 0.2 --fail-fast
    "$PY" mac_query_status.py --input input --pages "${3:-40}"
    ;;
  collect)
    "$PY" mac_query_status.py --input input --pages "${3:-40}" --require-complete
    "$PY" run_18k_collect.py --input input --output RESULTS_18K --per-class 6000 --max-new "${2:-1000}" --request-delay 0.2
    ;;
  setup-quantum) "$PY" -m pip install -r requirements-quantum.txt ;;
  prepare) "$PY" run_18k.py prepare --input input --output RESULTS_18K ;;
  smoke)
    "$PY" run_18k.py train --profile smoke --epochs 2 --max-train 90 --tag smoke --device cpu
    "$PY" run_18k.py analyze --profile smoke --bootstrap 300 --tag smoke
    ;;
  classical) "$PY" run_18k.py train --profile classical --epochs 40 --device cpu ;;
  analyze-classical) "$PY" run_18k.py analyze --profile classical --bootstrap 3000 ;;
  quantum-sanity) "$PY" run_18k.py train --profile matched --max-train 90 --epochs 2 --seeds 42 --tag quantum_sanity --device cpu ;;
  matched) "$PY" run_18k.py train --profile matched --max-train 3000 --epochs 40 --device cpu ;;
  analyze-matched) "$PY" run_18k.py analyze --profile matched --bootstrap 3000 ;;
  verify)
    [[ -f tla2tools.jar ]] || { echo 'Put official tla2tools.jar next to mac_run.sh first' >&2; exit 1; }
    command -v java >/dev/null || { echo 'Install a Java JDK first' >&2; exit 1; }
    "$PY" run_18k.py verify --jar ./tla2tools.jar --profile matched
    ;;
  tests) "$PY" -m pytest tests/test_18k.py tests/test_mac_fresh.py -q ;;
  *) echo "Unknown action: $ACTION (try bash mac_run.sh help)" >&2; exit 2 ;;
esac
