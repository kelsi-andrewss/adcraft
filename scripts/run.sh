#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

usage() {
    cat <<EOF
AdCraft Runner

Usage: ./scripts/run.sh <command> [options]

Commands:
  dashboard       Start the Streamlit dashboard (runs in foreground)
  test            Run pressure test (passes all args to pressure_test.py)
  both            Start dashboard in background, then run pressure test
  calibrate       Quick alias: run calibration only
  smoke           Quick alias: run smoke test (3 briefs)
  full            Quick alias: run full batch

Examples:
  ./scripts/run.sh dashboard
  ./scripts/run.sh test --stage smoke --rpm 8
  ./scripts/run.sh both --stage all
  ./scripts/run.sh calibrate
EOF
}

case "${1:-}" in
  dashboard)
    echo "Starting dashboard at http://localhost:8501"
    uv run streamlit run src/dashboard/app.py
    ;;
  test)
    shift
    uv run python scripts/pressure_test.py "$@"
    ;;
  both)
    shift || true
    echo "Starting dashboard in background..."
    uv run streamlit run src/dashboard/app.py &
    DASH_PID=$!
    echo "Dashboard PID: $DASH_PID (http://localhost:8501)"
    sleep 2
    echo ""
    echo "Running pressure test..."
    uv run python scripts/pressure_test.py "$@"
    echo ""
    echo "Dashboard still running at http://localhost:8501 (PID: $DASH_PID)"
    echo "Stop with: kill $DASH_PID"
    wait $DASH_PID 2>/dev/null || true
    ;;
  calibrate)
    uv run python scripts/pressure_test.py --stage calibrate
    ;;
  smoke)
    shift || true
    uv run python scripts/pressure_test.py --stage smoke "$@"
    ;;
  full)
    shift || true
    uv run python scripts/pressure_test.py --stage full "$@"
    ;;
  *)
    usage
    ;;
esac
