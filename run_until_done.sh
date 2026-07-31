#!/usr/bin/env bash
# =============================================================================
# run_until_done.sh — Auto-restarting benchmark runner
#
# Runs python run_all.py --config config.yaml --resume in a loop.  If the
# Python process crashes (segfault / C-level crash) the script restarts it
# with --resume so already-completed cells are skipped.  When the run
# completes cleanly (exit code 0) the loop exits.
#
# Usage:
#   bash run_until_done.sh
#   bash run_until_done.sh --full          # override config mode
#   bash run_until_done.sh --no-isolation  # run in-process (debug)
# =============================================================================

set -euo pipefail

PYTHON="/c/Users/13680/.conda/envs/chuantaoli/python.exe"
PROJECT_DIR="D:/imb_project"
RESTART_DELAY=10   # seconds to wait before restarting after a crash

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

# Ensure PYTHONWARNINGS and thread limits are set in the wrapper's environment
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "=============================================================================="
    echo " Attempt #${ATTEMPT} — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    set +e
    "$PYTHON" run_all.py --config config.yaml --resume "$@"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=============================================================================="
        echo " SUCCESS — benchmark completed (exit 0) at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================================================================="
        exit 0
    fi

    echo ""
    echo "------------------------------------------------------------------------------"
    echo " Python exited with code ${EXIT_CODE} — restarting in ${RESTART_DELAY}s ..."
    echo " (--resume will skip already-completed cells)"
    echo "------------------------------------------------------------------------------"
    sleep "$RESTART_DELAY"
done
