#!/bin/bash
# Reproduce main results (Table 1) from the DiagEval paper.
#
# Prerequisites:
#   1. pip install -e .
#   2. cp configs/config.yaml.example configs/config.yaml  (fill in API key)
#   3. bash scripts/download_data.sh
#
# This script runs the diagnostic retry evaluation on the test set.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== DiagEval: Reproducing main results ==="

# Check config exists
if [ ! -f configs/config.yaml ]; then
    echo "ERROR: configs/config.yaml not found."
    echo "Please copy configs/config.yaml.example and fill in your API key."
    exit 1
fi

# Run main evaluation with diagnostic retry (branching)
python experiments/run_test.py \
    --config configs/run_config.yaml

echo ""
echo "=== Results saved to work_dirs/ ==="
