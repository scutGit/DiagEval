#!/bin/bash
# Download pre-trained model weights and test data for DiagEval
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== DiagEval: Downloading data and model weights ==="

# 1. Icon detection model (OmniParser)
ICON_MODEL="$PROJECT_DIR/data/omniparser_icon_detect.pt"
if [ ! -f "$ICON_MODEL" ]; then
    echo "Downloading icon detection model..."
    # TODO: Replace with actual download URL after release
    echo "[INFO] Please download omniparser_icon_detect.pt manually and place it at:"
    echo "       $ICON_MODEL"
    echo "       (Required only for [ultra] mode with icon detection)"
else
    echo "Icon detection model already exists: $ICON_MODEL"
fi

# 2. Example test cases
EXAMPLE_DIR="$PROJECT_DIR/data/example"
if [ ! -f "$EXAMPLE_DIR/sample_case.json" ]; then
    echo "Example test cases already included in data/example/"
fi

echo ""
echo "=== Download complete ==="
echo "Next steps:"
echo "  1. Copy configs/config.yaml.example -> configs/config.yaml"
echo "  2. Fill in your API key"
echo "  3. Run: bash scripts/reproduce.sh"
