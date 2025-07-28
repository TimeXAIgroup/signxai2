#!/bin/bash

# Basic execution wrapper for image method comparison
# This script runs the Python comparison script which automatically discovers
# and tests all common methods between TensorFlow and PyTorch frameworks

echo "Starting image method comparison..."
echo "This will automatically discover and test all common methods between TensorFlow and PyTorch"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p results

# Run the comparison
echo "Executing comparison script..."
if python run_image_method_comparison.py; then
    echo ""
    echo "✅ Image method comparison completed successfully!"
    echo ""
    echo "Results saved in: $(pwd)/results/"
    echo "- Detailed report: results/values_image_method_comparison.txt"
    echo "- Comparison plots: results/*_comparison_tf_pt_vgg16.jpg"
    echo "- Metrics visualization: results/method_comparison_metrics.png"
else
    echo ""
    echo "❌ Image method comparison failed!"
    echo "Please check the error messages above for details."
    exit 1
fi