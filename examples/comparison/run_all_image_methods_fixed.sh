#!/bin/bash

# Comprehensive script to run ALL available image methods using the dynamic comparison script
# This will automatically discover and test all methods available in both TensorFlow and PyTorch frameworks

echo "=" * 80
echo "Starting comprehensive image method comparison..."
echo "This will automatically discover and test ALL common methods between TensorFlow and PyTorch"
echo "=" * 80

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p results

# Set up logging
log_file="results/comprehensive_image_comparison_log.txt"
echo "Comprehensive image comparison started at $(date)" > "$log_file"
echo "Running dynamic method discovery and comparison..." >> "$log_file"

# Function to display progress
show_progress() {
    echo "⏳ $1"
    echo "⏳ $1" >> "$log_file"
}

show_success() {
    echo "✅ $1"
    echo "✅ $1" >> "$log_file"
}

show_error() {
    echo "❌ $1"
    echo "❌ $1" >> "$log_file"
}

# Check Python environment
show_progress "Checking Python environment..."
if ! python -c "import signxai" 2>/dev/null; then
    show_error "SignXAI not found in Python environment"
    echo "Please ensure SignXAI is properly installed"
    exit 1
fi

# Check required files
show_progress "Checking required files..."
required_files=(
    "run_image_method_comparison.py"
    "../../examples/data/images/example.jpg"
    "../../examples/data/models/tensorflow/VGG16/model.h5"
    "../../examples/data/models/pytorch/VGG16/vgg16_ported_weights.pt"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        show_error "Required file not found: $file"
        echo "Please ensure all model files and example image are available"
        exit 1
    fi
done

show_success "All required files found"

# Run the comprehensive comparison
show_progress "Starting comprehensive method comparison..."
echo ""
echo "This will:"
echo "1. Dynamically discover all available methods in both frameworks"
echo "2. Test all common methods between TensorFlow and PyTorch"
echo "3. Generate comparison plots for successful methods"
echo "4. Create detailed analysis reports"
echo "5. Generate summary statistics and visualizations"
echo ""

# Execute the comparison script with proper error handling
start_time=$(date +%s)

if python run_image_method_comparison.py 2>&1 | tee -a "$log_file"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    show_success "Comprehensive comparison completed successfully"
    echo "Total execution time: ${duration} seconds" >> "$log_file"
    
    # Count results
    comparison_plots=$(find results -name "*_comparison_tf_pt_vgg16.jpg" 2>/dev/null | wc -l)
    
    echo ""
    echo "📊 RESULTS SUMMARY"
    echo "=================="
    echo "Execution time: ${duration} seconds"
    echo "Comparison plots generated: $comparison_plots"
    
    # Check for key result files
    if [[ -f "results/values_image_method_comparison.txt" ]]; then
        echo "📄 Detailed report: results/values_image_method_comparison.txt"
        
        # Extract key statistics from report
        if grep -q "Overall Success Rate" results/values_image_method_comparison.txt; then
            success_rate=$(grep "Overall Success Rate" results/values_image_method_comparison.txt | head -1)
            echo "🎯 $success_rate"
        fi
        
        if grep -q "Most similar implementation" results/values_image_method_comparison.txt; then
            best_method=$(grep "Most similar implementation" results/values_image_method_comparison.txt | head -1)
            echo "🥇 $best_method"
        fi
    fi
    
    if [[ -f "results/method_comparison_metrics.png" ]]; then
        echo "📈 Metrics visualization: results/method_comparison_metrics.png"
    fi
    
    echo ""
    echo "🔍 QUICK ANALYSIS"
    echo "================="
    
    # Get framework counts from log
    if grep -q "Discovered.*TensorFlow methods" "$log_file"; then
        tf_count=$(grep "Discovered.*TensorFlow methods" "$log_file" | tail -1 | grep -oE '[0-9]+')
        echo "TensorFlow methods available: $tf_count"
    fi
    
    if grep -q "Discovered.*PyTorch methods" "$log_file"; then
        pt_count=$(grep "Discovered.*PyTorch methods" "$log_file" | tail -1 | grep -oE '[0-9]+')
        echo "PyTorch methods available: $pt_count"
    fi
    
    if grep -q "Found.*common methods" "$log_file"; then
        common_count=$(grep "Found.*common methods" "$log_file" | tail -1 | grep -oE '[0-9]+')
        echo "Common methods tested: $common_count"
    fi
    
    # Show final success rates from log if available
    if grep -q "TensorFlow success rate" "$log_file"; then
        tf_success=$(grep "TensorFlow success rate" "$log_file" | tail -1)
        echo "TF: $tf_success"
    fi
    
    if grep -q "PyTorch success rate" "$log_file"; then
        pt_success=$(grep "PyTorch success rate" "$log_file" | tail -1)
        echo "PT: $pt_success"
    fi
    
    if grep -q "Both frameworks success rate" "$log_file"; then
        both_success=$(grep "Both frameworks success rate" "$log_file" | tail -1)
        echo "Both: $both_success"
    fi
    
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    show_error "Comprehensive comparison failed after ${duration} seconds"
    echo ""
    echo "📋 TROUBLESHOOTING"
    echo "=================="
    echo "1. Check the log file for detailed error information: $log_file"
    echo "2. Ensure all dependencies are installed (TensorFlow, PyTorch, SignXAI)"
    echo "3. Verify model files are accessible and not corrupted"
    echo "4. Check available disk space for result files"
    echo ""
    
    # Show last few lines of log for quick diagnosis
    echo "Last few log entries:"
    tail -10 "$log_file"
    
    exit 1
fi

echo ""
echo "📁 RESULT LOCATIONS"
echo "==================="
echo "All results are saved in: $(pwd)/results/"
echo "- Detailed report: results/values_image_method_comparison.txt"
echo "- Method visualizations: results/*_comparison_tf_pt_vgg16.jpg"
echo "- Summary metrics plot: results/method_comparison_metrics.png"
echo "- Execution log: results/comprehensive_image_comparison_log.txt"

echo ""
echo "🎉 COMPREHENSIVE COMPARISON COMPLETE!"
echo "Run 'ls -la results/' to see all generated files"

# Final timestamp
echo "Comprehensive comparison completed at $(date)" >> "$log_file"

# Optional: Display top 10 most successful methods if report exists
if [[ -f "results/values_image_method_comparison.txt" ]]; then
    echo ""
    echo "🏆 TOP PERFORMING METHODS (by success):"
    echo "======================================="
    
    # Extract successful methods from report
    if grep -A 100 "METHOD COMPARISONS" results/values_image_method_comparison.txt | grep "Success" | head -10 > /tmp/top_methods.txt; then
        if [[ -s /tmp/top_methods.txt ]]; then
            cat /tmp/top_methods.txt | while read line; do
                method_name=$(echo "$line" | awk '{print $1}')
                mae=$(echo "$line" | awk '{print $2}')
                echo "  • $method_name (MAE: $mae)"
            done
        fi
        rm -f /tmp/top_methods.txt
    fi
fi

echo ""
echo "For detailed analysis, view the complete report:"
echo "  cat results/values_image_method_comparison.txt"