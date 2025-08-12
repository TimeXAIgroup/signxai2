#!/bin/bash

# Script to run all curated timeseries methods using the extended comparison script
# This enforces the correct ECG record to pathology mapping:
#   03509 -> AVB
#   12131 -> ISCH
#   02906 -> RBBB
#   14493 -> LBBB

echo "Starting curated timeseries method comparison..."
echo "This will generate comparison plots for verified working methods"

# Create results directory
mkdir -p results

# Counter for progress tracking
count=0

# Define the correct ECG record to pathology mapping
declare -A record_to_pathology=(
    ["03509"]="AVB"
    ["12131"]="ISCH"
    ["02906"]="RBBB"
    ["14493"]="LBBB"
)

# Default to 03509_hr with AVB
record_id=${1:-"03509_hr"}

# Extract the numeric part of the record ID
record_num=$(echo $record_id | grep -oE '[0-9]+' | head -1)

# Get the correct pathology for this record
if [[ -n "${record_to_pathology[$record_num]}" ]]; then
    pathology="${record_to_pathology[$record_num]}"
    echo "Using record ID: $record_id with pathology: $pathology (correct mapping)"
else
    echo "ERROR: Unknown record ID: $record_id"
    echo "Valid record IDs and their pathologies:"
    echo "  03509_hr -> AVB"
    echo "  12131_hr -> ISCH"
    echo "  02906_hr -> RBBB"
    echo "  14493_hr -> LBBB"
    exit 1
fi

# Curated methods that work for timeseries in BOTH frameworks
methods=(
    # Core gradient-based methods (verified working)
    "gradient"
    "integrated_gradients"
    "smoothgrad"
    "guided_backprop"
    "deconvnet"
    "grad_cam"
    
    # Input/Sign variants (verified working)
    "input_t_gradient"
    "gradient_x_input"
    "gradient_x_sign"
    
    # Core LRP methods (verified working)
    "lrp_epsilon"
    "lrp_alpha_1_beta_0"
    "lrp_alpha_2_beta_1"
    "lrp_z"
    "lrp_flat"
    
    # LRP epsilon variants with std_x
    "lrp_epsilon_0_5_std_x"
    "lrp_epsilon_0_1_std_x"
    "lrp_epsilon_0_25_std_x"
    
    # LRP sign variants (working subset)
    "lrpsign_epsilon_0_5_std_x"
    "lrpsign_epsilon_0_1_std_x"
    "lrpsign_alpha_1_beta_0"
    
    # Additional working methods
    "vargrad"
    "deeplift"
    
    # GradCAM for timeseries
    "grad_cam_timeseries"
    
    # Sign variants that work
    "gradient_x_sign_mu_0"
    "smoothgrad_x_sign"
    "guided_backprop_x_sign"
    "deconvnet_x_sign"
)

total=${#methods[@]}

# Log file for tracking progress and errors
log_file="results/curated_timeseries_${pathology}_${record_id}_comparison_log.txt"
echo "Curated timeseries comparison started at $(date)" > "$log_file"
echo "Pathology: $pathology" >> "$log_file"
echo "Record ID: $record_id" >> "$log_file"
echo "Total methods to process: $total" >> "$log_file"

# Run each method
for method in "${methods[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] Running method: $method (pathology: $pathology, record: $record_id)"
    echo "[$count/$total] Running method: $method" >> "$log_file"
    
    # Run the comparison for this method
    if python run_timeseries_method_comparison.py --method "$method" --pathology "$pathology" --record_id "$record_id" 2>&1 | tee -a "$log_file"; then
        echo "✓ Success: $method" >> "$log_file"
    else
        echo "✗ Failed: $method" >> "$log_file"
        echo "Failed method: $method"
    fi
    
    echo "Completed $count/$total methods"
    echo ""
done

echo "Curated timeseries comparison completed at $(date)" >> "$log_file"
echo ""
echo "Curated timeseries method comparison completed!"
echo "Results saved in: results/"
echo "Log file: $log_file"
echo ""
echo "Generated comparison plots:"
ls -1 results/${pathology,,}_*_12leads_comparison_${record_id}.png 2>/dev/null | wc -l
echo ""
echo "Generated data files:"
ls -1 results/${pathology,,}_*_${record_id}_data.npz 2>/dev/null | wc -l
echo ""
echo "Usage examples:"
echo "  ./run_all_timeseries_methods_fixed.sh              # Run with default 03509_hr (AVB)"
echo "  ./run_all_timeseries_methods_fixed.sh 12131_hr     # Run with ISCH pathology"
echo "  ./run_all_timeseries_methods_fixed.sh 02906_hr     # Run with RBBB pathology"
echo "  ./run_all_timeseries_methods_fixed.sh 14493_hr     # Run with LBBB pathology"