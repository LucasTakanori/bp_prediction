#!/bin/bash
# Quick comparison script to see the difference between masked vs whole data

echo "🔍 Comparing Masked vs Whole Data Samples..."
echo "This shows exactly what data is filtered out by masking"
echo "=" * 60

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Set environment variables
export BP_DATA_ROOT="/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data"

# Run comparison for the same subjects that were used in training
python scripts/compare_mask_vs_whole_data.py \
    --subjects subject001 subject002 \
    --mask-type auto \
    --output-dir comparison_results

echo ""
echo "📊 Comparison complete! Check the results in comparison_results/ directory"
echo "📄 Files created:"
echo "  - detailed_comparison.csv (full data)"
echo "  - comparison_summary.txt (human-readable summary)"
echo "  - mask_vs_whole_data_comparison.png (visualization)" 