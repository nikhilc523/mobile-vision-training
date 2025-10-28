#!/bin/bash
# Complete Dataset Workflow Example
# This script demonstrates the full dataset preparation and validation workflow

set -e  # Exit on error

echo "========================================================================"
echo "📦 COMPLETE DATASET WORKFLOW"
echo "========================================================================"
echo ""

# Step 1: Prepare datasets
echo "Step 1: Preparing datasets (unzip, organize)..."
echo "------------------------------------------------------------------------"
python3 scripts/prepare_datasets.py
echo ""

# Step 2: Validate datasets (dry-run first)
echo "Step 2: Validating datasets (dry-run)..."
echo "------------------------------------------------------------------------"
python3 scripts/validate_and_cleanup_datasets.py --dry-run
echo ""

# Step 3: Review the report
echo "Step 3: Review validation report..."
echo "------------------------------------------------------------------------"
echo "Report location: docs/dataset_cleanup_report.md"
echo ""
echo "Summary:"
grep -A 5 "## Summary" docs/dataset_cleanup_report.md | head -10
echo ""

# Step 4: Ask user if they want to proceed with cleanup
echo "Step 4: Cleanup confirmation..."
echo "------------------------------------------------------------------------"
read -p "Do you want to proceed with cleanup? [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing cleanup..."
    python3 scripts/validate_and_cleanup_datasets.py --force
    echo ""
    echo "✅ Cleanup complete!"
else
    echo "❌ Cleanup cancelled"
    exit 0
fi

# Step 5: Parse Le2i annotations
echo ""
echo "Step 5: Parsing Le2i annotations..."
echo "------------------------------------------------------------------------"
python3 -m ml.data.parsers.le2i_annotations data/raw/le2i/Home_01
echo ""

# Step 6: Run example scripts
echo "Step 6: Running example scripts..."
echo "------------------------------------------------------------------------"
python3 examples/parse_le2i_example.py
echo ""

# Final summary
echo "========================================================================"
echo "✅ WORKFLOW COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review reports in docs/"
echo "  2. Extract video frames"
echo "  3. Run pose estimation"
echo "  4. Train LSTM model"
echo ""
echo "Documentation:"
echo "  - Dataset preparation: docs/dataset_notes.md"
echo "  - Validation report: docs/dataset_cleanup_report.md"
echo "  - Le2i parser: ml/data/parsers/README.md"
echo ""

