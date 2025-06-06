#!/bin/bash

# Comprehensive Coverage Analysis Script for CUDA Tests
# This script runs the full coverage analysis pipeline

set -e  # Exit on any error

echo "🚀 Starting Comprehensive Coverage Analysis"
echo "============================================"

# Check if we're in the right directory
if [ ! -d "logic/cuda" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

# Create coverage reports directory
mkdir -p coverage_reports

# Step 1: Generate basic coverage report
echo "📊 Step 1: Generating basic coverage report..."
python generate_coverage_report.py

# Step 2: Run advanced analysis
echo "🔬 Step 2: Running advanced coverage analysis..."
python advanced_coverage_analysis.py

# Step 3: Generate summary report
echo "📝 Step 3: Generating summary report..."
cat > coverage_reports/summary.md << EOF
# CUDA Module Coverage Report

Generated on: $(date)

## Quick Stats
- HTML Report: [coverage_html/index.html](../coverage_html/index.html)
- XML Report: [coverage.xml](../coverage.xml)

## How to Use This Report
1. Open the HTML report in your browser for interactive analysis
2. Review the advanced analysis output for specific recommendations
3. Focus on files with <80% coverage first
4. Add tests for uncovered branches and error conditions

## Next Steps
- [ ] Review uncovered lines in low-coverage files
- [ ] Add tests for error handling paths
- [ ] Improve branch coverage with conditional tests
- [ ] Set up automated coverage tracking

EOF

# Step 4: Open HTML report (if available)
if command -v xdg-open > /dev/null 2>&1; then
    echo "🌐 Opening HTML coverage report..."
    xdg-open coverage_html/index.html 2>/dev/null || true
elif command -v open > /dev/null 2>&1; then
    echo "🌐 Opening HTML coverage report..."
    open coverage_html/index.html 2>/dev/null || true
fi

echo "✅ Coverage analysis complete!"
echo "📊 Reports available in:"
echo "   - coverage_html/index.html (interactive)"
echo "   - coverage.xml (for CI/CD)"
echo "   - coverage_reports/summary.md (overview)" 