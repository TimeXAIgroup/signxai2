#!/bin/bash

# SignXAI2 Sphinx Documentation Update Script
# This script rebuilds the Sphinx documentation with the updated content

echo "================================================"
echo "SignXAI2 Sphinx Documentation Update"
echo "================================================"

# Check if we're in the project root
if [ ! -f "pyproject.toml" ] || [ ! -d "docs" ]; then
    echo "Error: This script must be run from the SignXAI2 project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Navigate to docs directory
cd docs || exit 1

echo ""
echo "Step 1: Cleaning previous build..."
echo "--------------------------------"
make clean
rm -rf _build
rm -rf _autosummary

echo ""
echo "Step 2: Updating API documentation..."
echo "------------------------------------"
# Generate API documentation from docstrings
sphinx-apidoc -f -o source/api ../signxai

echo ""
echo "Step 3: Building HTML documentation..."
echo "-------------------------------------"
make html

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Documentation built successfully!"
    echo ""
    echo "View the documentation:"
    echo "  - HTML: docs/_build/html/index.html"
    echo ""
    echo "To serve locally:"
    echo "  cd docs/_build/html && python -m http.server 8000"
    echo "  Then open: http://localhost:8000"
else
    echo ""
    echo "❌ Documentation build failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "Step 4: Building PDF documentation (optional)..."
echo "-----------------------------------------------"
# Check if LaTeX is installed
if command -v pdflatex &> /dev/null; then
    make latexpdf
    if [ $? -eq 0 ]; then
        echo "✅ PDF documentation built: docs/_build/latex/signxai2.pdf"
    else
        echo "⚠️  PDF build failed (non-critical)"
    fi
else
    echo "⚠️  LaTeX not installed, skipping PDF generation"
fi

echo ""
echo "Step 5: Checking for broken links..."
echo "-----------------------------------"
make linkcheck

echo ""
echo "================================================"
echo "Documentation update complete!"
echo "================================================"
echo ""
echo "Summary of updated content:"
echo "  - API Reference with dynamic method parsing"
echo "  - Updated tutorials (PyTorch & TensorFlow)"
echo "  - New quickstart examples"
echo "  - Updated README and CHANGELOG"
echo ""
echo "Deployment options:"
echo "  1. GitHub Pages: Push to gh-pages branch"
echo "  2. ReadTheDocs: Will auto-build on push"
echo "  3. Local: Serve from docs/_build/html/"
echo ""