#!/bin/bash

# SignXAI2 Build and Deploy Script
# This script rebuilds documentation and uploads the package to PyPI

echo "================================================"
echo "SignXAI2 Build and Deploy Pipeline"
echo "================================================"

# Check if we're in the project root
if [ ! -f "pyproject.toml" ] || [ ! -d "docs" ]; then
    echo "Error: This script must be run from the SignXAI2 project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Parse command line arguments
BUILD_DOCS=true
BUILD_PACKAGE=true
UPLOAD_PYPI=false
PYPI_REPO="pypi"  # or "testpypi" for testing

while [[ $# -gt 0 ]]; do
    case $1 in
        --docs-only)
            BUILD_PACKAGE=false
            shift
            ;;
        --package-only)
            BUILD_DOCS=false
            shift
            ;;
        --upload)
            UPLOAD_PYPI=true
            shift
            ;;
        --test-pypi)
            PYPI_REPO="testpypi"
            shift
            ;;
        --help)
            echo "Usage: ./update_sphinx_docs.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docs-only      Only build documentation"
            echo "  --package-only   Only build Python package"
            echo "  --upload         Upload package to PyPI after building"
            echo "  --test-pypi      Upload to TestPyPI instead of PyPI"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./update_sphinx_docs.sh                    # Build docs and package"
            echo "  ./update_sphinx_docs.sh --upload           # Build everything and upload to PyPI"
            echo "  ./update_sphinx_docs.sh --test-pypi --upload  # Test upload to TestPyPI"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Store the project root directory
PROJECT_ROOT=$(pwd)

# ========================================
# PART 1: BUILD DOCUMENTATION
# ========================================

if [ "$BUILD_DOCS" = true ]; then
    echo ""
    echo "================================================"
    echo "PART 1: Building Documentation"
    echo "================================================"
    
    # Navigate to docs directory
    cd docs || exit 1
    
    echo ""
    echo "Step 1.1: Cleaning previous documentation build..."
    echo "---------------------------------------------------"
    make clean
    rm -rf _build
    rm -rf _autosummary
    
    echo ""
    echo "Step 1.2: Updating API documentation..."
    echo "----------------------------------------"
    # Generate API documentation from docstrings
    sphinx-apidoc -f -o source/api ../signxai
    
    echo ""
    echo "Step 1.3: Building HTML documentation..."
    echo "-----------------------------------------"
    make html
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Documentation built successfully!"
        echo "   Location: docs/_build/html/index.html"
    else
        echo ""
        echo "❌ Documentation build failed!"
        echo "Please check the error messages above."
        exit 1
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
fi

# ========================================
# PART 2: BUILD PYTHON PACKAGE
# ========================================

if [ "$BUILD_PACKAGE" = true ]; then
    echo ""
    echo "================================================"
    echo "PART 2: Building Python Package"
    echo "================================================"
    
    echo ""
    echo "Step 2.1: Cleaning previous build artifacts..."
    echo "-----------------------------------------------"
    rm -rf build/ dist/ *.egg-info/ signxai2.egg-info/
    
    echo ""
    echo "Step 2.2: Checking required tools..."
    echo "-------------------------------------"
    
    # Check if build tool is installed
    if ! python -m pip show build > /dev/null 2>&1; then
        echo "Installing 'build' package..."
        python -m pip install --upgrade build
    fi
    
    # Check if twine is installed
    if ! python -m pip show twine > /dev/null 2>&1; then
        echo "Installing 'twine' package..."
        python -m pip install --upgrade twine
    fi
    
    echo ""
    echo "Step 2.3: Building distribution packages..."
    echo "-------------------------------------------"
    python -m build
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Package build failed!"
        exit 1
    fi
    
    echo ""
    echo "Step 2.4: Checking package integrity..."
    echo "----------------------------------------"
    python -m twine check dist/*
    
    if [ $? -eq 0 ]; then
        echo "✅ Package built successfully!"
        echo ""
        echo "Built packages:"
        ls -lh dist/
    else
        echo "❌ Package validation failed!"
        exit 1
    fi
fi

# ========================================
# PART 3: UPLOAD TO PYPI (OPTIONAL)
# ========================================

if [ "$UPLOAD_PYPI" = true ] && [ "$BUILD_PACKAGE" = true ]; then
    echo ""
    echo "================================================"
    echo "PART 3: Uploading to PyPI"
    echo "================================================"
    
    echo ""
    echo "Target repository: $PYPI_REPO"
    
    # Check for PyPI credentials
    if [ "$PYPI_REPO" = "testpypi" ]; then
        echo "Uploading to TestPyPI..."
        echo "Make sure you have configured your ~/.pypirc or will provide credentials"
        python -m twine upload --repository testpypi dist/*
    else
        echo ""
        echo "⚠️  WARNING: About to upload to production PyPI!"
        echo "This action cannot be undone. Version numbers cannot be reused."
        echo ""
        read -p "Are you sure you want to continue? (yes/no): " confirm
        
        if [ "$confirm" = "yes" ]; then
            echo "Uploading to PyPI..."
            python -m twine upload dist/*
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "🎉 Package successfully uploaded to PyPI!"
                echo "Install with: pip install signxai2[all]"
            else
                echo ""
                echo "❌ Upload failed!"
                exit 1
            fi
        else
            echo "Upload cancelled."
        fi
    fi
fi

# ========================================
# SUMMARY
# ========================================

echo ""
echo "================================================"
echo "Build Pipeline Complete!"
echo "================================================"
echo ""

if [ "$BUILD_DOCS" = true ]; then
    echo "📚 Documentation:"
    echo "   - HTML: docs/_build/html/index.html"
    echo "   - Serve locally: cd docs/_build/html && python -m http.server 8000"
fi

if [ "$BUILD_PACKAGE" = true ]; then
    echo ""
    echo "📦 Package:"
    echo "   - Distribution files in: dist/"
    if [ "$UPLOAD_PYPI" = false ]; then
        echo "   - To upload to TestPyPI: ./update_sphinx_docs.sh --test-pypi --upload"
        echo "   - To upload to PyPI: ./update_sphinx_docs.sh --upload"
    fi
fi

echo ""
echo "✅ All tasks completed successfully!"
echo ""