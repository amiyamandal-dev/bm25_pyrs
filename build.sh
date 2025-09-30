#!/bin/bash
# Simple build script for BM25-RS with Python 3.13 compatibility

set -e

echo "🚀 Building BM25-RS with Python 3.13 compatibility..."

# Set the PyO3 compatibility flag
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Build for development
echo "🔨 Building for development..."
maturin develop --release

echo "✅ Build completed successfully!"
echo "🧪 Testing import..."
python -c "import bm25_rs; print('✅ BM25-RS imported successfully!')"