#!/bin/bash

# Advanced Multi-Agent RAG System Setup Script
# This script sets up the project for development or production use

set -e

echo "🚀 Setting up Advanced Multi-Agent RAG System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your OpenAI API key"
else
    echo "✅ Environment file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/vector_db data/cache data/analytics logs

# Run quick functionality test
echo "🧪 Running quick functionality test..."
python quick_functionality_test.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run the system:"
echo "   - API Server: python start_server.py"
echo "   - Streamlit UI: streamlit run streamlit_app.py"
echo "   - Multi-Agent Demo: python run_multi_agent_demo.py"
echo "   - Docker: docker-compose up -d"
echo ""
echo "📖 Documentation: docs/"
echo "🔗 API Docs: http://localhost:8000/docs (after starting server)"
echo "🌐 UI: http://localhost:8501 (after starting Streamlit)"
echo ""
echo "Happy coding! 🚀"