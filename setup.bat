@echo off
REM Advanced Multi-Agent RAG System Setup Script for Windows
REM This script sets up the project for development or production use

echo 🚀 Setting up Advanced Multi-Agent RAG System...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python installation found

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Create environment file
if not exist ".env" (
    echo ⚙️ Creating environment file...
    copy .env.example .env
    echo 📝 Please edit .env file and add your OpenAI API key
) else (
    echo ✅ Environment file already exists
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\vector_db" mkdir data\vector_db
if not exist "data\cache" mkdir data\cache
if not exist "data\analytics" mkdir data\analytics
if not exist "logs" mkdir logs

REM Run quick functionality test
echo 🧪 Running quick functionality test...
python quick_functionality_test.py

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Run the system:
echo    - API Server: python start_server.py
echo    - Streamlit UI: streamlit run streamlit_app.py
echo    - Multi-Agent Demo: python run_multi_agent_demo.py
echo    - Docker: docker-compose up -d
echo.
echo 📖 Documentation: docs\
echo 🔗 API Docs: http://localhost:8000/docs (after starting server)
echo 🌐 UI: http://localhost:8501 (after starting Streamlit)
echo.
echo Happy coding! 🚀
pause