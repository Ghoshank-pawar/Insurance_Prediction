@echo off
REM ==============================================================================
REM InsurePredict AI - Universal One-Click Launcher (Windows)
REM Sets up virtual environment, installs requirements, and launches the app.
REM ==============================================================================

cd /d "%~dp0"

echo ==========================================================
echo  Starting InsurePredict AI Web Application
echo ==========================================================

REM 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Checking Python installation...
python --version

REM 2. Create virtual environment if needed
if not exist ".venv" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
)

REM 3. Install dependencies
echo Checking dependencies...
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.venv\Scripts\pip.exe install --quiet -r requirements.txt

REM 4. Launch app
echo.
echo Launching InsurePredict AI server...
echo Your web browser will open automatically in a moment.
echo Press Ctrl+C in this window to stop the server at any time.
echo ==========================================================
echo.

.venv\Scripts\python.exe wsgi.py

pause
