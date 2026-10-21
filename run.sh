#!/usr/bin/env bash
# ==============================================================================
# InsurePredict AI - Universal One-Click Launcher (macOS & Linux)
# Sets up virtual environment, installs requirements, and launches the app.
# ==============================================================================

set -e

# Change directory to the folder containing this script
cd "$(dirname "$0")"

echo "=========================================================="
echo " Starting InsurePredict AI Web Application"
echo "=========================================================="

# 1. Determine Python executable (prefer python3.12, python3.11, python3.10, python3, python)
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo " [ERROR] Python was not found on your system."
    echo "Please install Python 3.9+ from https://www.python.org/downloads/ and try again."
    exit 1
fi

echo " Using Python: $($PYTHON_CMD --version) ($PYTHON_CMD)"

# 2. Create virtual environment if it doesn't exist
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo " Creating fresh virtual environment in ./$VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# 3. Locate venv python and pip
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# 4. Install dependencies if needed
echo " Checking and installing project dependencies..."
"$VENV_PIP" install --quiet --upgrade pip
"$VENV_PIP" install --quiet -r requirements.txt

# 5. Launch the application
echo " Launching InsurePredict AI server..."
echo "Your default web browser will open automatically in a moment."
echo "Press Ctrl+C to stop the server at any time."
echo "=========================================================="

"$VENV_PYTHON" wsgi.py
