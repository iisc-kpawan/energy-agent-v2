#!/bin/bash
set -e

echo "============================================================"
echo " EnergyPlus MCP Agent Setup (Unix/macOS/Linux)"
echo "============================================================"

# 1. Check Python installation
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment (venv)..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment (venv) already exists."
fi

# 3. Install dependencies
echo "Installing dependencies from requirements.txt..."
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed successfully."

# 4. Create .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env configuration from template..."
    cp .env.example .env
    echo ""
    echo "[IMPORTANT] A new '.env' file has been created."
    echo "Please open '.env' and set your GOOGLE_API_KEY."
    echo ""
else
    echo "'.env' configuration file already exists."
fi

echo "============================================================"
echo " Setup Completed Successfully!"
echo "============================================================"
echo "To start the agent server, run:"
echo "   source venv/bin/activate && python agent.py"
echo "============================================================"
