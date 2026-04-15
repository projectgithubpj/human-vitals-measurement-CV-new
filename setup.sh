#!/usr/bin/env bash
# ============================================================
# setup.sh — Install dependencies and start the server
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      CardiacVision — rPPG Heart Rate Monitor     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Please install Python 3.10+"
  exit 1
fi

PYTHON=$(command -v python3)
echo "✓  Using Python: $($PYTHON --version)"

# Create venv if needed
if [ ! -d ".venv" ]; then
  echo "→  Creating virtual environment…"
  $PYTHON -m venv .venv
fi

# Activate
source .venv/bin/activate

echo "→  Installing dependencies (this may take a minute)…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅  Setup complete!"
echo ""
echo "→  Starting server on http://localhost:5000"
echo "   Press Ctrl+C to stop."
echo ""

python app.py
