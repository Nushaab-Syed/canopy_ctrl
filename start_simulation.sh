#!/bin/bash
# Startup script for Canopy Robot in simulation mode

echo "🤖 Starting Canopy Robot in SIMULATION MODE..."
echo "==================================================="
echo ""
echo "This mode provides:"
echo "  • Simulated motor movements (5x speed)"
echo "  • Synthetic camera feed with shingle patterns"
echo "  • Configurable dx offsets and noise"
echo "  • Failure injection for testing"
echo "  • Full FSM operation without hardware"
echo ""
echo "Web interface will be available at:"
echo "  http://localhost:5000"
echo ""
echo "To exit simulation mode, press Ctrl+C"
echo "==================================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python3 -m venv venv"
    echo "   ./venv/bin/pip install flask flask-cors opencv-python numpy pyserial"
    exit 1
fi

# Test dependencies
echo "🔍 Checking dependencies..."
./venv/bin/python -c "import flask, cv2, numpy, serial" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    ./venv/bin/pip install flask flask-cors opencv-python numpy pyserial
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies OK"
echo ""

# Set environment variable for simulation mode
export SIMULATION_MODE=true

# Start the application
echo "🚀 Starting simulation..."
./venv/bin/python app.py