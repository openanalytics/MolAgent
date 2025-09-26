#!/bin/bash
# MolAgent Data Server Startup Script for Unix/Linux
# This script starts the AutoMol data server (port 8000)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log "Starting MolAgent Data Server..."
echo "================================"

# Check if virtual environment exists
if [ ! -f "molagent_env/bin/activate" ]; then
    error "Virtual environment 'molagent_env' not found!"
    error "Please run the installation script first:"
    error "  ./install.sh"
    exit 1
fi

# Check if MCP directory exists
if [ ! -d "MCP" ]; then
    error "MCP directory not found!"
    error "Please ensure you're running this script from the MolAgent root directory."
    exit 1
fi

# Activate virtual environment
info "Activating virtual environment..."
source molagent_env/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    warn ".env file not found!"
    warn "Please copy .env.template to .env and configure your API keys."
    warn "The server will start but some features may not work properly."
    sleep 3
fi

# Check if required Python modules are available
info "Checking dependencies..."
python -c "
try:
    import fastmcp
    import pandas as pd
    print('✓ Core dependencies found')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    print('Please run the installation script or install missing packages.')
    exit(1)
" || exit 1

# Check if port 8000 is already in use
info "Checking port availability..."
if lsof -i :8000 >/dev/null 2>&1 || netstat -tulpn 2>/dev/null | grep -q :8000; then
    warn "Port 8000 is already in use!"
    echo
    echo "Found processes using port 8000:"
    if command -v lsof >/dev/null 2>&1; then
        lsof -i :8000
    else
        netstat -tulpn 2>/dev/null | grep :8000
    fi
    echo
    echo "Do you want to:"
    echo "[1] Kill existing process and start server"
    echo "[2] Exit and let you handle it manually"
    echo
    read -p "Enter your choice (1 or 2): " choice
    
    if [ "$choice" = "1" ]; then
        info "Attempting to kill existing process on port 8000..."
        if command -v lsof >/dev/null 2>&1; then
            # Use lsof on systems that have it
            lsof -ti :8000 | xargs -r kill -9
        else
            # Fallback for systems without lsof
            netstat -tulpn 2>/dev/null | grep :8000 | awk '{print $7}' | cut -d'/' -f1 | xargs -r kill -9
        fi
        sleep 2
        
        # Check again if port is free
        if lsof -i :8000 >/dev/null 2>&1 || netstat -tulpn 2>/dev/null | grep -q :8000; then
            error "Could not free port 8000. Please kill the process manually."
            exit 1
        else
            log "Port 8000 is now available."
        fi
    else
        info "Exiting. Please stop the existing process and try again."
        exit 1
    fi
fi

# Change to MCP directory and start server
info "Starting data server on port 8000..."
cd MCP

# Set Python path to include parent directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Start the server with error handling
python mcp_server/automol_data_server.py &
SERVER_PID=$!

# Function to handle cleanup on exit
cleanup() {
    log "Shutting down data server..."
    kill $SERVER_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

log "Data server started with PID: $SERVER_PID"
log "Server running at: http://localhost:8000"
log "Press Ctrl+C to stop the server"

# Wait for the server process
wait $SERVER_PID