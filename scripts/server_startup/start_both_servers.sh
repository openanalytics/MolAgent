#!/bin/bash
# MolAgent - Start Both Servers Script for Unix/Linux
# This script starts both data and model servers in the background

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

log "Starting MolAgent Servers..."
echo "============================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if startup scripts exist
if [ ! -f "$SCRIPT_DIR/start_data_server.sh" ] || [ ! -f "$SCRIPT_DIR/start_model_server.sh" ]; then
    error "Server startup scripts not found!"
    error "Please ensure start_data_server.sh and start_model_server.sh exist in $SCRIPT_DIR."
    exit 1
fi

# Make scripts executable
chmod +x "$SCRIPT_DIR/start_data_server.sh" "$SCRIPT_DIR/start_model_server.sh"

# Function to check if port is in use
check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:$port >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ln | grep ":$port " >/dev/null 2>&1
    else
        # Fallback: try to connect
        timeout 1 bash -c "</dev/tcp/localhost/$port" >/dev/null 2>&1
    fi
}

# Check if ports are already in use
if check_port 8000; then
    warn "Port 8000 is already in use. Please stop any existing data server."
fi

if check_port 8001; then
    warn "Port 8001 is already in use. Please stop any existing model server."
fi

# Start data server in background
info "Starting data server..."
"$SCRIPT_DIR/start_data_server.sh" > data_server.log 2>&1 &
DATA_SERVER_PID=$!

# Wait a moment for the data server to start
sleep 3

# Start model server in background
info "Starting model server..."
"$SCRIPT_DIR/start_model_server.sh" > model_server.log 2>&1 &
MODEL_SERVER_PID=$!

# Function to handle cleanup on exit
cleanup() {
    log "Shutting down servers..."
    kill $DATA_SERVER_PID 2>/dev/null || true
    kill $MODEL_SERVER_PID 2>/dev/null || true
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill if still running
    kill -9 $DATA_SERVER_PID 2>/dev/null || true
    kill -9 $MODEL_SERVER_PID 2>/dev/null || true
    
    log "Servers stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

log "Both servers are starting..."
log "Data server PID: $DATA_SERVER_PID (log: data_server.log)"
log "Model server PID: $MODEL_SERVER_PID (log: model_server.log)"

# Wait a moment and check if servers are running
sleep 5

if ! kill -0 $DATA_SERVER_PID 2>/dev/null; then
    error "Data server failed to start. Check data_server.log for details."
    exit 1
fi

if ! kill -0 $MODEL_SERVER_PID 2>/dev/null; then
    error "Model server failed to start. Check model_server.log for details."
    exit 1
fi

log "SUCCESS: Both servers are running!"
echo ""
info "Data Server: http://localhost:8000"
info "Model Server: http://localhost:8001"
echo ""
info "Logs:"
info "  Data server: tail -f data_server.log"
info "  Model server: tail -f model_server.log"
echo ""
info "For Claude Desktop integration:"
info "  claude mcp add --transport sse automoldata https://localhost:8000/sse"
info "  claude mcp add --transport sse automolmodelling https://localhost:8001/sse"
echo ""
log "Press Ctrl+C to stop both servers"

# Wait for either server to exit
wait $DATA_SERVER_PID $MODEL_SERVER_PID