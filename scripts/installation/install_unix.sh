#!/bin/bash
# MolAgent Installation Script for Unix-like systems (Linux/macOS)
# This script installs MolAgent using uv package manager

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main installation function
main() {
    log "Starting MolAgent installation..."
    
    # Check Python version
    if ! command_exists python3; then
        error "Python 3 is required but not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Found Python version: $PYTHON_VERSION"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    # Install uv if not present
    if ! command_exists uv; then
        log "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if ! command_exists uv; then
            error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi
    else
        log "Found uv package manager"
    fi
    
    # Create virtual environment
    ENV_NAME="${1:-.venv}"
    PYTHON_VER="${2:-3.12}"
    
    log "Creating virtual environment '$ENV_NAME' with Python $PYTHON_VER..."
    uv venv "$ENV_NAME" --python "$PYTHON_VER"
    
    # Activate environment  
    log "Activating virtual environment..."
    source "$ENV_NAME/bin/activate"
    
    # Install system dependencies
    log "Installing system dependencies..."
    if command_exists apt-get; then
        log "Detected Debian/Ubuntu system"
        if ! command_exists wkhtmltopdf; then
            warn "wkhtmltopdf not found. Installing..."
            sudo apt-get update
            sudo apt-get install -y wkhtmltopdf
        fi
    elif command_exists brew; then
        log "Detected macOS with Homebrew"
        if ! command_exists wkhtmltopdf; then
            warn "wkhtmltopdf not found. Installing..."
            brew install wkhtmltopdf
        fi
    elif command_exists yum; then
        log "Detected RHEL/CentOS system"
        if ! command_exists wkhtmltopdf; then
            warn "wkhtmltopdf not found. Installing..."
            sudo yum install -y wkhtmltopdf
        fi
    else
        warn "Could not detect package manager. Please install wkhtmltopdf manually."
        warn "See: https://wkhtmltopdf.org/downloads.html"
    fi
    
    # Install AutoMol submodule packages
    log "Installing AutoMol packages..."
    if [ -d "AutoMol/automol_resources" ]; then
        uv pip install AutoMol/automol_resources/
    else
        warn "AutoMol/automol_resources directory not found. Make sure to clone with submodules."
    fi
    
    if [ -d "AutoMol/automol" ]; then
        uv pip install AutoMol/automol/
    else
        warn "AutoMol/automol directory not found. Make sure to clone with submodules."
    fi
    
    # Install essential dependencies first
    log "Installing essential dependencies..."
    uv pip install "psutil>=5.9.0" "smolagents>=1.19.0"
    
    # Install MolAgent
    log "Installing MolAgent and dependencies..."
    uv pip install -r requirements.txt
    uv pip install pytdc
    uv pip install rdkit==2024.3.5
    
    # Verify installation
    log "Verifying installation..."
    python3 -c "
import pandas as pd
import numpy as np
try:
    import rdkit
    import torch
    import fastmcp
    print('✓ Core dependencies installed successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"
    
    # Success message
    log "Installation completed successfully!"
    info ""
    info "You can activate environment manually:"
    info "   source $ENV_NAME/bin/activate"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [environment_name] [python_version]"
            echo ""
            echo "Options:"
            echo "  environment_name    Name for virtual environment (default: .venv)"
            echo "  python_version      Python version to use (default: 3.12)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Use defaults"
            echo "  $0 my_env 3.11             # Custom environment and Python version"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Run main installation
main "$@"
