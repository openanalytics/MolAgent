#!/bin/bash
# MolAgent Installation Script - Platform Detection
# This script detects the platform and runs the appropriate installer

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

# Platform detection and installer selection
main() {
    log "Detecting platform..."
    chmod +x ./scripts/installation/install_unix.sh
    # Detect platform
    case "$(uname -s)" in
        Darwin*)
            log "Detected macOS - using Unix installer"
            exec ./scripts/installation/install_unix.sh "$@"
            ;;
        Linux*)
            log "Detected Linux - using Unix installer"
            exec ./scripts/installation/install_unix.sh "$@"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            log "Detected Windows (Git Bash/MSYS2) - using Unix installer"
            exec ./scripts/installation/install_unix.sh "$@"
            ;;
        *)
            warn "Unknown platform: $(uname -s)"
            warn "Attempting to use Unix installer..."
            exec ./scripts/installation/install_unix.sh "$@"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [environment_name] [python_version]"
            echo ""
            echo "Options:"
            echo "  environment_name    Name for virtual environment (default: molagent_env)"
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
