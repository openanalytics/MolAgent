#!/bin/bash
# MolAgent Installation Launcher
# This script launches the main installation script from the organized scripts folder

echo "Starting MolAgent Installation..."
echo "Launching installation script from scripts/installation/"
echo

# Make the script executable
chmod +x ./scripts/installation/install.sh

# Run the main installation script
./scripts/installation/install.sh "$@"
