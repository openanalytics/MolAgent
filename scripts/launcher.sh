#!/bin/bash
# MolAgent Master Script Launcher
# Interactive menu for all MolAgent operations

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_menu() {
    clear
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║        MolAgent Script Launcher      ║"
    echo "  ╠══════════════════════════════════════╣"
    echo "  ║                                      ║"
    echo "  ║  1. Install MolAgent                 ║"
    echo "  ║  2. Start Both Servers               ║"
    echo "  ║  3. Start Data Server Only           ║"
    echo "  ║  4. Start Model Server Only          ║"
    echo "  ║  5. View Installation Status         ║"
    echo "  ║  6. Open Scripts Directory           ║"
    echo "  ║  7. Help & Documentation             ║"
    echo "  ║  8. Exit                             ║"
    echo "  ║                                      ║"
    echo "  ╚══════════════════════════════════════╝"
    echo -e "${NC}"
    echo
}

install_molagent() {
    clear
    echo -e "${GREEN}Starting MolAgent Installation...${NC}"
    echo "================================"
    echo
    chmod +x installation/install_unix.sh
    ./installation/install_unix.sh
    echo
    echo -e "${YELLOW}Installation completed. Press any key to return to menu...${NC}"
    read -n 1
}

start_both_servers() {
    clear
    echo -e "${GREEN}Starting Both MolAgent Servers...${NC}"
    echo "================================="
    echo
    chmod +x server_startup/start_both_servers.sh
    ./server_startup/start_both_servers.sh
    echo
    echo -e "${YELLOW}Servers started. Press any key to return to menu...${NC}"
    read -n 1
}

start_data_server() {
    clear
    echo -e "${GREEN}Starting MolAgent Data Server...${NC}"
    echo "==============================="
    echo
    chmod +x server_startup/start_data_server.sh
    ./server_startup/start_data_server.sh
    echo
    echo -e "${YELLOW}Data server started. Press any key to return to menu...${NC}"
    read -n 1
}

start_model_server() {
    clear
    echo -e "${GREEN}Starting MolAgent Model Server...${NC}"
    echo "================================"
    echo
    chmod +x server_startup/start_model_server.sh
    ./server_startup/start_model_server.sh
    echo
    echo -e "${YELLOW}Model server started. Press any key to return to menu...${NC}"
    read -n 1
}

show_status() {
    clear
    echo -e "${BLUE}MolAgent Installation Status${NC}"
    echo "============================"
    echo

    # Check virtual environment
    if [ -d "molagent_env" ]; then
        echo -e "${GREEN}[✓] Virtual environment: Found${NC}"
    else
        echo -e "${RED}[✗] Virtual environment: Not found${NC}"
    fi

    # Check if servers are running
    if lsof -i :8000 >/dev/null 2>&1 || netstat -tulpn 2>/dev/null | grep -q :8000; then
        echo -e "${GREEN}[✓] Data Server (port 8000): Running${NC}"
    else
        echo -e "${RED}[✗] Data Server (port 8000): Not running${NC}"
    fi

    if lsof -i :8001 >/dev/null 2>&1 || netstat -tulpn 2>/dev/null | grep -q :8001; then
        echo -e "${GREEN}[✓] Model Server (port 8001): Running${NC}"
    else
        echo -e "${RED}[✗] Model Server (port 8001): Not running${NC}"
    fi

    # Check .env file
    if [ -f ".env" ]; then
        echo -e "${GREEN}[✓] Configuration file (.env): Found${NC}"
    else
        echo -e "${RED}[✗] Configuration file (.env): Not found${NC}"
    fi

    echo
    echo -e "${YELLOW}Press any key to return to menu...${NC}"
    read -n 1
}

open_scripts() {
    clear
    echo -e "${GREEN}Opening Scripts Directory...${NC}"
    echo "==========================="
    echo
    
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open .
    elif command -v open >/dev/null 2>&1; then
        open .
    else
        echo "Scripts directory: $(pwd)"
        ls -la
    fi
    
    echo "Scripts directory opened."
    echo -e "${YELLOW}Press any key to return to menu...${NC}"
    read -n 1
}

show_help() {
    clear
    echo -e "${BLUE}MolAgent Help & Documentation${NC}"
    echo "============================="
    echo
    echo "Available Documentation:"
    echo "- README.md (in scripts folder) - Complete script documentation"
    echo "- INSTALL.md (in root) - Installation guide"
    echo "- CLAUDE.md (in root) - Development commands"
    echo "- SERVER_STARTUP_GUIDE.md (in root) - Server management"
    echo
    echo "Quick Commands:"
    echo "- Installation: ./scripts/installation/install_unix.sh"
    echo "- Start Servers: ./scripts/server_startup/start_both_servers.sh"
    echo "- Data Server: ./scripts/server_startup/start_data_server.sh"
    echo "- Model Server: ./scripts/server_startup/start_model_server.sh"
    echo
    echo "For issues: Check logs and script output for error details"
    echo
    echo -e "${YELLOW}Press any key to return to menu...${NC}"
    read -n 1
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-8): " choice
    
    case $choice in
        1) install_molagent ;;
        2) start_both_servers ;;
        3) start_data_server ;;
        4) start_model_server ;;
        5) show_status ;;
        6) open_scripts ;;
        7) show_help ;;
        8) 
            echo
            echo -e "${GREEN}Thank you for using MolAgent!${NC}"
            echo
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please select 1-8.${NC}"
            sleep 2
            ;;
    esac
done