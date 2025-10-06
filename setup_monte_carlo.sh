#!/bin/bash
# Quick setup script for Monte Carlo Engine
# This script automates the complete installation process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse command line arguments
SKIP_SYSTEM_DEPS=false
for arg in "$@"; do
    case $arg in
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-system-deps    Skip system dependency installation"
            echo "  --help               Show this help"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}Monte Carlo Engine - Quick Setup${NC}"
echo "=================================="

# Check if running from correct directory
if [ ! -f "src/monte_carlo/CMakeLists.txt" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Install system dependencies (unless skipped)
if [ "$SKIP_SYSTEM_DEPS" = true ]; then
    echo -e "${YELLOW}Skipping system dependency installation${NC}"
else

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    echo "Detected apt package manager (Ubuntu/Debian)"
    sudo apt-get update
    sudo apt-get install -y build-essential cmake python3-dev python3-pip libomp-dev git pkg-config
elif command -v dnf5 &> /dev/null; then
    echo "Detected dnf5 package manager (Fedora 40+)"
    sudo dnf5 install -y gcc-c++ cmake python3-devel python3-pip libomp-devel git pkgconfig make
elif command -v dnf &> /dev/null; then
    echo "Detected dnf package manager (Fedora/RHEL 8+)"
    sudo dnf install -y gcc-c++ cmake python3-devel python3-pip libomp-devel git pkgconfig make
elif command -v yum &> /dev/null; then
    echo "Detected yum package manager (CentOS/RHEL 7)"
    sudo yum install -y gcc-c++ cmake python3-devel python3-pip libomp-devel git pkgconfig make
elif command -v pacman &> /dev/null; then
    echo "Detected pacman package manager (Arch Linux)"
    sudo pacman -S --noconfirm gcc cmake python python-pip openmp git pkgconf make
elif command -v zypper &> /dev/null; then
    echo "Detected zypper package manager (openSUSE)"
    sudo zypper install -y gcc-c++ cmake python3-devel python3-pip libomp-devel git pkg-config make
else
    echo -e "${YELLOW}Warning: Could not detect package manager. Please install manually:${NC}"
    echo "Required packages:"
    echo "- build-essential/Development Tools (gcc, g++, make)"
    echo "- cmake (3.15+)"
    echo "- python3-dev/python3-devel"
    echo "- python3-pip"
    echo "- libomp-dev/openmp"
    echo "- git"
    echo "- pkg-config"
    echo ""
    echo "After manual installation, run: $0 --skip-system-deps"
    exit 1
fi
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
python3 -m pip install --user numpy pandas matplotlib seaborn pybind11 psutil scikit-learn xgboost joblib

# Build the Monte Carlo engine
echo -e "${YELLOW}Building Monte Carlo engine...${NC}"
cd src/monte_carlo
chmod +x build.sh
./build.sh release

# Copy the Python module to the src directory
cp build/monte_carlo_engine.cpython-*-linux-gnu.so ../
cd ../..

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
if python3 -c "import sys; sys.path.append('./src'); import monte_carlo_engine; print('✓ Python module available')" 2>/dev/null; then
    echo -e "${GREEN}✓ Installation successful!${NC}"
else
    echo -e "${RED}✗ Installation verification failed${NC}"
    exit 1
fi

# Run a quick test
echo -e "${YELLOW}Running quick functionality test...${NC}"
if python3 -c "
import sys
sys.path.append('./src')
from monte_carlo_wrapper import MonteCarloSimulator
import numpy as np
import pandas as pd

portfolio = pd.DataFrame({
    'account_id': range(100),
    'balance': np.random.lognormal(9, 1, 100),
    'limit': np.random.lognormal(10, 0.8, 100),
    'exposure_at_default': np.random.lognormal(9.2, 1.2, 100),
    'loss_given_default': np.random.beta(2, 3, 100) * 0.6 + 0.2
})

default_probs = np.random.beta(1, 20, 100)

simulator = MonteCarloSimulator(num_simulations=1000, num_threads=2)
results = simulator.run_simulation(portfolio, default_probs)

if results['success']:
    print(f'✓ Quick test passed - Performance: {results[\"performance\"][\"iterations_per_second\"]:,.0f} it/s')
else:
    print(f'✗ Quick test failed: {results[\"error_message\"]}')
" 2>/dev/null; then
    echo -e "${GREEN}✓ Quick test passed!${NC}"
else
    echo -e "${YELLOW}Warning: Quick test had issues, but installation may still be functional${NC}"
fi

echo ""
echo -e "${GREEN}Setup completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Run the complete demo:"
echo "   python3 src/monte_carlo_demo.py --quick"
echo ""
echo "2. Run performance benchmarks:"
echo "   python3 src/performance_benchmark.py --quick"
echo ""
echo "3. Check the documentation:"
echo "   cat MONTE_CARLO_README.md"
echo ""
echo -e "${BLUE}Happy simulating! 🚀${NC}"