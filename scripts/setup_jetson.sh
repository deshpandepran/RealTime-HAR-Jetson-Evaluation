#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/setup_jetson.sh
# Purpose: Configure NVIDIA Jetson Xavier NX embedded platform (JetPack 5.1.1)
#          for RT-HARE and IMFE benchmarking.
# Reference: Section III-B & Section V-A in "Real-Time Human Action Recognition
#            on Embedded Platforms" (Wang et al., 2024).
#
# Hardware: Seeed Studio reComputer J2021 (NVIDIA Jetson Xavier NX, 8GB RAM)
# OS / SDK: JetPack 5.1.1 (L4T R35.3.1, Ubuntu 20.04 LTS, Kernel 5.10)
#
# Specific Paper Configuration:
#   - Power Mode: Mode 6 (20W, 2-Core) -> nvpmodel -m 6
#   - Clocks: DVFS governor disabled, CPU clocks maximized to 1.9 GHz -> jetson_clocks
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script requires superuser privileges to adjust nvpmodel and jetson_clocks."
        echo "Please re-run with: sudo bash $0 $@"
        exit 1
    fi
}

check_jetson_platform() {
    log_info "Verifying hardware platform..."
    if [[ ! -f /etc/nv_tegra_release ]]; then
        log_warning "File /etc/nv_tegra_release not found. This host may not be an NVIDIA Jetson device."
    else
        L4T_RELEASE=$(head -n 1 /etc/nv_tegra_release)
        log_success "Detected Jetson L4T Release: ${L4T_RELEASE}"
    fi

    if ! command -v nvpmodel &> /dev/null; then
        log_error "'nvpmodel' utility not found. Please verify JetPack installation."
        exit 1
    fi

    if ! command -v jetson_clocks &> /dev/null; then
        log_error "'jetson_clocks' utility not found. Please verify JetPack installation."
        exit 1
    fi
}

set_power_mode() {
    log_info "Configuring Power Mode 6 (20W, 2-Core)..."
    log_info "Paper Rationale: Mode 6 provides the shortest HAR inference latency on Jetson Xavier NX."

    nvpmodel -m 6
    log_success "nvpmodel set to Mode 6 successfully."

    if [[ -f /etc/nvpmodel.conf ]]; then
        log_info "Persisting power mode 6 in /etc/nvpmodel.conf..."
        nvpmodel -f /etc/nvpmodel.conf -m 6 || true
    fi

    echo "----------------------------------------"
    log_info "Current nvpmodel status:"
    nvpmodel -q
    echo "----------------------------------------"
}

set_max_clocks() {
    log_info "Disabling DVFS governor and locking clocks to maximum (CPU: 1.9 GHz)..."
    jetson_clocks
    log_success "jetson_clocks applied: CPU, GPU, and EMC frequencies locked to max."

    log_info "Maximizing cooling fan speed..."
    jetson_clocks --fan || true

    for cpu_gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -f "$cpu_gov" ]]; then
            echo "performance" > "$cpu_gov" 2>/dev/null || true
        fi
    done

    echo "----------------------------------------"
    log_info "Active Clocks Summary:"
    jetson_clocks --show
    echo "----------------------------------------"
}

show_status() {
    log_info "--- Jetson Hardware Status ---"
    if command -v nvpmodel &> /dev/null; then
        nvpmodel -q
    fi
    if command -v jetson_clocks &> /dev/null; then
        echo ""
        jetson_clocks --show
    fi
}

setup_python_env() {
    local VENV_DIR="${1:-$HOME/rt-hare-venv}"
    log_info "Setting up Python environment for JetPack 5.1.1 at ${VENV_DIR}..."

    apt-get update -y
    apt-get install -y python3-pip python3-dev python3-venv libopenblas-base libopenmpi-dev

    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv --system-site-packages "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install jetson-stats pycuda

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
        pip install -r "$REPO_ROOT/requirements.txt" || true
    fi

    log_success "Jetson Python environment initialized at ${VENV_DIR}"
    echo "To activate: source ${VENV_DIR}/bin/activate"
}

print_usage() {
    cat << EOF
Usage: sudo bash scripts/setup_jetson.sh [OPTION]

Configures the NVIDIA Jetson Xavier NX testbed for RT-HARE benchmarking:
  - Sets Power Mode 6 (20W, 2-Core)
  - Disables DVFS governor and maximizes CPU clocks to 1.9 GHz
  - Maximizes cooling fan speed

Options:
  --clocks-only       Only apply Power Mode 6 and maximize clocks (recommended before each benchmark).
  --all               Apply Power Mode 6, max clocks, and configure Python virtualenv.
  --env [PATH]        Configure Python virtualenv (default: \$HOME/rt-hare-venv).
  --status            Print current nvpmodel, clock speeds, and hardware status.
  -h, --help          Show this help message.
EOF
}

main() {
    local action="${1:---all}"

    case "$action" in
        --clocks-only)
            check_root
            check_jetson_platform
            set_power_mode
            set_max_clocks
            log_success "Jetson clocks and power mode configured successfully for benchmarking!"
            ;;
        --status)
            show_status
            ;;
        --env)
            check_root
            check_jetson_platform
            local venv_path="${2:-$HOME/rt-hare-venv}"
            setup_python_env "$venv_path"
            ;;
        --all)
            check_root
            check_jetson_platform
            set_power_mode
            set_max_clocks
            local venv_path="${2:-$HOME/rt-hare-venv}"
            setup_python_env "$venv_path"
            show_status
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $action"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
