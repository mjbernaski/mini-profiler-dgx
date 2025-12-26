#!/bin/bash

# Machine comparison script
LOCAL_HOST=$(hostname)
REMOTE_HOST="192.168.5.40"
OUTPUT_FILE="/home/mjbernaski/machine_comparison_$(date +%Y%m%d_%H%M%S).compare"

collect_info() {
    echo "=== HOSTNAME ==="
    hostname

    echo -e "\n=== OS/KERNEL ==="
    uname -a
    cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)="

    echo -e "\n=== CPU ==="
    lscpu | grep -E "^(Architecture|CPU\(s\)|Model name|CPU MHz)"

    echo -e "\n=== MEMORY ==="
    free -h | head -2

    echo -e "\n=== DISK ==="
    df -h | grep -E "^/dev"

    echo -e "\n=== NETWORK INTERFACES ==="
    ip -4 addr show | grep -E "^[0-9]+:|inet " | sed 's/^/  /'

    echo -e "\n=== GPU ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU or nvidia-smi not available"

    echo -e "\n=== INSTALLED PACKAGES (count) ==="
    dpkg -l 2>/dev/null | wc -l || rpm -qa 2>/dev/null | wc -l

    echo -e "\n=== RUNNING SERVICES ==="
    systemctl list-units --type=service --state=running 2>/dev/null | grep ".service" | wc -l

    echo -e "\n=== DOCKER ==="
    docker --version 2>/dev/null || echo "Docker not installed"
    docker ps 2>/dev/null | tail -n +2 | wc -l && echo "containers running" || true
}

{
    echo "######################################################"
    echo "# LOCAL MACHINE: $LOCAL_HOST"
    echo "######################################################"
    collect_info

    echo -e "\n\n######################################################"
    echo "# REMOTE MACHINE: $REMOTE_HOST"
    echo "######################################################"
    ssh $REMOTE_HOST "$(declare -f collect_info); collect_info"
} | tee "$OUTPUT_FILE"

echo -e "\n>>> Comparison saved to: $OUTPUT_FILE"
