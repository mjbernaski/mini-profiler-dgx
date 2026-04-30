#!/bin/bash

# Machine comparison script
LOCAL_HOST=$(hostname)
REMOTE_HOST="192.168.5.40"
OUTPUT_DIR="/home/mjbernaski/machine_compare"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/machine_comparison_$(date +%Y%m%d_%H%M%S).compare"

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

    echo -e "\n=== NVIDIA PACKAGES ==="
    echo "Count: $(dpkg -l | grep -i nvidia | wc -l)"
    dpkg -l | grep -i nvidia | awk '{print $2}' | sort

    echo -e "\n=== CUDA VERSION ==="
    nvcc --version 2>/dev/null | grep "release" || echo "nvcc not available"

    echo -e "\n=== APT UPDATE HISTORY ==="
    echo "Recent apt history files:"
    ls -lt /var/log/apt/history.log* 2>/dev/null | head -3
    echo -e "\nLast 20 package operations:"
    (cat /var/log/apt/history.log 2>/dev/null | tail -40) || echo "No apt history available"

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

# Generate package lists for comparison
echo -e "\n>>> Generating package lists for comparison..."
LOCAL_PKGS="/tmp/packages_local_$(hostname).txt"
REMOTE_PKGS="/tmp/packages_remote.txt"

dpkg --get-selections | awk '{print $1}' | sort > "$LOCAL_PKGS"
ssh $REMOTE_HOST "dpkg --get-selections" | awk '{print $1}' | sort > "$REMOTE_PKGS"

# Generate summary
{
    echo -e "\n\n######################################################"
    echo "# SUMMARY OF DIFFERENCES"
    echo "######################################################"

    echo -e "\n=== PACKAGE COUNT COMPARISON ==="
    LOCAL_COUNT=$(wc -l < "$LOCAL_PKGS")
    REMOTE_COUNT=$(wc -l < "$REMOTE_PKGS")
    echo "Local ($LOCAL_HOST): $LOCAL_COUNT packages"
    echo "Remote ($REMOTE_HOST): $REMOTE_COUNT packages"
    echo "Difference: $((LOCAL_COUNT - REMOTE_COUNT)) (local - remote)"

    echo -e "\n=== PACKAGES ONLY ON LOCAL ($LOCAL_HOST) ==="
    comm -23 "$LOCAL_PKGS" "$REMOTE_PKGS" | head -50
    LOCAL_ONLY=$(comm -23 "$LOCAL_PKGS" "$REMOTE_PKGS" | wc -l)
    echo "... Total packages only on local: $LOCAL_ONLY"

    echo -e "\n=== PACKAGES ONLY ON REMOTE ($REMOTE_HOST) ==="
    comm -13 "$LOCAL_PKGS" "$REMOTE_PKGS" | head -50
    REMOTE_ONLY=$(comm -13 "$LOCAL_PKGS" "$REMOTE_PKGS" | wc -l)
    echo "... Total packages only on remote: $REMOTE_ONLY"

    echo -e "\n=== NVIDIA PACKAGE DIFFERENCES ==="
    echo "NVIDIA packages only on local:"
    comm -23 "$LOCAL_PKGS" "$REMOTE_PKGS" | grep -i nvidia || echo "  (none)"
    echo "NVIDIA packages only on remote:"
    comm -13 "$LOCAL_PKGS" "$REMOTE_PKGS" | grep -i nvidia || echo "  (none)"

    echo -e "\n=== QUICK SUMMARY ==="
    echo "- Total packages: Local=$LOCAL_COUNT, Remote=$REMOTE_COUNT (diff: $((LOCAL_COUNT - REMOTE_COUNT)))"
    echo "- Packages unique to local: $LOCAL_ONLY"
    echo "- Packages unique to remote: $REMOTE_ONLY"
    echo "- Common packages: $((LOCAL_COUNT - LOCAL_ONLY))"

} | tee -a "$OUTPUT_FILE"

# Cleanup temp files
rm -f "$LOCAL_PKGS" "$REMOTE_PKGS"

echo -e "\n>>> Comparison saved to: $OUTPUT_FILE"

# Generate AI summary using ollama
SUMMARY_FILE="$OUTPUT_DIR/machine_comparison_summary_$(date +%Y%m%d_%H%M%S).txt"
echo -e "\n>>> Generating AI summary with ollama (gpt-oss:20b)..."

PROMPT="Analyze this machine comparison report between two systems. Provide a clear summary that highlights:
1. Overall health assessment of each machine (disk usage, memory, GPU status, services)
2. Key hardware differences (CPU, memory, GPU, disk)
3. Key software differences (OS, kernel, NVIDIA drivers, CUDA, Docker, package counts)
4. Any concerns or recommendations (low disk space, missing drivers, version mismatches, etc.)
Be concise and focus on what matters most.

Here is the comparison data:

$(cat "$OUTPUT_FILE")"

ollama run gpt-oss:20b "$PROMPT" 2>/dev/null | tee "$SUMMARY_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n>>> AI summary saved to: $SUMMARY_FILE"
else
    echo -e "\n>>> Warning: ollama summary generation failed. Is ollama running with gpt-oss:20b available?"
fi
