#!/usr/bin/env python3
"""Mini Profiler CLI - compact terminal status bars for RAM and GPU utilization."""

import subprocess
import sys
import time

import psutil

# ANSI colors
CYAN = "\033[36m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
CLEAR_LINE = "\033[2K"
UP = "\033[A"


def get_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        util = int(parts[0]) if parts[0] not in ("[N/A]", "N/A", "") else 0
        temp = int(parts[1]) if parts[1] not in ("[N/A]", "N/A", "") else 0
        power = float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else 0
        return util, temp, power
    except Exception:
        return 0, 0, 0.0


def bar(label, pct, width, color, detail=""):
    filled = int(pct / 100 * width)
    empty = width - filled
    pct_str = f"{pct:5.1f}%"
    return f"{BOLD}{label}{RESET} {color}{'█' * filled}{'░' * empty}{RESET} {pct_str} {DIM}{detail}{RESET}"


def main():
    width = 30
    print("\033[?25l", end="")  # hide cursor
    try:
        # Print initial blank lines to reserve space
        print()
        print()
        while True:
            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            ram_detail = f"{mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f}G"

            gpu_util, gpu_temp, gpu_power = get_gpu_util()
            gpu_detail = f"{gpu_temp}°C {gpu_power:.0f}W"

            # Move up 2 lines and overwrite
            sys.stdout.write(f"{UP}{UP}")
            sys.stdout.write(f"{CLEAR_LINE}{bar('RAM', ram_pct, width, CYAN, ram_detail)}\n")
            sys.stdout.write(f"{CLEAR_LINE}{bar('GPU', gpu_util, width, MAGENTA, gpu_detail)}\n")
            sys.stdout.flush()

            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="")  # show cursor


if __name__ == "__main__":
    main()
