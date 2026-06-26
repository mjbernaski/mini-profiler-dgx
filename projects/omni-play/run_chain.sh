#!/usr/bin/env bash
# Wait for the captioning run to finish, then narrate + rebuild everything.
# Launched detached so the full pipeline completes hands-free.
set -u
cd /home/mjbernaski/projects/omni-play

CAP_PID="${1:?need caption pid}"
SYS=/usr/bin/python3
XTTS=/home/mjbernaski/projects/debate-out-loud/.venv/bin/python3
SRV=/home/mjbernaski/projects/omni-play/.venv-serve/bin/python3

echo "[chain] waiting for captioning (pid $CAP_PID) to finish..."
while kill -0 "$CAP_PID" 2>/dev/null; do sleep 20; done
echo "[chain] captioning process exited at $(date)"

# Stop the read-xtts service if active so two XTTS models don't fight the GPU.
systemctl --user stop read-xtts.service 2>/dev/null || true

echo "[chain] 1/4 narrating captions -> WAVs"
"$XTTS" narrate_captions.py

echo "[chain] 2/4 rebuilding gallery index.html"
"$SYS" build_gallery.py

echo "[chain] 3/4 rebuilding slideshow.html"
"$SYS" build_slideshow.py

echo "[chain] 4/4 refreshing status + wip pages"
"$SRV" status.py
"$SRV" wip.py

echo "[chain] DONE at $(date)"
