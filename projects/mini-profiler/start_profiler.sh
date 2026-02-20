#!/bin/bash
# Start Mini Profiler - kills any prior instance, launches server + bare browser window

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=6048
URL="http://localhost:$PORT"

# Kill any existing instance on the port
existing=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$existing" ]; then
    echo "Killing existing process(es) on port $PORT: $existing"
    kill $existing 2>/dev/null
    sleep 1
    # Force kill if still running
    remaining=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$remaining" ]; then
        kill -9 $remaining 2>/dev/null
        sleep 1
    fi
fi

# Start the profiler in the background
python3 "$SCRIPT_DIR/mini_profiler.py" &
SERVER_PID=$!
echo "Mini Profiler started (PID: $SERVER_PID)"

# Wait for server to be ready
for i in $(seq 1 10); do
    curl -s "$URL/api/stats" > /dev/null 2>&1 && break
    sleep 0.5
done

# Open bare browser window (no address bar, menus, bookmarks)
if command -v chromium-browser &> /dev/null; then
    BROWSER=chromium-browser
elif command -v chromium &> /dev/null; then
    BROWSER=chromium
elif command -v google-chrome &> /dev/null; then
    BROWSER=google-chrome
elif command -v google-chrome-stable &> /dev/null; then
    BROWSER=google-chrome-stable
else
    echo "No Chrome/Chromium found. Open $URL manually."
    exit 0
fi

echo "Opening bare window with $BROWSER"
"$BROWSER" --app="$URL" --window-size=300,380 &> /dev/null &

echo "Dashboard: $URL"
