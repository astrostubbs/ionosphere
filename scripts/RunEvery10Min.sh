#!/bin/zsh
# RunEvery10Min.sh - Scheduler to run TakeBothTriplets.sh every 10 minutes
#
# Usage: ./RunEvery10Min.sh
#
# This script runs continuously, executing TakeBothTriplets.sh every 10 minutes.
# Press Ctrl+C to stop.
#
# For long-term monitoring, consider running in a screen/tmux session:
#   screen -S kiwi ./RunEvery10Min.sh
#   tmux new -s kiwi ./RunEvery10Min.sh
#
# Or use nohup for background execution:
#   nohup ./RunEvery10Min.sh > capture.log 2>&1 &

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INTERVAL_SECONDS=600  # 10 minutes

echo "Starting continuous capture every ${INTERVAL_SECONDS} seconds (10 minutes)"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC'): Starting capture..."
    "${SCRIPT_DIR}/TakeBothTriplets.sh"
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC'): Sleeping ${INTERVAL_SECONDS} seconds..."
    echo ""
    sleep $INTERVAL_SECONDS
done
