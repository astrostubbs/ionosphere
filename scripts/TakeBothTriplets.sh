#!/bin/zsh
# TakeBothTriplets.sh - Capture triplets from both Cambridge and Sudbury receivers
#
# Usage: ./TakeBothTriplets.sh
#
# This script runs Take3Freq.sh in parallel for two geographically separated
# KiwiSDR receivers, enabling geographic correlation analysis.
#
# Default receivers:
#   - 22463.proxy.kiwisdr.com (Sudbury, MA)
#   - 22350.proxy.kiwisdr.com (Cambridge, MA)
#
# Edit the URLs below to use different receivers.

# === RECEIVER CONFIGURATION ===
RECEIVER_1="http://22463.proxy.kiwisdr.com:8073"  # Sudbury, MA
RECEIVER_2="http://22350.proxy.kiwisdr.com:8073"  # Cambridge, MA

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Taking triplets at both stations..."
echo "  Receiver 1: $RECEIVER_1"
echo "  Receiver 2: $RECEIVER_2"

# Run Take3Freq.sh for both receivers in parallel
"${SCRIPT_DIR}/Take3Freq.sh" "$RECEIVER_1" &
"${SCRIPT_DIR}/Take3Freq.sh" "$RECEIVER_2" &

# Wait for both to complete
wait

echo "Both triplet captures complete."
