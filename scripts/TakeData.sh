#!/bin/zsh
# TakeData.sh - Capture a single WWV/CHU frequency from a KiwiSDR
#
# Usage: ./TakeData.sh <URL> <WWV_freq_kHz>
#   Example: ./TakeData.sh http://22463.proxy.kiwisdr.com:8073 20000
#   Example: ./TakeData.sh 22463.proxy.kiwisdr.com:8073 5000
#
# URL FORMAT:
#   The script accepts URLs in these formats:
#   - http://HOSTNAME:PORT   (full URL with protocol)
#   - HOSTNAME:PORT          (without protocol)
#   - HOSTNAME               (uses default port 8073)
#
# FREQUENCIES:
#   WWV broadcasts on: 2500, 5000, 10000, 15000, 20000, 25000 kHz
#   CHU broadcasts on: 3330, 7850, 14670 kHz
#
# OUTPUT:
#   Creates IQ WAV file named:
#   YYYYMMDD.HHMM.HOSTNAME.freqXXXX.SAMPLERATE.wav

# Load local environment
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
fi

set -euo pipefail

usage() {
    echo "Usage: $0 <URL> <WWV_freq_kHz>"
    echo "  <URL>           : e.g., http://22463.proxy.kiwisdr.com:8073"
    echo "  <WWV_freq_kHz>  : integer kHz (e.g., 5000, 10000, 20000)"
    echo ""
    echo "Examples:"
    echo "  $0 http://22463.proxy.kiwisdr.com:8073 20000"
    echo "  $0 22463.proxy.kiwisdr.com:8073 5000"
    exit 2
}

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -eq 2 ]] || usage

RAW_URL="$1"
FREQ_KHZ="$2"

# === URL PARSING ===
# Remove protocol prefix if present
CLEAN_HOST=${RAW_URL#*://}

# Extract host and port
if [[ "$CLEAN_HOST" == *:* ]]; then
    PORT=${CLEAN_HOST#*:}
    HOST=${CLEAN_HOST%:*}
else
    PORT=8073
    HOST=$CLEAN_HOST
fi

# Validate frequency is numeric
[[ "$FREQ_KHZ" =~ ^[0-9]+$ ]] || die "Frequency must be integer kHz"

# === RECORDING CONFIGURATION ===
OFFSET_KHZ=2              # Tune 2 kHz above carrier
CAPTURE_KHZ=$((FREQ_KHZ + OFFSET_KHZ))
DURATION_SEC=90
SAMPLE_RATE=20000
MODE="iq"
USER_NAME="kiwi_iq_phase_study"

# Timestamp in UTC
TS_UTC="$(date -u +%Y%m%d.%H%M)"
OUT="${TS_UTC}.${HOST}.freq${FREQ_KHZ}.${SAMPLE_RATE}.wav"

# Find kiwirecorder.py
if [[ -f "./kiwirecorder.py" ]]; then
    RECORDER="./kiwirecorder.py"
elif [[ -f "$HOME/Desktop/projects/ionosphere/kiwiclient/kiwirecorder.py" ]]; then
    RECORDER="$HOME/Desktop/projects/ionosphere/kiwiclient/kiwirecorder.py"
else
    die "kiwirecorder.py not found. Run from kiwiclient directory or set path."
fi

echo "=== Connection Details ==="
echo "Hostname       : ${HOST}"
echo "Port           : ${PORT}"
echo "WWV carrier    : ${FREQ_KHZ} kHz"
echo "Tuning capture : ${CAPTURE_KHZ} kHz (offset +${OFFSET_KHZ} kHz)"
echo "Sample Rate    : ${SAMPLE_RATE} Hz (IQ)"
echo "Duration       : ${DURATION_SEC} seconds"
echo "Output         : ${OUT}"
echo "=========================="

# Build command arguments
CMD_ARGS=(
    -s "$HOST"
    -p "$PORT"
    -f "$CAPTURE_KHZ"
    -m "$MODE"
    -u "$USER_NAME"
    --time-limit="$DURATION_SEC"
    -r "$SAMPLE_RATE"
    --agc-gain=50
)

echo "Executing: python3 $RECORDER ${CMD_ARGS[@]}"
echo "=========================="

# Create temp directory for capture
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

pushd "$TMPDIR" >/dev/null
python3 "$RECORDER" "${CMD_ARGS[@]}" || {
    echo "kiwirecorder failed. Check if the URL/Port is correct."
    popd >/dev/null
    exit 1
}

# Find the newest WAV file created
NEWEST_WAV=$(ls -t *.wav 2>/dev/null | head -n 1)
if [[ -z "$NEWEST_WAV" ]]; then
    echo "ERROR: No WAV file was generated."
    popd >/dev/null
    exit 1
fi

popd >/dev/null

# Move to current directory with standardized name
mv "$TMPDIR/$NEWEST_WAV" "./$OUT"

echo "Success! Data saved to: $OUT"
