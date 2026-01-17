#!/bin/zsh
# Take3Freq.sh - Capture 3 WWV frequencies simultaneously from a KiwiSDR
#
# Usage: ./Take3Freq.sh <URL>
#   Example: ./Take3Freq.sh http://22463.proxy.kiwisdr.com:8073
#   Example: ./Take3Freq.sh 22463.proxy.kiwisdr.com:8073
#
# URL FORMAT:
#   The script accepts URLs in these formats:
#   - http://HOSTNAME:PORT   (full URL with protocol)
#   - HOSTNAME:PORT          (without protocol)
#   - http://HOSTNAME        (uses default port 8073)
#
# OUTPUT FILES:
#   Creates 3 IQ WAV files named:
#   YYYYMMDD.HHMM.HOSTNAME.freqXXXX.SAMPLERATE.triplet.wav
#
# FREQUENCIES:
#   Edit FREQS_KHZ array below to change which frequencies to capture.
#   Default nighttime: 3330, 5000, 7850 kHz (CHU 3330, WWV 5000, CHU 7850)
#   Default daytime:   7850, 20000, 25000 kHz

# === FREQUENCY CONFIGURATION ===
FREQS_KHZ=(3330 5000 7850)      # Nighttime frequencies
# FREQS_KHZ=(7850 20000 25000)  # Daytime frequencies (uncomment to use)

# Load local environment
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
fi

set -euo pipefail

usage() {
    echo "Usage: $0 <URL>"
    echo "  Example: ./Take3Freq.sh http://22463.proxy.kiwisdr.com:8073"
    echo "  Example: ./Take3Freq.sh 22463.proxy.kiwisdr.com:8073"
    echo ""
    echo "URL can be with or without http:// prefix"
    echo "Port defaults to 8073 if not specified"
    exit 2
}

[[ $# -eq 1 ]] || usage

RAW_URL="$1"

# === URL PARSING ===
# Remove protocol prefix if present, then extract host and port
KIWI_HOST=$(echo "$RAW_URL" | sed -e 's|^[^/]*//||' -e 's|:.*$||')
KIWI_PORT=$(echo "$RAW_URL" | sed -e 's|^.*:||' -e 's|/.*$||')

# Default port if not specified
if [[ "$KIWI_PORT" == "$KIWI_HOST" ]]; then
    KIWI_PORT=8073
fi

# === RECORDING CONFIGURATION ===
OFFSET_KHZ=2          # Tune 2 kHz above carrier for beat frequency
SAMPLE_RATE=20000     # 20 kHz sample rate
DURATION=90           # 90 second recordings

# Timestamp in UTC
TIMESTAMP=$(date -u +%Y%m%d.%H%M)

echo "Starting synchronized capture (UT: ${TIMESTAMP}) on ${KIWI_HOST}:${KIWI_PORT}..."
echo "Frequencies: ${FREQS_KHZ[@]} kHz"

# === PARALLEL DATA COLLECTION ===
for FREQ in "${FREQS_KHZ[@]}"; do
    TARGET_KHZ=$(( FREQ + OFFSET_KHZ ))

    # Output filename format: timestamp.stationID.freqXXXX.samplerate.triplet.wav
    OUTFILE="${TIMESTAMP}.${KIWI_HOST}.freq${FREQ}.${SAMPLE_RATE}.triplet"

    echo "  -> Launching ${FREQ} kHz (tuned to ${TARGET_KHZ} kHz) -> ${OUTFILE}"

    # Execute kiwirecorder in background
    python3 kiwirecorder.py \
        -s "${KIWI_HOST}" \
        -p "${KIWI_PORT}" \
        -f "${TARGET_KHZ}" \
        -m iq \
        --resample "${SAMPLE_RATE}" \
        --time-limit "${DURATION}" \
        --agc-gain=50 \
        --filename "${OUTFILE}" &
done

# Wait for all background processes to complete
wait

echo "Synchronized capture complete."
