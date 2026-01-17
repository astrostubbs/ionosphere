# HF Ionosphere Analysis

Tools for studying HF radio propagation through the ionosphere, with emphasis on magnetoionic effects (O/X mode splitting), polarization, and fading mechanisms.

## Overview

This project provides:

1. **3D HF Ray Tracer** (`multihop_3d.py`) - Simulates O-mode and X-mode propagation including:
   - Differential D-region absorption
   - Elevation and azimuth splitting
   - Faraday rotation
   - Received polarization state (Stokes parameters)
   - Multi-hop paths

2. **Phase Stability Analysis** (`wwv_phase_analysis_final.py`) - Analyzes WWV/CHU recordings for:
   - Carrier phase tracking
   - Fade detection
   - Structure function D(τ) calculation
   - Assessment of adaptive correction feasibility

3. **Triplet Analysis** (`wwv_triplet_analysis.py`) - Simultaneous multi-frequency analysis:
   - Cross-frequency correlation
   - Comparative fading statistics
   - Day/night behavior comparison

4. **Geographic Correlation** (`wwv_geographic_correlation.py`) - Multi-receiver analysis:
   - Compare signals from geographically separated receivers
   - Assess spatial coherence of ionospheric effects

5. **Spectrum Analysis** (`check_spectrum.py`) - View carrier and sideband structure

6. **Data Collection Scripts** (`scripts/`) - Shell scripts for KiwiSDR data capture:
   - `Take3Freq.sh` - Capture 3 frequencies simultaneously from one receiver
   - `TakeBothTriplets.sh` - Capture from two geographically separated receivers
   - `TakeData.sh` - Single-frequency capture with verbose output
   - `RunEvery10Min.sh` - Scheduler for continuous monitoring

## Documentation

See the `docs/` folder for user guides:
- `HF3D_propagation_simulator_manual.pdf` - 3D HF propagation simulator manual
- `HF_Polarization_physics.pdf` - Comprehensive magnetoionic physics (~50 pages)
- `Phase_Analysis_Users_Guide.pdf` - Phase stability analysis guide
- `Phase_analysis_helper_programs_user_s_guide.pdf` - Helper tools guide
- `Geographic_Correlation_Users_Guide.pdf` - Multi-receiver geographic correlation guide

## Usage

### 3D Ray Tracer

```bash
# WWV to Boston, daytime
python3 multihop_3d.py \
    --tx-lat 40.68 --tx-lon -105.04 \
    --rx-lat 42.36 --rx-lon -71.06 \
    --frequencies 2.5 5 10 15 20 25 \
    --hops 1 \
    --date "2026-01-16T18:00:00" \
    --output wwv_boston_day
```

### Phase Analysis

```bash
python3 wwv_phase_analysis_final.py recording.wav
python3 wwv_phase_analysis_final.py --batch "*.wav"
```

### Triplet Analysis

```bash
python3 wwv_triplet_analysis.py 20260107.1044
```

### Geographic Correlation

```bash
# List available receiver pairs
python3 wwv_geographic_correlation.py --list

# Analyze specific timestamp
python3 wwv_geographic_correlation.py 20260101.1148

# Analyze all pairs from a specific date
python3 wwv_geographic_correlation.py --date 20260101
```

### Data Collection Scripts

The `scripts/` directory contains shell scripts for capturing IQ data from KiwiSDR receivers.

**Prerequisites:** These scripts require `kiwirecorder.py` from the [kiwiclient](https://github.com/jks-prv/kiwiclient) repository. Run from the kiwiclient directory or ensure kiwirecorder.py is in your PATH.

#### URL Format (Important!)

The scripts accept KiwiSDR URLs in these formats:

```bash
# Full URL with protocol and port (recommended)
http://22463.proxy.kiwisdr.com:8073

# Without protocol (http:// assumed)
22463.proxy.kiwisdr.com:8073

# Without port (8073 assumed)
http://22463.proxy.kiwisdr.com
```

**Common KiwiSDR proxy URLs:**
- `22463.proxy.kiwisdr.com:8073` - Sudbury, MA
- `22350.proxy.kiwisdr.com:8073` - Cambridge, MA

#### Take3Freq.sh - Triplet Capture

Captures 3 WWV/CHU frequencies simultaneously from a single receiver:

```bash
cd /path/to/kiwiclient
./scripts/Take3Freq.sh http://22463.proxy.kiwisdr.com:8073
```

Edit the `FREQS_KHZ` array in the script to change frequencies:
- Nighttime default: `(3330 5000 7850)` - CHU 3.33, WWV 5, CHU 7.85 MHz
- Daytime option: `(7850 20000 25000)` - CHU 7.85, WWV 20, WWV 25 MHz

#### TakeBothTriplets.sh - Geographic Correlation Capture

Runs Take3Freq.sh on two receivers simultaneously for geographic correlation studies:

```bash
./scripts/TakeBothTriplets.sh
```

Edit receiver URLs in the script to use different KiwiSDRs.

#### TakeData.sh - Single Frequency Capture

Captures one frequency with detailed output:

```bash
./scripts/TakeData.sh http://22463.proxy.kiwisdr.com:8073 20000
./scripts/TakeData.sh 22463.proxy.kiwisdr.com:8073 5000
```

#### RunEvery10Min.sh - Continuous Monitoring

Runs TakeBothTriplets.sh every 10 minutes for long-term monitoring:

```bash
# Run in foreground (Ctrl+C to stop)
./scripts/RunEvery10Min.sh

# Run in background with logging
nohup ./scripts/RunEvery10Min.sh > capture.log 2>&1 &

# Run in screen/tmux session
screen -S kiwi ./scripts/RunEvery10Min.sh
```

## Requirements

```bash
pip install numpy matplotlib scipy
```

## Author

Christopher Stubbs

Analysis developed with assistance from Claude Code.
