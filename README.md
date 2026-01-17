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

## Documentation

See the `docs/` folder for user guides:
- `Chris_s_quick_and_dirty_HF_propagator.pdf` - HF propagation simulator guide
- `HF_Polarization_physics.pdf` - Comprehensive physics (~50 pages)
- `Phase_Analysis_Users_Guide.pdf` - Phase stability analysis guide
- `Phase_analysis_helper_programs_user_s_guide.pdf` - Helper tools guide

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

## Requirements

```bash
pip install numpy matplotlib scipy
```

## Author

Christopher Stubbs

Analysis developed with assistance from Claude Code.
