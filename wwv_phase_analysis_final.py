#!/usr/bin/env python3
"""
WWV Phase Stability Analysis - Final Version
=============================================

PURPOSE:
    Analyze phase stability of WWV radio signals recorded via KiwiSDR to assess
    feasibility of adaptive ionospheric correction for agile HF communications.

    This is analogous to adaptive optics for astronomy - using pilot tones to
    measure and correct for ionospheric phase distortions in real-time.

BACKGROUND:
    WWV (NIST, Fort Collins, CO) broadcasts precise carriers at 2.5, 5, 10, 15,
    20, and 25 MHz. The signal includes:
    - Main carrier (continuous)
    - ±100 Hz BCD time code sidebands (amplitude modulated, intermittent)
    - ±600 Hz audio tones (when present, depends on frequency/time)

    KiwiSDR receivers digitize the signal as I/Q (complex) samples. By detuning
    the receiver ~2 kHz from the carrier, we get a beat signal that preserves
    ALL phase variations from the original RF carrier.

KEY METRICS:
    - Phase structure function D(τ) = <[φ(t+τ) - φ(t)]²>
      Measures how much phase varies over time lag τ
    - Critical timescale: τ = 20 ms (round-trip time for ~2000 km path)
    - Goal: RMS phase variation < 0.5 rad at 20 ms for viable correction

USAGE:
    # Analyze a specific file:
    python wwv_phase_analysis_final.py recording.wav

    # Analyze most recent .wav file in current directory:
    python wwv_phase_analysis_final.py

    # Batch analysis of all files:
    python wwv_phase_analysis_final.py --batch "*.wav"

OUTPUT:
    - Console summary with structure function values
    - PNG plot with amplitude, phase, zoom, and structure function panels
    - Detection of 600 Hz tones and their phase analysis when present

AUTHOR:
    Analysis developed with Claude Code, December 2024-January 2025

DEPENDENCIES:
    numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, welch
from scipy.optimize import curve_fit
import glob
import os
import re
import argparse
from datetime import datetime

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Time trimming - first few seconds often have transients
T_TRIM_START = 2.0  # seconds to trim from start of recording

# SNR threshold for valid data
# Samples more than this many dB below the 90th percentile are masked as "faded"
SNR_THRESHOLD_DB = 12.0

# Filter bandwidth for carrier extraction
# WIDE bandwidth preserves fast phase variations (essential for adaptive optics)
# 500 Hz allows seeing variations down to ~2 ms timescales
CARRIER_FILTER_BW = 500.0  # Hz

# Filter bandwidth for 600 Hz tone extraction
# Slightly narrower to reduce noise, but still wide enough for dynamics
TONE_600_FILTER_BW = 50.0  # Hz

# Critical timescale for round-trip correction
# For ~2000 km ionospheric path with multiple hops: ~20 ms
TAU_ROUNDTRIP = 0.020  # seconds

# Phase discontinuity detection parameters
# When signal fades, phase tracking is lost. On recovery, there's often a
# large phase jump that's not real ionospheric variation - we splice these out.
PHASE_JUMP_THRESHOLD = 1.5  # radians - jumps larger than this are spliced
FADE_MARGIN_SAMPLES = 100   # samples to exclude around detected fades

# Structure function evaluation points
TAU_VALUES = np.array([0.001, 0.002, 0.005, 0.010, 0.020, 0.050,
                       0.100, 0.200, 0.500, 1.0])  # seconds


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def load_wav_file(filename):
    """
    Load a KiwiSDR IQ WAV file.

    Parameters
    ----------
    filename : str
        Path to the WAV file (must be stereo with I and Q channels)

    Returns
    -------
    z : ndarray (complex)
        Complex IQ signal: z = I + j*Q
    sample_rate : int
        Sample rate in Hz
    freq_mhz : float or None
        WWV frequency in MHz, parsed from filename if present

    Notes
    -----
    KiwiSDR saves IQ data as stereo WAV with I in left channel, Q in right.
    The filename convention includes 'freqXXXXX' where XXXXX is frequency in kHz.
    """
    sample_rate, data = wavfile.read(filename)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("WAV file must be stereo (I/Q format)")

    # Convert to complex signal
    z = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)

    # Parse frequency from filename (e.g., "freq5000" -> 5.0 MHz)
    freq_match = re.search(r'freq(\d+)', filename)
    freq_mhz = int(freq_match.group(1)) / 1000 if freq_match else None

    return z, sample_rate, freq_mhz


def find_carrier_frequency(z, sample_rate, time):
    """
    Find the precise carrier frequency using FFT and iterative phase refinement.

    The carrier appears at an offset from 0 Hz due to receiver detuning
    (typically ~2 kHz). We need sub-Hz precision for accurate phase tracking.

    Parameters
    ----------
    z : ndarray (complex)
        Complex IQ signal
    sample_rate : int
        Sample rate in Hz
    time : ndarray
        Time vector in seconds

    Returns
    -------
    f_carrier : float
        Carrier frequency in Hz (relative to baseband)

    Algorithm
    ---------
    1. Coarse estimate via FFT peak detection
    2. Iterative refinement: shift to baseband, measure residual phase slope,
       adjust frequency estimate. Repeat 3x for convergence.
    """
    # Coarse FFT search
    n_fft = 2**18  # ~0.08 Hz resolution at 20 kHz sample rate
    freqs = np.fft.fftfreq(n_fft, d=1/sample_rate)
    mag_spec = np.abs(np.fft.fft(z, n=n_fft))
    f_est = freqs[np.argmax(mag_spec)]

    # Iterative phase-based refinement
    nyq = sample_rate / 2
    b, a = butter(4, 200 / nyq, btype='low')

    for _ in range(3):
        # Shift to baseband
        z_bb = filtfilt(b, a, z * np.exp(-1j * 2 * np.pi * f_est * time))
        # Measure residual frequency as phase slope
        phase = np.unwrap(np.angle(z_bb))
        slope, _ = np.polyfit(time, phase, 1)
        # Update estimate
        f_est += slope / (2 * np.pi)

    return f_est


def detect_fades(amplitude, sample_rate, threshold_db=SNR_THRESHOLD_DB):
    """
    Detect signal fade regions where phase tracking becomes unreliable.

    Parameters
    ----------
    amplitude : ndarray
        Signal amplitude (linear scale)
    sample_rate : int
        Sample rate in Hz
    threshold_db : float
        Threshold in dB below 90th percentile to consider "faded"

    Returns
    -------
    fade_mask : ndarray (bool)
        True where signal is faded (should be excluded from analysis)
    amp_db : ndarray
        Amplitude in dB
    threshold : float
        The actual threshold value used (dB)

    Notes
    -----
    We expand fade regions by FADE_MARGIN_SAMPLES on each side to exclude
    edge effects where phase tracking may be degraded but amplitude looks OK.
    """
    amp_db = 20 * np.log10(amplitude + 1e-12)
    ref_db = np.percentile(amp_db, 90)
    threshold = ref_db - threshold_db

    # Initial fade detection
    is_faded = amp_db < threshold

    # Expand fade regions by margin
    fade_mask = np.copy(is_faded)
    for i in range(len(fade_mask)):
        if is_faded[i]:
            start = max(0, i - FADE_MARGIN_SAMPLES)
            end = min(len(fade_mask), i + FADE_MARGIN_SAMPLES)
            fade_mask[start:end] = True

    return fade_mask, amp_db, threshold


def splice_phase(phase, time, fade_mask):
    """
    Splice out phase discontinuities that occur at fade boundaries.

    When signal fades out, the phase-locked loop loses lock. On recovery,
    the phase may jump by an arbitrary amount. These jumps are artifacts,
    not real ionospheric variations, so we remove them by adjusting the
    phase to be continuous across fade boundaries.

    Parameters
    ----------
    phase : ndarray
        Unwrapped phase in radians
    time : ndarray
        Time vector in seconds
    fade_mask : ndarray (bool)
        True where signal is faded

    Returns
    -------
    phase_spliced : ndarray
        Phase with discontinuities removed
    splice_times : list of float
        Times where splices were applied (for plotting)

    Algorithm
    ---------
    1. Find transitions from faded to valid (exit_fade_indices)
    2. At each transition, measure phase jump from last valid sample before fade
    3. If jump exceeds threshold, subtract it from all subsequent samples
    """
    phase_spliced = np.copy(phase)
    splice_times = []

    # Find transitions: fade -> valid
    fade_edges = np.diff(fade_mask.astype(int))
    exit_fade_indices = np.where(fade_edges == -1)[0] + 1

    for idx in exit_fade_indices:
        if idx < 10 or idx >= len(phase) - 10:
            continue

        # Find last valid sample before this fade
        pre_fade_idx = idx - 1
        while pre_fade_idx > 0 and fade_mask[pre_fade_idx]:
            pre_fade_idx -= 1

        if pre_fade_idx <= 0:
            continue

        # Calculate and apply splice if jump is large
        phase_before = phase_spliced[pre_fade_idx]
        phase_after = phase_spliced[idx]
        jump = phase_after - phase_before

        if abs(jump) > PHASE_JUMP_THRESHOLD:
            phase_spliced[idx:] -= jump
            splice_times.append(time[idx])

    return phase_spliced, splice_times


def detect_600hz_tones(z, sample_rate, f_carrier):
    """
    Check if 600 Hz audio tones are present in the recording.

    WWV transmits 600 Hz tones during certain minutes on certain frequencies.
    When present, they provide additional phase references for dispersion
    measurements across a 1200 Hz bandwidth.

    Parameters
    ----------
    z : ndarray (complex)
        Complex IQ signal
    sample_rate : int
        Sample rate in Hz
    f_carrier : float
        Carrier frequency in Hz

    Returns
    -------
    present : bool
        True if both ±600 Hz sidebands are detected

    Detection
    ---------
    Tones are considered present if their power is within 35 dB of the carrier.
    """
    n_fft = 2**16
    freqs = np.fft.fftfreq(n_fft, d=1/sample_rate)
    mag_spec = np.abs(np.fft.fft(z, n=n_fft))**2

    # Carrier power
    carrier_idx = np.argmin(np.abs(freqs - f_carrier))
    carrier_power = mag_spec[carrier_idx]

    # Sideband power (search in ±10 Hz window for peak)
    window = 10
    lsb_idx = np.argmin(np.abs(freqs - (f_carrier - 600)))
    usb_idx = np.argmin(np.abs(freqs - (f_carrier + 600)))

    lsb_power = np.max(mag_spec[max(0, lsb_idx-window):lsb_idx+window])
    usb_power = np.max(mag_spec[max(0, usb_idx-window):usb_idx+window])

    # Check if tones are within 35 dB of carrier
    lsb_present = 10*np.log10(lsb_power/carrier_power + 1e-12) > -35
    usb_present = 10*np.log10(usb_power/carrier_power + 1e-12) > -35

    return lsb_present and usb_present


def extract_signal(z, sample_rate, time, f_target, filter_bw):
    """
    Extract amplitude and phase of a signal at a specific frequency.

    Parameters
    ----------
    z : ndarray (complex)
        Complex IQ signal
    sample_rate : int
        Sample rate in Hz
    time : ndarray
        Time vector in seconds
    f_target : float
        Target frequency to extract (Hz)
    filter_bw : float
        Low-pass filter bandwidth after frequency shift (Hz)

    Returns
    -------
    amplitude : ndarray
        Signal amplitude (linear)
    phase : ndarray
        Unwrapped phase in radians

    Method
    ------
    1. Multiply by complex exponential to shift target frequency to baseband
    2. Low-pass filter to isolate the signal
    3. Extract amplitude and unwrapped phase
    """
    nyq = sample_rate / 2
    b, a = butter(4, filter_bw / nyq, btype='low')
    z_bb = filtfilt(b, a, z * np.exp(-1j * 2 * np.pi * f_target * time))

    return np.abs(z_bb), np.unwrap(np.angle(z_bb))


def compute_structure_function(phase, time, valid_mask, tau_values):
    """
    Compute the phase structure function D(τ).

    D(τ) = <[φ(t+τ) - φ(t)]²>

    This is the key metric for adaptive optics feasibility. It tells us
    how much the phase typically changes over a time lag τ.

    Parameters
    ----------
    phase : ndarray
        Phase values in radians
    time : ndarray
        Time vector in seconds
    valid_mask : ndarray (bool)
        True for valid (non-faded) samples
    tau_values : ndarray
        Time lags at which to evaluate D(τ)

    Returns
    -------
    D_tau : ndarray
        Structure function values (rad²) at each tau

    Interpretation
    --------------
    - D(τ) ~ τ^α indicates turbulent-like behavior
    - RMS phase change over lag τ is sqrt(D(τ))
    - For adaptive optics: want sqrt(D(τ_roundtrip)) < 0.5 rad
    """
    dt = time[1] - time[0]
    phase_valid = phase[valid_mask]

    D_tau = []
    for tau in tau_values:
        shift = int(tau / dt)
        if shift >= len(phase_valid) - 1:
            D_tau.append(np.nan)
            continue
        diff = phase_valid[shift:] - phase_valid[:-shift]
        D_tau.append(np.mean(diff**2))

    return np.array(D_tau)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_wwv(filename=None, make_plot=True, verbose=True):
    """
    Perform complete WWV phase stability analysis on a recording.

    Parameters
    ----------
    filename : str or None
        Path to WAV file. If None, uses most recent .wav in current directory.
    make_plot : bool
        Whether to generate and save diagnostic plots
    verbose : bool
        Whether to print progress and results to console

    Returns
    -------
    results : dict
        Dictionary containing all analysis results:
        - filename: input filename
        - freq_mhz: WWV frequency
        - sample_rate: sample rate
        - duration: recording duration (seconds)
        - pct_valid: percentage of samples that passed SNR threshold
        - n_splices: number of phase splices applied
        - has_600hz: whether 600 Hz tones were detected
        - tau_values: time lags for structure function
        - D_tau: structure function values
        - rms_20ms: RMS phase change at 20 ms (key metric)
        - rms_20ms_600: RMS for 600 Hz USB-LSB difference (if present)
    """

    # -------------------------------------------------------------------------
    # STEP 1: Load data
    # -------------------------------------------------------------------------
    if filename is None:
        wav_files = glob.glob('*.wav')
        if not wav_files:
            raise FileNotFoundError("No .wav files found in current directory")
        filename = max(wav_files, key=os.path.getmtime)

    if verbose:
        print(f"\n{'='*70}")
        print(f"WWV PHASE STABILITY ANALYSIS")
        print(f"{'='*70}")
        print(f"File: {os.path.basename(filename)}")

    z, sample_rate, freq_mhz = load_wav_file(filename)
    dt = 1.0 / sample_rate

    # Trim start
    start_idx = int(T_TRIM_START * sample_rate)
    z = z[start_idx:]
    time = np.arange(len(z)) / sample_rate
    duration = time[-1]

    if verbose:
        print(f"Frequency: {freq_mhz} MHz")
        print(f"Sample rate: {sample_rate} Hz (dt = {dt*1000:.3f} ms)")
        print(f"Duration: {duration:.1f} s (after {T_TRIM_START}s trim)")

    # -------------------------------------------------------------------------
    # STEP 2: Find carrier frequency
    # -------------------------------------------------------------------------
    f_carrier = find_carrier_frequency(z, sample_rate, time)
    if verbose:
        print(f"Carrier offset: {f_carrier:.2f} Hz")

    # -------------------------------------------------------------------------
    # STEP 3: Extract carrier with wide bandwidth
    # -------------------------------------------------------------------------
    amplitude, phase_raw = extract_signal(z, sample_rate, time,
                                          f_carrier, CARRIER_FILTER_BW)

    # Remove linear trend (residual frequency offset)
    coeffs = np.polyfit(time, phase_raw, 1)
    phase_detrend = phase_raw - np.polyval(coeffs, time)

    # -------------------------------------------------------------------------
    # STEP 4: Detect fades and create validity mask
    # -------------------------------------------------------------------------
    fade_mask, amp_db, fade_threshold = detect_fades(amplitude, sample_rate)
    valid_mask = ~fade_mask
    pct_valid = 100 * np.sum(valid_mask) / len(valid_mask)

    if verbose:
        print(f"Valid samples: {pct_valid:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 5: Splice phase discontinuities
    # -------------------------------------------------------------------------
    phase_spliced, splice_times = splice_phase(phase_detrend, time, fade_mask)

    if verbose:
        print(f"Phase splices applied: {len(splice_times)}")

    # -------------------------------------------------------------------------
    # STEP 6: Check for 600 Hz tones
    # -------------------------------------------------------------------------
    has_600hz = detect_600hz_tones(z, sample_rate, f_carrier)

    if verbose:
        print(f"600 Hz tones detected: {'YES' if has_600hz else 'NO'}")

    # -------------------------------------------------------------------------
    # STEP 7: Compute structure function
    # -------------------------------------------------------------------------
    D_tau = compute_structure_function(phase_spliced, time, valid_mask, TAU_VALUES)

    # Key metric: RMS at 20 ms
    idx_20ms = np.argmin(np.abs(TAU_VALUES - TAU_ROUNDTRIP))
    rms_20ms = np.sqrt(D_tau[idx_20ms]) if not np.isnan(D_tau[idx_20ms]) else np.nan

    if verbose:
        print(f"\nPhase Structure Function D(τ):")
        print(f"{'τ (ms)':>8} {'D(τ) rad²':>12} {'RMS rad':>10} {'RMS °':>8}")
        print("-" * 42)
        for tau, D in zip(TAU_VALUES, D_tau):
            if not np.isnan(D):
                rms = np.sqrt(D)
                marker = " <-- 20ms" if abs(tau - TAU_ROUNDTRIP) < 0.001 else ""
                print(f"{tau*1000:8.1f} {D:12.4f} {rms:10.4f} {np.degrees(rms):8.2f}{marker}")

        print(f"\n*** RMS phase @ 20ms: {rms_20ms:.4f} rad ({np.degrees(rms_20ms):.1f}°) ***")

        if rms_20ms < 0.5:
            print("    Assessment: GOOD - suitable for adaptive correction")
        elif rms_20ms < 1.0:
            print("    Assessment: MARGINAL - correction may be challenging")
        else:
            print("    Assessment: POOR - correction will be difficult")

    # -------------------------------------------------------------------------
    # STEP 8: Analyze 600 Hz tones if present
    # -------------------------------------------------------------------------
    rms_20ms_600 = np.nan
    tone_600_data = None

    if has_600hz:
        amp_lsb, phase_lsb = extract_signal(z, sample_rate, time,
                                            f_carrier - 600, TONE_600_FILTER_BW)
        amp_usb, phase_usb = extract_signal(z, sample_rate, time,
                                            f_carrier + 600, TONE_600_FILTER_BW)

        # Phase relative to carrier
        phase_lsb_rel = phase_lsb - phase_raw
        phase_usb_rel = phase_usb - phase_raw

        # Splice
        phase_lsb_spliced, _ = splice_phase(phase_lsb_rel, time, fade_mask)
        phase_usb_spliced, _ = splice_phase(phase_usb_rel, time, fade_mask)

        # USB - LSB difference (1200 Hz dispersion)
        phase_diff_600 = phase_usb_spliced - phase_lsb_spliced
        D_tau_600 = compute_structure_function(phase_diff_600, time,
                                               valid_mask, TAU_VALUES)
        rms_20ms_600 = np.sqrt(D_tau_600[idx_20ms]) if not np.isnan(D_tau_600[idx_20ms]) else np.nan

        tone_600_data = {
            'phase_diff': phase_diff_600,
            'D_tau': D_tau_600,
        }

        if verbose:
            print(f"\n600 Hz USB-LSB dispersion @ 20ms: {rms_20ms_600:.4f} rad ({np.degrees(rms_20ms_600):.1f}°)")

    # -------------------------------------------------------------------------
    # STEP 9: Generate plots
    # -------------------------------------------------------------------------
    if make_plot:
        n_panels = 5 if has_600hz else 4
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4*n_panels))

        # Panel 1: Amplitude with fade regions
        ax = axes[0]
        ax.plot(time, amp_db, 'b-', lw=0.5, alpha=0.7)
        ax.axhline(fade_threshold, color='r', ls='--', alpha=0.5,
                   label=f'Fade threshold ({fade_threshold:.0f} dB)')
        ax.fill_between(time, amp_db.min(), amp_db.max(), where=fade_mask,
                        alpha=0.3, color='red', label='Faded regions')
        for st in splice_times:
            ax.axvline(st, color='orange', lw=1, alpha=0.7)
        ax.set_ylabel('Amplitude (dB)')
        ax.set_title(f'{os.path.basename(filename)} - {freq_mhz} MHz | '
                     f'{pct_valid:.0f}% valid, {len(splice_times)} splices')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Panel 2: Spliced phase
        ax = axes[1]
        ax.plot(time[valid_mask], phase_spliced[valid_mask], 'b-', lw=0.3)
        for i, st in enumerate(splice_times):
            ax.axvline(st, color='orange', lw=1, alpha=0.7,
                       label='Splice' if i == 0 else '')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('Carrier Phase (spliced)')
        if splice_times:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 3: Zoomed phase with 20 ms markers
        ax = axes[2]
        t_start, t_end = 10.0, 10.5  # 500 ms window
        zoom_mask = valid_mask & (time >= t_start) & (time <= t_end)
        if np.any(zoom_mask):
            ax.plot(time[zoom_mask], phase_spliced[zoom_mask], 'b-',
                    lw=1, marker='.', markersize=1)
            for t_line in np.arange(t_start, t_end, TAU_ROUNDTRIP):
                ax.axvline(t_line, color='red', alpha=0.3, lw=0.5)
        ax.set_ylabel('Phase (rad)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Zoomed Phase (red lines = {TAU_ROUNDTRIP*1000:.0f} ms intervals)')
        ax.grid(True, alpha=0.3)

        # Panel 4: Structure function
        ax = axes[3]
        valid_D = ~np.isnan(D_tau)
        ax.loglog(TAU_VALUES[valid_D]*1000, D_tau[valid_D], 'bo-',
                  markersize=8, label='Carrier')
        ax.axvline(TAU_ROUNDTRIP*1000, color='red', ls='--', lw=2,
                   label=f'{TAU_ROUNDTRIP*1000:.0f} ms round-trip')
        ax.axhline(0.25, color='green', ls=':', alpha=0.5, label='0.5 rad RMS')
        ax.axhline(1.0, color='orange', ls=':', alpha=0.5, label='1.0 rad RMS')
        if has_600hz and tone_600_data is not None:
            valid_D600 = ~np.isnan(tone_600_data['D_tau'])
            ax.loglog(TAU_VALUES[valid_D600]*1000, tone_600_data['D_tau'][valid_D600],
                      'gs-', markersize=6, label='600Hz USB-LSB')
        ax.set_xlabel('τ (ms)')
        ax.set_ylabel('D(τ) (rad²)')
        ax.set_title(f'Phase Structure Function | RMS @ 20ms: {rms_20ms:.3f} rad ({np.degrees(rms_20ms):.1f}°)')
        ax.legend(loc='upper left')
        ax.grid(True, which='both', alpha=0.3)

        # Panel 5: 600 Hz dispersion (if present)
        if has_600hz and n_panels > 4:
            ax = axes[4]
            ax.plot(time[valid_mask], tone_600_data['phase_diff'][valid_mask],
                    'g-', lw=0.3)
            for st in splice_times:
                ax.axvline(st, color='orange', lw=1, alpha=0.7)
            ax.set_ylabel('Phase diff (rad)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'600 Hz USB-LSB Phase Difference (1200 Hz dispersion) | '
                         f'RMS @ 20ms: {rms_20ms_600:.3f} rad')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.splitext(filename)[0] + '_phase_analysis.png'
        plt.savefig(output_file, dpi=150)
        plt.close()

        if verbose:
            print(f"\nPlot saved: {output_file}")

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------
    return {
        'filename': filename,
        'freq_mhz': freq_mhz,
        'sample_rate': sample_rate,
        'duration': duration,
        'pct_valid': pct_valid,
        'n_splices': len(splice_times),
        'has_600hz': has_600hz,
        'tau_values': TAU_VALUES,
        'D_tau': D_tau,
        'rms_20ms': rms_20ms,
        'rms_20ms_600': rms_20ms_600,
    }


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def batch_analyze(pattern='*.wav', make_plots=True):
    """
    Analyze multiple WWV recordings and generate summary statistics.

    Parameters
    ----------
    pattern : str
        Glob pattern for files to analyze (default: '*.wav')
    make_plots : bool
        Whether to generate individual plots for each file

    Returns
    -------
    results : list of dict
        Analysis results for each file
    summary : list of dict
        Summary statistics grouped by frequency
    """
    files = sorted(glob.glob(pattern))
    print(f"\nBatch analysis: {len(files)} files matching '{pattern}'")

    results = []
    for f in files:
        try:
            r = analyze_wwv(f, make_plot=make_plots, verbose=True)
            results.append(r)
        except Exception as e:
            print(f"  ERROR processing {f}: {e}")

    # Summary by frequency
    print("\n" + "="*70)
    print("SUMMARY BY FREQUENCY")
    print("="*70)

    freqs = sorted(set(r['freq_mhz'] for r in results if r['freq_mhz'] is not None))

    print(f"\n{'Freq (MHz)':>10} {'N':>4} {'RMS@20ms (rad)':>18} {'RMS@20ms (°)':>14} {'600Hz?':>8}")
    print("-" * 58)

    summary = []
    for freq in freqs:
        freq_results = [r for r in results if r['freq_mhz'] == freq]
        rms_values = [r['rms_20ms'] for r in freq_results if not np.isnan(r['rms_20ms'])]
        has_600 = any(r['has_600hz'] for r in freq_results)

        if rms_values:
            mean_rms = np.mean(rms_values)
            std_rms = np.std(rms_values) if len(rms_values) > 1 else 0
            print(f"{freq:>10.1f} {len(freq_results):>4} "
                  f"{mean_rms:>8.3f} ± {std_rms:<7.3f} "
                  f"{np.degrees(mean_rms):>8.1f}° "
                  f"{'YES' if has_600 else 'NO':>8}")
            summary.append({
                'freq': freq,
                'n': len(freq_results),
                'mean_rms': mean_rms,
                'std_rms': std_rms,
                'has_600': has_600
            })

    # Generate summary plot
    if summary:
        fig, ax = plt.subplots(figsize=(10, 6))
        freqs_plot = [d['freq'] for d in summary]
        means = [d['mean_rms'] for d in summary]
        stds = [d['std_rms'] for d in summary]

        ax.errorbar(freqs_plot, means, yerr=stds, fmt='bo-',
                    markersize=10, capsize=5, capthick=2)
        ax.axhline(0.5, color='green', ls='--', alpha=0.5,
                   label='Good (< 0.5 rad)')
        ax.axhline(1.0, color='orange', ls='--', alpha=0.5,
                   label='Marginal (< 1.0 rad)')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('RMS Phase @ 20 ms (rad)')
        ax.set_title('WWV Phase Stability vs Frequency (20 ms timescale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(means) * 1.3 if means else 1)

        plt.tight_layout()
        plt.savefig('wwv_phase_stability_summary.png', dpi=150)
        print(f"\nSummary plot saved: wwv_phase_stability_summary.png")
        plt.close()

    return results, summary


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='WWV Phase Stability Analysis for Adaptive HF Communications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wwv_phase_analysis_final.py                    # Analyze most recent .wav
  python wwv_phase_analysis_final.py recording.wav      # Analyze specific file
  python wwv_phase_analysis_final.py --batch "*.wav"    # Batch analyze all files
  python wwv_phase_analysis_final.py --batch "*freq25000*"  # Only 25 MHz files
        """)

    parser.add_argument('filename', nargs='?', default=None,
                        help='WAV file to analyze (default: most recent)')
    parser.add_argument('--batch', metavar='PATTERN',
                        help='Batch analyze files matching pattern')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')

    args = parser.parse_args()

    if args.batch:
        results, summary = batch_analyze(args.batch, make_plots=not args.no_plot)
    else:
        results = analyze_wwv(args.filename, make_plot=not args.no_plot)
