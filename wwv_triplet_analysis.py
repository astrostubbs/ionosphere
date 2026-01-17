#!/usr/bin/env python3
"""
WWV Triplet Analysis - Multi-Frequency Synchronous Recordings
==============================================================

PURPOSE:
    Analyze synchronously-recorded WWV signals at multiple frequencies to study:
    1. Individual carrier phase stability at each frequency
    2. Cross-frequency fade correlation (are fades simultaneous or delayed?)
    3. Cross-frequency phase correlation (common ionospheric variations?)

BACKGROUND:
    With synchronous digitization of multiple WWV frequencies, we can determine
    whether ionospheric effects are correlated across frequency. This reveals:
    - Whether fading is caused by the same ionospheric structure at all frequencies
    - Propagation path differences between frequencies
    - Potential for multi-frequency adaptive correction

DATA FORMAT:
    Triplets share the same timestamp in filename but differ in frequency:
    - 20260101.1044.*.freq2500.20000.wav  (2.5 MHz)
    - 20260101.1044.*.freq5000.20000.wav  (5 MHz)
    - 20260101.1044.*.freq10000.20000.wav (10 MHz)

USAGE:
    # Analyze a triplet by timestamp
    python wwv_triplet_analysis.py 20260101.1044

    # Analyze most recent triplet
    python wwv_triplet_analysis.py

    # Batch analyze all triplets from a date
    python wwv_triplet_analysis.py --batch 20260101

AUTHOR:
    Analysis developed with Claude Code, January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, correlate, coherence
from scipy.stats import pearsonr
import glob
import os
import re
import argparse
from collections import defaultdict

# =============================================================================
# CONFIGURATION - Import settings from final analysis
# =============================================================================
T_TRIM_START = 2.0
SNR_THRESHOLD_DB = 12.0
CARRIER_FILTER_BW = 500.0
TONE_600_FILTER_BW = 50.0  # Hz - 600 Hz tones are narrow but allow for some variation
PHASE_JUMP_THRESHOLD = 1.5
FADE_MARGIN_SAMPLES = 100
TAU_VALUES = np.array([0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0])
TAU_ROUNDTRIP = 0.020

# Cross-correlation settings
XCORR_MAX_LAG_SEC = 5.0  # Maximum lag to compute for cross-correlation


# =============================================================================
# HELPER FUNCTIONS (reused from wwv_phase_analysis_final.py)
# =============================================================================

def load_wav_file(filename):
    """Load KiwiSDR IQ WAV file."""
    sample_rate, data = wavfile.read(filename)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("WAV must be stereo I/Q")
    z = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)
    freq_match = re.search(r'freq(\d+)', filename)
    freq_mhz = int(freq_match.group(1)) / 1000 if freq_match else None
    return z, sample_rate, freq_mhz


def find_carrier_frequency(z, sample_rate, time):
    """Find carrier via FFT + iterative refinement."""
    n_fft = 2**18
    freqs = np.fft.fftfreq(n_fft, d=1/sample_rate)
    mag_spec = np.abs(np.fft.fft(z, n=n_fft))
    f_est = freqs[np.argmax(mag_spec)]

    nyq = sample_rate / 2
    b, a = butter(4, 200 / nyq, btype='low')
    for _ in range(3):
        z_bb = filtfilt(b, a, z * np.exp(-1j * 2 * np.pi * f_est * time))
        phase = np.unwrap(np.angle(z_bb))
        slope, _ = np.polyfit(time, phase, 1)
        f_est += slope / (2 * np.pi)
    return f_est


def extract_signal(z, sample_rate, time, f_target, filter_bw):
    """Extract amplitude and phase at target frequency."""
    nyq = sample_rate / 2
    b, a = butter(4, filter_bw / nyq, btype='low')
    z_bb = filtfilt(b, a, z * np.exp(-1j * 2 * np.pi * f_target * time))
    return np.abs(z_bb), np.unwrap(np.angle(z_bb))


def detect_fades(amplitude, threshold_db=SNR_THRESHOLD_DB):
    """Detect fade regions."""
    amp_db = 20 * np.log10(amplitude + 1e-12)
    ref_db = np.percentile(amp_db, 90)
    threshold = ref_db - threshold_db
    is_faded = amp_db < threshold

    fade_mask = np.copy(is_faded)
    for i in range(len(fade_mask)):
        if is_faded[i]:
            start = max(0, i - FADE_MARGIN_SAMPLES)
            end = min(len(fade_mask), i + FADE_MARGIN_SAMPLES)
            fade_mask[start:end] = True

    return fade_mask, amp_db, threshold


def splice_phase(phase, time, fade_mask):
    """Splice phase discontinuities at fade boundaries."""
    phase_spliced = np.copy(phase)
    splice_times = []

    fade_edges = np.diff(fade_mask.astype(int))
    exit_fade_indices = np.where(fade_edges == -1)[0] + 1

    for idx in exit_fade_indices:
        if idx < 10 or idx >= len(phase) - 10:
            continue
        pre_fade_idx = idx - 1
        while pre_fade_idx > 0 and fade_mask[pre_fade_idx]:
            pre_fade_idx -= 1
        if pre_fade_idx <= 0:
            continue

        jump = phase_spliced[idx] - phase_spliced[pre_fade_idx]
        if abs(jump) > PHASE_JUMP_THRESHOLD:
            phase_spliced[idx:] -= jump
            splice_times.append(time[idx])

    return phase_spliced, splice_times


def compute_structure_function(phase, time, valid_mask, tau_values):
    """Compute phase structure function D(τ)."""
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
# TRIPLET-SPECIFIC FUNCTIONS
# =============================================================================

def find_triplets(pattern='*.wav'):
    """
    Find groups of files that form triplets (same timestamp, different frequencies).

    Returns dict: {timestamp: {freq_mhz: filename, ...}, ...}
    """
    files = glob.glob(pattern)
    triplets = defaultdict(dict)

    for f in files:
        # Extract timestamp (e.g., "20260101.1044")
        ts_match = re.search(r'(\d{8}\.\d{4})\.', f)
        freq_match = re.search(r'freq(\d+)', f)

        if ts_match and freq_match:
            timestamp = ts_match.group(1)
            freq_mhz = int(freq_match.group(1)) / 1000
            triplets[timestamp][freq_mhz] = f

    # Filter to only complete triplets (3 or more frequencies)
    complete = {ts: freqs for ts, freqs in triplets.items() if len(freqs) >= 2}
    return complete


def compute_cross_correlation(amp1, amp2, sample_rate, max_lag_sec=XCORR_MAX_LAG_SEC):
    """
    Compute normalized cross-correlation between two amplitude time series.

    Returns:
        lags: time lags in seconds
        xcorr: normalized cross-correlation values
        peak_lag: lag at maximum correlation
        peak_corr: maximum correlation value
    """
    # Normalize to zero mean, unit variance
    amp1_norm = (amp1 - np.mean(amp1)) / (np.std(amp1) + 1e-12)
    amp2_norm = (amp2 - np.mean(amp2)) / (np.std(amp2) + 1e-12)

    # Full cross-correlation
    xcorr_full = correlate(amp1_norm, amp2_norm, mode='full')
    xcorr_full /= len(amp1)  # Normalize

    # Create lag array
    lags_samples = np.arange(-len(amp1)+1, len(amp2))
    lags_sec = lags_samples / sample_rate

    # Trim to max_lag
    max_lag_samples = int(max_lag_sec * sample_rate)
    center = len(amp1) - 1
    trim_start = max(0, center - max_lag_samples)
    trim_end = min(len(xcorr_full), center + max_lag_samples + 1)

    lags = lags_sec[trim_start:trim_end]
    xcorr = xcorr_full[trim_start:trim_end]

    # Find peak
    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]
    peak_corr = xcorr[peak_idx]

    return lags, xcorr, peak_lag, peak_corr


def compute_fade_overlap(fade_mask1, fade_mask2):
    """
    Compute statistics on fade overlap between two channels.

    Returns:
        overlap_frac: fraction of time both are faded simultaneously
        fade1_frac: fraction of time channel 1 is faded
        fade2_frac: fraction of time channel 2 is faded
        conditional_prob: P(fade2 | fade1) - probability of fade2 given fade1
    """
    n = len(fade_mask1)
    both_faded = np.sum(fade_mask1 & fade_mask2)
    fade1_count = np.sum(fade_mask1)
    fade2_count = np.sum(fade_mask2)

    overlap_frac = both_faded / n
    fade1_frac = fade1_count / n
    fade2_frac = fade2_count / n
    conditional_prob = both_faded / fade1_count if fade1_count > 0 else 0

    return overlap_frac, fade1_frac, fade2_frac, conditional_prob


def analyze_single_frequency(filename, verbose=False):
    """
    Analyze a single frequency file and return results dict.
    (Simplified version of wwv_phase_analysis_final.analyze_wwv)
    """
    z, sample_rate, freq_mhz = load_wav_file(filename)

    # Trim
    start_idx = int(T_TRIM_START * sample_rate)
    z = z[start_idx:]
    time = np.arange(len(z)) / sample_rate

    # Find carrier
    f_carrier = find_carrier_frequency(z, sample_rate, time)

    # Extract carrier
    amplitude, phase_raw = extract_signal(z, sample_rate, time, f_carrier, CARRIER_FILTER_BW)

    # Detrend phase
    coeffs = np.polyfit(time, phase_raw, 1)
    phase_detrend = phase_raw - np.polyval(coeffs, time)

    # Detect fades
    fade_mask, amp_db, threshold = detect_fades(amplitude)
    valid_mask = ~fade_mask

    # Splice phase
    phase_spliced, splice_times = splice_phase(phase_detrend, time, fade_mask)

    # Structure function
    D_tau = compute_structure_function(phase_spliced, time, valid_mask, TAU_VALUES)
    idx_20ms = np.argmin(np.abs(TAU_VALUES - TAU_ROUNDTRIP))
    rms_20ms = np.sqrt(D_tau[idx_20ms]) if not np.isnan(D_tau[idx_20ms]) else np.nan

    pct_valid = 100 * np.sum(valid_mask) / len(valid_mask)

    # =========================================================================
    # 600 Hz TONE ANALYSIS (sidebands at ±600 Hz from carrier)
    # =========================================================================
    tone_600_results = None
    try:
        # Extract ±600 Hz sidebands
        amp_lsb, phase_lsb = extract_signal(z, sample_rate, time, f_carrier - 600, TONE_600_FILTER_BW)
        amp_usb, phase_usb = extract_signal(z, sample_rate, time, f_carrier + 600, TONE_600_FILTER_BW)

        # Convert to dB for masking
        amp_lsb_db = 20 * np.log10(amp_lsb + 1e-12)
        amp_usb_db = 20 * np.log10(amp_usb + 1e-12)

        # Check if 600 Hz tones are present (amplitude should be significant)
        carrier_amp_median = np.median(amplitude[valid_mask]) if np.any(valid_mask) else np.median(amplitude)
        lsb_amp_median = np.median(amp_lsb)
        usb_amp_median = np.median(amp_usb)

        # Tones present if within 30 dB of carrier (they're typically ~10-20 dB below)
        tone_threshold = carrier_amp_median / 30  # ~30 dB below carrier
        tones_present = (lsb_amp_median > tone_threshold) and (usb_amp_median > tone_threshold)

        if tones_present:
            # Create SEPARATE mask for tones based on BOTH tone amplitudes
            # Use same threshold approach as carrier but applied to tones
            lsb_ref_db = np.percentile(amp_lsb_db, 90)
            usb_ref_db = np.percentile(amp_usb_db, 90)
            lsb_valid = amp_lsb_db > (lsb_ref_db - SNR_THRESHOLD_DB)
            usb_valid = amp_usb_db > (usb_ref_db - SNR_THRESHOLD_DB)
            # Both tones must be valid
            tone_valid_mask = lsb_valid & usb_valid

            # Add margin around fade edges for tones too
            tone_fade_mask = ~tone_valid_mask
            tone_valid_expanded = np.copy(tone_valid_mask)
            for i in range(len(tone_fade_mask)):
                if tone_fade_mask[i]:
                    start = max(0, i - FADE_MARGIN_SAMPLES)
                    end = min(len(tone_fade_mask), i + FADE_MARGIN_SAMPLES)
                    tone_valid_expanded[start:end] = False
            tone_valid_mask = tone_valid_expanded

            pct_tone_valid = 100 * np.sum(tone_valid_mask) / len(tone_valid_mask)

            if np.sum(tone_valid_mask) > 1000:  # Need enough valid samples
                # Phase relative to carrier (removes common ionospheric phase)
                phase_lsb_rel = phase_lsb - phase_raw
                phase_usb_rel = phase_usb - phase_raw

                # USB - LSB phase difference (sensitive to dispersion across 1200 Hz)
                phase_usb_lsb_diff = phase_usb - phase_lsb

                # Detrend using TONE mask, not carrier mask (for plotting)
                coeffs_diff = np.polyfit(time[tone_valid_mask], phase_usb_lsb_diff[tone_valid_mask], 1)
                phase_diff_detrend = phase_usb_lsb_diff - np.polyval(coeffs_diff, time)

                # LONG-TERM RMS (dominated by ionospheric drift)
                rms_diff_total = np.std(phase_diff_detrend[tone_valid_mask])

                # SHORT-TERM RMS: Compute per-second variance, then take median
                # This isolates fast variations from slow ionospheric drift
                dt = time[1] - time[0]
                samples_per_sec = int(1.0 / dt)
                n_seconds = int(len(time) / samples_per_sec)

                per_sec_rms = []
                per_sec_times = []
                for sec_idx in range(n_seconds):
                    start = sec_idx * samples_per_sec
                    end = start + samples_per_sec
                    seg_mask = tone_valid_mask[start:end]
                    if np.sum(seg_mask) > samples_per_sec // 4:  # At least 25% valid
                        seg_phase = phase_usb_lsb_diff[start:end][seg_mask]
                        seg_time = time[start:end][seg_mask]
                        # Remove linear trend within this 1-second window
                        if len(seg_time) > 10:
                            seg_coeffs = np.polyfit(seg_time, seg_phase, 1)
                            seg_detrend = seg_phase - np.polyval(seg_coeffs, seg_time)
                            per_sec_rms.append(np.std(seg_detrend))
                            per_sec_times.append(time[start] + 0.5)

                per_sec_rms = np.array(per_sec_rms)
                per_sec_times = np.array(per_sec_times)
                rms_diff_1sec = np.median(per_sec_rms) if len(per_sec_rms) > 0 else np.nan

                tone_600_results = {
                    'present': True,
                    'amp_lsb': amp_lsb,
                    'amp_usb': amp_usb,
                    'amp_lsb_db': amp_lsb_db,
                    'amp_usb_db': amp_usb_db,
                    'tone_valid_mask': tone_valid_mask,
                    'pct_tone_valid': pct_tone_valid,
                    'phase_lsb_rel': phase_lsb_rel,
                    'phase_usb_rel': phase_usb_rel,
                    'phase_usb_lsb_diff': phase_diff_detrend,
                    'rms_usb_lsb_diff_total': rms_diff_total,
                    'rms_usb_lsb_diff_1sec': rms_diff_1sec,
                    'per_sec_rms': per_sec_rms,
                    'per_sec_times': per_sec_times,
                }
                if verbose:
                    print(f"    600 Hz tones: {pct_tone_valid:.0f}% valid, "
                          f"USB-LSB RMS(1s) = {rms_diff_1sec:.4f} rad ({np.degrees(rms_diff_1sec):.2f}°), "
                          f"total = {rms_diff_total:.2f} rad")
            else:
                tone_600_results = {'present': False, 'reason': 'insufficient valid samples'}
                if verbose:
                    print(f"    600 Hz tones: insufficient valid samples ({np.sum(tone_valid_mask)})")
        else:
            tone_600_results = {'present': False, 'reason': 'below threshold'}
            if verbose:
                print(f"    600 Hz tones: not detected (below threshold)")
    except Exception as e:
        tone_600_results = {'present': False, 'error': str(e)}
        if verbose:
            print(f"    600 Hz tones: analysis failed ({e})")

    if verbose:
        print(f"  {freq_mhz} MHz: {pct_valid:.0f}% valid, "
              f"RMS@20ms={rms_20ms:.3f} rad ({np.degrees(rms_20ms):.1f}°)")

    return {
        'filename': filename,
        'freq_mhz': freq_mhz,
        'sample_rate': sample_rate,
        'time': time,
        'f_carrier': f_carrier,
        'amplitude': amplitude,
        'amp_db': amp_db,
        'fade_mask': fade_mask,
        'valid_mask': valid_mask,
        'phase_spliced': phase_spliced,
        'D_tau': D_tau,
        'rms_20ms': rms_20ms,
        'pct_valid': pct_valid,
        'n_splices': len(splice_times),
        'splice_times': splice_times,
        'tone_600': tone_600_results,
    }


# =============================================================================
# MAIN TRIPLET ANALYSIS
# =============================================================================

def analyze_triplet(timestamp, file_dict, make_plot=True, verbose=True):
    """
    Analyze a triplet of synchronous multi-frequency recordings.

    Parameters
    ----------
    timestamp : str
        Triplet timestamp (e.g., "20260101.1044")
    file_dict : dict
        {freq_mhz: filename} mapping
    make_plot : bool
        Generate diagnostic plots
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Complete analysis results including cross-frequency correlations
    """
    # Extract station ID from filename early for logging
    first_filename = list(file_dict.values())[0]
    station_match = re.search(r'\.(\d{5})\.proxy', first_filename)
    if station_match:
        station_id = station_match.group(1)
    else:
        station_match = re.search(r'\.(\d{5})\.', first_filename)
        station_id = station_match.group(1) if station_match else 'unknown'

    if verbose:
        print(f"\n{'='*70}")
        print(f"TRIPLET ANALYSIS: {timestamp} (Station {station_id})")
        print(f"Frequencies: {sorted(file_dict.keys())} MHz")
        print('='*70)

    # Analyze each frequency
    freq_results = {}
    for freq_mhz in sorted(file_dict.keys()):
        filename = file_dict[freq_mhz]
        if verbose:
            print(f"\nAnalyzing {freq_mhz} MHz...")
        freq_results[freq_mhz] = analyze_single_frequency(filename, verbose=verbose)

    freqs = sorted(freq_results.keys())
    sample_rate = freq_results[freqs[0]]['sample_rate']

    # =========================================================================
    # CROSS-FREQUENCY ANALYSIS
    # =========================================================================
    if verbose:
        print(f"\n{'─'*50}")
        print("CROSS-FREQUENCY CORRELATION ANALYSIS")
        print('─'*50)

    xcorr_results = {}
    fade_overlap_results = {}

    # Analyze all pairs
    for i, freq1 in enumerate(freqs):
        for freq2 in freqs[i+1:]:
            pair_key = f"{freq1}-{freq2}"

            # Get amplitude time series (use amp_db for correlation)
            amp1 = freq_results[freq1]['amp_db']
            amp2 = freq_results[freq2]['amp_db']

            # Ensure same length (should be, but check)
            min_len = min(len(amp1), len(amp2))
            amp1 = amp1[:min_len]
            amp2 = amp2[:min_len]

            # Cross-correlation of amplitude (fade correlation)
            lags, xcorr, peak_lag, peak_corr = compute_cross_correlation(
                amp1, amp2, sample_rate
            )
            xcorr_results[pair_key] = {
                'lags': lags,
                'xcorr': xcorr,
                'peak_lag': peak_lag,
                'peak_corr': peak_corr,
            }

            # Fade overlap statistics
            fade1 = freq_results[freq1]['fade_mask'][:min_len]
            fade2 = freq_results[freq2]['fade_mask'][:min_len]
            overlap, f1_frac, f2_frac, cond_prob = compute_fade_overlap(fade1, fade2)
            fade_overlap_results[pair_key] = {
                'overlap_frac': overlap,
                'freq1_fade_frac': f1_frac,
                'freq2_fade_frac': f2_frac,
                'conditional_prob': cond_prob,
            }

            if verbose:
                print(f"\n  {freq1} MHz vs {freq2} MHz:")
                print(f"    Amplitude correlation: r={peak_corr:.3f} at lag={peak_lag*1000:.1f} ms")
                print(f"    Fade fractions: {freq1}MHz={f1_frac*100:.1f}%, {freq2}MHz={f2_frac*100:.1f}%")
                print(f"    Simultaneous fade: {overlap*100:.1f}%")
                print(f"    P(fade@{freq2}|fade@{freq1}): {cond_prob*100:.1f}%")

    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    if make_plot:
        if verbose:
            print(f"\nGenerating plots...")

        n_freqs = len(freqs)
        n_pairs = len(xcorr_results)
        colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red for clarity

        # =====================================================================
        # FIGURE 1: Amplitude time series (separate panels) + histograms
        # =====================================================================
        fig1, axes1 = plt.subplots(n_freqs, 2, figsize=(16, 3*n_freqs),
                                   gridspec_kw={'width_ratios': [3, 1]})
        fig1.suptitle(f'Triplet {timestamp} (Station {station_id}) - Amplitude Analysis', fontsize=14, y=1.02)

        for idx, freq in enumerate(freqs):
            r = freq_results[freq]
            color = colors[idx % len(colors)]

            # Left: Amplitude time series
            ax = axes1[idx, 0]
            ax.plot(r['time'], r['amp_db'], color=color, lw=0.5, alpha=0.8)
            # Mark fade threshold
            threshold = np.percentile(r['amp_db'], 90) - SNR_THRESHOLD_DB
            ax.axhline(threshold, color='red', ls='--', alpha=0.5, lw=1)
            # Shade faded regions
            ax.fill_between(r['time'], ax.get_ylim()[0], ax.get_ylim()[1],
                           where=r['fade_mask'], alpha=0.2, color='red')
            ax.set_ylabel('Amplitude (dB)')
            ax.set_title(f'{freq} MHz - {r["pct_valid"]:.0f}% valid')
            ax.grid(True, alpha=0.3)
            if idx == n_freqs - 1:
                ax.set_xlabel('Time (s)')

            # Right: Amplitude histogram
            ax = axes1[idx, 1]
            ax.hist(r['amp_db'], bins=50, orientation='horizontal', color=color,
                   alpha=0.7, density=True)
            ax.axhline(threshold, color='red', ls='--', alpha=0.5, lw=1,
                      label=f'Fade threshold')
            ax.set_xlabel('Density')
            ax.set_title('Distribution')
            ax.grid(True, alpha=0.3)
            # Match y-limits with time series
            ax.set_ylim(axes1[idx, 0].get_ylim())

        plt.tight_layout()
        output_file1 = f'triplet_{timestamp}_{station_id}_amplitude.png'
        plt.savefig(output_file1, dpi=150, bbox_inches='tight')
        plt.close()

        # =====================================================================
        # FIGURE 2: Fade patterns comparison
        # =====================================================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
        fig2.suptitle(f'Triplet {timestamp} (Station {station_id}) - Fade Analysis', fontsize=14)

        # Top left: Fade timeline (stacked)
        ax = axes2[0, 0]
        for idx, freq in enumerate(freqs):
            r = freq_results[freq]
            color = colors[idx % len(colors)]
            ax.fill_between(r['time'], idx, idx + r['fade_mask'].astype(float),
                           alpha=0.6, color=color, label=f'{freq} MHz')
        ax.set_ylabel('Frequency')
        ax.set_yticks(np.arange(n_freqs) + 0.5)
        ax.set_yticklabels([f'{f} MHz' for f in freqs])
        ax.set_xlabel('Time (s)')
        ax.set_title('Fade Regions (colored = faded)')
        ax.grid(True, alpha=0.3, axis='x')

        # Top right: Fade fraction bar chart
        ax = axes2[0, 1]
        fade_fracs = [100 * np.mean(freq_results[f]['fade_mask']) for f in freqs]
        bars = ax.bar([f'{f} MHz' for f in freqs], fade_fracs, color=colors[:n_freqs])
        ax.set_ylabel('Time Faded (%)')
        ax.set_title('Fade Fraction by Frequency')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, frac in zip(bars, fade_fracs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{frac:.0f}%', ha='center', va='bottom', fontsize=10)

        # Bottom left: Fade duration histograms
        ax = axes2[1, 0]
        for idx, freq in enumerate(freqs):
            r = freq_results[freq]
            color = colors[idx % len(colors)]
            # Find fade durations
            fade_starts = np.where(np.diff(r['fade_mask'].astype(int)) == 1)[0]
            fade_ends = np.where(np.diff(r['fade_mask'].astype(int)) == -1)[0]
            if r['fade_mask'][0]:
                fade_starts = np.insert(fade_starts, 0, 0)
            if r['fade_mask'][-1]:
                fade_ends = np.append(fade_ends, len(r['fade_mask'])-1)
            if len(fade_starts) > 0 and len(fade_ends) > 0:
                # Match starts and ends
                min_len = min(len(fade_starts), len(fade_ends))
                durations = (fade_ends[:min_len] - fade_starts[:min_len]) / sample_rate
                if len(durations) > 0:
                    ax.hist(durations, bins=30, alpha=0.5, color=color,
                           label=f'{freq} MHz (n={len(durations)})', density=True)
        ax.set_xlabel('Fade Duration (s)')
        ax.set_ylabel('Density')
        ax.set_title('Fade Duration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(10, ax.get_xlim()[1]))  # Cap at 10s for readability

        # Bottom right: Conditional probability matrix
        # P(column faded | row faded) - "If row freq is faded, what's probability column is also faded?"
        ax = axes2[1, 1]
        cond_prob_matrix = np.zeros((n_freqs, n_freqs))
        for i, f1 in enumerate(freqs):
            for j, f2 in enumerate(freqs):
                m1 = freq_results[f1]['fade_mask']
                m2 = freq_results[f2]['fade_mask']
                min_len = min(len(m1), len(m2))
                m1, m2 = m1[:min_len], m2[:min_len]
                if i == j:
                    # Diagonal: total fade % for this frequency
                    cond_prob_matrix[i, j] = 100 * np.mean(m1)
                else:
                    # Off-diagonal: P(f2 faded | f1 faded)
                    f1_faded_count = np.sum(m1)
                    if f1_faded_count > 0:
                        cond_prob_matrix[i, j] = 100 * np.sum(m1 & m2) / f1_faded_count
                    else:
                        cond_prob_matrix[i, j] = 0

        im = ax.imshow(cond_prob_matrix, cmap='YlOrRd', vmin=0, vmax=100)
        ax.set_xticks(range(n_freqs))
        ax.set_yticks(range(n_freqs))
        ax.set_xticklabels([f'{f} MHz' for f in freqs])
        ax.set_yticklabels([f'{f} MHz' for f in freqs])
        ax.set_title('P(col faded | row faded)\nDiagonal = total fade %\nExample: row=2.5, col=5 → P(5 faded | 2.5 faded)')
        plt.colorbar(im, ax=ax, label='%')
        # Add text annotations
        for i in range(n_freqs):
            for j in range(n_freqs):
                ax.text(j, i, f'{cond_prob_matrix[i,j]:.0f}%',
                       ha='center', va='center', fontsize=10,
                       color='white' if cond_prob_matrix[i,j] > 50 else 'black')

        plt.tight_layout()
        output_file2 = f'triplet_{timestamp}_{station_id}_fading.png'
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        plt.close()

        # =====================================================================
        # FIGURE 3: Phase and structure functions
        # =====================================================================
        fig3, axes3 = plt.subplots(n_freqs, 2, figsize=(14, 3*n_freqs))
        fig3.suptitle(f'Triplet {timestamp} (Station {station_id}) - Phase Analysis', fontsize=14, y=1.02)

        for idx, freq in enumerate(freqs):
            r = freq_results[freq]
            color = colors[idx % len(colors)]

            # Left: Phase time series
            ax = axes3[idx, 0]
            ax.plot(r['time'][r['valid_mask']], r['phase_spliced'][r['valid_mask']],
                   color=color, lw=0.3)
            # Add vertical red lines at splice locations
            for splice_t in r['splice_times']:
                ax.axvline(splice_t, color='red', alpha=0.5, lw=0.8)
            ax.set_ylabel('Phase (rad)')
            ax.set_title(f'{freq} MHz Phase (spliced, {r["n_splices"]} splices marked in red)')
            ax.grid(True, alpha=0.3)
            if idx == n_freqs - 1:
                ax.set_xlabel('Time (s)')

            # Right: Structure function
            ax = axes3[idx, 1]
            valid_D = ~np.isnan(r['D_tau'])
            ax.loglog(TAU_VALUES[valid_D]*1000, r['D_tau'][valid_D],
                     'o-', color=color, markersize=8)
            ax.axvline(20, color='red', ls='--', alpha=0.7, label='20 ms')
            ax.axhline(0.25, color='green', ls=':', alpha=0.5, label='0.5 rad RMS')
            ax.axhline(1.0, color='orange', ls=':', alpha=0.5, label='1.0 rad RMS')
            ax.set_xlabel('τ (ms)')
            ax.set_ylabel('D(τ) rad²')
            ax.set_title(f'{freq} MHz: RMS@20ms = {r["rms_20ms"]:.3f} rad ({np.degrees(r["rms_20ms"]):.1f}°)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        output_file3 = f'triplet_{timestamp}_{station_id}_phase.png'
        plt.savefig(output_file3, dpi=150, bbox_inches='tight')
        plt.close()

        # =====================================================================
        # FIGURE 4: Amplitude scatter plots (more informative than xcorr)
        # =====================================================================
        fig4, axes4 = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 5))
        if n_pairs == 1:
            axes4 = [axes4]
        fig4.suptitle(f'Triplet {timestamp} (Station {station_id}) - Cross-Frequency Amplitude Comparison', fontsize=14)

        pair_idx = 0
        for i, freq1 in enumerate(freqs):
            for freq2 in freqs[i+1:]:
                ax = axes4[pair_idx]
                amp1 = freq_results[freq1]['amp_db']
                amp2 = freq_results[freq2]['amp_db']
                min_len = min(len(amp1), len(amp2))

                # Subsample for plotting (every 100th point to avoid overplotting)
                step = max(1, min_len // 2000)
                amp1_sub = amp1[:min_len:step]
                amp2_sub = amp2[:min_len:step]

                # Compute Pearson correlation
                r, _ = pearsonr(amp1[:min_len], amp2[:min_len])

                ax.scatter(amp1_sub, amp2_sub, alpha=0.3, s=2, c='blue')
                ax.set_xlabel(f'{freq1} MHz (dB)')
                ax.set_ylabel(f'{freq2} MHz (dB)')
                ax.set_title(f'{freq1} vs {freq2} MHz\nr = {r:.3f}')
                ax.grid(True, alpha=0.3)

                # Add reference line (y=x shifted to means)
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'r--', alpha=0.5, lw=1, label='y=x')
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_aspect('equal')

                pair_idx += 1

        plt.tight_layout()
        output_file4 = f'triplet_{timestamp}_{station_id}_scatter.png'
        plt.savefig(output_file4, dpi=150, bbox_inches='tight')
        plt.close()

        # =====================================================================
        # FIGURE 5: 600 Hz Tone Analysis (if tones are present)
        # =====================================================================
        # Check which frequencies have 600 Hz tones
        freqs_with_tones = [f for f in freqs
                           if freq_results[f]['tone_600'] and freq_results[f]['tone_600'].get('present', False)]

        output_file5 = None
        if freqs_with_tones:
            n_tones = len(freqs_with_tones)
            # 4 columns: phase time series, phase histogram, LSB vs USB amp, tone vs carrier amp
            fig5, axes5 = plt.subplots(n_tones, 4, figsize=(20, 4*n_tones))
            if n_tones == 1:
                axes5 = axes5.reshape(1, -1)
            fig5.suptitle(f'Triplet {timestamp} (Station {station_id}) - 600 Hz Tone Analysis (Dispersion Probe)', fontsize=14, y=1.02)

            for idx, freq in enumerate(freqs_with_tones):
                r = freq_results[freq]
                tone = r['tone_600']
                tone_mask = tone['tone_valid_mask']
                color = colors[freqs.index(freq) % len(colors)]

                # Column 1: Per-second RMS time series (short-term stability)
                ax = axes5[idx, 0]
                if len(tone['per_sec_times']) > 0:
                    ax.plot(tone['per_sec_times'], tone['per_sec_rms'], 'o-',
                           color=color, markersize=3, lw=1)
                    ax.axhline(tone['rms_usb_lsb_diff_1sec'], color='red', ls='--', lw=2,
                              label=f'Median: {tone["rms_usb_lsb_diff_1sec"]:.4f} rad')
                ax.set_ylabel('USB-LSB RMS (rad)')
                ax.set_title(f'{freq} MHz: Per-Second Phase Stability\n'
                            f'Median RMS(1s) = {tone["rms_usb_lsb_diff_1sec"]:.4f} rad = {np.degrees(tone["rms_usb_lsb_diff_1sec"]):.2f}°')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                if idx == n_tones - 1:
                    ax.set_xlabel('Time (s)')

                # Column 2: Histogram of per-second RMS values
                ax = axes5[idx, 1]
                if len(tone['per_sec_rms']) > 0:
                    ax.hist(tone['per_sec_rms'], bins=30, color=color, alpha=0.7, density=True)
                    ax.axvline(tone['rms_usb_lsb_diff_1sec'], color='red', ls='--', lw=2,
                              label=f'Median: {tone["rms_usb_lsb_diff_1sec"]:.4f}')
                    ax.axvline(np.mean(tone['per_sec_rms']), color='orange', ls='-', lw=2,
                              label=f'Mean: {np.mean(tone["per_sec_rms"]):.4f}')
                ax.set_xlabel('Per-Second RMS (rad)')
                ax.set_ylabel('Density')
                ax.set_title(f'1-Second RMS Distribution')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # Column 3: LSB amplitude vs USB amplitude (are tones correlated?)
                ax = axes5[idx, 2]
                # Subsample for plotting
                step = max(1, len(tone['amp_lsb_db']) // 2000)
                ax.scatter(tone['amp_lsb_db'][::step], tone['amp_usb_db'][::step],
                          alpha=0.3, s=2, c='blue', label='All samples')
                # Highlight valid samples
                valid_idx = np.where(tone_mask)[0][::max(1, np.sum(tone_mask)//500)]
                ax.scatter(tone['amp_lsb_db'][valid_idx], tone['amp_usb_db'][valid_idx],
                          alpha=0.5, s=4, c='green', label='Valid')
                # Correlation
                r_tone, _ = pearsonr(tone['amp_lsb_db'], tone['amp_usb_db'])
                ax.set_xlabel('-600 Hz (LSB) Amp (dB)')
                ax.set_ylabel('+600 Hz (USB) Amp (dB)')
                ax.set_title(f'LSB vs USB Amplitude\nr = {r_tone:.3f}')
                ax.legend(fontsize=8, loc='lower right')
                ax.grid(True, alpha=0.3)
                # Equal aspect
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'r--', alpha=0.3, lw=1)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

                # Column 4: Mean tone amplitude vs carrier amplitude
                ax = axes5[idx, 3]
                mean_tone_db = (tone['amp_lsb_db'] + tone['amp_usb_db']) / 2
                carrier_db = r['amp_db']
                ax.scatter(carrier_db[::step], mean_tone_db[::step],
                          alpha=0.3, s=2, c='blue', label='All')
                ax.scatter(carrier_db[valid_idx], mean_tone_db[valid_idx],
                          alpha=0.5, s=4, c='green', label='Tone valid')
                # Correlation
                r_ct, _ = pearsonr(carrier_db, mean_tone_db)
                ax.set_xlabel('Carrier Amp (dB)')
                ax.set_ylabel('Mean Tone Amp (dB)')
                ax.set_title(f'Carrier vs Mean Tone\nr = {r_ct:.3f}')
                ax.legend(fontsize=8, loc='lower right')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file5 = f'triplet_{timestamp}_{station_id}_600Hz.png'
            plt.savefig(output_file5, dpi=150, bbox_inches='tight')
            plt.close()

        if verbose:
            print(f"  Saved: {output_file1}")
            print(f"  Saved: {output_file2}")
            print(f"  Saved: {output_file3}")
            print(f"  Saved: {output_file4}")
            if output_file5:
                print(f"  Saved: {output_file5}")

    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    return {
        'timestamp': timestamp,
        'station_id': station_id,
        'frequencies': freqs,
        'freq_results': freq_results,
        'xcorr_results': xcorr_results,
        'fade_overlap': fade_overlap_results,
    }


def batch_analyze_triplets(date_pattern, make_plots=True):
    """
    Analyze all triplets matching a date pattern.

    Parameters
    ----------
    date_pattern : str
        Date prefix (e.g., "20260101")
    """
    pattern = f'*{date_pattern}*.wav'
    triplets = find_triplets(pattern)

    print(f"\nFound {len(triplets)} triplets for pattern '{date_pattern}'")

    all_results = []
    for timestamp in sorted(triplets.keys()):
        results = analyze_triplet(timestamp, triplets[timestamp],
                                  make_plot=make_plots, verbose=True)
        all_results.append(results)

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("BATCH SUMMARY")
        print('='*70)

        print(f"\n{'Timestamp':<16} ", end='')
        for r in all_results:
            for freq in r['frequencies']:
                print(f"{freq:>8} MHz ", end='')
            break
        print()
        print("-" * (16 + 12 * len(all_results[0]['frequencies'])))

        for r in all_results:
            print(f"{r['timestamp']:<16} ", end='')
            for freq in r['frequencies']:
                rms = r['freq_results'][freq]['rms_20ms']
                print(f"{rms:>8.3f} rad ", end='')
            print()

        # Cross-correlation summary
        print(f"\n{'Timestamp':<16} ", end='')
        for pair in all_results[0]['xcorr_results'].keys():
            print(f"{pair:>12} ", end='')
        print()
        print("-" * (16 + 14 * len(all_results[0]['xcorr_results'])))

        for r in all_results:
            print(f"{r['timestamp']:<16} ", end='')
            for pair, xc in r['xcorr_results'].items():
                print(f"r={xc['peak_corr']:>5.2f} ", end='')
            print()

    return all_results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='WWV Triplet Analysis - Multi-Frequency Synchronous Recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wwv_triplet_analysis.py 20260101.1044    # Analyze specific triplet
  python wwv_triplet_analysis.py                   # Analyze most recent triplet
  python wwv_triplet_analysis.py --batch 20260101  # All triplets from date
        """)

    parser.add_argument('timestamp', nargs='?', default=None,
                        help='Triplet timestamp (e.g., 20260101.1044)')
    parser.add_argument('--batch', metavar='DATE',
                        help='Batch analyze all triplets from date (e.g., 20260101)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')

    args = parser.parse_args()

    if args.batch:
        results = batch_analyze_triplets(args.batch, make_plots=not args.no_plot)
    else:
        triplets = find_triplets('*.wav')
        if args.timestamp:
            # Find matching triplet
            matching = {ts: files for ts, files in triplets.items()
                       if args.timestamp in ts}
            if not matching:
                print(f"No triplet found matching '{args.timestamp}'")
                print(f"Available: {sorted(triplets.keys())}")
                exit(1)
            timestamp = list(matching.keys())[0]
            file_dict = matching[timestamp]
        else:
            # Most recent
            if not triplets:
                print("No triplets found")
                exit(1)
            timestamp = max(triplets.keys())
            file_dict = triplets[timestamp]

        results = analyze_triplet(timestamp, file_dict,
                                  make_plot=not args.no_plot, verbose=True)
