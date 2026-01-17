#!/usr/bin/env python3
"""
WWV Geographic Correlation Analysis
====================================

PURPOSE:
    Analyze correlations in fading and phase stability between geographically
    separated receivers observing the same WWV frequency simultaneously.

    Key questions:
    1. Are fades correlated between receivers at the same frequency?
    2. Is phase stability correlated between receivers?
    3. How do correlations depend on frequency (2.5, 5, 10 MHz)?

BACKGROUND:
    Two KiwiSDR receivers ~20 miles apart (Cambridge MA and Sudbury MA) observe
    the same WWV transmissions. If ionospheric structures are large compared to
    the receiver separation, fades should be correlated. If structures are small,
    fades will be independent.

    This reveals the spatial coherence scale of ionospheric disturbances,
    critical for understanding whether adaptive correction at one site would
    help nearby sites.

DATA FORMAT:
    Files with same timestamp but different receiver IDs:
    - 20260101.1148.22350.proxy.kiwisdr.com.freq5000.20000.wav  (Cambridge)
    - 20260101.1148.22463.proxy.kiwisdr.com.freq5000.20000.wav  (Sudbury)

USAGE:
    # Analyze all available receiver pairs
    python wwv_geographic_correlation.py

    # Analyze specific timestamp
    python wwv_geographic_correlation.py 20260101.1148

    # Analyze specific date
    python wwv_geographic_correlation.py --date 20260101

AUTHOR:
    Analysis developed with Claude Code, January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, correlate, welch
from scipy.stats import pearsonr
import glob
import os
import re
import argparse
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
T_TRIM_START = 2.0  # Seconds to trim from start
SNR_THRESHOLD_DB = 12.0  # Fade detection threshold
CARRIER_FILTER_BW = 500.0  # Hz - balances preserving phase dynamics vs noise
PHASE_JUMP_THRESHOLD = 1.5  # Radians
FADE_MARGIN_SAMPLES = 100
TAU_VALUES = np.array([0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0])
TAU_ROUNDTRIP = 0.020  # 20 ms roundtrip time

# Receiver locations (approximate)
RECEIVER_INFO = {
    '22350': {'name': 'Cambridge', 'location': 'Cambridge, MA'},
    '22463': {'name': 'Sudbury', 'location': 'Sudbury, MA'},
}
BASELINE_KM = 32  # Approximate distance between receivers

# Cross-correlation settings
XCORR_MAX_LAG_SEC = 10.0  # Maximum lag for cross-correlation


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_wav_file(filename):
    """Load KiwiSDR IQ WAV file."""
    sample_rate, data = wavfile.read(filename)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("WAV must be stereo I/Q")
    z = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)
    return z, sample_rate


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
    """Detect fade regions based on amplitude threshold."""
    amp_db = 20 * np.log10(amplitude + 1e-12)
    ref_db = np.percentile(amp_db, 90)
    threshold = ref_db - threshold_db
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


def compute_cross_correlation(sig1, sig2, sample_rate, max_lag_sec=XCORR_MAX_LAG_SEC):
    """
    Compute normalized cross-correlation between two signals.

    Returns:
        lags: time lags in seconds
        xcorr: normalized cross-correlation values
        peak_lag: lag at maximum correlation
        peak_corr: maximum correlation value
    """
    # Normalize to zero mean, unit variance
    sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-12)
    sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-12)

    # Full cross-correlation
    xcorr_full = correlate(sig1_norm, sig2_norm, mode='full')
    xcorr_full /= len(sig1)

    # Create lag array
    lags_samples = np.arange(-len(sig1)+1, len(sig2))
    lags_sec = lags_samples / sample_rate

    # Trim to max_lag
    max_lag_samples = int(max_lag_sec * sample_rate)
    center = len(sig1) - 1
    trim_start = max(0, center - max_lag_samples)
    trim_end = min(len(xcorr_full), center + max_lag_samples + 1)

    lags = lags_sec[trim_start:trim_end]
    xcorr = xcorr_full[trim_start:trim_end]

    # Find peak
    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]
    peak_corr = xcorr[peak_idx]

    return lags, xcorr, peak_lag, peak_corr


def compute_fade_statistics(fade_mask1, fade_mask2):
    """
    Compute detailed fade correlation statistics between two receivers.

    Returns dict with:
        - fade1_frac: fraction of time receiver 1 is faded
        - fade2_frac: fraction of time receiver 2 is faded
        - both_faded: fraction of time both are faded
        - either_faded: fraction of time at least one is faded
        - neither_faded: fraction of time both are clear
        - correlation: Pearson correlation of fade masks
        - p_2_given_1: P(fade2 | fade1)
        - p_1_given_2: P(fade1 | fade2)
        - jaccard: Jaccard similarity of fade regions
    """
    n = len(fade_mask1)
    f1 = fade_mask1.astype(float)
    f2 = fade_mask2.astype(float)

    fade1_count = np.sum(f1)
    fade2_count = np.sum(f2)
    both_count = np.sum(f1 * f2)
    either_count = np.sum(np.maximum(f1, f2))
    neither_count = n - either_count

    # Pearson correlation of binary fade signals
    if np.std(f1) > 0 and np.std(f2) > 0:
        corr, _ = pearsonr(f1, f2)
    else:
        corr = np.nan

    # Jaccard similarity: intersection / union
    jaccard = both_count / either_count if either_count > 0 else 0

    return {
        'fade1_frac': fade1_count / n,
        'fade2_frac': fade2_count / n,
        'both_faded': both_count / n,
        'either_faded': either_count / n,
        'neither_faded': neither_count / n,
        'correlation': corr,
        'p_2_given_1': both_count / fade1_count if fade1_count > 0 else np.nan,
        'p_1_given_2': both_count / fade2_count if fade2_count > 0 else np.nan,
        'jaccard': jaccard,
    }


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def parse_filename(filename):
    """Extract metadata from filename."""
    basename = os.path.basename(filename)

    # Pattern: YYYYMMDD.HHMM.RXID.proxy.kiwisdr.com.freqXXXXX.SRATE.wav
    ts_match = re.search(r'(\d{8}\.\d{4})\.(\d{5})\.', basename)
    freq_match = re.search(r'freq(\d+)', basename)
    rate_match = re.search(r'freq\d+\.(\d+)\.wav', basename)

    if ts_match and freq_match:
        return {
            'timestamp': ts_match.group(1),
            'receiver': ts_match.group(2),
            'freq_khz': int(freq_match.group(1)),
            'freq_mhz': int(freq_match.group(1)) / 1000,
            'sample_rate': int(rate_match.group(1)) if rate_match else None,
            'filename': filename,
        }
    return None


def find_receiver_pairs(pattern='*.wav'):
    """
    Find pairs of files from different receivers at the same time and frequency.

    Returns:
        dict: {(timestamp, freq_mhz): {receiver_id: filename, ...}, ...}
    """
    files = glob.glob(pattern)
    pairs = defaultdict(dict)

    for f in files:
        meta = parse_filename(f)
        if meta:
            key = (meta['timestamp'], meta['freq_mhz'])
            pairs[key][meta['receiver']] = f

    # Filter to only entries with 2+ receivers
    return {k: v for k, v in pairs.items() if len(v) >= 2}


def find_complete_sets(pattern='*.wav'):
    """
    Find timestamps where both receivers have all three frequencies.

    Returns:
        dict: {timestamp: {receiver: {freq: filename, ...}, ...}, ...}
    """
    files = glob.glob(pattern)
    by_timestamp = defaultdict(lambda: defaultdict(dict))

    for f in files:
        meta = parse_filename(f)
        if meta:
            by_timestamp[meta['timestamp']][meta['receiver']][meta['freq_mhz']] = f

    # Filter to timestamps where at least 2 receivers have 3 frequencies
    complete = {}
    for ts, receivers in by_timestamp.items():
        complete_receivers = {rx: freqs for rx, freqs in receivers.items()
                            if len(freqs) >= 3}
        if len(complete_receivers) >= 2:
            complete[ts] = complete_receivers

    return complete


# =============================================================================
# SINGLE-FILE ANALYSIS
# =============================================================================

def analyze_single_file(filename, verbose=False):
    """
    Analyze a single WAV file and return results.
    """
    z, sample_rate = load_wav_file(filename)

    # Trim start
    start_idx = int(T_TRIM_START * sample_rate)
    z = z[start_idx:]
    time = np.arange(len(z)) / sample_rate

    # Find and extract carrier
    f_carrier = find_carrier_frequency(z, sample_rate, time)
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

    # Compute carrier spectrum using Welch (2^18 samples for good freq resolution)
    nperseg = min(2**18, len(z))
    spec_freqs, spec_psd = welch(z, fs=sample_rate, nperseg=nperseg,
                                  return_onesided=False, scaling='density')
    # Shift to center carrier at 0
    spec_freqs_shifted = spec_freqs - f_carrier

    # Compute fade statistics
    fade_stats = compute_fade_statistics_single(amp_db, fade_mask, threshold, sample_rate)

    if verbose:
        meta = parse_filename(filename)
        print(f"  {meta['freq_mhz']} MHz: {pct_valid:.0f}% valid, RMS@20ms={rms_20ms:.3f} rad")

    return {
        'filename': filename,
        'sample_rate': sample_rate,
        'time': time,
        'amplitude': amplitude,
        'amp_db': amp_db,
        'fade_mask': fade_mask,
        'fade_threshold': threshold,
        'valid_mask': valid_mask,
        'phase_spliced': phase_spliced,
        'splice_times': splice_times,
        'D_tau': D_tau,
        'rms_20ms': rms_20ms,
        'pct_valid': pct_valid,
        'f_carrier': f_carrier,
        'spec_freqs': spec_freqs_shifted,
        'spec_psd': spec_psd,
        'fade_stats': fade_stats,
    }


def compute_fade_statistics_single(amp_db, fade_mask, threshold, sample_rate):
    """
    Compute detailed fade statistics for a single receiver.

    Returns statistics about fade depth, duration, and frequency.
    """
    # Basic amplitude statistics
    amp_mean = np.mean(amp_db)
    amp_std = np.std(amp_db)
    amp_min = np.min(amp_db)
    amp_max = np.max(amp_db)
    amp_median = np.median(amp_db)

    # Percentiles for CDF
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    amp_percentiles = {p: np.percentile(amp_db, p) for p in percentiles}

    # Fade event detection - find contiguous fade regions
    fade_diff = np.diff(fade_mask.astype(int))
    fade_starts = np.where(fade_diff == 1)[0] + 1
    fade_ends = np.where(fade_diff == -1)[0] + 1

    # Handle edge cases
    if fade_mask[0]:
        fade_starts = np.concatenate([[0], fade_starts])
    if fade_mask[-1]:
        fade_ends = np.concatenate([fade_ends, [len(fade_mask)]])

    # Compute fade durations
    if len(fade_starts) > 0 and len(fade_ends) > 0:
        n_fades = min(len(fade_starts), len(fade_ends))
        fade_durations = (fade_ends[:n_fades] - fade_starts[:n_fades]) / sample_rate

        fade_duration_mean = np.mean(fade_durations) if len(fade_durations) > 0 else 0
        fade_duration_median = np.median(fade_durations) if len(fade_durations) > 0 else 0
        fade_duration_max = np.max(fade_durations) if len(fade_durations) > 0 else 0
    else:
        n_fades = 0
        fade_durations = np.array([])
        fade_duration_mean = 0
        fade_duration_median = 0
        fade_duration_max = 0

    # Fade depth - how far below threshold during fades
    if np.any(fade_mask):
        fade_depths = threshold - amp_db[fade_mask]
        fade_depth_mean = np.mean(fade_depths)
        fade_depth_max = np.max(fade_depths)
    else:
        fade_depth_mean = 0
        fade_depth_max = 0

    # Total time statistics
    total_samples = len(fade_mask)
    total_time = total_samples / sample_rate
    faded_time = np.sum(fade_mask) / sample_rate
    faded_fraction = faded_time / total_time

    # Fade rate (fades per minute)
    fade_rate = n_fades / (total_time / 60) if total_time > 0 else 0

    return {
        'amp_mean': amp_mean,
        'amp_std': amp_std,
        'amp_min': amp_min,
        'amp_max': amp_max,
        'amp_median': amp_median,
        'amp_percentiles': amp_percentiles,
        'threshold': threshold,
        'n_fades': n_fades,
        'fade_durations': fade_durations,
        'fade_duration_mean': fade_duration_mean,
        'fade_duration_median': fade_duration_median,
        'fade_duration_max': fade_duration_max,
        'fade_depth_mean': fade_depth_mean,
        'fade_depth_max': fade_depth_max,
        'total_time': total_time,
        'faded_time': faded_time,
        'faded_fraction': faded_fraction,
        'fade_rate': fade_rate,
    }


# =============================================================================
# RECEIVER PAIR ANALYSIS
# =============================================================================

def analyze_receiver_pair(file1, file2, verbose=True):
    """
    Analyze a pair of files from different receivers at the same frequency.

    Parameters
    ----------
    file1, file2 : str
        Paths to WAV files from two receivers

    Returns
    -------
    dict : Analysis results including correlations
    """
    meta1 = parse_filename(file1)
    meta2 = parse_filename(file2)

    rx1_name = RECEIVER_INFO.get(meta1['receiver'], {}).get('name', meta1['receiver'])
    rx2_name = RECEIVER_INFO.get(meta2['receiver'], {}).get('name', meta2['receiver'])

    if verbose:
        print(f"\n  Analyzing {meta1['freq_mhz']} MHz: {rx1_name} vs {rx2_name}")

    # Analyze each file
    r1 = analyze_single_file(file1)
    r2 = analyze_single_file(file2)

    # Ensure same length
    min_len = min(len(r1['amplitude']), len(r2['amplitude']))

    # Amplitude cross-correlation
    amp_lags, amp_xcorr, amp_peak_lag, amp_peak_corr = compute_cross_correlation(
        r1['amp_db'][:min_len], r2['amp_db'][:min_len], r1['sample_rate']
    )

    # Fade statistics
    fade_stats = compute_fade_statistics(
        r1['fade_mask'][:min_len], r2['fade_mask'][:min_len]
    )

    # Phase correlation during jointly valid periods
    joint_valid = r1['valid_mask'][:min_len] & r2['valid_mask'][:min_len]
    if np.sum(joint_valid) > 1000:
        phase1_valid = r1['phase_spliced'][:min_len][joint_valid]
        phase2_valid = r2['phase_spliced'][:min_len][joint_valid]

        # Remove individual trends for fair comparison
        phase1_detrend = phase1_valid - np.polyval(np.polyfit(np.arange(len(phase1_valid)), phase1_valid, 1), np.arange(len(phase1_valid)))
        phase2_detrend = phase2_valid - np.polyval(np.polyfit(np.arange(len(phase2_valid)), phase2_valid, 1), np.arange(len(phase2_valid)))

        phase_corr, _ = pearsonr(phase1_detrend, phase2_detrend)

        # Cross-correlation of phase derivatives (rate of change)
        phase1_diff = np.diff(phase1_detrend)
        phase2_diff = np.diff(phase2_detrend)
        phase_diff_corr, _ = pearsonr(phase1_diff, phase2_diff)
    else:
        phase_corr = np.nan
        phase_diff_corr = np.nan

    if verbose:
        print(f"    {rx1_name}: {r1['pct_valid']:.0f}% valid, RMS@20ms={r1['rms_20ms']:.3f} rad")
        print(f"    {rx2_name}: {r2['pct_valid']:.0f}% valid, RMS@20ms={r2['rms_20ms']:.3f} rad")
        print(f"    Amplitude correlation: r={amp_peak_corr:.3f} at lag={amp_peak_lag*1000:.1f} ms")
        print(f"    Fade correlation: r={fade_stats['correlation']:.3f}, "
              f"Jaccard={fade_stats['jaccard']:.3f}")
        print(f"    Joint valid: {100*np.mean(joint_valid):.0f}%")
        if not np.isnan(phase_corr):
            print(f"    Phase correlation: r={phase_corr:.3f}")
            print(f"    Phase derivative correlation: r={phase_diff_corr:.3f}")

    return {
        'timestamp': meta1['timestamp'],
        'freq_mhz': meta1['freq_mhz'],
        'receiver1': meta1['receiver'],
        'receiver2': meta2['receiver'],
        'rx1_name': rx1_name,
        'rx2_name': rx2_name,
        'r1': r1,
        'r2': r2,
        'min_len': min_len,
        'amp_lags': amp_lags,
        'amp_xcorr': amp_xcorr,
        'amp_peak_lag': amp_peak_lag,
        'amp_peak_corr': amp_peak_corr,
        'fade_stats': fade_stats,
        'joint_valid_frac': np.mean(joint_valid),
        'phase_corr': phase_corr,
        'phase_diff_corr': phase_diff_corr,
    }


# =============================================================================
# FULL TIMESTAMP ANALYSIS
# =============================================================================

def analyze_timestamp(timestamp, receiver_files, make_plot=True, verbose=True):
    """
    Analyze all receiver pairs at a given timestamp across all frequencies.

    Parameters
    ----------
    timestamp : str
        Timestamp identifier (e.g., "20260101.1148")
    receiver_files : dict
        {receiver_id: {freq_mhz: filename, ...}, ...}

    Returns
    -------
    dict : Complete analysis results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"GEOGRAPHIC CORRELATION ANALYSIS: {timestamp}")
        print(f"Receivers: {[RECEIVER_INFO.get(rx, {}).get('name', rx) for rx in receiver_files.keys()]}")
        print(f"Baseline: ~{BASELINE_KM} km")
        print('='*70)

    receivers = sorted(receiver_files.keys())
    frequencies = sorted(set.intersection(*[set(f.keys()) for f in receiver_files.values()]))

    if verbose:
        print(f"Common frequencies: {frequencies} MHz")

    # Analyze each frequency
    pair_results = {}
    for freq in frequencies:
        # Get files from each receiver
        files = {rx: receiver_files[rx][freq] for rx in receivers}

        # For now, compare first two receivers
        rx_list = list(files.keys())
        result = analyze_receiver_pair(files[rx_list[0]], files[rx_list[1]], verbose=verbose)
        pair_results[freq] = result

    # Generate plots
    if make_plot:
        _generate_geographic_plots(timestamp, pair_results, frequencies)

    return {
        'timestamp': timestamp,
        'receivers': receivers,
        'frequencies': frequencies,
        'pair_results': pair_results,
    }


def _generate_geographic_plots(timestamp, pair_results, frequencies):
    """Generate diagnostic plots for geographic correlation analysis."""
    n_freqs = len(frequencies)

    # Get receiver names from first result
    first_result = pair_results[frequencies[0]]
    rx1_name = first_result['rx1_name']
    rx2_name = first_result['rx2_name']

    colors = {'amp1': '#1f77b4', 'amp2': '#d62728', 'fade': '#ff7f0e'}

    # =========================================================================
    # FIGURE 1: Amplitude time series comparison
    # =========================================================================
    fig1, axes1 = plt.subplots(n_freqs, 1, figsize=(16, 3*n_freqs), sharex=True)
    if n_freqs == 1:
        axes1 = [axes1]
    fig1.suptitle(f'Geographic Comparison {timestamp}\n{rx1_name} vs {rx2_name} (~{BASELINE_KM} km baseline)',
                  fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]
        ax = axes1[idx]

        time = r['r1']['time'][:r['min_len']]
        amp1 = r['r1']['amp_db'][:r['min_len']]
        amp2 = r['r2']['amp_db'][:r['min_len']]

        ax.plot(time, amp1, color=colors['amp1'], lw=0.5, alpha=0.7, label=rx1_name)
        ax.plot(time, amp2, color=colors['amp2'], lw=0.5, alpha=0.7, label=rx2_name)

        ax.set_ylabel('Amplitude (dB)')
        ax.set_title(f'{freq} MHz - Amp corr: r={r["amp_peak_corr"]:.3f}, '
                    f'Fade corr: r={r["fade_stats"]["correlation"]:.3f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes1[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_amplitude.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 2: Fade pattern comparison
    # =========================================================================
    fig2, axes2 = plt.subplots(n_freqs, 2, figsize=(14, 3*n_freqs),
                               gridspec_kw={'width_ratios': [3, 1]})
    if n_freqs == 1:
        axes2 = axes2.reshape(1, -1)
    fig2.suptitle(f'Fade Correlation {timestamp} - {rx1_name} vs {rx2_name}', fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]
        time = r['r1']['time'][:r['min_len']]
        fade1 = r['r1']['fade_mask'][:r['min_len']]
        fade2 = r['r2']['fade_mask'][:r['min_len']]

        # Left: Fade timeline
        ax = axes2[idx, 0]
        ax.fill_between(time, 0, fade1.astype(float), alpha=0.5,
                       color=colors['amp1'], label=f'{rx1_name} faded')
        ax.fill_between(time, 1, 1 + fade2.astype(float), alpha=0.5,
                       color=colors['amp2'], label=f'{rx2_name} faded')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([rx1_name, rx2_name])
        ax.set_ylim(-0.1, 2.1)
        ax.set_title(f'{freq} MHz - Jaccard similarity: {r["fade_stats"]["jaccard"]:.3f}')
        ax.grid(True, alpha=0.3, axis='x')
        if idx == n_freqs - 1:
            ax.set_xlabel('Time (s)')

        # Right: Fade statistics
        ax = axes2[idx, 1]
        stats = r['fade_stats']
        categories = ['Both\nfaded', 'Only\n'+rx1_name, 'Only\n'+rx2_name, 'Neither\nfaded']
        both_only = stats['both_faded']
        rx1_only = stats['fade1_frac'] - stats['both_faded']
        rx2_only = stats['fade2_frac'] - stats['both_faded']
        neither = stats['neither_faded']
        values = [both_only, rx1_only, rx2_only, neither]
        colors_bar = ['purple', colors['amp1'], colors['amp2'], 'green']

        bars = ax.bar(categories, [v*100 for v in values], color=colors_bar, alpha=0.7)
        ax.set_ylabel('Time (%)')
        ax.set_title('Fade breakdown')
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val*100:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_fading.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 3: Cross-correlation functions
    # =========================================================================
    fig3, axes3 = plt.subplots(1, n_freqs, figsize=(5*n_freqs, 4))
    if n_freqs == 1:
        axes3 = [axes3]
    fig3.suptitle(f'Amplitude Cross-Correlation {timestamp}', fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]
        ax = axes3[idx]

        ax.plot(r['amp_lags'], r['amp_xcorr'], 'b-', lw=1)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.axvline(r['amp_peak_lag'], color='red', ls='-', alpha=0.7,
                  label=f'Peak: {r["amp_peak_lag"]*1000:.0f} ms')
        ax.axhline(r['amp_peak_corr'], color='red', ls=':', alpha=0.5)

        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'{freq} MHz\nr = {r["amp_peak_corr"]:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_xcorr.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 4: Summary scatter plots
    # =========================================================================
    fig4, axes4 = plt.subplots(1, n_freqs, figsize=(5*n_freqs, 5))
    if n_freqs == 1:
        axes4 = [axes4]
    fig4.suptitle(f'Amplitude Scatter {timestamp} - {rx1_name} vs {rx2_name}', fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]
        ax = axes4[idx]

        amp1 = r['r1']['amp_db'][:r['min_len']]
        amp2 = r['r2']['amp_db'][:r['min_len']]

        # Subsample for plotting
        step = max(1, len(amp1) // 3000)
        ax.scatter(amp1[::step], amp2[::step], alpha=0.3, s=2, c='blue')

        # Correlation
        corr, _ = pearsonr(amp1, amp2)

        ax.set_xlabel(f'{rx1_name} (dB)')
        ax.set_ylabel(f'{rx2_name} (dB)')
        ax.set_title(f'{freq} MHz\nr = {corr:.3f}')
        ax.grid(True, alpha=0.3)

        # Reference line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 5: Phase correlation (if available)
    # =========================================================================
    has_phase_data = any(not np.isnan(pair_results[f]['phase_corr']) for f in frequencies)

    if has_phase_data:
        fig5, axes5 = plt.subplots(1, n_freqs, figsize=(5*n_freqs, 4))
        if n_freqs == 1:
            axes5 = [axes5]
        fig5.suptitle(f'Phase Correlation {timestamp} (joint valid periods only)', fontsize=14)

        for idx, freq in enumerate(frequencies):
            r = pair_results[freq]
            ax = axes5[idx]

            if np.isnan(r['phase_corr']):
                ax.text(0.5, 0.5, 'Insufficient\njoint valid data',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{freq} MHz')
            else:
                # Get joint valid phase data
                min_len = r['min_len']
                joint_valid = r['r1']['valid_mask'][:min_len] & r['r2']['valid_mask'][:min_len]

                phase1 = r['r1']['phase_spliced'][:min_len][joint_valid]
                phase2 = r['r2']['phase_spliced'][:min_len][joint_valid]

                # Detrend
                phase1 = phase1 - np.polyval(np.polyfit(np.arange(len(phase1)), phase1, 1), np.arange(len(phase1)))
                phase2 = phase2 - np.polyval(np.polyfit(np.arange(len(phase2)), phase2, 1), np.arange(len(phase2)))

                # Subsample
                step = max(1, len(phase1) // 2000)
                ax.scatter(phase1[::step], phase2[::step], alpha=0.3, s=2, c='green')

                ax.set_xlabel(f'{rx1_name} phase (rad)')
                ax.set_ylabel(f'{rx2_name} phase (rad)')
                ax.set_title(f'{freq} MHz\nr = {r["phase_corr"]:.3f}')
                ax.grid(True, alpha=0.3)

            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'geo_{timestamp}_phase.png', dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # FIGURE 6: Phase stability analysis for each station (like wwv_phase_analysis_final)
    # =========================================================================
    # For each frequency, create a multi-panel plot showing phase stability at each station
    for freq in frequencies:
        r = pair_results[freq]
        rx1_name_short = r['rx1_name']
        rx2_name_short = r['rx2_name']

        fig6, axes6 = plt.subplots(4, 2, figsize=(16, 14))
        fig6.suptitle(f'Phase Stability Analysis - {freq} MHz - {timestamp}\n'
                     f'{rx1_name_short} (left) vs {rx2_name_short} (right) | '
                     f'Round-trip time: 20 ms', fontsize=14)

        # Get data for each station
        for col, (rx_data, rx_name, rx_color) in enumerate([
            (r['r1'], rx1_name_short, colors['amp1']),
            (r['r2'], rx2_name_short, colors['amp2'])
        ]):
            time_data = rx_data['time']
            amp_db = rx_data['amp_db']
            phase = rx_data['phase_spliced']
            valid_mask = rx_data['valid_mask']
            fade_mask = rx_data['fade_mask']
            splice_times = rx_data.get('splice_times', [])
            D_tau = rx_data['D_tau']
            rms_20ms = rx_data['rms_20ms']
            pct_valid = rx_data['pct_valid']

            # Row 1: Amplitude with fade regions and splice markers
            ax = axes6[0, col]
            ax.plot(time_data, amp_db, color=rx_color, lw=0.5, alpha=0.7)
            ax.fill_between(time_data, amp_db.min(), amp_db.max(),
                           where=fade_mask, alpha=0.3, color='red')
            for i, st in enumerate(splice_times):
                ax.axvline(st, color='orange', lw=1, alpha=0.7,
                          label='Splice' if i == 0 else '')
            ax.set_ylabel('Amplitude (dB)')
            ax.set_title(f'{rx_name} - {pct_valid:.0f}% valid, {len(splice_times)} splices')
            if splice_times:
                ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Row 2: Phase time series (valid only) with splice markers
            ax = axes6[1, col]
            ax.plot(time_data[valid_mask], phase[valid_mask], color=rx_color, lw=0.3)
            for i, st in enumerate(splice_times):
                ax.axvline(st, color='orange', lw=1, alpha=0.7,
                          label='Splice' if i == 0 else '')
            ax.set_ylabel('Phase (rad)')
            ax.set_title(f'Carrier Phase (spliced) - {len(splice_times)} discontinuities removed')
            if splice_times:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Row 3: Zoomed phase with 20 ms markers
            ax = axes6[2, col]
            t_start, t_end = 10.0, 10.5  # 500 ms window
            zoom_mask = valid_mask & (time_data >= t_start) & (time_data <= t_end)
            if np.any(zoom_mask):
                ax.plot(time_data[zoom_mask], phase[zoom_mask], color=rx_color,
                       lw=1, marker='.', markersize=1)
                for t_line in np.arange(t_start, t_end, TAU_ROUNDTRIP):
                    ax.axvline(t_line, color='red', alpha=0.3, lw=0.5)
            ax.set_ylabel('Phase (rad)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Zoomed Phase (red = {TAU_ROUNDTRIP*1000:.0f} ms intervals)')
            ax.grid(True, alpha=0.3)

            # Row 4: Structure function D(τ)
            ax = axes6[3, col]
            valid_D = ~np.isnan(D_tau)
            if np.any(valid_D):
                ax.loglog(TAU_VALUES[valid_D]*1000, D_tau[valid_D], 'o-',
                         color=rx_color, markersize=8)
            ax.axvline(TAU_ROUNDTRIP*1000, color='red', ls='--', lw=2,
                      label=f'{TAU_ROUNDTRIP*1000:.0f} ms round-trip')
            ax.axhline(0.25, color='green', ls=':', alpha=0.5, label='0.5 rad RMS')
            ax.axhline(1.0, color='orange', ls=':', alpha=0.5, label='1.0 rad RMS')
            ax.set_xlabel('τ (ms)')
            ax.set_ylabel('D(τ) (rad²)')

            # Assessment
            if not np.isnan(rms_20ms):
                if rms_20ms < 0.5:
                    assessment = 'GOOD'
                elif rms_20ms < 1.0:
                    assessment = 'MARGINAL'
                else:
                    assessment = 'POOR'
                ax.set_title(f'Structure Function | RMS@20ms: {rms_20ms:.3f} rad ({np.degrees(rms_20ms):.1f}°) - {assessment}')
            else:
                ax.set_title('Structure Function (insufficient data)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlim(1, 1000)

        plt.tight_layout()
        plt.savefig(f'geo_{timestamp}_{freq}MHz_phase_stability.png', dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # FIGURE 7: Combined structure function comparison (all freqs, both stations)
    # =========================================================================
    fig7, axes7 = plt.subplots(1, n_freqs, figsize=(5*n_freqs, 5))
    if n_freqs == 1:
        axes7 = [axes7]
    fig7.suptitle(f'Phase Structure Function Comparison - {timestamp}\n'
                 f'{rx1_name} vs {rx2_name}', fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]
        ax = axes7[idx]

        # Plot D(τ) for both stations
        for rx_data, rx_name, rx_color, marker in [
            (r['r1'], r['rx1_name'], colors['amp1'], 'o'),
            (r['r2'], r['rx2_name'], colors['amp2'], 's')
        ]:
            D_tau = rx_data['D_tau']
            rms_20ms = rx_data['rms_20ms']
            valid_D = ~np.isnan(D_tau)
            if np.any(valid_D):
                label = f'{rx_name}: {rms_20ms:.3f} rad' if not np.isnan(rms_20ms) else rx_name
                ax.loglog(TAU_VALUES[valid_D]*1000, D_tau[valid_D], marker+'-',
                         color=rx_color, markersize=6, label=label)

        ax.axvline(TAU_ROUNDTRIP*1000, color='red', ls='--', lw=2, alpha=0.7)
        ax.axhline(0.25, color='green', ls=':', alpha=0.5)
        ax.axhline(1.0, color='orange', ls=':', alpha=0.5)

        ax.set_xlabel('τ (ms)')
        ax.set_ylabel('D(τ) (rad²)')
        ax.set_title(f'{freq} MHz')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1, 1000)

    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_structure_function.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 8: Carrier spectrum near peak (high resolution Welch)
    # =========================================================================
    fig8, axes8 = plt.subplots(n_freqs, 2, figsize=(14, 4*n_freqs))
    if n_freqs == 1:
        axes8 = axes8.reshape(1, -1)
    fig8.suptitle(f'Carrier Power Spectrum - {timestamp}\n'
                 f'{rx1_name} (left) vs {rx2_name} (right) | Welch 2^18 samples',
                 fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]

        for col, (rx_data, rx_name, rx_color) in enumerate([
            (r['r1'], r['rx1_name'], colors['amp1']),
            (r['r2'], r['rx2_name'], colors['amp2'])
        ]):
            ax = axes8[idx, col]
            spec_freqs = rx_data['spec_freqs']
            spec_psd = rx_data['spec_psd']
            f_carrier = rx_data['f_carrier']

            # Plot spectrum in dB, centered on carrier
            spec_db = 10 * np.log10(np.abs(spec_psd) + 1e-20)

            # Sort by frequency for proper plotting
            sort_idx = np.argsort(spec_freqs)
            spec_freqs_sorted = spec_freqs[sort_idx]
            spec_db_sorted = spec_db[sort_idx]

            # Zoom to ±1000 Hz around carrier
            zoom_mask = (spec_freqs_sorted > -1000) & (spec_freqs_sorted < 1000)
            ax.plot(spec_freqs_sorted[zoom_mask], spec_db_sorted[zoom_mask],
                   color=rx_color, lw=0.5)

            # Mark key frequencies
            ax.axvline(0, color='red', ls='-', lw=1, alpha=0.7, label='Carrier')
            ax.axvline(-100, color='green', ls='--', alpha=0.5, label='±100 Hz (BCD)')
            ax.axvline(100, color='green', ls='--', alpha=0.5)
            ax.axvline(-600, color='orange', ls=':', alpha=0.5, label='±600 Hz (tone)')
            ax.axvline(600, color='orange', ls=':', alpha=0.5)

            ax.set_xlabel('Frequency offset from carrier (Hz)')
            ax.set_ylabel('Power (dB)')
            ax.set_title(f'{freq} MHz - {rx_name}\nCarrier at {f_carrier:.1f} Hz')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1000, 1000)

    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 9: Amplitude CDF and fade statistics
    # =========================================================================
    fig9, axes9 = plt.subplots(n_freqs, 2, figsize=(14, 4*n_freqs))
    if n_freqs == 1:
        axes9 = axes9.reshape(1, -1)
    fig9.suptitle(f'Amplitude Distribution & Fade Statistics - {timestamp}\n'
                 f'{rx1_name} vs {rx2_name}', fontsize=14)

    for idx, freq in enumerate(frequencies):
        r = pair_results[freq]

        # Left panel: CDF comparison
        ax = axes9[idx, 0]
        for rx_data, rx_name, rx_color in [
            (r['r1'], r['rx1_name'], colors['amp1']),
            (r['r2'], r['rx2_name'], colors['amp2'])
        ]:
            amp_db = rx_data['amp_db']
            threshold = rx_data['fade_threshold']

            # Compute CDF
            sorted_amp = np.sort(amp_db)
            cdf = np.arange(1, len(sorted_amp) + 1) / len(sorted_amp)

            ax.plot(sorted_amp, cdf * 100, color=rx_color, lw=1.5,
                   label=f'{rx_name} (med={np.median(amp_db):.1f} dB)')
            ax.axvline(threshold, color=rx_color, ls='--', alpha=0.5)

        ax.set_xlabel('Amplitude (dB)')
        ax.set_ylabel('Cumulative Probability (%)')
        ax.set_title(f'{freq} MHz - Amplitude CDF\n(dashed = fade threshold)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # Right panel: Fade statistics text
        ax = axes9[idx, 1]
        ax.axis('off')

        stats_text = f"{freq} MHz Fade Statistics\n" + "="*40 + "\n\n"

        for rx_data, rx_name in [(r['r1'], r['rx1_name']), (r['r2'], r['rx2_name'])]:
            fs = rx_data['fade_stats']
            stats_text += f"{rx_name}:\n"
            stats_text += f"  Amplitude: {fs['amp_mean']:.1f} ± {fs['amp_std']:.1f} dB\n"
            stats_text += f"  Range: {fs['amp_min']:.1f} to {fs['amp_max']:.1f} dB\n"
            stats_text += f"  Fade threshold: {fs['threshold']:.1f} dB\n"
            stats_text += f"  Faded: {fs['faded_fraction']*100:.1f}% ({fs['faded_time']:.1f}s of {fs['total_time']:.1f}s)\n"
            stats_text += f"  Fade events: {fs['n_fades']} ({fs['fade_rate']:.1f}/min)\n"
            if fs['n_fades'] > 0:
                stats_text += f"  Fade duration: {fs['fade_duration_mean']*1000:.0f} ms mean, {fs['fade_duration_max']*1000:.0f} ms max\n"
                stats_text += f"  Fade depth: {fs['fade_depth_mean']:.1f} dB mean, {fs['fade_depth_max']:.1f} dB max\n"
            stats_text += "\n"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'geo_{timestamp}_fade_stats.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # OUTPUT DATA FILE: Tab-delimited with metadata
    # =========================================================================
    output_filename = f'geo_{timestamp}_data.txt'
    with open(output_filename, 'w') as f:
        # Header with metadata
        f.write(f"# WWV Geographic Correlation Analysis\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Receiver 1: {rx1_name}\n")
        f.write(f"# Receiver 2: {rx2_name}\n")
        f.write(f"# Baseline: ~32 km\n")
        f.write(f"# Analysis date: {timestamp[:8]}\n")
        f.write(f"# Filter bandwidth: {CARRIER_FILTER_BW} Hz\n")
        f.write(f"# SNR threshold: {SNR_THRESHOLD_DB} dB\n")
        f.write(f"#\n")

        # Summary table header
        f.write("# SUMMARY BY FREQUENCY\n")
        f.write("Freq_MHz\t")
        f.write("RX1_pct_valid\tRX1_RMS_20ms_rad\tRX1_n_fades\tRX1_fade_rate_per_min\t")
        f.write("RX2_pct_valid\tRX2_RMS_20ms_rad\tRX2_n_fades\tRX2_fade_rate_per_min\t")
        f.write("Amp_corr\tFade_corr\tJaccard\tPhase_corr\n")

        for freq in frequencies:
            r = pair_results[freq]
            fs1 = r['r1']['fade_stats']
            fs2 = r['r2']['fade_stats']

            f.write(f"{freq:.1f}\t")
            f.write(f"{r['r1']['pct_valid']:.1f}\t{r['r1']['rms_20ms']:.4f}\t{fs1['n_fades']}\t{fs1['fade_rate']:.2f}\t")
            f.write(f"{r['r2']['pct_valid']:.1f}\t{r['r2']['rms_20ms']:.4f}\t{fs2['n_fades']}\t{fs2['fade_rate']:.2f}\t")
            f.write(f"{r['amp_peak_corr']:.4f}\t{r['fade_stats']['correlation']:.4f}\t{r['fade_stats']['jaccard']:.4f}\t")
            phase_corr = r.get('phase_corr', np.nan)
            f.write(f"{phase_corr:.4f}\n")

        f.write("\n")

        # Structure function data
        f.write("# STRUCTURE FUNCTION D(tau)\n")
        f.write("# tau_ms = time lag in milliseconds\n")
        f.write("# D_tau = phase structure function in rad^2\n")
        f.write("# RMS = sqrt(D_tau) in radians\n")
        f.write("tau_ms\t")
        for freq in frequencies:
            f.write(f"{freq}MHz_{rx1_name}_D_tau\t{freq}MHz_{rx1_name}_RMS\t")
            f.write(f"{freq}MHz_{rx2_name}_D_tau\t{freq}MHz_{rx2_name}_RMS\t")
        f.write("\n")

        for i, tau in enumerate(TAU_VALUES):
            f.write(f"{tau*1000:.1f}\t")
            for freq in frequencies:
                r = pair_results[freq]
                D1 = r['r1']['D_tau'][i]
                D2 = r['r2']['D_tau'][i]
                rms1 = np.sqrt(D1) if not np.isnan(D1) else np.nan
                rms2 = np.sqrt(D2) if not np.isnan(D2) else np.nan
                f.write(f"{D1:.6f}\t{rms1:.6f}\t{D2:.6f}\t{rms2:.6f}\t")
            f.write("\n")

        f.write("\n")

        # Detailed fade statistics
        f.write("# DETAILED FADE STATISTICS\n")
        f.write("Freq_MHz\tReceiver\t")
        f.write("Amp_mean_dB\tAmp_std_dB\tAmp_min_dB\tAmp_max_dB\t")
        f.write("Threshold_dB\tFaded_pct\tN_fades\tFade_rate_per_min\t")
        f.write("Fade_dur_mean_ms\tFade_dur_max_ms\tFade_depth_mean_dB\tFade_depth_max_dB\n")

        for freq in frequencies:
            r = pair_results[freq]
            for rx_data, rx_name in [(r['r1'], r['rx1_name']), (r['r2'], r['rx2_name'])]:
                fs = rx_data['fade_stats']
                f.write(f"{freq:.1f}\t{rx_name:<9}\t")
                f.write(f"{fs['amp_mean']:.2f}\t{fs['amp_std']:.2f}\t{fs['amp_min']:.2f}\t{fs['amp_max']:.2f}\t")
                f.write(f"{fs['threshold']:.2f}\t{fs['faded_fraction']*100:.2f}\t{fs['n_fades']}\t{fs['fade_rate']:.2f}\t")
                f.write(f"{fs['fade_duration_mean']*1000:.1f}\t{fs['fade_duration_max']*1000:.1f}\t")
                f.write(f"{fs['fade_depth_mean']:.2f}\t{fs['fade_depth_max']:.2f}\n")

        f.write("\n")

        # Amplitude percentiles
        f.write("# AMPLITUDE PERCENTILES (dB)\n")
        f.write("Freq_MHz\tReceiver\tP1\tP5\tP10\tP25\tP50\tP75\tP90\tP95\tP99\n")
        for freq in frequencies:
            r = pair_results[freq]
            for rx_data, rx_name in [(r['r1'], r['rx1_name']), (r['r2'], r['rx2_name'])]:
                fs = rx_data['fade_stats']
                f.write(f"{freq:.1f}\t{rx_name:<9}\t")
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    f.write(f"{fs['amp_percentiles'][p]:.2f}\t")
                f.write("\n")

    print(f"\n  Saved: geo_{timestamp}_amplitude.png")
    print(f"  Saved: geo_{timestamp}_fading.png")
    print(f"  Saved: geo_{timestamp}_xcorr.png")
    print(f"  Saved: geo_{timestamp}_scatter.png")
    if has_phase_data:
        print(f"  Saved: geo_{timestamp}_phase.png")
    for freq in frequencies:
        print(f"  Saved: geo_{timestamp}_{freq}MHz_phase_stability.png")
    print(f"  Saved: geo_{timestamp}_structure_function.png")
    print(f"  Saved: geo_{timestamp}_spectrum.png")
    print(f"  Saved: geo_{timestamp}_fade_stats.png")
    print(f"  Saved: {output_filename}")


# =============================================================================
# BATCH ANALYSIS AND SUMMARY
# =============================================================================

def batch_analyze(date_pattern=None, make_plots=True, verbose=True):
    """
    Analyze all available receiver pairs.

    Parameters
    ----------
    date_pattern : str, optional
        Filter to specific date (e.g., "20260101")
    """
    pattern = f'*{date_pattern}*.wav' if date_pattern else '*.wav'
    complete_sets = find_complete_sets(pattern)

    if not complete_sets:
        print(f"No complete receiver pair sets found for pattern '{pattern}'")
        return None

    print(f"\nFound {len(complete_sets)} timestamps with complete receiver pairs")

    all_results = []
    for timestamp in sorted(complete_sets.keys()):
        results = analyze_timestamp(timestamp, complete_sets[timestamp],
                                   make_plot=make_plots, verbose=verbose)
        all_results.append(results)

    # Generate summary
    if len(all_results) > 1:
        _generate_summary(all_results)

    return all_results


def _generate_summary(all_results):
    """Generate summary plots and statistics across all analyzed timestamps."""
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print('='*70)

    # Collect statistics by frequency
    freq_stats = defaultdict(lambda: {
        'amp_corr': [], 'fade_corr': [], 'jaccard': [],
        'phase_corr': [], 'phase_diff_corr': [],
        'joint_valid': [], 'timestamps': []
    })

    for result in all_results:
        for freq, pair in result['pair_results'].items():
            freq_stats[freq]['amp_corr'].append(pair['amp_peak_corr'])
            freq_stats[freq]['fade_corr'].append(pair['fade_stats']['correlation'])
            freq_stats[freq]['jaccard'].append(pair['fade_stats']['jaccard'])
            freq_stats[freq]['phase_corr'].append(pair['phase_corr'])
            freq_stats[freq]['phase_diff_corr'].append(pair['phase_diff_corr'])
            freq_stats[freq]['joint_valid'].append(pair['joint_valid_frac'])
            freq_stats[freq]['timestamps'].append(result['timestamp'])

    # Print summary table
    print(f"\n{'Frequency':<12} {'Amp Corr':<12} {'Fade Corr':<12} {'Jaccard':<12} {'Phase Corr':<12}")
    print("-" * 60)

    for freq in sorted(freq_stats.keys()):
        stats = freq_stats[freq]
        amp_mean = np.nanmean(stats['amp_corr'])
        fade_mean = np.nanmean(stats['fade_corr'])
        jaccard_mean = np.nanmean(stats['jaccard'])
        phase_mean = np.nanmean(stats['phase_corr'])

        print(f"{freq:<12.1f} {amp_mean:<12.3f} {fade_mean:<12.3f} "
              f"{jaccard_mean:<12.3f} {phase_mean:<12.3f}")

    # Summary figure
    frequencies = sorted(freq_stats.keys())
    n_freqs = len(frequencies)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Geographic Correlation Summary ({len(all_results)} timestamps)\n'
                f'~{BASELINE_KM} km baseline', fontsize=14)

    # Amplitude correlation by frequency
    ax = axes[0, 0]
    for freq in frequencies:
        data = freq_stats[freq]['amp_corr']
        ax.scatter([freq] * len(data), data, alpha=0.5, s=50)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Amplitude Correlation')
    ax.set_title('Amplitude Correlation vs Frequency')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # Fade correlation by frequency
    ax = axes[0, 1]
    for freq in frequencies:
        data = [x for x in freq_stats[freq]['fade_corr'] if not np.isnan(x)]
        if data:
            ax.scatter([freq] * len(data), data, alpha=0.5, s=50)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Fade Correlation')
    ax.set_title('Fade Correlation vs Frequency')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # Jaccard similarity by frequency
    ax = axes[1, 0]
    box_data = [freq_stats[f]['jaccard'] for f in frequencies]
    bp = ax.boxplot(box_data, labels=[f'{f}' for f in frequencies])
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('Fade Overlap (Jaccard) vs Frequency\n(higher = more correlated fading)')
    ax.grid(True, alpha=0.3)

    # Phase correlation by frequency
    ax = axes[1, 1]
    for freq in frequencies:
        data = [x for x in freq_stats[freq]['phase_corr'] if not np.isnan(x)]
        if data:
            ax.scatter([freq] * len(data), data, alpha=0.5, s=50, label=f'{freq} MHz')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Phase Correlation')
    ax.set_title('Phase Correlation vs Frequency\n(joint valid periods)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('geo_correlation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: geo_correlation_summary.png")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='WWV Geographic Correlation Analysis - Compare receivers at same frequency',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wwv_geographic_correlation.py                    # Analyze all available pairs
  python wwv_geographic_correlation.py 20260101.1148      # Specific timestamp
  python wwv_geographic_correlation.py --date 20260101    # All from specific date
  python wwv_geographic_correlation.py --list             # List available pairs
        """)

    parser.add_argument('timestamp', nargs='?', default=None,
                        help='Specific timestamp to analyze (e.g., 20260101.1148)')
    parser.add_argument('--date', metavar='DATE',
                        help='Analyze all pairs from date (e.g., 20260101)')
    parser.add_argument('--list', action='store_true',
                        help='List available receiver pairs')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')

    args = parser.parse_args()

    if args.list:
        complete_sets = find_complete_sets('*.wav')
        print(f"\nAvailable timestamps with complete receiver pairs:")
        print("-" * 60)
        for ts in sorted(complete_sets.keys()):
            receivers = complete_sets[ts]
            rx_names = [RECEIVER_INFO.get(rx, {}).get('name', rx) for rx in receivers.keys()]
            freqs = sorted(set.intersection(*[set(f.keys()) for f in receivers.values()]))
            print(f"  {ts}: {rx_names} @ {freqs} MHz")
    elif args.timestamp:
        # Find matching complete set
        complete_sets = find_complete_sets('*.wav')
        matching = {ts: files for ts, files in complete_sets.items()
                   if args.timestamp in ts}
        if not matching:
            print(f"No complete receiver pair found for '{args.timestamp}'")
            print(f"Available: {sorted(complete_sets.keys())}")
            exit(1)
        timestamp = list(matching.keys())[0]
        analyze_timestamp(timestamp, matching[timestamp],
                         make_plot=not args.no_plot, verbose=True)
    elif args.date:
        batch_analyze(args.date, make_plots=not args.no_plot, verbose=True)
    else:
        # Analyze all available
        batch_analyze(make_plots=not args.no_plot, verbose=True)
