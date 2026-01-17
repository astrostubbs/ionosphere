#!/usr/bin/env python3
"""Check spectrum for sideband amplitudes"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os

# Load most recent WAV
wav_files = glob.glob('*.wav')
wav_filename = max(wav_files, key=os.path.getmtime)
print(f"Loading: {wav_filename}")
sample_rate, data = wavfile.read(wav_filename)

I = data[:, 0].astype(np.float64)
Q = data[:, 1].astype(np.float64)
z = I + 1j * Q

# Trim start
start_idx = int(2.0 * sample_rate)
z = z[start_idx:]
time = np.arange(len(z)) / sample_rate

# FFT for spectrum
n_fft = 2**18
freqs = np.fft.fftfreq(n_fft, d=1/sample_rate)
fft_data = np.fft.fft(z, n=n_fft)
mag_spec = np.abs(fft_data)**2

# Find carrier peak
idx_peak = np.argmax(mag_spec)
f_carrier = freqs[idx_peak]
print(f"Carrier at: {f_carrier:.2f} Hz")

# Plot spectrum around carrier - two panels: full view and zoomed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: full view showing ±600 Hz sidebands
mask_full = (freqs > f_carrier - 700) & (freqs < f_carrier + 700)
ax1.semilogy(freqs[mask_full], mag_spec[mask_full])
ax1.axvline(f_carrier, color='k', ls='--', lw=0.5, alpha=0.5)
ax1.axvline(f_carrier - 100, color='r', ls='--', lw=0.5, alpha=0.7)
ax1.axvline(f_carrier + 100, color='r', ls='--', lw=0.5, alpha=0.7)
ax1.axvline(f_carrier - 600, color='g', ls='--', lw=0.5, alpha=0.7)
ax1.axvline(f_carrier + 600, color='g', ls='--', lw=0.5, alpha=0.7)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power')
ax1.set_title('Full view: carrier ±700 Hz')
ax1.grid(True, alpha=0.3)

# Right panel: zoomed view around carrier ±150 Hz
mask_zoom = (freqs > f_carrier - 150) & (freqs < f_carrier + 150)
ax2.semilogy(freqs[mask_zoom], mag_spec[mask_zoom])
ax2.axvline(f_carrier, color='k', ls='--', lw=0.5, alpha=0.5, label=f'Carrier: {f_carrier:.1f} Hz')
ax2.axvline(f_carrier - 100, color='r', ls='--', lw=0.5, alpha=0.7, label='±100 Hz (BCD)')
ax2.axvline(f_carrier + 100, color='r', ls='--', lw=0.5, alpha=0.7)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power')
ax2.set_title('Zoomed: carrier ±150 Hz (BCD sidebands)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrum_sidebands.png', dpi=150)
print("Saved: spectrum_sidebands.png")

# Check amplitude at sideband frequencies
print("\nSideband power levels (relative to carrier):")
carrier_power = mag_spec[idx_peak]
for offset in [-600, -100, 0, 100, 600]:
    f_target = f_carrier + offset
    # Search for peak near target frequency
    search_mask = (freqs > f_target - 10) & (freqs < f_target + 10)
    if np.any(search_mask):
        local_peak = np.max(mag_spec[search_mask])
        rel_db = 10*np.log10(local_peak / carrier_power + 1e-12)
        print(f"  {offset:+4d} Hz: {rel_db:+.1f} dB relative to carrier")
