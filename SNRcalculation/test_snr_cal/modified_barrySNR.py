import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.signal import welch

def compute_inner_product(a, b, psd, df=1./0.1, freq_range=None, domain='fourier'):
    N = len(a)
    if domain == 'time':
        a = np.fft.fft(a) / np.sqrt(N)
        b = np.fft.fft(b) / np.sqrt(N)

    freqs_data = np.fft.fftfreq(N, df)
    freqs_psd = np.linspace(0, 1/(2*df), len(psd))
    conj_b = np.conjugate(b)
    psd_interpolation_func = interp1d(freqs_psd, psd, kind='linear', fill_value="extrapolate")

    if freq_range is not None:
        freq_data_indices = np.where((np.abs(freqs_data) > freq_range[0]) & (np.abs(freqs_data) < freq_range[1]))[0]
        freqs_psd_indices = np.where((np.abs(freqs_psd) > freq_range[0]) & (np.abs(freqs_psd) < freq_range[1]))[0]
    else:
        freq_data_indices = np.arange(len(freqs_data))
        freqs_psd_indices = np.arange(len(freqs_psd))

    conj_b = conj_b[freq_data_indices]
    b = b[freq_data_indices]
    psd_interp = psd_interpolation_func(np.abs(freqs_data[freq_data_indices]))

    if len(freqs_psd_indices) > 0:
        psd_interp = np.clip(psd_interp, np.min(psd[freqs_psd_indices]), np.max(psd[freqs_psd_indices]))
    else:
        print("⚠️ freq_range results in empty PSD index set.")

    if np.isnan(psd_interp).any():
        print("⚠️ NaNs in PSD interpolation")

    integrand = a * conj_b / psd_interp * df
    return np.real(np.sum(integrand))

def calculate_snr(signal_ft, template_ft, noise_psd, df=1./0.1, freq_range=None, domain='fourier'):
    return np.abs(compute_inner_product(signal_ft, template_ft, noise_psd, df, freq_range, domain))


