import numpy as np
from scipy.interpolate import interp1d

def snr_calculation(data_vec, sig_vec, freqs_psd, psd_vec, fs):
    """
    Calculate the SNR of a signal in data given the PSD of the noise.
    Parameters:
    data_vec : array_like
        The data vector containing the signal and noise.
    sig_vec : array_like
        The pure signal vector without noise.
    freqs_psd : array_like
        corresponding frequencies of the PSD values.
    psd_vec : array_like
        The PSD values of the noise with two-sided definition.
    fs : float
        Sampling frequency of the data and signal.
    Returns:    
    snr : float
        The calculated SNR value.

    Notice: 
        1. Corresponding frequencies of the PSD values positively span from 0 to fs/2.
        2. There uses two‐sided PSD definition. So if you use welch method to estimate the PSD, you need to divide the PSD values by 2.
        3. The function assumes the signal and data have the same length and sampling rate.

    Clara Liu, Sept 2025
    """
    # Number of samples
    n_samp = len(data_vec)

    # Attain PSD values at the DFT frequencies(f = k*fs/N, k=0,1,...,N-1)
    freqs_data = np.fft.fftfreq(n_samp, 1/fs) 
    func_interpo_psd = interp1d(freqs_psd, psd_vec, kind='linear', fill_value="extrapolate")
    psd_vec = func_interpo_psd(np.abs(freqs_data))

    # Compute FFTs of data and signal
    fft_sig = np.fft.fft(sig_vec)
    fft_data = np.fft.fft(data_vec)

    # Compute factor from the definition of inner product
    fac_inn_prod = 1 / (fs * n_samp) 

    # Normalize the pure signal in the frequency domain
    fac_sig_norm = np.sqrt(np.real(np.sum(fft_sig * np.conj(fft_sig) / psd_vec) * fac_inn_prod))
    fft_sig = fft_sig / fac_sig_norm

    # Compute the SNR
    snr = np.real(np.sum(fft_data * np.conj(fft_sig)/ psd_vec) * fac_inn_prod)

    return snr