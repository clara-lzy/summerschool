import numpy as np
from scipy import signal

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

def crcbgenqcsig(dataX, snr, qcCoefs):
    """
    Generate a quadratic chirp signal
    
    Parameters:
    -----------
    dataX : array-like
        Vector of time stamps at which the samples of the signal are to be computed
    snr : float
        Matched filtering signal-to-noise ratio
    qcCoefs : array-like
        Vector of three coefficients [a1, a2, a3] that parametrize the phase 
        of the signal: a1*t + a2*t^2 + a3*t^3
    
    Returns:
    --------
    sigVec : numpy.ndarray
        
    """
    # Convert inputs to numpy arrays for element-wise operations
    dataX = np.array(dataX)
    qcCoefs = np.array(qcCoefs)

    # print('dataX:', dataX)
    # print('qcCoefs:', qcCoefs)
    
    # Calculate phase vector: a1*t + a2*t^2 + a3*t^3
    phaseVec = qcCoefs[0] * dataX + qcCoefs[1] * dataX**2 + qcCoefs[2] * dataX**3
    
    # Generate the signal using sine function
    sigVec = np.sin(2 * np.pi * phaseVec)
    
    # Normalize and scale by SNR
    sigVec = snr * sigVec / np.linalg.norm(sigVec)
    
    return sigVec


def statgaussnoisegen(n_samples, psd_vals, fltr_order, samp_freq):
    """
    Generate a realization of stationary Gaussian noise with given 2-sided PSD
    
    Parameters:
    n_samples : int
        Number of samples to generate
    psd_vals : array-like, shape (M, 2)
        Matrix containing frequencies and corresponding PSD values
        First column: frequencies, Second column: PSD values
        Frequencies must start from 0 and end at Fs/2
    fltr_order : int
        Order of the FIR filter to be used
    samp_freq : float
        Sampling frequency
        
    Returns:
    out_noise : ndarray
        Generated colored noise realization
    """
    
    # Design FIR filter with T(f) = square root of target PSD
    freq_vec = psd_vals[:, 0]
    sqrt_psd = np.sqrt(psd_vals[:, 1])

    
    # Design FIR filter using firwin2 
    b = signal.firwin2(fltr_order + 1, freq_vec, sqrt_psd, fs=samp_freq)
    
    # Generate white Gaussian noise realization
    in_noise = np.random.randn(n_samples)
    
    # Pass through the designed filter and scale
    out_noise = np.sqrt(samp_freq) * signal.fftconvolve(in_noise,b, mode='same')
    return out_noise


def innerprod_psd(x_vec, y_vec, samp_freq, psd_vals):
    """
    P = INNERPROD_PSD(X, Y, Fs, Sn)
    
    Calculates the inner product of vectors X and Y for the case of Gaussian
    stationary noise having a specified power spectral density. Sn is a vector
    containing PSD values at the positive frequencies in the DFT of X
    and Y. The sampling frequency of X and Y is Fs.
    
    Parameters:
    -----------
    x_vec : array_like
        First input vector
    y_vec : array_like
        Second input vector
    samp_freq : float
        Sampling frequency of X and Y
    psd_vals : array_like
        PSD values at positive DFT frequencies[0,fs/2]
    
    Returns:
    --------
    inn_prod : float
        Inner product value
        
    Raises:
    -------
    ValueError
        If vectors are not the same length or PSD values don't match expected length
        
    """
    
    x_vec = np.array(x_vec)
    y_vec = np.array(y_vec)
    psd_vals = np.array(psd_vals)

    n_samples = len(x_vec)

    if len(y_vec) != n_samples:
        raise ValueError('Vectors must be of the same length')

    k_nyq = n_samples // 2 +1 

    if len(psd_vals) != k_nyq:
        raise ValueError('PSD values must be specified at positive DFT frequencies')
    
    fft_x = np.fft.fft(x_vec)
    fft_y = np.fft.fft(y_vec)
        
    # We take care of even or odd number of samples when replicating PSD values
    # for negative frequencies
    neg_f_strt = 1 - (n_samples % 2) #even~1,odd~0
    psd_vec_4_norm = np.concatenate([psd_vals, psd_vals[( (k_nyq-1) - neg_f_strt):0:-1]])

    
    data_len = samp_freq * n_samples #duration of the signal in seconds
    inn_prod = (1 / data_len) * np.sum((fft_x / psd_vec_4_norm) * np.conj(fft_y))
    inn_prod = np.real(inn_prod)
    
    return inn_prod




def normsig4psd(sig_vec, samp_freq, psd_vec, snr):
    """
    Normalize a given signal to have a specified SNR in specified noise PSD
    
    Parameters:
    -----------
    sig_vec : array_like
        Signal vector to be normalized to have signal to noise ratio SNR
    samp_freq : float
        Sampling frequency
    psd_vec : array_like
        Noise PSD vector. Should be specified at the positive DFT frequencies[0, fs/2]
        corresponding to the length of sig_vec and sampling frequency samp_freq
    snr : float
        Desired signal to noise ratio
        
    Returns:
    --------
    norm_sig_vec : ndarray
        Normalized signal vector
    norm_fac : float
        Normalization factor
        
    Notes:
    ------
    The PSD should be specified at the positive DFT frequencies corresponding 
    to the length of sig_vec and sampling frequency samp_freq. The normalized 
    signal vector is returned along with the normalization factor.
    
    """
    
    # Convert inputs to numpy arrays
    sig_vec = np.asarray(sig_vec)
    psd_vec = np.asarray(psd_vec)
    
    # PSD length must be commensurate with the length of the signal DFT
    n_samples = len(sig_vec)
    k_nyq = n_samples // 2 +1
    
    if len(psd_vec) != k_nyq:
        raise ValueError('Length of PSD is not correct')
    
    # Norm of signal squared is inner product of signal with itself
    norm_sig_sqrd = innerprod_psd(sig_vec, sig_vec, samp_freq, psd_vec)
    # print('norm_sig_sqrd:', norm_sig_sqrd)
    
    # Normalization factor
    norm_fac = snr / np.sqrt(norm_sig_sqrd)
    
    # Normalize signal to specified SNR
    norm_sig_vec = norm_fac * sig_vec
    
    return norm_sig_vec, norm_fac



def glrtqcsig(time_vec,data_vec, fs,psd_vec,qcCoefs):
    '''

    calculate the GLRT for a quadratic chirp signal with unknown amplitude

    psd_vec: for positive DFT frequencies[0, fs/2]

    qcCoefs : array-like Vector of three coefficients [a1, a2, a3] that parametrize the phase of the quadratic chirp signal: a1*t + a2*t^2 + a3*t^3
    
    '''
    
    # Generate unit norm template
    sig_vec = crcbgenqcsig(time_vec, 1, qcCoefs)
    templateVec, _ = normsig4psd(sig_vec, fs, psd_vec, 1)

    # Calculate inner product of data with unit norm template
    llr = innerprod_psd(data_vec, templateVec, fs, psd_vec)

    return llr**2