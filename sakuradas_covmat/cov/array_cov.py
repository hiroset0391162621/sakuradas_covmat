import obspy
import numpy as np
#from numba import jit
from functools import partial
from scipy import signal, stats
import scipy.fft
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
import scipy.fftpack as sf

class ArrayStream(obspy.core.stream.Stream):
    

    def __init__(self, *args, **kwargs):
        r"""Subclassing."""
        super(ArrayStream, self).__init__(*args, **kwargs)

    
    def preprocess(self, domain="spectral", **kwargs):
        r"""Pre-process each trace in temporal or spectral domain."""
        kwargs.setdefault("epsilon", 1e-10)
        
        
        if domain == "spectral":
            whiten(self, **kwargs)
        elif domain == "temporal":
            normalize(self, **kwargs)
        else:
            raise ValueError(
                "Invalid preprocessing domain {} - please specify 'spectral' or 'temporal'".format(
                    domain
                )
            )
        pass

    def times(self, station_index=0, **kwargs):
        
        return self[0].times(**kwargs)


def read(pathname_or_url=None, **kwargs):
    
    stream = obspy.read(pathname_or_url, **kwargs)
    stream = ArrayStream(stream)
    return stream


def whiten(
    stream,
    method="onebit",
    window_duration_sec=2,
    smooth_length=11,
    smooth_order=1,
    epsilon=1e-10,
):
    if method == "onebit":
        whiten_method = phase
    elif method == "smooth":
        whiten_method = partial(
            detrend_spectrum, smooth=smooth_length, order=smooth_order, epsilon=epsilon
        )
    else:
        raise ValueError("Unknown method {}".format(method))
    
    fft_size = int(window_duration_sec * stream[0].stats.sampling_rate)
    print('window size for preprocessing [sec]', fft_size/stream[0].stats.sampling_rate)
    for index, trace in enumerate(stream):
        data = trace.data
        if np.nanmax(np.abs(data))>1.0:
            _, _, data_fft = signal.stft(data, nperseg=fft_size)
            data_fft = whiten_method(data_fft)
            _, data = signal.istft(data_fft, nperseg=fft_size)
            trace.data = data
        else:
            trace.data = np.zeros(len(data))*np.nan

    


def detrend_spectrum(x, smooth=None, order=None, epsilon=1e-10):
    
    n_frequencies, n_times = x.shape
    for t in range(n_times):
        x_smooth = signal.savgol_filter(np.abs(x[:, t]), smooth, order)
        x[:, t] /= x_smooth + epsilon
    return x


def normalize(stream, method="onebit", smooth_length=11, smooth_order=1, epsilon=1e-10):
    
    
    if method == "onebit":
        for trace in stream:
            trace.data = np.sign(trace.data) #trace.data / (np.abs(trace.data) + epsilon)

    elif method == "smooth":
        for trace in stream:
            trace_env_smooth = signal.savgol_filter(
                np.abs(trace.data), smooth_length, smooth_order
            )
            trace.data = trace.data / (trace_env_smooth + epsilon)

    elif method == "mad":
        for trace in stream:
            trace.data = trace.data / (
                stats.median_absolute_deviation(trace.data) + epsilon
            )
        

    else:
        raise ValueError("Unknown method {}".format(method))

def rms(y):
    N = len(y)
    y = np.array(y)
    y -= np.nanmean(y)
    rms = np.sqrt( np.sum(y**2) / N )
    return rms

def phase(x):
    
    return np.exp(1j * np.angle(x))