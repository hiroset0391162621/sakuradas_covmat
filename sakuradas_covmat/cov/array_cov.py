#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read and pre-process seismic data.

Todo
----
- Implement the new :meth:`~covseisnet.arraystream.ArrayStream.synchronize` method.
- Implement a :meth:`covseisnet.arraystream.ArrayStream.check_synchronicity` method.
"""

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
    """List-like object of multiple ObsPy trace objects (synchroneous).

    This class is a subclass of the :class:`obspy.core.stream.Stream` class,
    with additional methods pre-processing methods. The main idea is to
    gather traces with an exact same amount of samples per traces in order
    to perform array-procesing methods onto it. The synchronization of the
    different traces objects is not automatic. There are two options for
    synchronizing the stream:

    (1) use the :meth:`~covseisnet.arraystream.ArrayStream.synchronize`

        >>> import covseisnet as cn
        >>> stream = cn.arraystream.read()
        >>> stream.synchronize()

    (2) perform a manual synchronization before turning the stream
        into an :class:`~covseisnet.arraystream.ArrayStream` object with

        >>> import covseisnet as cn
        >>> stream = obspy.read()
        >>> # manually synchronize
        >>> stream = cn.arraystream.ArrayStream(stream)

    Note
    -----
    All the original methods of :class:`obspy.core.stream.Stream` objects
    remain available in :class:`~covseisnet.arraystream.ArrayStream` objects. For more
    information, please visit the ObsPy documentation at
    https://examples.obspy.org.
    """

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
        """Common time vector of the ArrayStream.

        Because the :class:`~covseisnet.arraystream.ArrayStream` is supposed to handle
        traces with exact same number of samples (and sampling frequency), the
        time vector of each traces is supposed to be the same. This function
        only returns the times of one of the traces, accessible from the
        :meth:`obspy.core.trace.Trace.times` method.

        Keyword arguments
        -----------------

        station_index: int, optional
            The trace index to extract the time vector from. This has no
            influence on the returned time vector is the traces have indeed
            the same sampling, otherwise, you should consider synchronizing
            the traces first. By default, the first trace is considered.

        **kwargs: dict, optional
            Additional keyword arguments are directly passed to the
            :meth:`obspy.core.trace.Trace.times` (for instance,
            ``type="matplotlib"`` allows to recover matplotlib timestamps
            provided by the :func:`matplotlib.dates.date2num` function.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`list`.
            An array of timestamps in a :class:`numpy.ndarray` or in a
            :class:`list`.

        Note
        ----
        If the times are not synchroneous, the time vector will only correspond
        to the times of the trace indexed with ``station_index``. The user
        should ensure that the traces are synchroneous first.

        Tip
        ---
        In order to extract times in matplotlib format, you can set the
        ``type`` parameter of the
        :meth:`~obspy.core.trace.Trace.times` method such as

        >>> import covseisnet as cn
        >>> st = cn.arraystream.read()
        >>> st.times(type='matplotlib')
        array([ 733643.01392361,  733643.01392373,  733643.01392384, ...,
        733643.01427049,  733643.0142706 ,  733643.01427072])
        """
        return self[0].times(**kwargs)


def read(pathname_or_url=None, **kwargs):
    """Read seismic waveforms files into an ArrayStream object.

    This function uses the :func:`obspy.core.stream.read` function to read
    the streams. A detailed list of arguments and options are available at
    https://docs.obspy.org. This function opens either one or multiple
    waveform files given via file name or URL using the ``pathname_or_url``
    attribute. The format of the waveform file will be automatically detected
    if not given. See the `Supported Formats` section in
    the :func:`obspy.core.stream.read` function.

    This function returns an :class:`~covseisnet.arraystream.ArrayStream` object, an
    object directly inherited from the :class:`obspy.core.stream.Stream`
    object.

    Keyword arguments
    -----------------
    pathname_or_url: str or io.BytesIO or None
        String containing a file name or a URL or a open file-like object.
        Wildcards are allowed for a file name. If this attribute is omitted,
        an example :class:`~covseisnet.arraystream.ArrayStream` object will be
        returned.

    Other parameters
    ----------------
    **kwargs: dict
        Other parameters are passed to the :func:`obspy.core.stream.read`
        directly.

    Returns
    -------
    :class:`~covseisnet.arraystream.ArrayStream`
        An :class:`~covseisnet.arraystream.ArrayStream` object.

    Example
    -------

    In most cases a filename is specified as the only argument to
    :func:`obspy.core.stream.read`. For a quick start you may omit all
    arguments and ObsPy will create and return a basic example seismogram.
    Further usages of this function can be seen in the ObsPy documentation.

    >>> import covseisnet as cn
    >>> stream = cn.arraystream.read()
    >>> print(stream)
    3 Trace(s) in Stream:
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples

    .. rubric:: _`Further Examples`

    Example waveform files may be retrieved via https://examples.obspy.org.
    """
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

    
def whitening(matsign, Nfft, tau, frebas, frehaut, plot=False):
    """This function takes 1-dimensional *matsign* timeseries array,
    goes to frequency domain using fft, whitens the amplitude of the spectrum
    in frequency domain between *frebas* and *frehaut*
    and returns the whitened fft.

    Parameters
    ----------
    matsign : numpy.ndarray
        Contains the 1D time series to whiten
    Nfft : int
        The number of points to compute the FFT
    tau : int
        The sampling frequency of the `matsign`
    frebas : int
        The lower frequency bound
    frehaut : int
        The upper frequency bound
    plot : bool
        Whether to show a raw plot of the action (default: False)



    Returns
    -------
    data : numpy.ndarray
        The FFT of the input trace, whitened between the frequency bounds
    """
    # if len(matsign)/2 %2 != 0:
        # matsign = np.append(matsign,[0,0])

    if plot:
        plt.subplot(411)
        plt.plot(np.arange(len(matsign)) * tau, matsign)
        plt.xlim(0, len(matsign) * tau)
        plt.title('Input trace')

    Napod = 300

    #freqVec = np.arange(0., Nfft / 2.0) / (tau * (Nfft - 1))
    freqVec = sf.fftfreq(Nfft, 1.0/tau)
    J = np.where((freqVec >= frebas) & (freqVec <= frehaut))[0]
    low = J[0] - Napod
    if low < 0:
        low = 0

    porte1 = J[0]
    porte2 = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = Nfft / 2

    FFTRawSign = sf.fft(matsign, Nfft)


    if plot:
        plt.subplot(412)
        axis = np.arange(len(FFTRawSign))
        plt.plot(axis[1:], np.abs(FFTRawSign[1:]))
        plt.xlim(0, max(axis))
        plt.title('FFTRawSign')

    # Apodisation a gauche en cos2
    FFTRawSign[0:low] *= 0
    FFTRawSign[low:porte1] = np.cos(np.linspace(np.pi / 2., np.pi, porte1 - low)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[low:porte1]))
    # Porte
    FFTRawSign[porte1:porte2] = np.exp(
        1j * np.angle(FFTRawSign[porte1:porte2]))
    # Apodisation a droite en cos2
    FFTRawSign[porte2:high] = np.cos(np.linspace(0., np.pi / 2., high - porte2)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[porte2:high]))

    if low == 0:
        low = 1

    FFTRawSign[-low:] *= 0
    # Apodisation a gauche en cos2
    FFTRawSign[-porte1:-low] = np.cos(np.linspace(0., np.pi / 2., porte1 - low)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[-porte1:-low]))
    # Porte
    FFTRawSign[-porte2:-porte1] = np.exp(
        1j * np.angle(FFTRawSign[-porte2:-porte1]))
    # ~ # Apodisation a droite en cos2
    FFTRawSign[-high:-porte2] = np.cos(np.linspace(np.pi / 2., np.pi, high - porte2)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[-high:-porte2]))

    FFTRawSign[high:-high] *= 0

    FFTRawSign[-1] *= 0.
    if plot:
        plt.subplot(413)
        axis = np.arange(len(FFTRawSign))
        plt.axvline(low, c='g')
        plt.axvline(porte1, c='g')
        plt.axvline(porte2, c='r')
        plt.axvline(high, c='r')

        plt.axvline(Nfft - high, c='r')
        plt.axvline(Nfft - porte2, c='r')
        plt.axvline(Nfft - porte1, c='g')
        plt.axvline(Nfft - low, c='g')

        plt.plot(axis, np.abs(FFTRawSign))
        plt.xlim(0, max(axis))

    wmatsign = np.real(sf.ifft(FFTRawSign))
    del matsign
    if plot:
        plt.subplot(414)
        plt.plot(np.arange(len(wmatsign)) * tau, wmatsign)
        plt.xlim(0, len(wmatsign) * tau)
        plt.show()
    
    return wmatsign



def detrend_spectrum(x, smooth=None, order=None, epsilon=1e-10):
    r"""Smooth modulus spectrum.

    Arugments
    --------
    x: :class:`np.ndarray`
        The spectra to detrend. Must be of shape `(n_frequencies, n_times)`.

    smooth: int
        Smoothing window size in points.

    order: int
        Smoothing order. Please check the :func:`savitzky_golay` function
        for more details.

    Keyword arguments
    -----------------
    epsilon: float, optional
        A regularizer for avoiding zero division.

    Returns
    -------
    The spectrum divided by the smooth modulus spectrum.
    """
    n_frequencies, n_times = x.shape
    for t in range(n_times):
        x_smooth = signal.savgol_filter(np.abs(x[:, t]), smooth, order)
        x[:, t] /= x_smooth + epsilon
    return x


def normalize(stream, method="onebit", smooth_length=11, smooth_order=1, epsilon=1e-10):
    r"""Normalize the seismic traces in temporal domain.

    Considering :math:`x_i(t)` being the seismic trace :math:`x_i(t)`, the
    normalized trace :math:`\tilde{x}_i(t)` is obtained with

    .. math::
        \tilde{x}_i(t) = \frac{x_i(t)}{Fx_i(t) + \epsilon}

    where :math:`Fx` is a characteristic of the trace :math:`x` that
    depends on the ``method`` argument, and :math:`\epsilon > 0` is a
    regularization value to avoid division by 0, set by the ``epsilon``
    keyword argument.

    Keyword arguments
    -----------------
    method : str, optional
        Must be one of "onebit" (default), "mad", or "smooth".

        - "onebit" compress the seismic trace into a series of 0 and 1.
          In this case, :math:`F` is defined as :math:`Fx(t) = |x(t)|`.

        - "mad" normalize each trace by its median absolute deviation.
          In this case, :math:`F` delivers a scalar value defined as
          :math:`Fx(t) = \text{MAD}x(t) =
          \text{median}(|x(t) - \langle x(t)\rangle|)`, where
          :math:`\langle x(t)\rangle)` is the signal's average.

        - "smooth" normalize each trace by a smooth version of its
          envelope. In this case, :math:`F` is obtained from the
          signal's Hilbert envelope.

    smooth_length: int, optional
        If the ``method`` keyword argument is set to "smooth", the
        normalization is performed with the smoothed trace envelopes,
        calculated over a sliding window of `smooth_length` samples.


    smooth_order: int, optional
        If the ``method`` keyword argument is set to "smooth", the
        normalization is performed with the smoothed trace envelopes.
        The smoothing order is set by the ``smooth_order`` parameter.


    epsilon: float, optional
        Regularization parameter in division, set to ``1e-10`` by default.

    """
    
    
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
    
    elif method == "tft":
        
        windL = 240.0
        for windowed_st in stream.slide(window_length=windL, step=windL):   
            
            amp_level = [] 
            for trace in windowed_st:
                
                amp_level.append([ rms(trace.data) + epsilon])

            
            rms_level = np.nanmean(amp_level)
            
            for trace in windowed_st:
                trace.data /= rms_level
        
    


    else:
        raise ValueError("Unknown method {}".format(method))

def rms(y):
    N = len(y)
    y = np.array(y)
    y -= np.nanmean(y)
    rms = np.sqrt( np.sum(y**2) / N )
    return rms

def phase(x):
    r"""Complex phase extraction.

    Given a complex number (or complex-valued array)
    :math:`x = r e^{\imath \phi}`, where :math:`r` is the complex modulus
    and :math:`phi` the complex phase, the function returns the unitary-modulus
    complex number such as

    .. math::

            \tilde{x} = e^{\imath \phi}

    Arguments
    ---------
    x: :class:`np.ndarray`
        The complex-valued data to extract the complex phase from.
    """
    return np.exp(1j * np.angle(x))