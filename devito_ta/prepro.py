# Got these from:
# https://github.com/mrava87/Devito-fwi/blob/main/devitofwi/preproc/filtering.py
# https://github.com/mrava87/Devito-fwi/blob/main/devitofwi/devito/source.py

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, correlate, freqs, hilbert
from examples.seismic.utils import sources, PointSource


def create_filter(nfilt, fmin, fmax, dt, plotflag=False):
    """Create a Butterworth filter (lowpass, highpass, or bandpass).

    This function designs a digital Butterworth filter using SciPy and returns:
    - analog filter coefficients (b, a) for frequency-response visualization
    - second-order sections (sos) for numerically stable filtering

    The filter type is chosen based on which cutoff frequencies are provided:
    - lowpass:  fmin is None, fmax is not None
    - highpass: fmax is None, fmin is not None
    - bandpass: both fmin and fmax are provided

    Args:
        nfilt (int): Filter order (size/steepness).
        fmin (float | None): Minimum cutoff frequency in Hz. If None, designs a lowpass filter.
        fmax (float | None): Maximum cutoff frequency in Hz. If None, designs a highpass filter.
        dt (float): Sampling interval in seconds.
        plotflag (bool, optional): If True, plots the analog frequency response
            of the filter. Defaults to False.

    Returns:
        tuple:
            - b (numpy.ndarray): Numerator coefficients (analog prototype, for plotting).
            - a (numpy.ndarray): Denominator coefficients (analog prototype, for plotting).
            - sos (numpy.ndarray): Digital filter in second-order sections form (for filtering).
    """
    if fmin is None:
        b, a = butter(nfilt, fmax, 'lowpass', analog=True)
        sos = butter(nfilt, fmax, 'lowpass', fs=1 / dt, output='sos')
    elif fmax is None:
        b, a = butter(nfilt, fmin, 'highpass', analog=True)
        sos = butter(nfilt, fmin, 'highpass', fs=1 / dt, output='sos')
    else:
        b, a = butter(nfilt, [fmin, fmax], 'bandpass', analog=True)
        sos = butter(nfilt, [fmin, fmax], 'bandpass', fs=1 / dt, output='sos')

    if plotflag:
        w, h = freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)), 'k', lw=2)
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(fmax, color='green')  # cutoff frequency

    return b, a, sos


def apply_filter(sos, inp):
    """Apply a digital filter to input data using zero-phase filtering.

    Filtering is performed using `sosfiltfilt`, which applies the filter
    forward and backward to avoid phase distortion.

    Args:
        sos (numpy.ndarray): Second-order sections representation of the filter.
        inp (numpy.ndarray): Input array to filter. Filtering is applied along
            the last axis (`axis=-1`).

    Returns:
        numpy.ndarray: Filtered output array with the same shape as `inp`.
    """
    filtered = sosfiltfilt(sos, inp, axis=-1)
    return filtered


def filter_data(nfilt, fmin, fmax, dt, inp, plotflag=False):
    """Apply a Butterworth filter to a dataset.

    This is a convenience wrapper around `create_filter()` and `apply_filter()`.
    It creates the filter (low/high/bandpass) and returns both the filter
    coefficients and the filtered output.

    Args:
        nfilt (int): Filter order (size/steepness).
        fmin (float | None): Minimum cutoff frequency in Hz.
        fmax (float | None): Maximum cutoff frequency in Hz.
        dt (float): Sampling interval in seconds.
        inp (numpy.ndarray): Input data array (e.g., shape (nx, nt) or (..., nt)).
            Filtering is applied along the last axis (time).
        plotflag (bool, optional): If True, plots filter frequency response.
            Defaults to False.

    Returns:
        tuple:
            - b (numpy.ndarray): Numerator coefficients (analog prototype, for plotting).
            - a (numpy.ndarray): Denominator coefficients (analog prototype, for plotting).
            - sos (numpy.ndarray): Digital filter in SOS form.
            - filtered (numpy.ndarray): Filtered data, same shape as `inp`.
    """
    b, a, sos = create_filter(nfilt, fmin, fmax, dt, plotflag=plotflag)
    filtered = apply_filter(sos, inp)

    return b, a, sos, filtered


class Filter:
    """Manage a sequence of lowpass filters for multiscale processing.

    This class defines a list of filters (typically lowpass) based on a list
    of cutoff frequencies and corresponding filter orders. It is useful for
    multiscale FWI workflows where the data/wavelet is progressively filtered
    from low to higher frequencies.

    Args:
        freqs (list[float]): Cutoff frequencies (Hz) for each filter.
        nfilts (list[int]): Filter orders corresponding to each cutoff frequency.
        dt (float): Sampling interval in seconds.
        plotflag (bool, optional): If True, plots the frequency response of the
            filters when created. Defaults to False.

    Attributes:
        freqs (list[float]): Cutoff frequencies (Hz).
        nfilts (list[int]): Filter orders.
        dt (float): Sampling interval (s).
        plotflag (bool): Whether to plot filter responses.
        filters (list[numpy.ndarray]): List of SOS filters.
    """

    def __init__(self, freqs, nfilts, dt, plotflag=False):
        self.freqs = freqs
        self.nfilts = nfilts
        self.dt = dt
        self.plotflag = plotflag
        self.filters = self._create_filters()

    def _create_filters(self):
        """Create SOS filters for all configured cutoff frequencies.

        Returns:
            list[numpy.ndarray]: List of SOS filters.
        """
        filters = []

        for freq, nfilt in zip(self.freqs, self.nfilts):
            filters.append(create_filter(nfilt, None, freq, self.dt, plotflag=self.plotflag)[-1])
        return filters

    def apply_filter(self, inp, ifilt=0):
        """Apply one of the pre-built filters to input data.

        Args:
            inp (numpy.ndarray): Input array to filter (filtering along last axis).
            ifilt (int, optional): Index of the filter in the internal filter list.
                Defaults to 0.

        Returns:
            numpy.ndarray: Filtered data.
        """
        return apply_filter(self.filters[ifilt], inp)

    def find_optimal_t0(self, inp, pad=400, thresh=1e-2):
        """Estimate padding needed to avoid acausal energy after filtering.

        Filtering (especially zero-phase filtering) can introduce acausal artifacts
        near the start of the signal if padding is insufficient. This method
        estimates the minimal left-padding required so that the filtered signal
        remains effectively causal according to an envelope threshold.

        Args:
            inp (numpy.ndarray): Input 1D signal (e.g., wavelet) to analyze.
            pad (int, optional): Initial symmetric padding size (samples) applied
                before filtering. Defaults to 400.
            thresh (float, optional): Threshold ratio relative to peak envelope
                used to detect onset time. Defaults to 1e-2.

        Returns:
            int: Estimated optimal padding (samples) to use on the left side.
        """
        inppad = np.pad(inp, (pad, pad))
        itmax = np.argmax(np.abs(inppad))
        it0 = np.where(np.abs(inppad[:itmax]) < thresh * np.abs(inppad[itmax]))[0][-1]
        for ifilt in range(len(self.filters)):
            inpfilt = apply_filter(self.filters[ifilt], inppad)
            inpfiltenv = np.abs(hilbert(inpfilt))
            it0filt = np.where(np.abs(inpfiltenv[:itmax]) < thresh * inpfiltenv[itmax])[0][-1]
            it0 = min(it0, it0filt)
        optimalpad = pad - it0
        return optimalpad


class CustomSource(PointSource):
    """Devito PointSource variant with a user-provided wavelet.

    This class extends Devito's `PointSource` by allowing you to pass a custom
    source time function through the keyword argument `wav`.

    Typical usage:
        Create a `CustomSource` and provide `wav` as a 1D numpy array matching
        the source time axis length.

    Args:
        name (str): Symbol name.
        grid (devito.Grid): Computational grid/domain.
        time_range (examples.seismic.source.TimeAxis): Time axis for the source.
        wav (numpy.ndarray): User-provided wavelet, shape (nt,).

    Attributes:
        wav (numpy.ndarray): Stored wavelet provided by the user.
    """

    __rkwargs__ = PointSource.__rkwargs__ + ['wav']

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        """Configure default argument behavior for PointSource construction.

        Notes:
            Sets `npoint` to 1 by default unless overridden.
        """
        kwargs.setdefault('npoint', 1)

        return super().__args_setup__(*args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        """Finalize initialization and inject the user-defined wavelet into data."""
        super().__init_finalize__(*args, **kwargs)

        self.wav = kwargs.get('wav')

        if not self.alias:
            for p in range(kwargs['npoint']):
                self.data[:, p] = self.wavelet

    @property
    def wavelet(self):
        """Return the user-provided wavelet."""
        return self.wav

    def show(self, idx=0, wavelet=None):
        """Plot the wavelet of the specified source point.

        Args:
            idx (int, optional): Index of the source point. Defaults to 0.
            wavelet (numpy.ndarray | Callable | None, optional): Optional wavelet
                override. If None, uses `self.data[:, idx]`. Defaults to None.

        Returns:
            None
        """
        wavelet = wavelet or self.data[:, idx]
        plt.figure()
        plt.plot(self.time_values, wavelet)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()


sources['CustomSource'] = CustomSource
