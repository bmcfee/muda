#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-24 16:10:11 by Brian McFee <brian.mcfee@nyu.edu>
"""Impulse response"""

import six
import numpy as np
import pandas as pd
import scipy

import soundfile as psf
import librosa
import jams

from ..base import BaseTransformer


def median_group_delay(y, sr, n_fft=2048, rolloff_value=-24):
    """Compute the average group delay for a signal

    Parameters
    ----------
    y : np.ndarray
        the signal

    sr : int > 0
        the sampling rate of `y`

    n_fft : int > 0
        the FFT window size

    rolloff_value : int > 0
        If provided, only estimate the groupd delay of the passband that
        above the threshold, which is the rolloff_value below the peak
        on frequency response.

    Returns
    -------
    mean_delay : float
        The mediant group delay of `y` (in seconds)

    """
    if rolloff_value > 0:
        # rolloff_value must be strictly negative
        rolloff_value = -rolloff_value

    # frequency response
    _, h_ir = scipy.signal.freqz(y, a=1, worN=n_fft, whole=False, plot=None)

    # convert to dB(clip function avoids the zero value in log computation)
    power_ir = 20 * np.log10(np.clip(np.abs(h_ir), 1e-8, 1e100))

    # set up threshold for valid range
    threshold = max(power_ir) + rolloff_value

    _, gd_ir = scipy.signal.group_delay((y, 1), n_fft)

    return np.median(gd_ir[power_ir > threshold]) / sr


class IRConvolution(BaseTransformer):
    """Impulse response filtering"""

    def __init__(self, ir_files, n_fft=2048, rolloff_value=-24):
        """Impulse response filtering

        Parameters
        ----------

        ir_files : str or list of str
            Path to audio files on disk containing the impulse responses

        n_fft : int > 0
            FFT window size

        rolloff_value : int > 0
            If provided, only estimate the groupd delay of the passband that
            above threshold which is 'rolloff_value' below the peak
            on frequency response.
            Positive input will be changed to negative
        """

        if isinstance(ir_files, six.string_types):
            ir_files = [ir_files]

        BaseTransformer.__init__(self)
        self.ir_files = ir_files
        self.n_fft = n_fft
        self.rolloff_value = rolloff_value
        self._register(".*", self.deform_times)

    @staticmethod
    def metadata(metadata, state):
        # Extend the annotation time
        metadata.duration += state['ir_groupdelay']

    @staticmethod
    def deform_times(annotation, state):
        # Deform time values for all annotations.

        # Extend the observation duration by the estimated delay
        if annotation.duration is not None:
            annotation.duration += state['ir_groupdelay']

        # Shift all observations forward in time
        for obs in annotation.pop_data():
            annotation.append(
                time=obs.time + state['ir_groupdelay'],
                duration=obs.duration,
                value=obs.value,
                confidence=obs.confidence,
            )

    def states(self, jam):
        mudabox = jam.sandbox.muda

        for fname in self.ir_files:
            # load and resample ir
            y_ir, sr_ir = librosa.load(fname, sr=mudabox._audio["sr"])
            estimated_group_delay = median_group_delay(
                y=y_ir, sr=sr_ir, n_fft=self.n_fft, rolloff_value=self.rolloff_value
            )
            yield dict(filename=fname, ir_groupdelay=estimated_group_delay)

    def audio(self, mudabox, state):
        # Deform the audio
        fname = state["filename"]

        y_ir, sr_ir = librosa.load(fname, sr=mudabox._audio["sr"])

        mudabox._audio["y"] = scipy.signal.convolve(
            mudabox._audio["y"], y_ir, mode="full"
        )
