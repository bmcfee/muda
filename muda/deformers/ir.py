#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-24 16:10:11 by Brian McFee <brian.mcfee@nyu.edu>
'''Impulse response'''

import six
import numpy as np
import pandas as pd
import scipy

import librosa
import jams

from ..base import BaseTransformer


def mean_group_delay(y, sr=22050, n_fft=2048):
    '''Compute the average group delay for a signal

    Parameters
    ----------
    y : np.ndarray
        the signal

    sr : int > 0
        the sampling rate of `y`

    n_fft : int > 0
        the FFT resolution

    Returns
    -------
    mean_delay : float
        The mean group delay of `y` (in seconds)

    '''

    yfft = np.fft.rfft(y, n=n_fft)

    phase = np.unwrap(np.angle(yfft))

    bin_width = 2 * np.pi * sr / float(n_fft)

    delay = - librosa.feature.delta(phase / bin_width, width=3, axis=0)

    return np.mean(delay)


class ImpulseResponse(BaseTransformer):
    '''Impulse response filtering'''

    def __init__(self, files=None, estimate_delay=False):
        '''Impulse response filtering

        Parameters
        ----------

        files : str or list of str
            Path to audio files on disk containing the impulse responses

        '''

        if isinstance(files, six.string_types):
            files = [files]

        BaseTransformer.__init__(self)
        self.files = files
        self.estimate_delay = estimate_delay

        self.ir_ = []
        self.delay_ = []
        for fname in files:
            self.ir_.append(librosa.load(fname)[0])
            self.delay_.append(0.0)
            if estimate_delay:
                self.delay_[-1] = mean_group_delay(self.ir_[-1])

        self._register('.*', self.deform_times)

    def states(self, jam):
        '''Iterate the impulse respones states'''

        state = dict()
        mudabox = jam.sandbox.muda
        state['duration'] = librosa.get_duration(y=mudabox._audio['y'],
                                                 sr=mudabox._audio['sr'])

        for i in range(len(self.ir_)):
            state['index'] = i
            yield state

    def audio(self, mudabox, state):
        '''Audio deformation for impulse responses'''
        idx = state['index']

        # If the input signal isn't big enough, pad it out first
        n = len(mudabox._audio['y'])
        if n < len(self.ir_[idx]):
            mudabox._audio['y'] = librosa.util.fix_length(mudabox._audio['y'],
                                                          self.ir_[idx])

        mudabox._audio['y'] = scipy.signal.fftconvolve(mudabox._audio['y'],
                                                       self.ir_[idx],
                                                       mode='same')

        # Trim back to the original duration
        mudabox._audio['y'] = mudabox._audio['y'][:n]

    def deform_times(self, annotation, state):
        '''Apply group delay for the selected filter'''

        duration = pd.to_timedelta(state['duration'], unit='s')

        idx = state['index']
        delay = self.delay_[idx]
        data = annotation.data

        # Shift by the delay
        data.time = [pd.to_timedelta(x.total_seconds() + delay, unit='s')
                     for x in data.time]

        # Drop anything that fell off the end
        data = data[data['time'] <= duration]

        # And convert back to jamsframe
        annotation.data = jams.JamsFrame.from_dataframe(data)
