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

    def __init__(self, files):
        '''Impulse response filtering

        Parameters
        ----------

        files : str or list of str
            Path to audio files on disk containing the impulse responses

        '''

        if isinstance(files, six.string_types):
            files = [files]

        BaseTransformer.__init__(self)
        self.n_samples = len(files)
        self.files = files

        self.ir_ = []
        self.delay_ = []
        for fname in files:
            self.ir_.append(librosa.load(fname)[0])
            self.delay_.append(mean_group_delay(self.ir_[-1]))

        self.dispatch['.*'] = self.deform_times

    def get_state(self, jam):
        '''Build the ir state'''

        state = BaseTransformer.get_state(self, jam)

        if not len(self._state):
            state['index'] = 0
            state['duration'] = librosa.get_duration(y=jam.sandbox.muda['y'],
                                                     sr=jam.sandbox.muda['sr'])
        else:
            state.update(self._state)
            state['index'] += 1

        return state

    def audio(self, mudabox):
        '''Audio deformation for impulse responses'''
        idx = self._state['index']

        # If the input signal isn't big enough, pad it out first
        n = len(mudabox['y'])
        if n < len(self.ir_[idx]):
            mudabox['y'] = librosa.util.fix_length(mudabox['y'],
                                                   self.ir_[idx])

        mudabox['y'] = scipy.signal.fftconvolve(mudabox['y'],
                                                self.ir_[idx],
                                                mode='same')

        # Trim back to the original duration
        mudabox['y'] = mudabox['y'][:n]

    def deform_times(self, annotation):
        '''Apply group delay for the selected filter'''

        state = self._state
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
