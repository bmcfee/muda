#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import librosa
import numpy as np
import pandas as pd

from ..base import BaseTransformer

__all__ = ['TimeStretch', 'RandomTimeStretch']


class AbstractTimeStretch(BaseTransformer):
    '''Abstract base class for time stretching'''

    def __init__(self):
        '''
        '''
        BaseTransformer.__init__(self)

        # Build the annotation mappers
        self.dispatch['.*'] = self.deform_times
        self.dispatch['tempo'] = self.deform_tempo

    def audio(self, mudabox):
        '''Deform the audio and metadata'''
        mudabox['y'] = librosa.effects.time_stretch(mudabox['y'],
                                                    self._state['rate'])

    def file_metadata(self, metadata):
        '''Deform the metadata'''
        metadata.duration /= self._state['rate']

    def deform_tempo(self, annotation):
        '''Deform a tempo annotation'''

        annotation.data.value *= self._state['rate']

    def deform_times(self, annotation):
        '''Deform time values for all annotations.'''

        annotation.data.time = [pd.to_timedelta(x.total_seconds() / self._state['rate'],
                                                unit='s')
                                for x in annotation.data.time]

        annotation.data.duration = [pd.to_timedelta(x.total_seconds() / self._state['rate'],
                                                    unit='s')
                                    for x in annotation.data.duration]


class TimeStretch(AbstractTimeStretch):
    '''Static time stretching by a fixed rate'''
    def __init__(self, rate):
        '''Time stretching

        Parameters
        ----------
        rate : float > 0
            The rate at which to speedup the audio.

            rate > 1 speeds up,
            rate < 1 slows down.
        '''
        AbstractTimeStretch.__init__(self)

        self.rate = float(rate)
        if rate <= 0:
            raise ValueError('rate parameter must be strictly positive.')


class RandomTimeStretch(AbstractTimeStretch):
    '''Random time stretching'''
    def __init__(self, n_samples, location=0.0, scale=1.0e-1):
        '''Generate randomly stretched examples.

        For each deformation, the rate parameter is drawn from a
        log-normal distribution with parameters `(location, scale)`
        '''

        AbstractTimeStretch.__init__(self)

        if scale <= 0:
            raise ValueError('scale parameter must be strictly positive.')

        if not (n_samples > 0 or n_samples is None):
            raise ValueError('n_samples must be None or positive')

        self.n_samples = n_samples
        self.location = location
        self.scale = scale

        # Build the annotation mappers
        self.dispatch['.*'] = self.deform_times
        self.dispatch['tempo'] = self.deform_tempo

    def get_state(self):
        '''Set the state for a transformation object.

        For a random time stretch, this corresponds to sampling
        from the stretch distribution.
        '''

        return dict(rate=np.random.lognormal(mean=self.location,
                                             sigma=self.scale,
                                             size=None))
