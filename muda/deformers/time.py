#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import librosa
import numpy as np
import pandas as pd

from ..base import BaseTransformer, IterTransformer

__all__ = ['TimeStretch', 'RandomTimeStretch']


class TimeStretch(BaseTransformer):
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
        BaseTransformer.__init__(self)

        self.rate = float(rate)
        if rate <= 0:
            raise ValueError('rate parameter must be strictly positive.')

        # Build the annotation mappers
        self.dispatch['.*'] = self.deform_times
        self.dispatch['tempo'] = self.deform_tempo

    def audio(self, mudabox, metadata):
        '''Deform the audio and metadata'''
        mudabox['y'] = librosa.effects.time_stretch(mudabox['y'], self.rate)

        #metadata.duration /= self.rate

    def deform_tempo(self, annotation):
        '''Deform a tempo annotation'''

        annotation.data.value *= self.rate

    def deform_times(self, annotation):
        '''Deform time values for all annotations.'''

        annotation.data.time = [pd.to_timedelta(x.total_seconds() / self.rate,
                                                unit='s')
                                for x in annotation.data.time]

        annotation.data.duration = [pd.to_timedelta(x.total_seconds() / self.rate,
                                                    unit='s')
                                    for x in annotation.data.duration]


class RandomTimeStretch(IterTransformer):
    '''Random time stretching'''
    def __init__(self, n_samples, location=0.0, scale=1.0e-1):
        '''Generate randomly stretched examples.

        For each deformation, the rate parameter is drawn from a
        log-normal distribution with parameters `(location, scale)`
        '''

        IterTransformer.__init__(self, n_samples)

        if scale <= 0:
            raise ValueError('scale parameter must be strictly positive.')

        self.location = location
        self.scale = scale

        # Build the annotation mappers
        self.dispatch['.*'] = self.deform_times
        self.dispatch['tempo'] = self.deform_tempo

    def audio(self, mudabox, metadata):
        '''Deform the audio and metadata'''

        self._state['rate'] = np.random.lognormal(mean=self.location,
                                                  sigma=self.scale,
                                                  size=None)

        mudabox['y'] = librosa.effects.time_stretch(mudabox['y'],
                                                    self._state['rate'])

        # metadata.duration /= self._state['rate']

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
