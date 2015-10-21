#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import pyrubberband as pyrb
import numpy as np
import pandas as pd

from ..base import BaseTransformer

__all__ = ['TimeStretch',
           'RandomTimeStretch',
           'LogspaceTimeStretch']


class AbstractTimeStretch(BaseTransformer):
    '''Abstract base class for time stretching'''

    def __init__(self):
        '''Abstract base class for time stretching.

        This contains the deformation functions and
        annotation query mapping, but does not manage
        state or parameters.
        '''
        BaseTransformer.__init__(self)

        # Build the annotation mappers
        self._register('.*', self.deform_times)
        self._register('tempo', self.deform_tempo)

    @staticmethod
    def audio(mudabox, state):
        '''Deform the audio and metadata'''
        mudabox._audio['y'] = pyrb.time_stretch(mudabox._audio['y'],
                                                mudabox._audio['sr'],
                                                state['rate'])

    @staticmethod
    def metadata(metadata, state):
        '''Deform the metadata'''
        metadata.duration /= state['rate']

    @staticmethod
    def deform_tempo(annotation, state):
        '''Deform a tempo annotation'''

        annotation.data.value *= state['rate']

    @staticmethod
    def deform_times(ann, state):
        '''Deform time values for all annotations.'''

        ann.time /= state['rate']
        ann.data.time = [pd.to_timedelta(x.total_seconds() / state['rate'],
                                         unit='s')
                         for x in ann.data.time]

        if ann.duration is not None:
            ann.duration /= state['rate']

        ann.data.duration = [pd.to_timedelta(x.total_seconds() / state['rate'],
                                             unit='s')
                             for x in ann.data.duration]


class TimeStretch(AbstractTimeStretch):
    '''Static time stretching by a fixed rate'''
    def __init__(self, rate=1.2):
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

    def states(self, jam):
        yield dict(rate=self.rate)


class LogspaceTimeStretch(AbstractTimeStretch):
    '''Logarithmically spaced time stretching'''
    def __init__(self, n_samples=3, lower=0.8, upper=1.2):
        '''Generate stretched examples distributed uniformly
        in log-time.

        '''

        AbstractTimeStretch.__init__(self)

        if upper <= lower:
            raise ValueError('upper must be strictly larger than lower')

        if n_samples <= 0:
            raise ValueError('n_samples must be strictly positive')

        self.n_samples = n_samples
        self.lower = float(lower)
        self.upper = float(upper)

    def states(self, jam):
        '''Set the state for the transformation object.'''

        rates = 2.0**np.linspace(self.lower,
                                 self.upper,
                                 num=self.n_samples,
                                 endpoint=True)

        for rate in rates:
            yield dict(rate=rate)


class RandomTimeStretch(AbstractTimeStretch):
    '''Random time stretching'''
    def __init__(self, n_samples=3, location=0.0, scale=1.0e-1):
        '''Generate randomly stretched examples.

        For each deformation, the rate parameter is drawn from a
        log-normal distribution with parameters `(location, scale)`
        '''

        AbstractTimeStretch.__init__(self)

        if scale <= 0:
            raise ValueError('scale parameter must be strictly positive.')

        if n_samples <= 0:
            raise ValueError('n_samples must be strictly positive')

        self.n_samples = n_samples
        self.location = location
        self.scale = scale

    def states(self, jam):
        '''Set the state for a transformation object.

        For a random time stretch, this corresponds to sampling
        from the stretch distribution.
        '''

        rates = np.random.lognormal(mean=self.location,
                                    sigma=self.scale,
                                    size=self.n_samples)

        for rate in rates:
            yield dict(rate=rate)
