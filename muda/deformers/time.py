#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import librosa
import pyrubberband as pyrb
import numpy as np
import pandas as pd

from ..base import BaseTransformer

__all__ = ['TimeStretch',
           'RandomTimeStretch',
           'LogspaceTimeStretch',
           'AnnotationBlur']


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


class AnnotationBlur(BaseTransformer):
    '''Randomly perturb the timing of observations.'''

    def __init__(self, n_samples=3, mean=0.0, sigma=1.0,
                 time=True, duration=False):
        '''Randomly perturb the timing of observations in a JAMS annotation.

        Parameters
        ----------
        n_samples : int > 0 or None
            The number of perturbations to generate

        mean : float
        sigma: float > 0
            The mean and standard deviation of timing noise

        time : bool
        duration : bool
            Whether to perturb `time` or `duration` fields.
            Note that duration fields may change near the end of the track,
            even when `duration` is false, in order to ensure that
            `time + duration <= track_duration`.
        '''

        BaseTransformer.__init__(self)

        if sigma <= 0:
            raise ValueError('sigma must be strictly positive')

        if not (n_samples > 0 or n_samples is None):
            raise ValueError('n_samples must be None or positive')

        self.n_samples = n_samples
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.time = time
        self.duration = duration

        self._register('.*', self.deform_annotation)

    def states(self, jam):
        '''Get the state information from the jam'''

        state = dict()
        mudabox = jam.sandbox.muda
        state['duration'] = librosa.get_duration(y=mudabox._audio['y'],
                                                 sr=mudabox._audio['sr'])
        yield state

    def deform_annotation(self, annotation, state):
        '''Deform the annotation'''

        track_duration = state['duration']

        # Get the time in seconds
        t = np.asarray([x.total_seconds() for x in annotation.data.time])
        if self.time:
            # Deform
            t += np.random.normal(loc=self.mean,
                                  scale=self.sigma,
                                  size=t.shape)

        # Clip to the track duration
        t = np.clip(t, 0, track_duration)
        annotation.data.time = pd.to_timedelta(t, unit='s')

        # Get the time in seconds
        d = np.asarray([x.total_seconds() for x in annotation.data.duration])
        if self.duration:
            # Deform
            d += np.random.normal(loc=self.mean,
                                  scale=self.sigma,
                                  size=d.shape)

        # Clip to the track duration - interval start
        d = [np.clip(d_i, 0, track_duration - t_i) for (d_i, t_i) in zip(d, t)]
        annotation.data.duration = pd.to_timedelta(d, unit='s')
