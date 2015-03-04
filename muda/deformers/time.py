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
        self.dispatch['.*'] = self.deform_times
        self.dispatch['tempo'] = self.deform_tempo

    def audio(self, mudabox):
        '''Deform the audio and metadata'''
        mudabox['y'] = pyrb.time_stretch(mudabox['y'], mudabox['sr'],
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


class LogspaceTimeStretch(AbstractTimeStretch):
    '''Logarithmically spaced time stretching'''
    def __init__(self, n_samples, lower, upper):
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

    def get_state(self, jam):
        '''Set the state for the transformation object.'''

        if not len(self._state):
            times = 2.0**np.linspace(self.lower,
                                     self.upper,
                                     num=self.n_samples,
                                     endpoint=True)

            return dict(times=times,
                        index=0,
                        rate=times[0])

        else:
            state = dict()
            state.update(self._state)
            state['index'] = (state['index'] + 1) % len(state['times'])
            state['rate'] = state['times'][state['index']]

            return state


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

    def get_state(self, jam):
        '''Set the state for a transformation object.

        For a random time stretch, this corresponds to sampling
        from the stretch distribution.
        '''

        return dict(rate=np.random.lognormal(mean=self.location,
                                             sigma=self.scale,
                                             size=None))


class AnnotationBlur(BaseTransformer):
    '''Randomly perturb the timing of observations.'''

    def __init__(self, n_samples, mean=0.0, sigma=1.0,
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

        self.dispatch['.*'] = self.deform_annotation

    def get_state(self, jam):
        '''Get the state information from the jam'''

        state = BaseTransformer.get_state(self, jam)

        state['duration'] = librosa.get_duration(y=jam.sandbox.muda['y'],
                                                 sr=jam.sandbox.muda['sr'])

        return state

    def deform_annotation(self, annotation):
        '''Deform the annotation'''

        track_duration = self._state['duration']

        # Get the time in seconds
        t = np.asarray([x.total_seconds() for x in annotation.data.time])
        if self.time:
            # Deform
            t += np.random.normal(loc=self._state['mean'],
                                  scale=self._state['sigma'],
                                  size=t.shape)

        # Clip to the track duration
        t = np.clip(t, 0, track_duration)
        annotation.data.time = pd.to_timedelta(t, unit='s')

        # Get the time in seconds
        d = np.asarray([x.total_seconds() for x in annotation.data.duration])
        if self.duration:
            # Deform
            d += np.random.normal(loc=self._state['mean'],
                                  scale=self._state['sigma'],
                                  size=d.shape)

        # Clip to the track duration - interval start
        d = [np.clip(d_i, 0, track_duration - t_i) for (d_i, t_i) in zip(d, t)]
        annotation.data.duration = pd.to_timedelta(d, unit='s')
