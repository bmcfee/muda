#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import jams
import librosa
import pyrubberband as pyrb
import numpy as np
import pandas as pd

from ..base import BaseTransformer

__all__ = ['TimeStretch',
           'RandomTimeStretch',
           'LogspaceTimeStretch',
           'AnnotationBlur',
           'Splitter']


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

    def audio(self, mudabox, state):
        '''Deform the audio and metadata'''
        mudabox['y'] = pyrb.time_stretch(mudabox['y'], mudabox['sr'],
                                         state['rate'])

    def metadata(self, metadata, state):
        '''Deform the metadata'''
        metadata.duration /= state['rate']

    def deform_tempo(self, annotation, state):
        '''Deform a tempo annotation'''

        annotation.data.value *= state['rate']

    def deform_times(self, ann, state):
        '''Deform time values for all annotations.'''

        ann.data.time = [pd.to_timedelta(x.total_seconds() / state['rate'],
                                         unit='s')
                         for x in ann.data.time]

        ann.data.duration = [pd.to_timedelta(x.total_seconds() / state['rate'],
                                             unit='s')
                             for x in ann.data.duration]


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

    def states(self, jam):
        yield dict(rate=self.rate)


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

    def states(self, jam):
        '''Get the state information from the jam'''

        state = dict()
        state['duration'] = librosa.get_duration(y=jam.sandbox.muda['y'],
                                                 sr=jam.sandbox.muda['sr'])
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


class Splitter(BaseTransformer):
    '''Split a single jams object into multiple small tiles'''

    def __init__(self, duration, stride, min_duration=0.5):
        '''
        Parameters
        ----------
        duration : float > 0
            The (maximum) length (in seconds) of the sampled objects

        stride : float > 0
            The amount (in seconds) to advance between each sample

        min_duration : float >= 0
            The minimum duration to allow.  If the cropped example is too
            small, it will not be generated.

        '''

        BaseTransformer.__init__(self)

        if duration <= 0:
            raise ValueError('duration must be strictly positive')

        if stride <= 0:
            raise ValueError('stride must be strictly positive')

        if min_duration < 0:
            raise ValueError('min_duration must be non-negative')

        self.duration = duration
        self.stride = stride
        self.min_duration = min_duration

        self.dispatch['.*'] = self.crop_times

    def states(self, jam):
        '''Set the state for the transformation object'''

        state = dict()

        mudabox = jam.sandbox.muda

        state['track_duration'] = librosa.get_duration(y=mudabox['y'],
                                                       sr=mudabox['sr'])

        offsets = np.arange(start=0,
                            stop=(state['track_duration'] - self.min_duration),
                            step=self.stride)

        for t in offsets:
            state['offset'] = t
            yield state

    def metadata(self, metadata, state):
        '''Adjust the metadata'''

        metadata.duration = np.minimum(self.duration,
                                       state['track_duration'] - state['offset'])

    def audio(self, mudabox, state):
        '''Crop the audio'''

        offset_idx = int(state['offset'] * mudabox['sr'])
        duration = int(self.duration * mudabox['sr'])

        mudabox['y'] = mudabox['y'][offset_idx:offset_idx + duration]

    def crop_times(self, annotation, state):
        '''Crop the annotation object'''

        # Convert timings to td64
        min_time = pd.to_timedelta(state['offset'], unit='s')
        duration = pd.to_timedelta(self.duration, unit='s')

        # Get all the rows where
        #   min_time <= time + duration
        #   time <= state.offset[state.index] + state.duration
        data = annotation.data
        data = data[data['time'] + data['duration'] >= min_time]
        data = data[data['time'] <= min_time + duration]

        # Move any partially contained intervals up to the feasible range
        #  [   |    )
        #  t   s    t+d1 = s + d2
        #  d2 = d1 + (t - s)
        #      s = max(t, min_time)
        #  d2 = d1 - (max(t, min_time) - t)
        #     = d1 - max(t - t, min_time - t)
        #     = d1 - max(0, min_time - t)
        shift = np.maximum(0, min_time - data['time'])
        data['duration'] -= shift

        # And now reset everything to the new t=0
        # time -= min_time
        data['time'] -= min_time
        data['time'] = data['time'].clip(lower=pd.to_timedelta(0, unit='s'))

        # For any rows with time + duration > self.duration:
        #   [  |   )
        #   t  d2  d1
        # t + d2 <= duration
        # d2 <= duration - t
        # d2 = min(d1, duration - t)
        #   duration = min(duration, self.duration - time)
        data['duration'] = np.minimum(data['duration'],
                                      duration - data['time'])

        data = data.reset_index()
        annotation.data = jams.JamsFrame.from_dataframe(data)
