#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import librosa
import numpy as np
import pandas as pd

from ..base import BaseTransformer, IterTransformer

__all__ = ['TimeStretch', 'RandomTimeStretch']


def deform_tempo(annotation, rate):
    '''Deform tempo annotation object by a specified rate.

    Parameters
    ----------
    annotation : pyjams.Annotation
        An annotation object for tempo measurements.

    rate : float > 0
        The speedup rate.

    Returns
    -------
    None
        `annotation` is modified in-place.

    Raises
    ------
    ValueError
        if rate <= 0
    '''

    if rate <= 0:
        raise ValueError('Time stretch rate must '
                         'be strictly positive')

    annotation.data.value *= rate


def deform_time(annotation, rate):
    '''Deform annotation object by a specified rate.

    This scales all time and duration values by rate.

    Parameters
    ----------
    annotation : pyjams.Annotation
        An annotation object

    rate : float > 0
        The speedup rate.

    Returns
    -------
    None
        `annotation` is modified in-place.

    Raises
    ------
    ValueError
        if rate <= 0
    '''

    if rate <= 0:
        raise ValueError('Time stretch rate must '
                         'be strictly positive')

    annotation.data.time = [pd.to_timedelta(x.total_seconds() / rate,
                                            unit='s')
                            for x in annotation.data.time]

    annotation.data.duration = [pd.to_timedelta(x.total_seconds() / rate,
                                                unit='s')
                                for x in annotation.data.duration]


def deform_audio(y, rate):
    '''Time-stretch an audio signal by a speedup

    Parameters
    ----------
    y : np.ndarray [shape=(t,)]
        Audio time series

    rate : float > 0
        The speedup rate

    Returns
    -------
    y_speedup : np.ndarray
        The time-stretched audio buffer

    Raises
    ------
    ValueError
        if rate <= 0

    '''

    if rate <= 0:
        raise ValueError('Time stretch rate must '
                         'be strictly positive')

    return librosa.effects.time_stretch(y, rate)


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

    def audio(self, mudabox):
        '''Deform the audio and metadata'''
        mudabox['y'] = deform_audio(mudabox['y'], self.rate)

    def file_metadata(self, metadata):
        '''Deform the metadata'''
        metadata.duration /= self.rate

    def deform_tempo(self, annotation):
        '''Deform a tempo annotation'''

        deform_tempo(annotation, self.rate)

    def deform_times(self, annotation):
        '''Deform time values for all annotations.'''

        deform_time(annotation, self.rate)


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

    def audio(self, mudabox):
        '''Deform the audio and metadata'''

        self._state['rate'] = np.random.lognormal(mean=self.location,
                                                  sigma=self.scale,
                                                  size=None)

        mudabox['y'] = deform_audio(mudabox['y'], self._state['rate'])

    def file_metadata(self, metadata):
        '''Deform the metadata'''
        metadata.duration /= self._state['rate']

    def deform_tempo(self, annotation):
        '''Deform a tempo annotation'''

        deform_tempo(annotation, self._state['rate'])

    def deform_times(self, annotation):
        '''Deform time values for all annotations.'''

        deform_time(annotation, self._state['rate'])
