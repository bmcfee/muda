#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 10:09:43 by Brian McFee <brian.mcfee@nyu.edu>
'''Time stretching deformations'''

import librosa
import pandas as pd

from ..base import BaseTransformer, IterTransformer

__all__ = ['TimeStretch']#, 'RandomTimeStretch']


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

    def audio(self, mudabox, metadata):
        '''Deform the audio and metadata'''
        mudabox['y'] = librosa.effects.time_stretch(mudabox['y'], self.rate)

        #metadata.duration /= self.rate

    def deform_times(self, annotation):
        '''Deform time values for all annotations.'''

        annotation.data.time = [pd.to_timedelta(x.total_seconds() / self.rate,
                                                unit='s')
                                for x in annotation.data.time]

        annotation.data.duration = [pd.to_timedelta(x.total_seconds() / self.rate,
                                                    unit='s')
                                    for x in annotation.data.duration]

