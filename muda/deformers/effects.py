#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-05 17:11:00 by Brian McFee <brian.mcfee@nyu.edu>
'''Audio effects'''

import collections
import librosa

from ..base import BaseTransformer


class Resample(BaseTransformer):
    '''Resampling deformations'''

    def __init__(self, rates, res_type='sinc_fastest'):
        '''Resampling deformations.

        For each rate parameter `r`, the signal is resampled to `r`
        and back to its original sample rate.


        Parameters
        ----------
        rates : iterable of int
            Sample rates

        res_type : str
            Resampling method (optional)
        '''

        if not isinstance(rates, collections.Iterable):
            raise ValueError('rates must be iterable')

        BaseTransformer.__init__(self)

        self.rates = rates
        self.res_type = res_type

    def states(self, jam):
        for rate in self.rates:
            yield dict(resample_rate=rate)

    def audio(self, mudabox, state):
        '''Deform the audio by resampling'''

        sr = state['resample_rate']

        y = librosa.resample(mudabox._audio['y'], mudabox._audio['sr'],
                             sr, res_type=self.res_type)
        mudabox._audio['y'] = librosa.resample(y, sr, mudabox._audio['sr'],
                                               res_type=self.res_type)
