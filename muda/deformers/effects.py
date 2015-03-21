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
        self.n_samples = len(rates)

    def get_state(self, jam):
        '''Build the resampling state'''

        state = BaseTransformer.get_state(self, jam)

        if not len(self._state):
            state['rates'] = list(self.rates)
            state['index'] = 0
        else:
            state.update(self._state)
            state['index'] += 1

        state['resample_rate'] = state['rates'][state['index']]
        return state

    def audio(self, mudabox):
        '''Deform the audio by resampling'''

        sr = self._state['resample_rate']

        y = librosa.resample(mudabox['y'], mudabox['sr'],
                             sr, res_type=self.res_type)
        mudabox['y'] = librosa.resample(y, sr, mudabox['sr'],
                                        res_type=self.res_type)
