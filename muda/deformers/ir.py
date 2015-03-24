#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-24 16:10:11 by Brian McFee <brian.mcfee@nyu.edu>
'''Impulse response'''

import librosa
import scipy
import six

from ..base import BaseTransformer


class ImpulseResponse(BaseTransformer):
    '''Impulse response filtering'''

    def __init__(self, files):
        '''Impulse response filtering

        Parameters
        ----------

        files : str or list of str
            Path to audio files on disk containing the impulse responses

        '''

        if isinstance(files, six.string_types):
            files = [files]

        BaseTransformer.__init__(self)
        self.n_samples = len(files)
        self.files = files

        self.ir_ = []
        for fname in files:
            self.ir_.append(librosa.load(fname)[0])
            # This would be a good spot to cache the group delays

    def get_state(self, jam):
        '''Build the ir state'''

        state = BaseTransformer.get_state(self, jam)

        if not len(self._state):
            state['index'] = 0

        else:
            state.update(self._state)
            state['index'] += 1

        return state

    def audio(self, mudabox):
        '''Audio deformation for impulse responses'''
        idx = self._state['index']

        # If the input signal isn't big enough, pad it out first
        n = len(mudabox['y'])
        if n < len(self.ir_[idx]):
            mudabox['y'] = librosa.util.fix_length(mudabox['y'],
                                                   self.ir_[idx])

        mudabox['y'] = scipy.signal.fftconvolve(mudabox['y'],
                                                self.ir_[idx],
                                                mode='same')

        # Trim back to the original duration
        mudabox['y'] = mudabox['y'][:n]
