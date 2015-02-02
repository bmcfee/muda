#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 11:07:07 by Brian McFee <brian.mcfee@nyu.edu>
'''Pitch deformation algorithms'''

import librosa
import re
import numpy as np

from ..base import BaseTransformer, IterTransformer

__all__ = ['PitchShift', 'RandomPitchShift']


def transpose(label, n_steps):
    '''Transpose a chord label by some number of semitones
    
    Parameters
    ----------
    label : str
        A chord string

    n_steps : float
        The number of semitones to move `label`

    Returns
    -------
    label_transpose : str
        The transposed chord label

    '''

    # Otherwise, split off the note from the modifier
    match = re.match('(?P<note>[A-G][b#]*)(?P<mod>.*)', label)

    if not match:
        return label

    note = match.group('note')

    new_note = librosa.midi_to_note(librosa.note_to_midi(note) + n_steps,
                                    octave=False)

    return new_note + match.group('mod')


class PitchShift(BaseTransformer):
    '''Static pitch shifting by (fractional) semitones'''
    def __init__(self, n_steps, bins_per_octave=12):
        '''Pitch shifting

        Parameters
        ----------
        n_steps : float
            The number of steps to transpose the signal.
            Can be positive, negative, integral, or fractional.

        bins_per_octave : int > 0
            The number of bins per octave
        '''

        BaseTransformer.__init__(self)

        if bins_per_octave <= 0:
            raise ValueError('bins_per_octave must be strictly positive')

        self.n_steps = float(n_steps)
        self.bins_per_octave = bins_per_octave

        self.dispatch['chord_harte'] = self.deform_chord
        self.dispatch['melody_hz'] = self.deform_frequency

    def audio(self, mudabox, *args):
        '''Deform the audio'''

        # First, estimate the original tuning
        self._state['tuning'] = librosa.estimate_tuning(y=mudabox['y'],
                                                        sr=mudabox['sr'],
                                                        bins_per_octave=self.bins_per_octave)

        mudabox['y'] = librosa.effects.pitch_shift(mudabox['y'],
                                                   mudabox['sr'],
                                                   self.n_steps,
                                                   self.bins_per_octave)

    def deform_frequency(self, annotation):
        '''Deform frequency-valued annotations'''

        shift = 2.0 ** (self.n_steps / self.bins_per_octave)

        annotation.data.value *= shift


    def deform_chord(self, annotation):
        '''Deform chord annotations'''

        if self.bins_per_octave != 12:
            raise RuntimeError('Harte chord deformation only '
                               'defined for bins_per_octave=12')

        # First, figure out the tuning after deformation
        if -0.5 <= (self._state['tuning'] + self.n_steps) < 0.5:
            # If our tuning was off by more than the deformation,
            # then no label modification is necessary
            return

        annotation.data.values = [transpose(l, self.n_steps)
                                  for l in annotation.data.values]


class RandomPitchShift(IterTransformer):
    '''Randomized pitch shifter'''
    def __init__(self, n_samples, mean=0.0, sigma=1.0, bins_per_octave=12):
        '''Randomized pitch shifting.

        Pitch is transposed by a normally distributed random variable.

        Parameters
        ----------
        n_samples : int > 0 or None
            The number of samples to generate per input

        mean : float
        sigma : float > 0
            The parameters of the normal distribution for sampling
            pitch shifts

        bins_per_octave : int > 0
            The number of scale bins per octave
        '''
        IterTransformer.__init__(self, n_samples)

        if bins_per_octave <= 0:
            raise ValueError('bins_per_octave must be strictly positive')
        if sigma <= 0:
            raise ValueError('sigma must be strictly positive')

        self.mean = float(mean)
        self.sigma = float(sigma)
        self.bins_per_octave = bins_per_octave

        self.dispatch['chord_harte'] = self.deform_chord
        self.dispatch['melody_hz'] = self.deform_frequency

    def audio(self, mudabox, *args):
        '''Deform the audio'''

        # Sample the deformation
        self._state['n_steps'] = np.random.normal(loc=self.mean,
                                                  scale=self.sigma,
                                                  size=None)

        # First, estimate the original tuning
        self._state['tuning'] = librosa.estimate_tuning(y=mudabox['y'],
                                                        sr=mudabox['sr'],
                                                        bins_per_octave=self.bins_per_octave)

        mudabox['y'] = librosa.effects.pitch_shift(mudabox['y'],
                                                   mudabox['sr'],
                                                   self._state['n_steps'],
                                                   self.bins_per_octave)

    def deform_frequency(self, annotation):
        '''Deform frequency-valued annotations'''

        n_steps = self._state['n_steps']

        shift = 2.0 ** (n_steps / self.bins_per_octave)

        annotation.data.value *= shift

    def deform_chord(self, annotation):
        '''Deform chord annotations'''

        n_steps = self._state['n_steps']

        if self.bins_per_octave != 12:
            raise RuntimeError('Harte chord deformation only '
                               'defined for bins_per_octave=12')

        # First, figure out the tuning after deformation
        if -0.5 <= (self._state['tuning'] + n_steps) < 0.5:
            # If our tuning was off by more than the deformation,
            # then no label modification is necessary
            return

        # Otherwise, split the chord labels
        # Transpose
        # Rejoin

        annotation.data.values = [transpose(l, n_steps)
                                  for l in annotation.data.values]
