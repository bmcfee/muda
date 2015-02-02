#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 11:07:07 by Brian McFee <brian.mcfee@nyu.edu>
'''Pitch deformation algorithms'''

import librosa
import pandas as pd
import re

from ..base import BaseTransformer, IterTransformer

__all__ = ['PitchShift']


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

    def transpose(self, label):
        '''Transpose a chord label'''

        # Otherwise, split off the note from the modifier
        match = re.match('(?P<note>[A-G][b#]*)(?P<mod>.*)', label)

        if not match:
            return label

        note = match.group('note')

        new_note = librosa.midi_to_note(librosa.note_to_midi(note) + self.n_steps,
                                        octave=False)

        return new_note + match.group('mod')

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

        # Otherwise, split the chord labels
        # Transpose
        # Rejoin

        annotation.data.values = [self.transpose(l)
                                  for l in annotation.data.values]
