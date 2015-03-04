#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 11:07:07 by Brian McFee <brian.mcfee@nyu.edu>
'''Pitch deformation algorithms'''

import librosa
import pyrubberband as pyrb
import re
import numpy as np

from ..base import BaseTransformer

__all__ = ['PitchShift', 'RandomPitchShift', 'LinearPitchShift']


def transpose(label, n_semitones):
    '''Transpose a chord label by some number of semitones

    Parameters
    ----------
    label : str
        A chord string

    n_semitones : float
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

    new_note = librosa.midi_to_note(librosa.note_to_midi(note) + n_semitones,
                                    octave=False)

    return new_note + match.group('mod')


class AbstractPitchShift(BaseTransformer):
    '''Abstract base class for pitch shifting transformations'''

    def __init__(self):
        '''Abstract base class for pitch shifting.

        This implements the deformations, but does not manage state.
        '''

        BaseTransformer.__init__(self)

        # Build the annotation mapping
        self.dispatch['key_mode|chord_harte'] = self.deform_note
        self.dispatch['melody_hz'] = self.deform_frequency

    def get_state(self, jam):
        '''Build the pitch shift state'''

        state = BaseTransformer.get_state(self, jam)

        mudabox = jam.sandbox.muda
        state['tuning'] = librosa.estimate_tuning(y=mudabox['y'],
                                                  sr=mudabox['sr'])

        return state

    def audio(self, mudabox):
        '''Deform the audio'''

        mudabox['y'] = pyrb.pitch_shift(mudabox['y'], mudabox['sr'],
                                        self._state['n_semitones'])

    def deform_frequency(self, annotation):
        '''Deform frequency-valued annotations'''

        annotation.data.value *= 2.0 ** (self._state['n_semitones'] / 12.0)

    def deform_note(self, annotation):
        '''Deform note-valued annotations (chord or key)'''

        # First, figure out the tuning after deformation
        if -0.5 <= (self._state['tuning'] + self._state['n_semitones']) < 0.5:
            # If our tuning was off by more than the deformation,
            # then no label modification is necessary
            return

        annotation.data.values = [transpose(l, self._state['n_semitones'])
                                  for l in annotation.data.values]


class PitchShift(AbstractPitchShift):
    '''Static pitch shifting by (fractional) semitones'''
    def __init__(self, n_semitones):
        '''Pitch shifting

        Parameters
        ----------
        n_semitones : float
            The number of semitones to transpose the signal.
            Can be positive, negative, integral, or fractional.
        '''

        AbstractPitchShift.__init__(self)
        self.n_semitones = float(n_semitones)


class RandomPitchShift(AbstractPitchShift):
    '''Randomized pitch shifter'''
    def __init__(self, n_samples, mean=0.0, sigma=1.0):
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
        '''
        AbstractPitchShift.__init__(self)

        if sigma <= 0:
            raise ValueError('sigma must be strictly positive')

        if not (n_samples > 0 or n_samples is None):
            raise ValueError('n_samples must be None or positive')

        self.n_samples = n_samples
        self.mean = float(mean)
        self.sigma = float(sigma)

    def get_state(self, jam):
        '''Get the randomized state for this transformation instance'''

        state = AbstractPitchShift.get_state(self, jam)

        # Sample the deformation
        state['n_semitones'] = np.random.normal(loc=self.mean,
                                                scale=self.sigma,
                                                size=None)
        return state


class LinearPitchShift(AbstractPitchShift):
    '''Linearly spaced pitch shift generator'''
    def __init__(self, n_samples, lower, upper):
        '''Generate pitch-shifted examples spaced linearly'''

        AbstractPitchShift.__init__(self)

        if upper <= lower:
            raise ValueError('upper must be strictly larger than lower')

        if n_samples <= 0:
            raise ValueError('n_samples must be strictly positive')

        self.n_samples = n_samples
        self.lower = float(lower)
        self.upper = float(upper)

    def get_state(self, jam):
        '''Set the state for the transformation object'''

        if not len(self._state):
            shifts = np.linspace(self.lower,
                                 self.upper,
                                 num=self.n_samples,
                                 endpoint=True)

            return dict(shifts=shifts,
                        index=0,
                        n_semitones=shifts[0])

        else:
            state = dict()
            state.update(self._state)
            state['index'] += 1
            state['n_semitones'] = state['shifts'][state['index']]

            return state
