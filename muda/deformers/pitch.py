#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-02 11:07:07 by Brian McFee <brian.mcfee@nyu.edu>
'''Pitch deformation algorithms'''

import librosa
import pyrubberband as pyrb
import re
import numpy as np
import six

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
    match = re.match(six.text_type('(?P<note>[A-G][b#]*)(?P<mod>.*)'),
                     six.text_type(label))

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
        self._register('key_mode|chord|chord_harte', self.deform_note)
        self._register('pitch_hz', self.deform_frequency)
        self._register('pitch_midi', self.deform_midi)
        self._register('chord_roman|pitch_class', self.deform_tonic)

    def states(self, jam):
        mudabox = jam.sandbox.muda
        state = dict(tuning=librosa.estimate_tuning(y=mudabox._audio['y'],
                                                    sr=mudabox._audio['sr']))
        yield state

    @staticmethod
    def audio(mudabox, state):
        mudabox._audio['y'] = pyrb.pitch_shift(mudabox._audio['y'],
                                               mudabox._audio['sr'],
                                               state['n_semitones'])

    @staticmethod
    def deform_frequency(annotation, state):
        annotation.data.value *= 2.0 ** (state['n_semitones'] / 12.0)

    @staticmethod
    def deform_midi(annotation, state):
        annotation.data.value += state['n_semitones']

    @staticmethod
    def deform_tonic(annotation, state):
        # First, figure out the tuning after deformation
        if -0.5 < (state['tuning'] + state['n_semitones']) <= 0.5:
            # If our tuning was off by more than the deformation,
            # then no label modification is necessary
            return

        for obs in annotation.data.value:
            obs['tonic'] = transpose(obs['tonic'], state['n_semitones'])

    @staticmethod
    def deform_note(annotation, state):
        # First, figure out the tuning after deformation
        if -0.5 < (state['tuning'] + state['n_semitones']) <= 0.5:
            # If our tuning was off by more than the deformation,
            # then no label modification is necessary
            return

        annotation.data.value = [transpose(obs, state['n_semitones'])
                                 for obs in annotation.data.value]


class PitchShift(AbstractPitchShift):
    '''Static pitch shifting by (fractional) semitones

    This transformation affects the following attributes:

    - Annotations
        - key_mode
        - chord, chord_harte, chord_roman
        - pitch_hz, pitch_midi, pitch_class
    - Audio

    Attributes
    ----------
    n_semitones : float or list of float
        The number of semitones to transpose the signal.
        Can be positive, negative, integral, or fractional.

    See Also
    --------
    RandomPitchShift
    LinearPitchShift

    Examples
    --------
    >>> # Shift down by a quarter-tone
    >>> D = muda.deformers.PitchShift(n_semitones=-0.5)
    '''

    def __init__(self, n_semitones=1):
        AbstractPitchShift.__init__(self)
        self.n_semitones = np.atleast_1d(n_semitones).flatten().tolist()

    def states(self, jam):
        for state in AbstractPitchShift.states(self, jam):
            for semitones in self.n_semitones:
                state['n_semitones'] = semitones
                yield state


class RandomPitchShift(AbstractPitchShift):
    '''Randomized pitch shifter

    Pitch is transposed by a normally distributed random variable.

    This transformation affects the following attributes:

    - Annotations
        - key_mode
        - chord, chord_harte, chord_roman
        - pitch_hz, pitch_midi, pitch_class
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate per input

    mean : float
    sigma : float > 0
        The parameters of the normal distribution for sampling
        pitch shifts

    See Also
    --------
    PitchShift
    LinearPitchShift

    Examples
    --------
    >>> # 5 random shifts with unit variance and mean of 1 semitone
    >>> D = muda.deformers.PitchShift(n_samples=5, mean=1.0, sigma=1)
    '''
    def __init__(self, n_samples=3, mean=0.0, sigma=1.0):
        AbstractPitchShift.__init__(self)

        if sigma <= 0:
            raise ValueError('sigma must be strictly positive')

        if n_samples <= 0:
            raise ValueError('n_samples must be None or positive')

        self.n_samples = n_samples
        self.mean = float(mean)
        self.sigma = float(sigma)

    def states(self, jam):
        # Sample the deformation
        for state in AbstractPitchShift.states(self, jam):
            for _ in range(self.n_samples):
                state['n_semitones'] = np.random.normal(loc=self.mean,
                                                        scale=self.sigma,
                                                        size=None)
                yield state


class LinearPitchShift(AbstractPitchShift):
    '''Linearly spaced pitch shift generator

    This transformation affects the following attributes:

    - Annotations
        - key_mode
        - chord, chord_harte, chord_roman
        - pitch_hz, pitch_midi, pitch_class
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate per input

    lower : float
    upper : float
        The lower and upper bounds for the shift sampling

    See Also
    --------
    PitchShift
    RandomPitchShift

    Examples
    --------
    >>> # 5 shifts spaced between -2 and +2 semitones
    >>> D = muda.deformers.LinearPitchShift(n_samples=5, lower=-2, upper=2)
    '''
    def __init__(self, n_samples=3, lower=-1, upper=1):
        AbstractPitchShift.__init__(self)

        if upper <= lower:
            raise ValueError('upper must be strictly larger than lower')

        if n_samples <= 0:
            raise ValueError('n_samples must be strictly positive')

        self.n_samples = n_samples
        self.lower = float(lower)
        self.upper = float(upper)

    def states(self, jam):
        shifts = np.linspace(self.lower,
                             self.upper,
                             num=self.n_samples,
                             endpoint=True)

        for state in AbstractPitchShift.states(self, jam):
            for n_semitones in shifts:
                state['n_semitones'] = n_semitones
                yield state
