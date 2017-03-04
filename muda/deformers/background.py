#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-03 21:29:49 by Brian McFee <brian.mcfee@nyu.edu>
'''Additive background noise'''

import soundfile as psf
import librosa
import numpy as np
import os
import six

from ..base import BaseTransformer


def sample_clip_indices(filename, n_samples, sr):
    '''Calculate the indices at which to sample a fragment of audio from a file.

    Parameters
    ----------
    filename : str
        Path to the input file

    n_samples : int > 0
        The number of samples to load

    sr : int > 0
        The target sampling rate

    Returns
    -------
    start : int
        The sample index from `filename` at which the audio fragment starts
    stop : int
        The sample index from `filename` at which the audio fragment stops (e.g. y = audio[start:stop])
    '''

    with psf.SoundFile(str(filename), mode='r') as soundf:

        n_target = int(np.ceil(n_samples * soundf.samplerate / float(sr)))

        # Draw a random clip
        start = np.random.randint(0, len(soundf) - n_target)
        stop = start + n_target

        return start, stop


def slice_clip(filename, start, stop, n_samples, sr, mono=True):
    '''Slice a fragment of audio from a file.

    This uses pysoundfile to efficiently seek without
    loading the entire stream.

    Parameters
    ----------
    filename : str
        Path to the input file

    start : int
        The sample index of `filename` at which the audio fragment should start

    stop : int
        The sample index of `filename` at which the audio fragment should stop (e.g. y = audio[start:stop])

    n_samples : int > 0
        The number of samples to load

    sr : int > 0
        The target sampling rate

    mono : bool
        Ensure monophonic audio

    Returns
    -------
    y : np.ndarray [shape=(n_samples,)]
        A fragment of audio sampled from `filename`

    Raises
    ------
    ValueError
        If the source file is shorter than the requested length

    '''

    with psf.SoundFile(str(filename), mode='r') as soundf:
        n_target = stop - start

        soundf.seek(start)

        y = soundf.read(n_target).T

        if mono:
            y = librosa.to_mono(y)

        # Resample to initial sr
        y = librosa.resample(y, soundf.samplerate, sr)

        # Clip to the target length exactly
        y = librosa.util.fix_length(y, n_samples)

        return y


class BackgroundNoise(BaseTransformer):
    '''Additive background noise deformations.

    From each background noise signal, `n_samples` clips are randomly
    extracted and mixed with the input audio with a random mixing coefficient
    sampled uniformly between `weight_min` and `weight_max`.

    This transformation affects the following attributes:

    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate with each noise source

    files : str or list of str
        Path to audio file(s) on disk containing background signals

    weight_min : float in (0.0, 1.0)
    weight_max : float in (0.0, 1.0)
        The minimum and maximum weight to combine input signals

        `y_out = (1 - weight) * y + weight * y_noise`
    '''

    def __init__(self, n_samples=1, files=None, weight_min=0.1, weight_max=0.5):
        if n_samples <= 0:
            raise ValueError('n_samples must be strictly positive')

        if not 0 < weight_min < weight_max < 1.0:
            raise ValueError('weights must be in the range (0.0, 1.0)')

        if isinstance(files, six.string_types):
            files = [files]

        for fname in files:
            if not os.path.exists(fname):
                raise RuntimeError('file not found: {}'.format(fname))

        BaseTransformer.__init__(self)

        self.n_samples = n_samples
        self.files = files
        self.weight_min = weight_min
        self.weight_max = weight_max

    def states(self, jam):
        mudabox = jam.sandbox.muda
        for fname in self.files:
            for _ in range(self.n_samples):
                start, stop = sample_clip_indices(fname, len(mudabox._audio['y']), mudabox._audio['sr'])
                yield dict(filename=fname,
                           weight=np.random.uniform(low=self.weight_min,
                                                    high=self.weight_max,
                                                    size=None),
                           start=start,
                           stop=stop)

    def audio(self, mudabox, state):
        weight = state['weight']
        fname = state['filename']
        start = state['start']
        stop = state['stop']

        noise = slice_clip(fname, start, stop, len(mudabox._audio['y']),
                           mudabox._audio['sr'],
                           mono=mudabox._audio['y'].ndim == 1)

        # Normalize the data
        mudabox._audio['y'] = librosa.util.normalize(mudabox._audio['y'])
        noise = librosa.util.normalize(noise)

        mudabox._audio['y'] = ((1.0 - weight) * mudabox._audio['y'] +
                               weight * noise)
