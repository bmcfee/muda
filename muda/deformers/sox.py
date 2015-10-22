#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-25 08:13:01 by Brian McFee <brian.mcfee@nyu.edu>
'''Sox-based deformations'''

import os
import six
import subprocess
import tempfile
import librosa
import json
import soundfile as psf
from pkg_resources import resource_filename

from ..base import BaseTransformer

__all__ = ['DynamicRangeCompression', 'PRESETS']

# Presets pulled from
# https://web.archive.org/web/20140828122713/http://forum.doom9.org/showthread.php?t=165807
# and http://forum.doom9.org/showthread.php?p=779165#post779165
PRESETS = json.load(open(resource_filename(__name__, 'data/drc_presets.json')))


def __sox(y, sr, *args):
    '''Execute sox

    Parameters
    ----------
    y : np.ndarray
        Audio time series

    sr : int > 0
        Sampling rate of `y`

    *args
        Additional arguments to sox

    Returns
    -------
    y_out : np.ndarray
        `y` after sox transformation
    '''

    assert sr > 0

    fdesc, infile = tempfile.mkstemp(suffix='.wav')
    os.close(fdesc)
    fdesc, outfile = tempfile.mkstemp(suffix='.wav')
    os.close(fdesc)

    # Dump the audio
    librosa.output.write_wav(infile, y, sr)

    try:
        arguments = ['sox', infile, outfile, '-q']
        arguments.extend(args)

        subprocess.check_call(arguments)

        y_out, sr = psf.read(outfile)
        y_out = y_out.T
        if y.ndim == 1:
            y_out = librosa.to_mono(y_out)

    finally:
        os.unlink(infile)
        os.unlink(outfile)

    return y_out


def drc(y, sr, preset):
    '''Apply a preset DRC

    Parameters
    ----------
    y : np.ndarray
        Audio time series

    sr : int > 0
        Sampling rate of `y`

    preset : str
        Preset keyword (see PRESETS)

    Returns
    -------
    y_out : np.ndarray
        `y` after applying preset DRC
    '''

    return __sox(y, sr, 'compand', *PRESETS[preset])


class DynamicRangeCompression(BaseTransformer):
    '''Dynamic range compression.

    For each DRC preset configuration, one deformation is generated.

    This transformation affects the following attributes:

    - Audio

    Attributes
    ----------
    preset : str or list of str
        One or more supported preset values:
        - radio
        - film standard
        - film light
        - music standard
        - music light
        - speech


    Examples
    --------
    >>> # A single preset
    >>> drc = muda.deformers.DynamicRangeCompression(preset='radio')
    >>> # Multiple presets
    >>> drc = muda.deformers.DynamicRangeCompression(preset=['film standard',
    ...                                                      'film light'])
    >>> # All presets
    >>> drc = muda.deformers.DynamicRangeCompression(preset=muda.deformers.PRESETS.keys())
    '''

    def __init__(self, preset=None):
        BaseTransformer.__init__(self)

        if isinstance(preset, six.string_types):
            preset = [preset]

        for p in preset:
            assert p in PRESETS

        self.preset = preset

    def states(self, jam):

        for p in self.preset:
            yield dict(preset=p)

    @staticmethod
    def audio(mudabox, state):
        mudabox._audio['y'] = drc(mudabox._audio['y'],
                                  mudabox._audio['sr'],
                                  state['preset'])
