#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CREATED:2015-02-01 19:25:59 by Brian McFee <brian.mcfee@nyu.edu>
'''Core functionality for muda'''

import jams
import librosa
import pysoundfile as psf
import jsonpickle

import six

from .base import *
import warnings


def jam_pack(jam, **kwargs):
    '''Pack data into a jams sandbox.

    Parameters
    ----------
    jam : jams.JAMS
        A JAMS object

    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> jam = jams.JAMS()
    >>> muda.jam_pack(jam, y=y, sr=sr)
    >>> print muda
    '''

    if not hasattr(jam.sandbox, 'muda'):
        jam.sandbox.muda = jams.Sandbox(history=[], state=[])

    jam.sandbox.muda.update(**kwargs)

    return jam


def load_jam_audio(jam_in, audio_file, **kwargs):
    '''Load a jam and pack it with audio.

    Parameters
    ----------
    jam_in : str or jams.JAMS
        JAM filename to load

    audio_file : str
        Audio filename to load

    kwargs : additional keyword arguments
        See `librosa.load`

    Returns
    -------
    jam : jams.JAMS
        A jams object with audio data in the top-level sandbox

    '''

    if isinstance(jam_in, six.string_types):
        jam = jams.load(jam_in)
    elif isinstance(jam_in, jams.JAMS):
        jam = jam_in
    else:
        raise TypeError('Invalid input type: ' + type(jam_in))

    y, sr = librosa.load(audio_file, **kwargs)

    return jam_pack(jam, _audio=dict(y=y, sr=sr))


def save(filename_audio, filename_jam, jam, strict=True, **kwargs):
    '''Save a muda jam to disk

    Parameters
    ----------
    filename_audio: str
        The path to store the audio file

    filename_jam: str
        The path to store the jams object

    strict: bool
        Strict safety checking for jams output

    kwargs
        Additional parameters to `pysoundfile.write`

    '''

    y = jam.sandbox.muda._audio['y']
    sr = jam.sandbox.muda._audio['sr']

    # First, dump the audio file
    psf.write(y, filename_audio, sr, **kwargs)

    # Then dump the jam
    jam.save(filename_jam, strict=strict)


def __reconstruct(params):
    '''Reconstruct a transformation or pipeline given a parameter dump.'''

    if isinstance(params, dict):
        if '__class__' in params:
            cls = params['__class__']
            data = __reconstruct(params['params'])
            return cls(**data)
        else:
            data = dict()
            for k, v in six.iteritems(params):
                data[k] = __reconstruct(v)
            return data

    elif isinstance(params, (list, tuple)):
        return [__reconstruct(v) for v in params]

    else:
        return params


def serialize(transform, **kwargs):
    '''Serialize a transformation object or pipeline.

    Parameters
    ----------
    transform : BaseTransform or Pipeline
        The transformation object to be serialized

    kwargs
        Additional keyword arguments to `jsonpickle.encode()`

    Returns
    -------
    json_str : str
        A JSON encoding of the transformation

    See Also
    --------
    deserialize
    '''

    params = transform.get_params()
    return jsonpickle.encode(params, **kwargs)


def deserialize(encoded, **kwargs):
    '''Construct a muda transformation from a JSON encoded string.

    Parameters
    ----------
    encoded : str
        JSON encoding of the transformation or pipeline

    kwargs
        Additional keyword arguments to `jsonpickle.decode()`

    Returns
    -------
    obj
        The transformation

    See Also
    --------
    serialize
    '''

    params = jsonpickle.decode(encoded, **kwargs)

    return __reconstruct(params)
