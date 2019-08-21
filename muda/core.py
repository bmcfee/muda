#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CREATED:2015-02-01 19:25:59 by Brian McFee <brian.mcfee@nyu.edu>
"""Core functionality for muda"""

import jams
import librosa
import soundfile as psf
import jsonpickle

import six

from .version import version
from . import deformers

__all__ = ["load_jam_audio", "save", "jam_pack", "serialize", "deserialize", "replay"]


def jam_pack(jam, **kwargs):
    """Pack data into a jams sandbox.

    If not already present, this creates a `muda` field within `jam.sandbox`,
    along with `history`, `state`, and version arrays which are populated by
    deformation objects.

    Any additional fields can be added to the `muda` sandbox by supplying
    keyword arguments.

    Parameters
    ----------
    jam : jams.JAMS
        A JAMS object

    Returns
    -------
    jam : jams.JAMS
        The updated JAMS object

    Examples
    --------
    >>> jam = jams.JAMS()
    >>> muda.jam_pack(jam, my_data=dict(foo=5, bar=None))
    >>> jam.sandbox
    <Sandbox: muda>
    >>> jam.sandbox.muda
    <Sandbox: state, version, my_data, history>
    >>> jam.sandbox.muda.my_data
    {'foo': 5, 'bar': None}
    """

    if not hasattr(jam.sandbox, "muda"):
        # If there's no mudabox, create one
        jam.sandbox.muda = jams.Sandbox(
            history=[],
            state=[],
            version=dict(
                muda=version,
                librosa=librosa.__version__,
                jams=jams.__version__,
                pysoundfile=psf.__version__,
            ),
        )

    elif not isinstance(jam.sandbox.muda, jams.Sandbox):
        # If there is a muda entry, but it's not a sandbox, coerce it
        jam.sandbox.muda = jams.Sandbox(**jam.sandbox.muda)

    jam.sandbox.muda.update(**kwargs)

    return jam


def load_jam_audio(
    jam_in, audio_file, validate=True, strict=True, fmt="auto", **kwargs
):
    """Load a jam and pack it with audio.

    Parameters
    ----------
    jam_in : str, file descriptor, jams.JAMS, or None
        JAMS filename, open file-descriptor, or object to load.
        See ``jams.load`` for acceptable formats.

        If `None` is provided, an empty JAMS object is constructed.

    audio_file : str
        Audio filename to load

    validate : bool
    strict : bool
    fmt : str
        Parameters to `jams.load`

    kwargs : additional keyword arguments
        See `librosa.load`

    Returns
    -------
    jam : jams.JAMS
        A jams object with audio data in the top-level sandbox

    Notes
    -----
    This operation can modify the `file_metadata.duration` field of `jam_in`:
    If it is not currently set, it will be populated with the duration of the
    audio file.

    See Also
    --------
    jams.load
    librosa.core.load


    Examples
    --------

    Load a JAMS object and audio from disk

    >>> jam = muda.load_jam_audio('my_file.jams', 'my_file.wav')


    Load an audio file with no jams annotation

    >>> jam = muda.load_jam_audio(None, 'my_file.wav')

    """

    if isinstance(jam_in, jams.JAMS):
        jam = jam_in
    elif jam_in is None:
        jam = jams.JAMS()
    else:
        jam = jams.load(jam_in, validate=validate, strict=strict, fmt=fmt)

    y, sr = librosa.load(audio_file, **kwargs)

    if jam.file_metadata.duration is None:
        jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr)

    return jam_pack(jam, _audio=dict(y=y, sr=sr))


def save(filename_audio, filename_jam, jam, strict=True, fmt="auto", **kwargs):
    """Save a muda jam to disk

    Parameters
    ----------
    filename_audio: str
        The path to store the audio file

    filename_jam: str
        The path to store the jams object

    strict: bool
        Strict safety checking for jams output

    fmt : str
        Output format parameter for `jams.JAMS.save`

    kwargs
        Additional parameters to `soundfile.write`
    """

    y = jam.sandbox.muda._audio["y"]
    sr = jam.sandbox.muda._audio["sr"]

    # First, dump the audio file
    psf.write(filename_audio, y, sr, **kwargs)

    # Then dump the jam
    jam.save(filename_jam, strict=strict, fmt=fmt)


def __reconstruct(params):
    """Reconstruct a transformation or pipeline given a parameter dump."""

    if isinstance(params, dict):
        if "__class__" in params:
            cls = params["__class__"]
            data = __reconstruct(params["params"])
            return cls(**data)
        else:
            data = dict()
            for key, value in six.iteritems(params):
                data[key] = __reconstruct(value)
            return data

    elif isinstance(params, (list, tuple)):
        return [__reconstruct(v) for v in params]

    else:
        return params


def serialize(transform, **kwargs):
    """Serialize a transformation object or pipeline.

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

    Examples
    --------
    >>> D = muda.deformers.TimeStretch(rate=1.5)
    >>> muda.serialize(D)
    '{"params": {"rate": 1.5},
      "__class__": {"py/type": "muda.deformers.time.TimeStretch"}}'
    """

    params = transform.get_params()
    return jsonpickle.encode(params, **kwargs)


def deserialize(encoded, **kwargs):
    """Construct a muda transformation from a JSON encoded string.

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

    Examples
    --------
    >>> D = muda.deformers.TimeStretch(rate=1.5)
    >>> D_serial = muda.serialize(D)
    >>> D2 = muda.deserialize(D_serial)
    >>> D2
    TimeStretch(rate=1.5)
    """

    params = jsonpickle.decode(encoded, **kwargs)

    return __reconstruct(params)


def replay(jam_muda, jam_new):
    """Re-apply the deformation history from one MUDA output on a new
    JAMS object.

    This is primarily useful when reconstructing the deformed audio from
    running MUDA when only the original audio and the deformed annotation (JAMS)
    are given.

    Parameters
    ----------
    jam_muda : jams.JAMS
        The JAMS object produced from a previous run of MUDA.

    jam_new : jams.JAMS
        The JAMS object on which to replay the deformation history.

        .. note:: This should be produced by `jam_pack` or `load_jam_audio`,
                  and contain the original audio.

    Returns
    -------
    jam_replay : jams.JAMS
        The result of applying `jam_muda`'s deformation history to `jam_new`.

    Examples
    --------

    Shift an example by one semitone, and then replay that transformation

    >>> jam = muda.load_jam_audio('my_file.jams', 'my_file.wav')
    >>> pitch = muda.deformers.PitchShift(n_semitones=+1)
    >>> jam_muda = next(pitch.transform(jam))
    >>> # Some time later, replay that transformation just from jam_muda
    >>> jam_new = muda.load_jam_audio('my_file.jams', 'my_file.wav')
    >>> jam_out = muda.replay(jam_muda, jam_new)
    >>> # jam_out now contains the deformed audio matching jam_muda
    """

    jam_re = jam_new

    for step in jam_muda.sandbox.muda.history:
        defclass = step["transformer"]["__class__"]
        params = step["transformer"]["params"]
        deformer = getattr(deformers, defclass)(**params)
        state = step["state"]
        jam_re = deformer._transform(jam_re, state)

    return jam_re
