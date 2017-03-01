#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''MUDA core tests'''

import numpy as np

import muda

import jams
import librosa

import tempfile
import os
import six

from nose.tools import eq_, raises

def test_jam_pack():

    jam = jams.JAMS()

    sr = 22050
    y = np.zeros(sr)

    muda.jam_pack(jam, y=y, sr=sr)

    # Make sure the jam now has a mudabox
    assert hasattr(jam.sandbox, 'muda')
    assert hasattr(jam.sandbox.muda, 'history')
    assert hasattr(jam.sandbox.muda, 'state')

    assert jam.sandbox.muda['y'] is y
    assert jam.sandbox.muda['sr'] == sr


def test_load_jam_audio():

    def __test(jam_in, audio_file):

        jam = muda.load_jam_audio(jam_in, audio_file)

        assert hasattr(jam.sandbox, 'muda')

        eq_(jam.file_metadata.duration,
            librosa.get_duration(**jam.sandbox.muda._audio))

    # Add an empty jams test for missing duration
    yield __test, jams.JAMS(), 'data/fixture.wav'

    yield __test, 'data/fixture.jams', 'data/fixture.wav'

    yield __test, jams.load('data/fixture.jams'), 'data/fixture.wav'

    with open('data/fixture.jams', 'r') as fdesc:
        yield __test, fdesc, 'data/fixture.wav'


def test_save():

    jam = muda.load_jam_audio('data/fixture.jams',
                              'data/fixture.wav')

    _, jamfile = tempfile.mkstemp(suffix='.jams')
    _, audfile = tempfile.mkstemp(suffix='.wav')

    muda.save(audfile, jamfile, jam)

    jam2 = muda.load_jam_audio(jamfile, audfile)
    jam2_raw = jams.load(jamfile)

    os.unlink(audfile)
    os.unlink(jamfile)

    assert hasattr(jam2.sandbox, 'muda')
    assert '_audio' in jam2.sandbox.muda
    assert '_audio' not in jam2_raw.sandbox.muda

    eq_(jam2.file_metadata.duration,
        librosa.get_duration(**jam2.sandbox.muda['_audio']))


def test_serialize_deformer():

    D = muda.deformers.LogspaceTimeStretch()
    D_ser = muda.serialize(D)
    D2 = muda.deserialize(D_ser)

    eq_(D.get_params(), D2.get_params())

    assert D is not D2


def test_serialize_pipeline():

    D1 = muda.deformers.LogspaceTimeStretch()
    D2 = muda.deformers.LogspaceTimeStretch()
    P_orig = muda.Pipeline([('stretch_1', D1),
                            ('stretch_2', D2)])
    P_ser = muda.serialize(P_orig)

    P_new = muda.deserialize(P_ser)

    eq_(P_orig.get_params(), P_new.get_params())

    assert P_orig is not P_new


def test_reload_jampack():

    # This test is to address #42, where mudaboxes reload as dict
    # instead of Sandbox
    jam = muda.load_jam_audio('data/fixture.jams', 'data/fixture.wav')

    jam2 = muda.load_jam_audio(six.StringIO(jam.dumps()), 'data/fixture.wav')
    assert isinstance(jam.sandbox.muda, jams.Sandbox)
    assert isinstance(jam2.sandbox.muda, jams.Sandbox)
