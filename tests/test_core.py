#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''MUDA core tests'''

import numpy as np

import muda

import jams
import librosa

import tempfile
import os

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



