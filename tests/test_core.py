#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''MUDA core tests'''
import tempfile
import os
import six

import numpy as np

import jams
import librosa

import muda

import pytest


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


@pytest.fixture()
def audio_file():
    return 'tests/data/fixture.wav'


@pytest.fixture()
def jam_in():
    return 'tests/data/fixture.jams'


@pytest.fixture(params=[0, 1, 2, 3, 4],
                ids=['JAMS()',
                     'fixture.jams',
                     "JAMS('fixture.jams')",
                     'fdesc',
                     'None'])
def jam_loader(request):
    if request.param == 0:
        yield jams.JAMS()

    elif request.param == 1:
        yield 'tests/data/fixture.jams'

    elif request.param == 2:
        yield jams.load('tests/data/fixture.jams')

    elif request.param == 3:
        with open('tests/data/fixture.jams', 'r') as fdesc:
            yield fdesc
    else:
        yield None


@pytest.mark.parametrize('validate', [False, True])
@pytest.mark.parametrize('strict', [False, True])
# gzip will throw an IOError on py2, and OSError on py3.
@pytest.mark.parametrize('fmt', ['auto', 'jams',
                                 pytest.mark.xfail('jamz',
                                                   raises=(OSError, IOError))])
def test_load_jam_audio(jam_loader, audio_file, validate, strict, fmt):

    jam = muda.load_jam_audio(jam_loader, audio_file,
                              validate=validate, strict=strict, fmt=fmt)

    assert hasattr(jam.sandbox, 'muda')

    duration = librosa.get_duration(**jam.sandbox.muda._audio)
    assert jam.file_metadata.duration == duration


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('fmt', ['auto', 'jams', 'jamz'])
def test_save(jam_in, audio_file, strict, fmt):

    jam = muda.load_jam_audio(jam_in, audio_file)

    _, jamfile = tempfile.mkstemp(suffix='.jams')
    _, audfile = tempfile.mkstemp(suffix='.wav')

    muda.save(audfile, jamfile, jam, strict=strict, fmt=fmt)

    jam2 = muda.load_jam_audio(jamfile, audfile, fmt=fmt)
    jam2_raw = jams.load(jamfile, fmt=fmt)

    os.unlink(audfile)
    os.unlink(jamfile)

    assert hasattr(jam2.sandbox, 'muda')
    assert '_audio' in jam2.sandbox.muda
    assert '_audio' not in jam2_raw.sandbox.muda

    duration = librosa.get_duration(**jam2.sandbox.muda['_audio'])

    assert jam2.file_metadata.duration == duration


def test_serialize_pipeline():

    D1 = muda.deformers.LogspaceTimeStretch()
    D2 = muda.deformers.LogspaceTimeStretch()
    P_orig = muda.Pipeline([('stretch_1', D1),
                            ('stretch_2', D2)])
    P_ser = muda.serialize(P_orig)

    P_new = muda.deserialize(P_ser)

    assert P_orig is not P_new
    assert P_orig.get_params() == P_new.get_params()


def test_serialize_union():

    D1 = muda.deformers.LogspaceTimeStretch()
    D2 = muda.deformers.LogspaceTimeStretch()
    U_orig = muda.Union([('stretch_1', D1),
                         ('stretch_2', D2)])
    U_ser = muda.serialize(U_orig)

    U_new = muda.deserialize(U_ser)

    assert U_orig is not U_new
    assert U_orig.get_params() == U_new.get_params()


def test_reload_jampack(jam_in, audio_file):

    # This test is to address #42, where mudaboxes reload as dict
    # instead of Sandbox
    jam = muda.load_jam_audio(jam_in, audio_file)

    jam2 = muda.load_jam_audio(six.StringIO(jam.dumps()), audio_file)
    assert isinstance(jam.sandbox.muda, jams.Sandbox)
    assert isinstance(jam2.sandbox.muda, jams.Sandbox)
