#!/usr/bin/env python

from __future__ import print_function

import librosa
import numpy as np
import jams

import re
import six

import muda
from copy import deepcopy

from nose.tools import eq_, raises

def ap_(a, b, msg=None, rtol=1e-5, atol=1e-5):
    """Shorthand for 'assert np.allclose(a, b, rtol, atol), "%r != %r" % (a, b)
    """
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(msg or "{} != {}".format(a, b))


jam_fixture = muda.load_jam_audio('data/fixture.jams', 'data/fixture.wav')


@raises(RuntimeError)
def test_raw():

    jam_raw = jams.load('data/fixture.jams')
    D = muda.deformers.TimeStretch(rate=2.0)

    six.next(D.transform(jam_raw))


def __test_time(jam_orig, jam_new, rate):

    # Test the track length
    ap_(librosa.get_duration(**jam_orig.sandbox.muda['_audio']),
        rate * librosa.get_duration(**jam_new.sandbox.muda['_audio']))

    # Test the metadata
    ap_(jam_orig.file_metadata.duration,
        rate * jam_new.file_metadata.duration)

    # Test each annotation
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        # JAMS 0.2.1 support
        if hasattr(ann_orig, 'time'):
            ap_(ann_orig.time, rate * ann_new.time)
            ap_(ann_orig.duration, rate * ann_new.duration)

        ap_(ann_orig.data.time.values.astype(float),
            rate * ann_new.data.time.values.astype(float))
        ap_(ann_orig.data.duration.values.astype(float),
            rate * ann_new.data.duration.values.astype(float))

        if ann_orig.namespace == 'tempo':
            ap_(rate * ann_orig.data.value, ann_new.data.value)


def __test_deformer_history(deformer, history):

    d_trans = history['transformer']
    params = deformer.get_params()

    eq_(d_trans['params'], params['params'])
    eq_(d_trans['__class__'], params['__class__'].__name__)


def test_timestretch():

    def __test(rate, jam):
        D = muda.deformers.TimeStretch(rate=rate)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_time(jam_orig, jam, 1.0)

            # Verify that the state and history objects are intact
            __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_rate = d_state['rate']
            ap_(rate, d_rate)

            __test_time(jam_orig, jam_new, d_rate)


    for rate in [0.5, 1.0, 2.0]:
        yield __test, rate, jam_fixture

    for bad_rate in [-1, -0.5, 0.0]:
        yield raises(ValueError)(__test), bad_rate, jam_fixture


def test_log_timestretch():

    def __test(n, lower, upper, jam):
        D = muda.deformers.LogspaceTimeStretch(n_samples=n, lower=lower, upper=upper)

        jam_orig = deepcopy(jam)

        n_samples = 0
        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_time(jam_orig, jam, 1.0)

            # Verify that the state and history objects are intact
            __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_rate = d_state['rate']
            assert 2.0**lower <= d_rate <= 2.0**upper

            __test_time(jam_orig, jam_new, d_rate)
            n_samples += 1

        eq_(n, n_samples)


    for n in [1, 3, 5]:
        for lower in [-1, -0.5, 0.0]:
            for upper in [0.5, 1.0]:
                yield __test, n, lower, upper, jam_fixture

    for bad_samples in [-3, 0]:
        yield raises(ValueError)(__test), bad_samples, -1, 1, jam_fixture

    for bad_int in [(-1, -3), (2, 1)]:
        yield raises(ValueError)(__test), 3, bad_int[0], bad_int[1], jam_fixture


def test_random_timestretch():

    def __test(n_samples, jam):
        np.random.seed(0)
        D = muda.deformers.RandomTimeStretch(n_samples=n_samples)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_time(jam_orig, jam, 1.0)

            # Verify that the state and history objects are intact
            __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_rate = d_state['rate']

            __test_time(jam_orig, jam_new, d_rate)

    @raises(ValueError)
    def __test_negative_scale():
        muda.deformers.RandomTimeStretch(scale=-1)

    for n in [1, 3, 5]:
        yield __test, n, jam_fixture

    for bad_n in [-1, 0]:
        yield raises(ValueError)(__test), bad_n, jam_fixture

    yield __test_negative_scale


def test_bypass():

    def __test(rate, jam):
        _D = muda.deformers.TimeStretch(rate=rate)
        D = muda.deformers.Bypass(transformer=_D)

        jam_orig = deepcopy(jam)

        generator = D.transform(jam)
        jam_new = six.next(generator)
        assert jam_new is jam
        __test_time(jam_orig, jam, 1.0)

        for jam_new in generator:
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_time(jam_orig, jam, 1.0)

            # Verify that the state and history objects are intact
            __test_deformer_history(_D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_rate = d_state['rate']
            ap_(rate, d_rate)

            __test_time(jam_orig, jam_new, d_rate)

    @raises(TypeError)
    def bad_test():
        D = muda.deformers.Bypass(None)

    for rate in [0.5, 1.0, 2.0]:
        yield __test, rate, jam_fixture

    for bad_rate in [-1, -0.5, 0.0]:
        yield raises(ValueError)(__test), bad_rate, jam_fixture

    yield bad_test


def pstrip(x):

    root = re.match(six.text_type('([A-G][b#]*).*'),
                    six.text_type(x)).groups()[0]

    return librosa.note_to_midi(root)


def __test_note(ann_orig, ann_new, n):

    # Get the value strings
    v_orig = np.asarray([pstrip(_) for _ in ann_orig.data.value])
    v_new  = np.asarray([pstrip(_) for _ in ann_new.data.value])

    v_orig = np.mod(np.round(np.mod(v_orig + n, 12)), 12)
    v_new = np.mod(np.round(np.mod(v_new, 12)), 12)
    ap_(v_orig, v_new)


def __test_tonic(ann_orig, ann_new, n):

    v_orig = np.asarray([pstrip(_['tonic']) for _ in ann_orig.data.value])
    v_new  = np.asarray([pstrip(_['tonic']) for _ in ann_new.data.value])

    v_orig = np.mod(np.round(np.mod(v_orig + n, 12)), 12)
    v_new = np.mod(np.round(np.mod(v_new, 12)), 12)
    ap_(v_orig, v_new)


def __test_hz(ann_orig, ann_new, n):

    scale = 2.0**(float(n) / 12)

    ap_(ann_orig.data.value * scale, ann_new.data.value)


def __test_midi(ann_orig, ann_new, n):

    ap_(ann_orig.data.value + n, ann_new.data.value)


def __test_pitch(jam_orig, jam_new, n_semitones, tuning):

    if -0.5 < tuning + n_semitones <= 0.5:
        q_tones = 0.0
    else:
        q_tones = n_semitones

    # Test each annotation
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        if ann_orig.namespace in ['chord', 'chord_harte', 'key_mode']:
            __test_note(ann_orig, ann_new, q_tones)
        elif ann_orig.namespace in ['pitch_class', 'chord_roman']:
            __test_tonic(ann_orig, ann_new, q_tones)
        elif ann_orig.namespace == 'pitch_hz':
            __test_hz(ann_orig, ann_new, n_semitones)
        elif ann_orig.namespace == 'pitch_midi':
            __test_midi(ann_orig, ann_new, n_semitones)


def test_pitchshift():
    def __test(n_semitones, jam):
        np.random.seed(0)
        D = muda.deformers.PitchShift(n_semitones=n_semitones)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_pitch(jam_orig, jam, 0.0, 0)

            # Verify that the state and history objects are intact
            __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_tones = d_state['n_semitones']
            tuning = d_state['tuning']
            ap_(n_semitones, d_tones)
            __test_pitch(jam_orig, jam_new, d_tones, tuning)

    for n in [-2, -1, -0.5, -0.25, 0, 0.25, 1.0, 1.5]:
        yield __test, n, jam_fixture

def test_random_pitchshift():

    def __test(n_samples, jam):
        D = muda.deformers.RandomPitchShift(n_samples=n_samples)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_pitch(jam_orig, jam, 0.0, 0.0)

            # Verify that the state and history objects are intact
            __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_tones = d_state['n_semitones']
            tuning = d_state['tuning']
            __test_pitch(jam_orig, jam_new, d_tones, tuning)

    @raises(ValueError)
    def __test_negative_scale(sigma):
        muda.deformers.RandomPitchShift(sigma=sigma)

    for n in [1, 3, 5]:
        yield __test, n, jam_fixture

    for bad_n in [-1, 0]:
        yield raises(ValueError)(__test), bad_n, jam_fixture
    
    for bad_sigma in [-1, 0]:
        yield __test_negative_scale, bad_sigma


def test_linear_pitchshift():

    def __test(n, lower, upper, jam):
        D = muda.deformers.LinearPitchShift(n_samples=n, lower=lower, upper=upper)

        jam_orig = deepcopy(jam)

        n_samples = 0
        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_pitch(jam_orig, jam, 0.0, 0.0)

            # Verify that the state and history objects are intact
            __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_tones = d_state['n_semitones']
            tuning = d_state['tuning']
            assert lower <= d_tones <= 2.0**upper

            __test_pitch(jam_orig, jam_new, d_tones, tuning)
            n_samples += 1

        eq_(n, n_samples)


    for n in [1, 3, 5]:
        for lower in [-3, -1, 0.0]:
            for upper in [1, 3]:
                yield __test, n, lower, upper, jam_fixture

    for bad_samples in [-3, 0]:
        yield raises(ValueError)(__test), bad_samples, -1, 1, jam_fixture

    for bad_int in [(-1, -3), (2, 1)]:
        yield raises(ValueError)(__test), 3, bad_int[0], bad_int[1], jam_fixture


def __test_effect(jam_orig, jam_new):


    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        eq_(ann_orig, ann_new)


def test_drc():

    def __test(preset, jam):

        D = muda.deformers.DynamicRangeCompression(preset=preset)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam_orig):

            assert jam_new is not jam
            __test_effect(jam_orig, jam)

            assert not np.allclose(jam_orig.sandbox.muda['_audio']['y'],
                                   jam_new.sandbox.muda['_audio']['y'])

            __test_effect(jam_orig, jam_new)

    for preset in muda.deformers.sox.PRESETS:
        yield __test, preset, jam_fixture

    yield __test, muda.deformers.sox.PRESETS, jam_fixture


def test_background():

    def __test(noise_sample, n_samples, weight_min, weight_max, jam):

        D = muda.deformers.BackgroundNoise(files=noise_sample,
                                           n_samples=n_samples,
                                           weight_min=weight_min,
                                           weight_max=weight_max)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam_orig):

            assert jam_new is not jam
            __test_effect(jam_orig, jam)

            assert not np.allclose(jam_orig.sandbox.muda['_audio']['y'],
                                   jam_new.sandbox.muda['_audio']['y'])

            __test_effect(jam_orig, jam_new)

    noise = 'data/noise_sample.ogg'

    for weight_min in [0.01, 0.1, 0.5]:
        for weight_max in [0.6, 0.8, 0.99]:
            yield __test, noise, 3, weight_min, weight_max, jam_fixture
            yield __test, [noise], 3, weight_min, weight_max, jam_fixture

    yield raises(RuntimeError)(__test), 'nonexistant_file.ogg', 3, weight_min, weight_max, jam_fixture

    for bad_n in [-1, 0]:
        yield raises(ValueError)(__test), noise, bad_n, 0.25, 0.75, jam_fixture

    for bad_int in [(0, 0.5), (0.5, 1), (-1, 0.5), (0.5, 1.5), (0.75, 0.25)]:
        yield raises(ValueError)(__test), noise, 1, bad_int[0], bad_int[1], jam_fixture



