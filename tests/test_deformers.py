#!/usr/bin/env python

import librosa
import numpy as np
import jams


import muda
from copy import deepcopy

from nose.tools import eq_, raises

def ap_(a, b, msg=None, rtol=1e-5, atol=1e-5):
    """Shorthand for 'assert np.allclose(a, b, rtol, atol), "%r != %r" % (a, b)
    """
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(msg or "{} != {}".format(a, b))


jam_fixture = muda.load_jam_audio('data/fixture.jams', 'data/fixture.wav')

def __test_time(jam_orig, jam_new, rate):

    # Test the track length
    ap_(librosa.get_duration(**jam_orig.sandbox.muda['_audio']),
        rate * librosa.get_duration(**jam_new.sandbox.muda['_audio']))

    # Test the metadata
    ap_(jam_orig.file_metadata.duration,
        rate * jam_new.file_metadata.duration)

    # Test each annotation
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        # FIXME: time deformer should modify this
        #eq_(ann_orig.time, rate * ann_new.time)
        #eq_(ann_orig.duration, rate * ann_new.duration)

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
        for lower in [0.5, 0.8]:
            for upper in [1.0, 1.25]:
                yield __test, n, lower, upper, jam_fixture

    for bad_samples in [-3, 0]:
        yield raises(ValueError)(__test), bad_samples, 0.8, 0.75, jam_fixture

    for bad_int in [(-1, -3), (2, 1)]:
        yield raises(ValueError)(__test), 3, bad_int[0], bad_int[1], jam_fixture
