#!/usr/bin/env python

import librosa
import numpy as np
import jams


import muda
from copy import deepcopy

from nose.tools import eq_, raises

jam_fixture = muda.load_jam_audio('data/fixture.jams', 'data/fixture.wav')

def __test_time(jam_orig, jam_new, rate):

    # Test the track length
    eq_(librosa.get_duration(**jam_orig.sandbox.muda['_audio']),
        rate * librosa.get_duration(**jam_new.sandbox.muda['_audio']))

    # Test the metadata
    eq_(jam_orig.file_metadata.duration,
        rate * jam_new.file_metadata.duration)

    # Test each annotation
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        # FIXME: time deformer should modify this
        #eq_(ann_orig.time, rate * ann_new.time)
        #eq_(ann_orig.duration, rate * ann_new.duration)

        assert np.allclose(ann_orig.data.time.values.astype(float),
                           rate * ann_new.data.time.values.astype(float))
        assert np.allclose(ann_orig.data.duration.values.astype(float),
                           rate * ann_new.data.duration.values.astype(float))

        if ann_orig.namespace == 'tempo':
            assert np.allclose(rate * ann_orig.data.value,
                               ann_new.data.value)

def test_timestretch():

    def __test(rate, jam):
        D = muda.deformers.TimeStretch(rate=rate)

        jam_orig = deepcopy(jam)

        for jam_new in D.transform(jam):
            # Verify that the original jam reference hasn't changed
            assert jam_new is not jam
            __test_time(jam_orig, jam, 1.0)

            # Verify that the state and history objects are intact
            d_trans = jam_new.sandbox.muda.history[-1]['transformer']
            eq_(d_trans['params'], D.get_params()['params'])
            d_state = jam_new.sandbox.muda.history[-1]['state']
            d_rate = d_state['rate']
            eq_(rate, d_rate)

            __test_time(jam_orig, jam_new, d_rate)


    for rate in [0.5, 1.0, 2.0]:
        yield __test, rate, jam_fixture

    for bad_rate in [-1, -0.5, 0.0]:
        yield raises(ValueError)(__test), bad_rate, jam_fixture
