#!/usr/bin/env python

from copy import deepcopy
import re
import six

import numpy as np
import soundfile as psf

import jams
import librosa

import muda
from scipy import fft
from scipy import signal

import pytest
import scipy

from contextlib import contextmanager

@contextmanager
def does_not_raise():
    yield

def ap_(a, b, msg=None, rtol=1e-5, atol=1e-5):
    """Shorthand for 'assert np.allclose(a, b, rtol, atol), "%r != %r" % (a, b)
    """
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(msg or "{} != {}".format(a, b))


""" Input JAMS object fixture for multiple tests """
@pytest.fixture(scope='module')
def jam_fixture():
    return muda.load_jam_audio('tests/data/fixture.jams',
                               'tests/data/fixture.wav')

@pytest.fixture(scope='module')
def jam_mixture():
    return muda.load_jam_audio('tests/data/mixture.jams',
                               'tests/data/mixture.wav')


@pytest.fixture(scope='module')
def jam_impulse():
    sr=22050
    
    impulse = np.zeros(round(1.5*sr))
    impulse[len(impulse)//2]= 1.0

    #make jam object for this audio - for testing purposes
    freq_dict = {
        50.0: [(0.0,0.6),(0.8,1.2)],
        100.0: [(0.0,0.6),(1.0,1.1)],
        400.0: [(0.0,0.9),(1.0,1.1)],
        800.0: [(0.5,0.9),(1.2,1.5)],
        1200.0: [(1.2,1.5)]
    }

    jam = make_jam(freq_dict,sr,1.5)
    
    if jam.file_metadata.duration is None:
        jam.file_metadata.duration = 1.5

    return muda.jam_pack(jam, _audio=dict(y=impulse, sr=sr))

@pytest.fixture(scope='module')
def jam_raw():
    return jams.load('tests/data/fixture.jams')


@pytest.mark.xfail(raises=RuntimeError)
def test_raw(jam_raw):

    D = muda.deformers.TimeStretch(rate=2.0)
    six.next(D.transform(jam_raw))

def make_jam(freq_dict,sr,track_duration):
    """
    this function creates a jam according to a dictionary that specifies 
    each frequency's presence 

    dict: keys are frequencies
          values are list of tuples (start_time, duration) of that frequency
    """
    jam = jams.JAMS()

    # Store the track duration
    jam.file_metadata.duration = track_duration

    pitch_co = jams.Annotation(namespace='pitch_contour')
    note_h = jams.Annotation(namespace='note_hz')
    note_m = jams.Annotation(namespace='note_midi')
    pitch_cl = jams.Annotation(namespace='pitch_class')
    pitch_h = jams.Annotation(namespace='pitch_hz')
    pitch_m = jams.Annotation(namespace='pitch_midi')
    
    pitch_co.annotation_metadata = jams.AnnotationMetadata(data_source='synth')
    note_h.annotation_metadata = jams.AnnotationMetadata(data_source='synth')
    note_m.annotation_metadata = jams.AnnotationMetadata(data_source='synth')
    pitch_cl.annotation_metadata = jams.AnnotationMetadata(data_source='synth')
    pitch_h.annotation_metadata = jams.AnnotationMetadata(data_source='synth')
    pitch_m.annotation_metadata = jams.AnnotationMetadata(data_source='synth')


    #assign frequencies to each start_time
    freqs = freq_dict.keys()
    for f in freqs:
        time_dur = freq_dict[f] #list of tuples (start_time,duration)
        for t, dur in time_dur:
            pitch_co.append(time=t, duration=dur, value={"index":0,"frequency":f,"voiced":True})
            note_h.append(time=t, duration=dur,value=f)
            note_m.append(time=t, duration=dur, value=librosa.hz_to_midi(f))
            pclass = librosa.hz_to_note(f)
            pitch_cl.append(time=t, duration=dur,value={"tonic":pclass[:-1],"pitch":int(pclass[-1])})
            pitch_h.append(time=t, duration=dur,value=f)
            pitch_m.append(time=t, duration=dur, value=librosa.hz_to_midi(f))
    # Store the new annotation in the jam
    jam.annotations.append(pitch_co)
    jam.annotations.append(note_h)
    jam.annotations.append(note_m)
    jam.annotations.append(pitch_cl)
    jam.annotations.append(pitch_h)
    jam.annotations.append(pitch_m)

    return jam

""" Helper functions -- used across deformers """
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

        assert len(ann_orig.data) == len(ann_new.data)

        for obs1, obs2 in zip(ann_orig, ann_new):

            ap_(obs1.time, rate * obs2.time)
            ap_(obs1.duration, rate * obs2.duration)

            if ann_orig.namespace == 'tempo':
                ap_(rate * obs1.value, obs2.value)


def __test_deformer_history(deformer, history):

    d_trans = history['transformer']
    params = deformer.get_params()

    assert d_trans['params'] == params['params']
    assert d_trans['__class__'] == params['__class__'].__name__


def __test_params(D1, D2):

    p1 = D1.get_params()['params']
    r1 = p1.pop('rng', None)

    p2 = D2.get_params()['params']
    r2 = p2.pop('rng', None)

    # Make sure that all parameters are preserved
    assert p1 == p2

    r1 = muda.base._get_rng(r1)
    r2 = muda.base._get_rng(r2)
    # Comparing random states is a pain
    if r1 is not None or r2 is not None:
        for (v1, v2) in zip(r1.get_state(), r2.get_state()):
            if isinstance(v1, six.string_types):
                assert v1 == v2
            else:
                assert np.allclose(v1, v2)

                
""" Deformer: Timestretch """
# Timestretch
@pytest.mark.parametrize('rate', [0.5, 1.0, 2.0, [1.0, 1.5],
                                  pytest.mark.xfail(-1, raises=ValueError),
                                  pytest.mark.xfail(-0.5, raises=ValueError),
                                  pytest.mark.xfail(0.0, raises=ValueError)])
def test_timestretch(rate, jam_fixture):

    D = muda.deformers.TimeStretch(rate=rate)

    jam_orig = deepcopy(jam_fixture)

    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_fixture
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_rate = d_state['rate']
        if isinstance(rate, list):
            assert d_rate in rate
        else:
            assert d_rate == rate

        __test_time(jam_orig, jam_new, d_rate)

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


# Fixture for number of samples -- used across deformers
@pytest.fixture(params=[1, 3, 5,
                        pytest.mark.xfail(-3, raises=ValueError),
                        pytest.mark.xfail(0, raises=ValueError)])
def n_samples(request):
    return request.param


# LogspaceTimestretch
@pytest.mark.parametrize('lower, upper',
                         [(-1, 0.5), (0.0, 1.0),
                          pytest.mark.xfail((-1, -3), raises=ValueError),
                          pytest.mark.xfail((2, 1), raises=ValueError)])
def test_log_timestretch(n_samples, lower, upper, jam_fixture):

    D = muda.deformers.LogspaceTimeStretch(n_samples=n_samples,
                                           lower=lower,
                                           upper=upper)

    jam_orig = deepcopy(jam_fixture)

    n_out = 0
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_fixture
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_rate = d_state['rate']
        assert 2.0**lower <= d_rate <= 2.0**upper

        __test_time(jam_orig, jam_new, d_rate)
        n_out += 1

    assert n_samples == n_out
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


# RandomTimestretch
@pytest.mark.parametrize('scale',
                         [0.1,
                          pytest.mark.xfail(0, raises=ValueError),
                          pytest.mark.xfail(-1, raises=ValueError)])
def test_random_timestretch(n_samples, scale, jam_fixture):

    D = muda.deformers.RandomTimeStretch(n_samples=n_samples, scale=scale, rng=0)

    jam_orig = deepcopy(jam_fixture)

    n_out = 0
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_rate = d_state['rate']

        __test_time(jam_orig, jam_new, d_rate)
        n_out += 1

    assert n_samples == n_out
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


""" Deformer: Bypass """
@pytest.fixture(scope='module',
                params=[0.5,
                        pytest.mark.xfail(None, raises=TypeError)])
def D_simple(request):
    if request.param is None:
        return None
    else:
        return muda.deformers.TimeStretch(rate=request.param)


def test_bypass(D_simple, jam_fixture):

    D = muda.deformers.Bypass(transformer=D_simple)

    jam_orig = deepcopy(jam_fixture)

    generator = D.transform(jam_orig)
    jam_new = six.next(generator)
    assert jam_new is jam_orig
    __test_time(jam_orig, jam_fixture, 1.0)

    for jam_new in generator:
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig

        # Verify that the state and history objects are intact
        __test_deformer_history(D_simple, jam_new.sandbox.muda.history[-1])
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)

    
""" Deformer: PitchShift """
# Helper functions
def pstrip(x):

    root = re.match(six.text_type('([A-G][b#]*).*'),
                    six.text_type(x)).groups()[0]

    return librosa.note_to_midi(root)


def __test_note(ann_orig, ann_new, n):

    for obs1, obs2 in zip(ann_orig, ann_new):
        v_orig = pstrip(obs1.value)
        v_new = pstrip(obs2.value)
        v_orig = np.mod(np.round(np.mod(v_orig + n, 12)), 12)
        v_new = np.mod(np.round(np.mod(v_new, 12)), 12)
        ap_(v_orig, v_new)


def __test_tonic(ann_orig, ann_new, n):

    for obs1, obs2 in zip(ann_orig, ann_new):
        v_orig = pstrip(obs1.value['tonic'])
        v_new = pstrip(obs2.value['tonic'])

        v_orig = np.mod(np.round(np.mod(v_orig + n, 12)), 12)
        v_new = np.mod(np.round(np.mod(v_new, 12)), 12)
        ap_(v_orig, v_new)


def __test_contour(ann_orig, ann_new, n):

    scale = 2.0**(float(n) / 12)

    for obs1, obs2 in zip(ann_orig, ann_new):
        ap_(obs1.value['frequency'] * scale, obs2.value['frequency'])


def __test_hz(ann_orig, ann_new, n):

    scale = 2.0**(float(n) / 12)

    for obs1, obs2 in zip(ann_orig, ann_new):
        ap_(obs1.value * scale, obs2.value)


def __test_midi(ann_orig, ann_new, n):

    for obs1, obs2 in zip(ann_orig, ann_new):
        ap_(obs1.value + n, obs2.value)


def __test_pitch(jam_orig, jam_new, n_semitones, tuning):

    if -0.5 < tuning + n_semitones <= 0.5:
        q_tones = 0.0
    else:
        q_tones = n_semitones

    # Test each annotation
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        assert len(ann_orig) == len(ann_new)

        if ann_orig.namespace in ['chord', 'chord_harte', 'key_mode']:
            __test_note(ann_orig, ann_new, q_tones)
        elif ann_orig.namespace in ['pitch_class', 'chord_roman']:
            __test_tonic(ann_orig, ann_new, q_tones)
        elif ann_orig.namespace == 'pitch_contour':
            __test_contour(ann_orig, ann_new, n_semitones)
        elif ann_orig.namespace == 'pitch_hz':
            __test_hz(ann_orig, ann_new, n_semitones)
        elif ann_orig.namespace == 'pitch_midi':
            __test_midi(ann_orig, ann_new, n_semitones)


# PitchShift
@pytest.mark.parametrize('n_semitones',
                         [-2, -1, -0.5, -0.25, 0, 0.25, 1.0, 1.5, [-1, 1]])
def test_pitchshift(n_semitones, jam_fixture):
    D = muda.deformers.PitchShift(n_semitones=n_semitones)

    jam_orig = deepcopy(jam_fixture)

    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
        __test_pitch(jam_orig, jam_fixture, 0.0, 0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_tones = d_state['n_semitones']
        tuning = d_state['tuning']
        if isinstance(n_semitones, list):
            assert d_tones in n_semitones
        else:
            assert d_tones == n_semitones

        __test_pitch(jam_orig, jam_new, d_tones, tuning)
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


# RandomPitchShift
@pytest.mark.parametrize('sigma',
                         [0.5,
                          pytest.mark.xfail(-1, raises=ValueError),
                          pytest.mark.xfail(0, raises=ValueError)])
def test_random_pitchshift(n_samples, sigma, jam_fixture):

    D = muda.deformers.RandomPitchShift(n_samples=n_samples, sigma=sigma)

    jam_orig = deepcopy(jam_fixture)

    n_out = 0
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
        __test_pitch(jam_orig, jam_fixture, 0.0, 0.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_tones = d_state['n_semitones']
        tuning = d_state['tuning']
        __test_pitch(jam_orig, jam_new, d_tones, tuning)
        n_out += 1

    assert n_out == n_samples
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


# LinearPitchShift
@pytest.mark.parametrize('lower, upper',
                         [(-3, 1), (0.0, 3.0),
                          pytest.mark.xfail((-1, -3), raises=ValueError),
                          pytest.mark.xfail((2, 1), raises=ValueError)])
def test_linear_pitchshift(n_samples, lower, upper, jam_fixture):
    D = muda.deformers.LinearPitchShift(n_samples=n_samples,
                                        lower=lower,
                                        upper=upper)

    jam_orig = deepcopy(jam_fixture)

    n_out = 0
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
        __test_pitch(jam_orig, jam_fixture, 0.0, 0.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_tones = d_state['n_semitones']
        tuning = d_state['tuning']
        assert lower <= d_tones <= upper

        __test_pitch(jam_orig, jam_new, d_tones, tuning)
        n_out += 1

    assert n_out == n_samples
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


""" Deformer: Dynamic Range Compression """
def __test_effect(jam_orig, jam_new):
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        assert ann_orig == ann_new


@pytest.fixture(params=[p for p in muda.deformers.sox.PRESETS] +
                list(muda.deformers.sox.PRESETS.keys()))
def preset(request):
    return request.param


def test_drc(preset, jam_fixture):

    D = muda.deformers.DynamicRangeCompression(preset=preset)

    jam_orig = deepcopy(jam_fixture)

    for jam_new in D.transform(jam_orig):

        assert jam_new is not jam_fixture
        __test_effect(jam_orig, jam_fixture)

        assert not np.allclose(jam_orig.sandbox.muda['_audio']['y'],
                               jam_new.sandbox.muda['_audio']['y'])

        __test_effect(jam_orig, jam_new)
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


""" Deformer: Background Noise """
@pytest.mark.parametrize('noise', ['tests/data/noise_sample.ogg',
                                   ['tests/data/noise_sample.ogg']])
@pytest.mark.parametrize('weight_min, weight_max',
                         [(0.01, 0.6), (0.1, 0.8), (0.5, 0.99),
                          pytest.mark.xfail((0.0, 0.5), raises=ValueError),
                          pytest.mark.xfail((-1, 0.5), raises=ValueError),
                          pytest.mark.xfail((0.5, 1.5), raises=ValueError),
                          pytest.mark.xfail((0.75, 0.25), raises=ValueError)])
def test_background(noise, n_samples, weight_min, weight_max, jam_fixture):

    D = muda.deformers.BackgroundNoise(files=noise,
                                       n_samples=n_samples,
                                       weight_min=weight_min,
                                       weight_max=weight_max)

    jam_orig = deepcopy(jam_fixture)
    orig_duration = librosa.get_duration(**jam_orig.sandbox.muda['_audio'])

    n_out = 0
    for jam_new in D.transform(jam_orig):

        assert jam_new is not jam_fixture
        __test_effect(jam_orig, jam_fixture)

        assert not np.allclose(jam_orig.sandbox.muda['_audio']['y'],
                               jam_new.sandbox.muda['_audio']['y'])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        filename = d_state['filename']
        start = d_state['start']
        stop = d_state['stop']

        with psf.SoundFile(str(filename), mode='r') as soundf:
            max_index = len(soundf)
            noise_sr = soundf.samplerate

        assert 0 <= start < stop
        assert start < stop <= max_index
        assert ((stop - start) / float(noise_sr)) == orig_duration

        __test_effect(jam_orig, jam_new)
        n_out += 1

    assert n_out == n_samples
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


@pytest.mark.xfail(raises=RuntimeError)
def test_background_no_file():
    muda.deformers.BackgroundNoise(files='does-not-exist.ogg', n_samples=1)


@pytest.mark.xfail(raises=RuntimeError)
def test_background_short_file():
    D = muda.deformers.BackgroundNoise(files='tests/data/fixture.wav')
    jam_orig = muda.load_jam_audio('tests/data/fixture.jams',
                                   'tests/data/noise_sample.ogg')
    jam_new = next(D.transform(jam_orig))

    
""" Deformer: Colored Noise """
# Helper functions
def isclose_(a, b, rtol=1e-5, atol=2.5e-1):
    """Shorthand for 'assert np.isclose(a, b, rtol, atol)"""
    if not np.isclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError("{}(Expectation)!= {}(Estimation)".format(a, b))

def __test_color_slope(jam_orig, jam_new, color):

    colored_noise_data = jam_new.sandbox.muda['_audio']['y']
    colored_noise_sr = jam_new.sandbox.muda['_audio']['sr']

    #Verify that the sampling rate hasn't changed
    assert jam_orig.sandbox.muda['_audio']['sr'] == colored_noise_sr
    #estimate the power spectrum slope on log-log scale
    n_frames = len(colored_noise_data)
    y_power = np.absolute(np.fft.rfft(colored_noise_data)) ** 2
    freqs = np.fft.rfftfreq(n_frames, 1/float(colored_noise_sr))
    x = np.log(freqs[1:])
    y = np.log(y_power[1:])
    #rounded off to the 3 digits after the decimal point
    estimated_slope = round(scipy.stats.linregress(x,y)[0],3)
    if color == 'white':
        expected_slope = -0.0
        isclose_(expected_slope,estimated_slope)
    elif color == 'pink':
        expected_slope = -1.0
        isclose_(expected_slope,estimated_slope)
    elif color == 'brownian':
        expected_slope = -2.0
        isclose_(expected_slope,estimated_slope)
    else:
        raise ValueError('Unknown noise color\n')


# Input JAMS
@pytest.fixture(scope='module')
def jam_silence_96k():
    return muda.load_jam_audio('tests/data/silence_96k.jams',
                               'tests/data/silence_96k.wav')


@pytest.fixture(scope='module')
def jam_silence_8k():
    return muda.load_jam_audio('tests/data/silence_8k.jams',
                               'tests/data/silence_8k.wav')


@pytest.mark.parametrize('jam_test_silence', [(jam_silence_96k()),(jam_silence_8k())])
@pytest.mark.parametrize('color', [['white'],['pink'],['brownian'],
                          pytest.mark.xfail(['unknown'], raises=ValueError)])
@pytest.mark.parametrize('weight_min, weight_max',
                         [(0.01, 0.6), (0.1, 0.8), (0.5, 0.99),
                          pytest.mark.xfail((0.0, 0.5), raises=ValueError),
                          pytest.mark.xfail((-1, 0.5), raises=ValueError),
                          pytest.mark.xfail((0.5, 1.5), raises=ValueError),
                          pytest.mark.xfail((0.75, 0.25), raises=ValueError)])
def test_colorednoise(n_samples, color, weight_min, weight_max, jam_test_silence):

    D = muda.deformers.ColoredNoise(n_samples=n_samples,
                                    color=color,
                                    weight_min=weight_min,
                                    weight_max=weight_max,
                                    rng=0)
    jam_orig = deepcopy(jam_test_silence)

    orig_duration = librosa.get_duration(**jam_orig.sandbox.muda['_audio'])

    n_out = 0
    for jam_new in D.transform(jam_orig):
        assert jam_new is not jam_test_silence
        __test_effect(jam_orig, jam_test_silence)

        assert not np.allclose(jam_orig.sandbox.muda['_audio']['y'],
                               jam_new.sandbox.muda['_audio']['y'])
        # verify that duration hasn't changed
        assert librosa.get_duration(**jam_new.sandbox.muda['_audio']) == orig_duration

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        __test_effect(jam_orig, jam_new)

        # Verify the colored noise has desired slope for its log-log
        # scale power spectrum
        color = jam_new.sandbox.muda.history[-1]['state']['color']
        __test_color_slope(jam_orig, jam_new, color)

        n_out += 1
    assert n_out == n_samples
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)
    
    
""" Deformer: IR Convolution"""

'''Not used'''
def __test_duration(jam_orig, jam_shifted, orig_duration):
    #Verify the duration of last delayed annotation is in valid range
    #Verify the total duration hasn't changed
    assert (librosa.get_duration(**jam_shifted.sandbox.muda['_audio'])) == orig_duration

    shifted_data = jam_shifted.search(namespace='chord')[0].data
    #the expected duration of last annotation = Duration - Onset of last annotation
    ref_duration = orig_duration - shifted_data[-1][0] #[-1][0] indicates the 'time' of last observation
    #deformed duration:
    derformed_duration = shifted_data[-1][1] #[-1][0] indicates the 'duration' of last observation
    isclose_(ref_duration,derformed_duration,rtol=1e-5, atol=1e-1)

    
def __test_shifted_impulse(jam_orig, jam_new, ir_files, orig_duration, n_fft, rolloff_value):

    #delayed impulse
    with psf.SoundFile(str(ir_files), mode='r') as soundf:
        ir_data = soundf.read()
        ir_sr = soundf.samplerate

    #delay the impulse signal by zero-padding 1-second long zeros
    ir_data_delayed = np.pad(ir_data,(ir_sr,0),mode = 'constant')

    #dump the delayed audio file
    psf.write('tests/data/ir_file_delayed.wav', ir_data_delayed, ir_sr)

    D_delayed = muda.deformers.IRConvolution(ir_files = 'tests/data/ir_file_delayed.wav',
                                             n_fft=n_fft, rolloff_value = rolloff_value)

    for jam_shifted in D_delayed.transform(jam_orig):

        #Verify the duration that delayed annotations(Using chords here) are in valid range
        #__test_duration(jam_orig, jam_shifted, orig_duration)

        shifted_data = jam_shifted.search(namespace='chord')[0].data
        delayed_data = jam_new.search(namespace='chord')[0].data

        for i in range(len(shifted_data)):
            #For each observation, verify its onset time has been shifted 1s
            isclose_(1.00,shifted_data[i][0] - delayed_data[i][0])

            
@pytest.mark.parametrize('ir_files', ['tests/data/ir2_48k.wav',
                                   'tests/data/ir1_96k.wav'])
@pytest.mark.parametrize('n_fft', [256,1024])
@pytest.mark.parametrize('rolloff_value', [-36,12])

def test_ir_convolution(ir_files,jam_fixture,n_fft,rolloff_value):
    D = muda.deformers.IRConvolution(ir_files = ir_files, n_fft=n_fft, rolloff_value = rolloff_value)

    jam_orig = deepcopy(jam_fixture)
    orig_duration = librosa.get_duration(**jam_orig.sandbox.muda['_audio'])

    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig

        #Testing with shifted impulse
        __test_shifted_impulse(jam_orig, jam_new, ir_files, orig_duration,n_fft=n_fft, rolloff_value = rolloff_value)

        #Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)

    
""" MUDA Interface Objects"""
def test_pipeline(jam_fixture):
    D1 = muda.deformers.TimeStretch(rate=2.0)
    D2 = muda.deformers.TimeStretch(rate=1.5)

    P = muda.Pipeline([('stretch_1', D1),
                       ('stretch_2', D2)])

    jam_orig = deepcopy(jam_fixture)

    for jam_new in P.transform(jam_orig):
        assert jam_new is not jam_orig
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D1, jam_new.sandbox.muda.history[0])
        __test_deformer_history(D2, jam_new.sandbox.muda.history[-1])

        __test_time(jam_orig, jam_new, D1.rate[0] * D2.rate[0])


def test_union(jam_fixture):
    D1 = muda.deformers.TimeStretch(rate=[1.0, 2.0, 3.0])
    D2 = muda.deformers.TimeStretch(rate=[0.5, 1.5, 2.5])

    rates = [1.0, 0.5, 2.0, 1.5, 3.0, 2.5]

    union = muda.Union([('stretch_1', D1),
                        ('stretch_2', D2)])

    jam_orig = deepcopy(jam_fixture)

    for i, jam_new in enumerate(union.transform(jam_orig)):
        assert jam_new is not jam_orig
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        if i % 2:
            __test_deformer_history(D2, jam_new.sandbox.muda.history[-1])
        else:
            __test_deformer_history(D1, jam_new.sandbox.muda.history[-1])

        __test_time(jam_orig, jam_new, rates[i])


@pytest.mark.xfail(raises=ValueError)
def test_bad_pipeline_unique():
    D1 = muda.deformers.TimeStretch(rate=2.0)
    D2 = muda.deformers.TimeStretch(rate=1.5)

    muda.Pipeline([('stretch', D1), ('stretch', D2)])


@pytest.mark.xfail(raises=ValueError)
def test_bad_union_unique():
    D1 = muda.deformers.TimeStretch(rate=2.0)
    D2 = muda.deformers.TimeStretch(rate=1.5)

    muda.Union([('stretch', D1), ('stretch', D2)])


@pytest.mark.xfail(raises=TypeError)
def test_bad_pipeline_object():
    D = muda.deformers.TimeStretch(rate=2.0)

    muda.Pipeline([('stretch1', D),
                   ('stretch2', 'not a basetransformer')])


@pytest.mark.xfail(raises=TypeError)
def test_bad_union_object():
    D = muda.deformers.TimeStretch(rate=2.0)

    muda.Union([('stretch1', D),
                ('stretch2', 'not a basetransformer')])


@pytest.mark.xfail(raises=NotImplementedError)
def test_base_transformer():

    D = muda.BaseTransformer()

    six.next(D.transform(jam_fixture))



"""Deformer: Filtering"""
# Helper function 

def __test_tonic_filter(ann_orig, ann_new, cutoff):
    # raise error if original note now out of range is still included in annotation 
    for obs in ann_new:
        v_new = librosa.note_to_hz(obs.value["tonic"]+str(obs.value['pitch']))
        assert cutoff[0] < v_new < cutoff[1]

    # ensure number of new annotations is less than or equal to the original  
    assert len(ann_new) <= len(ann_orig)






def __test_contour_filter(ann_orig, ann_new, cutoff):

    for obs1,obs2 in zip(ann_orig,ann_new):
        v_orig = obs1.value['frequency']
        v_new = obs2.value['frequency']
        if cutoff[0] < v_orig and cutoff[1] > v_orig:
            ap_(v_orig,v_new)
        else:
            assert v_new == None





def __test_hz_filter(ann_orig, ann_new, cutoff):

    for obs in ann_new:
        v_new = obs.value
        assert cutoff[0] < v_new < cutoff[1]
    assert len(ann_new) <= len(ann_orig)



def __test_midi_filter(ann_orig, ann_new, cutoff):

    for obs in ann_new:
        v_new = librosa.midi_to_hz(obs.value)
        assert cutoff[0] < v_new < cutoff[1]
    assert len(ann_new) <= len(ann_orig)


def __test_pitch_filter(jam_orig, jam_new, cutoff):


    # Test each annotation
    for ann_orig, ann_new in zip(jam_orig.annotations, jam_new.annotations):
        #assert len(ann_orig) == len(ann_new)

    
        if ann_orig.namespace == 'pitch_class':
            __test_tonic_filter(ann_orig, ann_new, cutoff)
        elif ann_orig.namespace == 'pitch_contour':
            assert len(ann_orig) == len(ann_new)
            __test_contour_filter(ann_orig, ann_new, cutoff)
        elif ann_orig.namespace in ['pitch_hz','note_hz']:
            __test_hz_filter(ann_orig, ann_new, cutoff)
        elif ann_orig.namespace in ['pitch_midi','note_midi']:
            __test_midi_filter(ann_orig, ann_new, cutoff)

def __testsound(attenuation,cutoff_freq,audio_new,audio_orig,sr,btype):
    #this attenuation should be the ultimate one
    N = len(audio_orig)
    T = 1.0 / sr
    if btype == "bandpass":
        low,high = cutoff_freq
        
    elif btype == "low":
        low = 0
        high = cutoff_freq
    else:
        low = cutoff_freq
        high = sr/2
    
    #bin number of cutoff frequencies
    idx_low = round(low // (sr/N)) # bin number of the passband
    idx_high = round(high // (sr/N))
    
    yf_orig = fft(audio_orig)
    yf_filt = fft(audio_new)
    
    #db of fft coefficients
    db_orig = 20 * np.log10(2.0/N * np.abs(yf_orig))
    db_filt = 20 * np.log10(2.0/N * np.abs(yf_filt))
    
    #check passband (if number of bins greater than threshold is equal)
    
    
    if btype == "low":
        stop_filt = db_filt[idx_high:N//2]
        pass_filt = db_filt[:idx_low]
        pass_orig = db_orig[:idx_low]
        stop_orig = db_orig[idx_high:N//2]
    elif btype == "high":
        stop_filt = db_filt[:idx_low]
        pass_filt = db_filt[idx_high:N//2]
        pass_orig = db_orig[idx_high:N//2]
        stop_orig = db_orig[:idx_low]
    else:
        stop_filt = np.array(list(db_filt[:idx_low]) + list(db_filt[idx_high:N//2]))
        pass_filt = db_filt[idx_low:idx_high]
        pass_orig = db_orig[idx_low:idx_high]
        stop_orig = np.array(list(db_orig[:idx_low]) + list(db_orig[idx_high:N//2]))
    
    #check passband
    assert sum(pass_orig > -attenuation) == sum(pass_filt > -attenuation)
     
    #check stopband
    assert sum(stop_filt > -attenuation) == 0


    
#test filtering
@pytest.mark.parametrize('btype,cutoff',
                         [("high",20.0),
                          ("high",100.0),
                          ("high",500.0), 
                          ("low",1000.0),
                          ("low",500.0),
                          ("low",[100.0,300.0,900.0]),
                          ("bandpass",[(25.0,45.0),(30.0,500.0),(200.0,1000.0),(900.0,2000)]),
                          ("bandpass",(1300,1500.0)),


                        pytest.mark.xfail(("bandpass", [[40,100],[100,200,300]]), raises=ValueError),
                        pytest.mark.xfail(("bandpass", [[100,40],[100,200]]), raises=ValueError),
                        pytest.mark.xfail(("high", -50), raises=ValueError),
                        pytest.mark.xfail(("low", 0.0), raises=ValueError),
                        pytest.mark.xfail(("low", (20,100)), raises=ValueError),
                        pytest.mark.xfail(("low", [(20,50),(40,100)]), raises=ValueError),
                        pytest.mark.xfail(("bandpass", 1.0), raises=ValueError),
                        pytest.mark.xfail(("bandpass", -30.0), raises=ValueError),
                        pytest.mark.xfail(("bandpass", [40, 100]), raises=ValueError)
                        ])


@pytest.mark.parametrize('attenuation', #input is the ultimate one
                         [10.0,30.0,80.0, 
  
                         pytest.mark.xfail(-20.0, raises=ValueError)
                         ])

            

def test_filtering(btype, cutoff, jam_impulse,attenuation):
   
    D = muda.deformers.Filter(btype=btype, 
                                order=4, 
                                attenuation=attenuation,
                                cutoff=cutoff)

  
    jam_orig = deepcopy(jam_impulse)
  
    
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
        assert not np.allclose(jam_orig.sandbox.muda['_audio']['y'],
                               jam_new.sandbox.muda['_audio']['y'])
    
        d_state = jam_new.sandbox.muda.history[-1]['state']
        nyquist = d_state["nyquist"]
        order = d_state['order']
        atten = d_state['attenuation'] #this is the halfed one for the filter parameter

        
        if btype == "bandpass":
            low,high = d_state['cut_off']
        elif btype == "low": 
            low = 0
            high = d_state['cut_off']
        else:
            low = d_state['cut_off']
            high = nyquist

        
        assert order > 0
        assert isinstance(order,int)
        assert atten > 0

        if btype == "bandpass":
            assert 0 < low < high < nyquist
            
        else:
            assert 0 <= low < high <= nyquist
           

        __test_pitch_filter(jam_orig, jam_new, [low,high])

        #test sound
        __testsound(attenuation, #this is the ultimate one
            d_state['cut_off'],
            jam_new.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['sr'],
            btype)

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)

        
#test random lowpass
@pytest.mark.parametrize('cutoff',
                         [50.0,100.0,400.0,1000.0,

                         pytest.mark.xfail(-50.0, raises=ValueError),
                         pytest.mark.xfail([0.0,-20], raises=ValueError),
                         pytest.mark.xfail([(50.0,1000.0),(1000.0,2000.0)], raises=ValueError)
                         ])



@pytest.mark.parametrize('attenuation', #input is the ultimate one
                         [10.0,30.0,80.0, 
                        
                         pytest.mark.xfail(-20.0, raises=ValueError)
                         ])

@pytest.mark.parametrize('n_samples', #input is the ultimate one
                         [1,3,5, 
                         pytest.mark.xfail(0, raises=ValueError),
                         pytest.mark.xfail(-1, raises=ValueError)
                         ])


def test_randomlpfiltering(n_samples,cutoff, jam_impulse,attenuation):

    D = muda.deformers.RandomLPFilter(n_samples=n_samples,
                                         order=4, 
                                         attenuation=attenuation, 
                                         cutoff=cutoff,
                                         sigma=1.0,
                                         rng=0)

    
    jam_orig = deepcopy(jam_impulse)
    orig_duration = librosa.get_duration(**jam_orig.sandbox.muda['_audio'])
    
    n_out = 0 
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        nyquist = d_state["nyquist"]
        low = 0
        high = d_state['cut_off']
        order = d_state['order']
        atten = d_state['attenuation']

        assert order > 0
        assert isinstance(order,int)
        assert atten > 0

       
        assert 0 <= high <= nyquist
       
        __test_pitch_filter(jam_orig, jam_new, [low,high])
          #test sound
        __testsound(attenuation,
            d_state['cut_off'],
            jam_new.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['sr'],
            "low")
        n_out += 1

    assert n_samples == n_out

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)
        


@pytest.mark.parametrize('cutoff',
                         [90.0,300.0,1000.0,1300.0,

                         pytest.mark.xfail(-50, raises=ValueError),
                         pytest.mark.xfail([10.0,70.0,700.0], raises=ValueError),
                         pytest.mark.xfail([0.0,-20], raises=ValueError),
                         pytest.mark.xfail([(50.0,1000.0),(1000.0,2000.0)], raises=ValueError)
                         ])



@pytest.mark.parametrize('attenuation', #input is the ultimate one
                         [10.0,30.0,80.0, 
                     
                         pytest.mark.xfail(-20.0, raises=ValueError)
                         ])

@pytest.mark.parametrize('n_samples', #input is the ultimate one
                         [1,3,5, 
                         pytest.mark.xfail(0, raises=ValueError),
                         pytest.mark.xfail(-1, raises=ValueError)
                         ])

def test_randomhpfiltering(cutoff, n_samples,jam_impulse,attenuation):
    
    D = muda.deformers.RandomHPFilter(n_samples=n_samples,
                                         order=4, 
                                         attenuation=attenuation, 
                                         cutoff=cutoff,
                                         sigma=1.0,
                                         rng=0)

  
    jam_orig = deepcopy(jam_impulse)
    orig_duration = librosa.get_duration(**jam_orig.sandbox.muda['_audio'])
    
    n_out = 0 
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig


        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        nyquist = d_state["nyquist"]
        low = d_state['cut_off']
        high = nyquist
       
        order = d_state['order']
        atten = d_state['attenuation']

        assert order > 0
        assert isinstance(order,int)
        assert atten > 0

       
        assert 0 < low < nyquist

        __test_pitch_filter(jam_orig, jam_new, [low,high])
        #test sound
        __testsound(attenuation,
            d_state['cut_off'],
            jam_new.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['sr'],
            "high")

        n_out += 1

    assert n_samples == n_out

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)




@pytest.mark.parametrize('cutoff',
                         [(50.0,300.0),(900.0,1300.0),

                         pytest.mark.xfail(-50, raises=ValueError),
                         pytest.mark.xfail([700.0,600.0], raises=ValueError),
                         pytest.mark.xfail([0.0,-20], raises=ValueError),
                         pytest.mark.xfail([(50.0,1000.0),(1000.0,2000.0)], raises=ValueError)
                         ])



@pytest.mark.parametrize('attenuation', #input is the ultimate one
                         [10.0,30.0,80.0, 
                        
                         pytest.mark.xfail(-20.0, raises=ValueError)
                         ])

@pytest.mark.parametrize('n_samples', #input is the ultimate one
                         [1,3,5, 
                         pytest.mark.xfail(0, raises=ValueError),
                         pytest.mark.xfail(-1, raises=ValueError)
                         ])


def test_randombpfiltering(cutoff, n_samples, jam_impulse,attenuation):
    if type(cutoff) != tuple:
        raise ValueError("cut off frequency for random bandpass filter must be a tuple") 
    else:
         low,high = cutoff
    
    D = muda.deformers.RandomBPFilter(n_samples=n_samples,
                                         order=4, 
                                         attenuation=attenuation, 
                                         cutoff_low=low,
                                         cutoff_high = high,
                                         sigma=1.0,
                                         rng=0)

  
    jam_orig = deepcopy(jam_impulse)
    orig_duration = librosa.get_duration(**jam_orig.sandbox.muda['_audio'])
    
    n_out = 0
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
       

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        low,high = d_state['cut_off']
        nyquist = d_state["nyquist"]
        order = d_state['order']
        atten = d_state['attenuation']

        assert order > 0
        assert isinstance(order,int)
        assert atten > 0

        assert 0 < low < high < nyquist

        __test_pitch_filter(jam_orig, jam_new, [low,high])
          #test sound
        __testsound(attenuation,
            d_state['cut_off'],
            jam_new.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['y'],
            jam_orig.sandbox.muda['_audio']['sr'],
            "bandpass")

        n_out += 1

    assert n_samples == n_out

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)

        




""" Deformer: Clipping """
# Helper function
def __test_clipped_buffer(jam_orig, jam_new, clip_limit):
    
    # Get Audio Buffer
    y_orig = jam_orig.sandbox.muda['_audio']['y']
    y_new = jam_new.sandbox.muda['_audio']['y']
    
    assert min(y_orig)*clip_limit <= y_new.all() <= max(y_orig)*clip_limit
    
    
# Clipping
@pytest.mark.parametrize('clip_limit, expectation', [(0.4, does_not_raise()), (0.8, does_not_raise()), 
                                                     ([0.3, 0.9], does_not_raise()),
# Old marker style - deprecated
#                                   pytest.mark.xfail(-1, raises=ValueError),
#                                   pytest.mark.xfail(-0.1, raises=ValueError),
#                                   pytest.mark.xfail(0.0, raises=ValueError),
#                                   pytest.mark.xfail(1.1, raises=ValueError),
#                                   pytest.mark.xfail([0.2, 1.0], raises=ValueError)])
# New marker style
                                  pytest.param(-1, pytest.raises(ValueError), marks=pytest.mark.xfail),
                                  pytest.param(-0.1, pytest.raises(ValueError), marks=pytest.mark.xfail),
                                  pytest.param(0.0, pytest.raises(ValueError), marks=pytest.mark.xfail),
                                  pytest.param(1.1, pytest.raises(ValueError), marks=pytest.mark.xfail),
                                  pytest.param([0.2, 1.0], pytest.raises(ValueError), marks=pytest.mark.xfail)])
def test_clipping(clip_limit, expectation, jam_fixture):

    with expectation: 
        D = muda.deformers.Clipping(clip_limit=clip_limit)

    jam_orig = deepcopy(jam_fixture)

    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_fixture
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_clip_limit = d_state['clip_limit']
        if isinstance(clip_limit, list):
            assert d_clip_limit in clip_limit
        else:
            assert d_clip_limit == clip_limit

        # Verify clipping outcome
        __test_clipped_buffer(jam_orig, jam_new, d_clip_limit)

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)

# LinearClipping
@pytest.mark.parametrize('lower, upper',
                         [(0.3, 0.5), (0.1, 0.9),
# Old marker style - deprecated
#                           pytest.mark.xfail((-0.1, 0.2), raises=ValueError),
#                           pytest.mark.xfail((1.0, 1.2), raises=ValueError),
#                           pytest.mark.xfail((0.8, 0.6), raises=ValueError),
#                           pytest.mark.xfail((0.6, 1.0), raises=ValueError)])
# New marker style
                          pytest.param(-0.1, 0.2, marks=pytest.mark.xfail),
                          pytest.param(1.0, 1.2, marks=pytest.mark.xfail),
                          pytest.param(0.8, 0.6, marks=pytest.mark.xfail),
                          pytest.param(0.6, 1.0, marks=pytest.mark.xfail)])
def test_linear_clipping(n_samples, lower, upper, jam_fixture):

    D = muda.deformers.LinearClipping(n_samples=n_samples,
                                               lower=lower,
                                               upper=upper)

    jam_orig = deepcopy(jam_fixture)

    n_out = 0    
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_fixture
        __test_time(jam_orig, jam_fixture, 1.0)

        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_clip_limit = d_state['clip_limit']
        assert lower <= d_clip_limit <= upper

        # Verify clipping outcome
        __test_clipped_buffer(jam_orig, jam_new, d_clip_limit)
        n_out += 1

    assert n_samples == n_out

    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)

# RandomClipping
@pytest.mark.parametrize('a, b',
                         [(0.5, 0.5), (5.0, 1.0), (1.0, 3.0),
# Old marker style - deprecated
#                           pytest.mark.xfail((0.0,0.5), raises=ValueError),
#                           pytest.mark.xfail((0.5,0.0), raises=ValueError),
#                           pytest.mark.xfail((-0.1,1.0), raises=ValueError),
#                           pytest.mark.xfail((1.0,-0.1), raises=ValueError),
#                           pytest.mark.xfail((-0.5,-0.5), raises=ValueError)])
# New marker style
                          pytest.param(0.0, 0.5, marks=pytest.mark.xfail),
                          pytest.param(0.5, 0.0, marks=pytest.mark.xfail),
                          pytest.param(-0.1, 1.0, marks=pytest.mark.xfail),
                          pytest.param(1.0, -0.1, marks=pytest.mark.xfail),
                          pytest.param(-0.5, -0.5, marks=pytest.mark.xfail)])
def test_random_clipping(n_samples, a, b, jam_fixture):

    D = muda.deformers.RandomClipping(n_samples=n_samples, a=a, b=b, rng=0)

    jam_orig = deepcopy(jam_fixture)

    n_out = 0
    for jam_new in D.transform(jam_orig):
        # Verify that the original jam reference hasn't changed
        assert jam_new is not jam_orig
       
        # Verify that the state and history objects are intact
        __test_deformer_history(D, jam_new.sandbox.muda.history[-1])

        d_state = jam_new.sandbox.muda.history[-1]['state']
        d_clip_limit = d_state['clip_limit']

        # Verify clipping outcome
        __test_clipped_buffer(jam_orig, jam_new, d_clip_limit)
        n_out += 1

    assert n_samples == n_out
    
    # Serialization test
    D2 = muda.deserialize(muda.serialize(D))
    __test_params(D, D2)


