#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-03-24 16:10:11 by Brian McFee <brian.mcfee@nyu.edu>
'''Impulse response'''

import six
import numpy as np
import pandas as pd
import scipy

import soundfile as psf
import librosa
import jams

from ..base import BaseTransformer


def median_group_delay(y, sr, n_fft=2048, rolloff_value = -24):
    '''Compute the average group delay for a signal

    Parameters
    ----------
    y : np.ndarray
        the signal

    sr : int > 0
        the sampling rate of `y`

    n_fft : int > 0
        the FFT resolution

    rolloff_value : int > 0
        If provided, only estimate the groupd delay of the passband that
        above the threshold, which is the rolloff_value below the peak
        on frequency response.

    Returns
    -------
    mean_delay : float
        The mediant group delay of `y` (in seconds)

    '''
    if rolloff_value > 0:
        raise ParameterError('rolloff_value must be strictly negative')

    # frequency response
    _, h_ir = scipy.signal.freqz(y, a=1, worN=n_fft, whole=False, plot=None)

    #convert to dB(clip function avoids the zero value in log computation)
    power_ir = 20*np.log10(np.clip(np.abs(h_ir), 1e-8, 1e100))

    #set up threshold for valid range
    threshold = max(power_ir) + rolloff_value

    _, gd_ir = scipy.signal.group_delay((y, 1), n_fft)

    return np.median(gd_ir[power_ir > threshold])/sr


class ImpulseResponse(BaseTransformer):
    '''Impulse response filtering'''

    def __init__(self, files=None):
        '''Impulse response filtering

        Parameters
        ----------

        files : str or list of str
            Path to audio files on disk containing the impulse responses

        '''

        if isinstance(files, six.string_types):
            files = [files]

        BaseTransformer.__init__(self)
        self.files = files
        self.delay_ = []

        for fname in files:
            with psf.SoundFile(str(fname), mode='r') as soundf:
                ir_data = soundf.read()
                ir_sr = soundf.samplerate
                #ir_data = np.pad(ir_data,(ir_sr,0),mode = 'constant') #This is the test of a delayed impulse response            
            self.delay_.append(0.0)
            self.delay_[-1] = median_group_delay(y=ir_data,
                                                 sr=ir_sr)
        self._register('.*', self.deform_times)

    def states(self, jam):
        '''Iterate the impulse respones states'''

        state = dict()
        mudabox = jam.sandbox.muda
        state['duration'] = librosa.get_duration(y=mudabox._audio['y'],
                                                 sr=mudabox._audio['sr'])
        for i in range(len(self.files)):
            state['index'] = i
            yield state

    def audio(self, mudabox, state):
        '''Audio deformation for impulse responses'''
        idx = state['index']
        #load coresponding ir file
        with psf.SoundFile(self.files[idx], mode='r') as soundf:
            ir_data = soundf.read()

        # If the input signal isn't big enough, pad it out first
        n = len(mudabox._audio['y'])
        if n < len(ir_data):
            mudabox._audio['y'] = librosa.util.fix_length(mudabox._audio['y'],
                                                          ir_data)

        mudabox._audio['y'] = scipy.signal.fftconvolve(mudabox._audio['y'],
                                                       ir_data,
                                                       mode='full')

        # Trim back to the original duration
        mudabox._audio['y'] = mudabox._audio['y'][:n]

    def deform_times(self, annotation, state):
        '''Apply group delay for the selected filter'''
        #duration = pd.to_timedelta(state['duration'], unit='s')
        idx = state['index']
        delay = self.delay_[idx]

        for obs in annotation.pop_data():
            # Drop obervation that fell off the end
            if obs.time + delay > annotation.duration:
                #Slice annotation from start time to the annotations duration
                annotation = annotation.slice(0, annotation.duration, strict=False)
            else:
                #truncate observation's duration if its offset fell off the end of annotation
                if obs.time + obs.duration + delay > annotation.duration:
                    deformed_duration = annotation.duration - obs.time - delay
                else:
                    deformed_duration = obs.duration

            annotation.append(time=obs.time + delay,
                       duration=deformed_duration,
                       value=obs.value, confidence=obs.confidence)
