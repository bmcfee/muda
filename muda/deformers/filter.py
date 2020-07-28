#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2020-07-11 by Han Han <hh2263@nyu.edu>
"""Filtering (low/band/high-pass) algorithms"""

import numpy as np
from scipy import signal
from copy import deepcopy
import numpy as np

from ..base import BaseTransformer, _get_rng

__all__ = ["Filter", "RandomLPFilter", "RandomHPFilter","RandomBPFilter"]


def checkfreqinband(freq,state,datatype):
    #check if frequency falls into the passband, frequency received in hertz
    #convert frequency to hertz
    if datatype == "midi":
        frequency = 2**((freq-69)/12)*440
    elif datatype == "hz":
        frequency = freq
    elif datatype == "pitchclass":
        frequency = pitch2freq(freq)
    #check if it falls into the passband
    if state["btype"] == "bandpass":
        low,high = state["cut_off"]
    elif state["btype"] == "low":
        high = state["cut_off"]
        low = 0
    else:
        high = state["nyquist"]
        low = state["cut_off"]

    if frequency <=low:
        return None,False
    elif frequency >= high:
        return None,False
    else:
        return freq, True

def pitch2freq(pitchclass):
    A4 = 440
    C0 = A4*pow(2, -4.75)
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    tonic = pitchclass[:-1]
    pitch = int(pitchclass[-1])
    n = name.index(tonic)
    h = pitch * 12 + n
    freq = C0 * 2 **(h/12)
    return freq


class AbstractFilter(BaseTransformer):
    """Abstract base class for Filtering transformations"""


    def __init__(self):
        """Abstract base class for Filtering Transformations.

        This implements the deformations, but does not manage state.
        """

        BaseTransformer.__init__(self)

        # Build the annotation mapping 
        self._register("pitch_contour", self.filter_contour)
        self._register("pitch_hz|note_hz", self.filter_hz)
        self._register("pitch_midi|note_midi", self.filter_midi)
        self._register("pitch_class", self.filter_class)


    def states(self,jam):   
        mudabox = jam.sandbox.muda
        state = dict( 
            nyquist=mudabox._audio["sr"]/2 
        )
        yield state

    @staticmethod
    def audio(mudabox, state):

        if state["btype"] == "bandpass":
            low,high = state["cut_off"]
            sos = signal.cheby2(
                state["order"], 
                state["attenuation"]/2, 
                [low,high],
                btype=state["btype"], 
                output='sos',
                fs=mudabox._audio["sr"])
        else:
            sos = signal.cheby2(
                state["order"], 
                state["attenuation"]/2, 
                state["cut_off"],
                btype=state["btype"], 
                output='sos',
                fs=mudabox._audio["sr"])

        mudabox._audio["y"] = signal.sosfiltfilt(sos, mudabox._audio["y"])

    

    @staticmethod
    def filter_contour(annotation, state): 
    #same length after modification
        for obs in annotation.pop_data():
            new_freq,voice = checkfreqinband(obs.value["frequency"],state,datatype="hz")
            annotation.append(
                time=obs.time,
                duration=obs.duration,
                confidence=obs.confidence,
                value={
                    "index": obs.value["index"],
                    "frequency": new_freq,
                    "voiced": voice,
                },
            )

    @staticmethod
    def filter_hz(annotation, state):
        #non-existent pitch removed 
        for obs in annotation.pop_data():
            new_freq,voice = checkfreqinband(obs.value,state,datatype="hz")
            if voice:
                annotation.append(
                    time=obs.time,
                    duration=obs.duration,
                    confidence=obs.confidence,
                    value=new_freq,
                )

    @staticmethod
    def filter_midi(annotation, state):
         #non-existent pitch removed 
        for obs in annotation.pop_data():
            new_midi,voice = checkfreqinband(obs.value,state,datatype="midi")
            if voice:
                annotation.append(
                    time=obs.time,
                    duration=obs.duration,
                    confidence=obs.confidence,
                    value=new_midi,
                )

    @staticmethod
    def filter_class(annotation, state):
        #non-existent pitch removed 
        for obs in annotation.pop_data():
            value = deepcopy(obs.value)
            
            new_freq, voice = checkfreqinband(value["tonic"]+str(value["pitch"]),state,"pitchclass")
            if voice:
                value["tonic"] = new_freq[:-1]
                value["pitch"] = int(new_freq[-1])
                annotation.append(
                time=obs.time,
                duration=obs.duration,
                confidence=obs.confidence,
                value=value,
            )
            
           


class Filter(AbstractFilter):
    """ Filtering by cheby2 iir filter
    
    This transformation affects the following attributes:

    - Annotations 
        - pitch_contour, pitch_hz, pitch_midi, pitch_class
        - note_hz, note_midi
    - Audio

    Attributes
    ----------
    type: "low" or "high" or "bandpass"
    order: int > 0
        order of the filter
    attenutation: float > 0
        The minimum attenuation required in the stop band. 
        Specified in decibels, as a positive number.

    cutoff: in hz
        can be float, list of float, or list of tuples in the case of bandpass filter

    make one or more filters of the same type, but customized cutoff frequencies

    See Also
    --------
    RandomFilter

    Examples
    --------
    >>> # Shift down by a quarter-tone
    >>> D = muda.deformers.Filter(btype,order,attenuation,cutoff)
    """

    def __init__(self,btype="low",order=4, attenuation=60.0, cutoff=4000):
        AbstractFilter.__init__(self)
        self.btype = btype
        self.order = order
        self.attenuation = attenuation
        if self.btype == "bandpass":
            if type(cutoff) == tuple:
                self.cutoff = [cutoff]
            elif type(cutoff) == list:
                if all(type(i) == tuple for i in cutoff):
                    self.cutoff = cutoff
                elif all(type(i) == list for i in cutoff):
                    if all(len(i) == 2 for i in cutoff): # [[a,b],[c,d]]
                        self.cutoff = [tuple(c) for c in cutoff]
                    else:
                        raise ValueError("bandpass filter cutoff must be tuple or list of tuples")
                else: 
                    raise ValueError("bandpass filter cutoff must be tuple or list of tuples")
            else:
                raise ValueError("bandpass filter cutoff must be tuple or list of tuples")

        else:
            if type(cutoff) == tuple:
                raise ValueError("low/high pass filter cutoff must be float or list of floats")
            elif type(cutoff)==list and type(cutoff[0])==tuple:
                raise ValueError("low/high pass filter cutoff must be float or list of floats")

            else:
                self.cutoff = np.atleast_1d(cutoff).flatten().tolist()
        
 
    def states(self, jam):
        for state in AbstractFilter.states(self, jam):
            if self.btype == "bandpass":
                for low,high in self.cutoff:
                    if low > high:
                        raise ValueError("cutoff_low must be smaller than cutoff_high")
                    else:
                        state["cut_off"] = (low,high)
                        state["order"] = self.order
                        state["attenuation"] = self.attenuation
                        state["btype"] = self.btype
                        yield state
            elif self.btype == "low":
                for freq in self.cutoff:
                    if freq <= 0:
                        raise ValueError("cutoff frequency for lowpass filter must be strictly positive")
                    else:
                        state["cut_off"] = freq
                        state["order"] = self.order
                        state["attenuation"] = self.attenuation
                        state["btype"] = self.btype
                        yield state
            elif self.btype == "high":
                for freq in self.cutoff:
                    if freq <= 0 or freq >= state["nyquist"]:
                        raise ValueError("cutoff frequency for high pass filter must be strictly positive and smaller than nyquist frequency")
                    else:
                        state["cut_off"] = freq
                        state["order"] = self.order
                        state["attenuation"] = self.attenuation
                        state["btype"] = self.btype
                        yield state



class RandomLPFilter(AbstractFilter):
    """ Filtering by cheby2 iir filter

    This transformation affects the following attributes:

    - Annotations 
        - pitch_contour, pitch_hz, pitch_midi, pitch_class
        - note_hz, note_midi
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate per input

    order: int > 0
        order of the filter
    attenuation: float > 0
        The minimum attenuation required in the stop band. 
        Specified in decibels, as a positive number.
    cutoff: float in hz
        low pass cutoff frequency
    
    
    sigma : float > 0
        The parameters of the normal distribution for sampling
        pitch shifts

    rng : None, int, or np.random.RandomState
        The random number generator state.

        If `None`, then `np.random` is used.

        If `int`, then `rng` becomes the seed for the random state.

    See Also
    --------
    Filter

    Examples
    --------
    >>> # Shift down by a quarter-tone
    >>> D = muda.deformers.RandomLPFilter(n_samples,order,attenuation,cutoff,sigma)
    """

    def __init__(self, n_samples=3, order=4, attenuation=60.0, cutoff=8000,sigma=1.0,rng=0):
        AbstractFilter.__init__(self)
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")

        if n_samples <= 0:
            raise ValueError("n_samples must be None or positive")

        if type(cutoff)==float and cutoff<=0:
            raise ValueError("cutoff frequency must be None or positive")

        if type(cutoff)==list and sum(np.array(cutoff)<=0)>0:
            raise ValueError("cutoff frequency must be None or positive")

        if order <= 0 or attenuation <= 0:
            raise ValueError("order and attenuation must be None or positive")



        self.n_samples = n_samples
        self.order = order
        self.attenuation = attenuation
        self.sigma = float(sigma)
        self.rng = rng
        self._rng = _get_rng(rng)
        self.cutoff = float(cutoff)
      


    #specify and stores the type/parameters of the augmentation
    def states(self, jam):
        for state in AbstractFilter.states(self, jam):
            for _ in range(self.n_samples):
                state["btype"] = "low"
                state["order"] = self.order
                state["attenuation"] = self.attenuation
                state["cut_off"] = self._rng.normal(
                        loc=self.cutoff, scale=self.sigma, size=None
                    )
                
                yield state




class RandomHPFilter(AbstractFilter):
    """ Filtering by cheby2 iir filter

    This transformation affects the following attributes:

    - Annotations 
        - pitch_contour, pitch_hz, pitch_midi, pitch_class
        - note_hz, note_midi
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate per input

    order: int > 0
        order of the filter
    attenuation: float > 0
        The minimum attenuation required in the stop band. 
        Specified in decibels, as a positive number.
    
    cutoff: float in hz
        high pass cutoff frequency
    
    sigma : float > 0
        The parameters of the normal distribution for sampling
        pitch shifts

    rng : None, int, or np.random.RandomState
        The random number generator state.

        If `None`, then `np.random` is used.

        If `int`, then `rng` becomes the seed for the random state.


    See Also
    --------
    Filter

    Examples
    --------
    >>> # Shift down by a quarter-tone
    >>> D = muda.deformers.RandomLPFilter(m_samples,order,attenuation,cutoff,sigma)
    """

    def __init__(self, n_samples=3, order=4, attenuation=60.0, cutoff=8000,sigma=1.0,rng=0):
        AbstractFilter.__init__(self)
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")

        if n_samples <= 0:
            raise ValueError("n_samples must be None or positive")

        if order <= 0 or attenuation <= 0:
            raise ValueError("order and attenuation must be None or positive")

        if type(cutoff)==list or cutoff<=0:
            raise ValueError("high pass cutoff frequency must be strictly positive and lower than nyquist frequency")


        self.n_samples = n_samples
        self.order = order
        self.attenuation = attenuation
        self.sigma = float(sigma)
        self.rng = rng
        self._rng = _get_rng(rng)
        self.cutoff = float(cutoff)
      


    #specify and stores the type/parameters of the augmentation

    def states(self, jam):
        for state in AbstractFilter.states(self, jam):
            for _ in range(self.n_samples):
                state["btype"] = "high"
                state["order"] = self.order
                state["attenuation"] = self.attenuation
                state["cut_off"] = self._rng.normal(
                        loc=self.cutoff, scale=self.sigma, size=None
                    )
                
                yield state


class RandomBPFilter(AbstractFilter):
    """ Filtering by cheby2 iir filter

    This transformation affects the following attributes:

    - Annotations 
        - pitch_contour, pitch_hz, pitch_midi, pitch_class
        - note_hz, note_midi
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate per input

    order: int > 0
        order of the filter
    attenuation: float > 0
        The minimum attenuation required in the stop band. 
        Specified in decibels, as a positive number.
    
    cutoff: float in hz
        high pass cutoff frequency
        
    
    
    sigma : float > 0
        The parameters of the normal distribution for sampling
        pitch shifts

    rng : None, int, or np.random.RandomState
        The random number generator state.

        If `None`, then `np.random` is used.

        If `int`, then `rng` becomes the seed for the random state.


    See Also
    --------
    Filter, RandomHPFilter, RandomLPFilter

    Examples
    --------
    >>> # Shift down by a quarter-tone
    >>> D = muda.deformers.RandomBPFilter(n_samples,order,attenuation,cutoff_low,cutoff_high,sigma)
    """

    def __init__(self, n_samples=3, order=4, attenuation=60.0, cutoff_low=4000, cutoff_high=8000,sigma=1.0,rng=0):
        AbstractFilter.__init__(self)
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")

        if n_samples <= 0:
            raise ValueError("n_samples must be None or positive")

        if order <= 0 or attenuation <= 0:
            raise ValueError("order and attenuation must be None or positive")

        if cutoff_low >= cutoff_high:
            raise ValueError("band pass higher cutoff frequency must be strictly greater than lower cutoff frequency")
        if cutoff_low<=0 or cutoff_high <=0:
            raise ValueError("band pass cutoff frequency must be strictly greater than zero")

        self.n_samples = n_samples
        self.order = order
        self.attenuation = attenuation
        self.sigma = float(sigma)
        self.rng = rng
        self._rng = _get_rng(rng)
        self.cutoff_low = float(cutoff_low)
        self.cutoff_high = float(cutoff_high)


    #specify and stores the type/parameters of the augmentation

    def states(self, jam):
        for state in AbstractFilter.states(self, jam):
            for _ in range(self.n_samples):

                #make sure higher bound is lower than lower bound
                high = self._rng.normal(
                        loc=self.cutoff_high, scale=self.sigma, size=None
                    )
                low = self._rng.normal(
                        loc=self.cutoff_low, scale=self.sigma, size=None
                    )
                while high <= low:
                    high = self._rng.normal(
                        loc=self.cutoff_high, scale=self.sigma, size=None
                    )
                    low = self._rng.normal(
                        loc=self.cutoff_low, scale=self.sigma, size=None
                    )
                state["btype"] = "bandpass"
                state["cut_off"] = (low,high)
                state["order"] = self.order
                state["attenuation"] = self.attenuation
                

                
                yield state


