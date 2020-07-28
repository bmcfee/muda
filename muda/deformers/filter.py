#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2020-07-26 by Jatin Khilnani <Jatin.Khilnani@nyu.edu>
"""Filter (low-, band-, high-pass) transformations"""

import numpy as np
from scipy.signal import cheby2, sosfiltfilt

from ..base import BaseTransformer, _get_rng

__all__ = ["Filter", "LinearFilter", "RandomFilter"]


class AbstractFilter(BaseTransformer):
    """Abstract base class for filtering

    This contains the deformation function 
    but does not manage state or parameters.
    """

    def __init__(self):
        BaseTransformer.__init__(self)

    @staticmethod
    def audio(mudabox, state):
        # Filter parameters
        fs = mudabox._audio["sr"]
        sos = cheby2(N=5, rs=18, Wn=2*np.array(state['cutoff'])/fs, \
                     btype=state['filter_type'], analog=False, output='sos', fs=fs)

        # Deform the audio
        mudabox._audio["y"] = sosfiltfilt(sos, mudabox._audio["y"])


class Filter(AbstractFilter):
    """Static filter around fixed limit(s)

    This transformation affects the following attributes:
    - Audio


    Attributes
    ----------
    cutoff : float, list of floats or list of float pairs, strictly positive
        The cutoff frequency for the filter.
    filter_type : string
        The argument for type of filter low-, band- or high-pass

    Examples
    --------
    >>> D = muda.deformers.Filter(filter_type='highpass', cutoff=2)
    >>> out_jams = list(D.transform(jam_in))

    See Also
    --------
    LinearFilter
    RandomFilter
    """

    def __init__(self, filter_type='lowpass', cutoff=2):
        AbstractFilter.__init__(self)

        # Validation
        if filter_type not in ['lowpass', 'highpass', 'bandpass']:
            raise ValueError('filter_type not valid.')

        if filter_type == 'bandpass':
            self.cutoff = np.atleast_2d(cutoff)

            for cutoff in self.cutoff:
                if len(cutoff) != 2:
                    raise ValueError('cutoff should be (fmin, fmax) for band-pass filter')
        else:
            self.cutoff = np.atleast_1d(cutoff)

        if np.any(self.cutoff.flatten() <= 0.0):
            raise ValueError("cutoff must be strictly positive.")

        self.filter_type = filter_type
        self.cutoff = self.cutoff.tolist()

    def states(self, jam):
        for cutoff in self.cutoff:
            yield dict(filter_type=self.filter_type,
                       cutoff=cutoff)


class LinearFilter(AbstractFilter):
    """Linearly spaced filtering.

    `n_samples` are generated with cutoff spaced linearly
    between `lower` and `upper`.

    This transformation affects the following attributes:
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        Number of deformations to generate

    lower : float > 0.0
    upper : float in (lower, 1.0)
        Minimum and maximum bounds on the clip parameters

    See Also
    --------
    Filter
    RandomFilter
    """

    def __init__(self, filter_type='lowpass', n_samples=3, lower=2, upper=20):
        AbstractFilter.__init__(self)

        # Validation
        if filter_type not in ['lowpass', 'highpass', 'bandpass']:
            raise ValueError('filter_type not valid.')
        
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive.")
        
        if lower <= 0.0:
            raise ValueError("lower must be strictly positive.")

        if upper <= lower:
            raise ValueError("upper must be strictly larger than lower.")

        self.filter_type = filter_type
        if filter_type == 'bandpass':
            self.n_samples = n_samples + 1
        else:
            self.n_samples = n_samples
        self.lower = float(lower)
        self.upper = float(upper)

    def states(self, jam):
        cutoff_list = np.linspace(
            self.lower, self.upper, num=self.n_samples, endpoint=True
        )

        for i, cutoff in enumerate(cutoff_list):
            cutoff = list(cutoff)
            if self.filter_type == 'bandpass':
                if i == len(cutoff_list)-1: break
                cutoff.append(cutoff[i+1])

            yield dict(
                        filter_type=self.filter_type,
                        cutoff=cutoff
                      )


class RandomFilter(AbstractFilter):
    """Random filter

    For each deformation, the cutoff parameter is drawn from a
    Beta distribution with parameters `(a, b)`

    This transformation affects the following attributes:
    - Audio

    Attributes
    ----------
    n_samples : int > 0
        The number of samples to generate per input

    a : float > 0.0
    b : float > 0.0
        Parameters of the Beta distribution from which
        cutoff parameter is sampled.

    rng : None, int, or np.random.RandomState
        The random number generator state.

        If `None`, then `np.random` is used.

        If `int`, then `rng` becomes the seed for the random state.

    See Also
    --------
    Filter
    LinearFilter
    """

    def __init__(self, filter_type='lowpass', n_samples=3, a=1.0, b=1.0, rng=None):
        AbstractFilter.__init__(self)

        # Validation
        if filter_type not in ['lowpass', 'highpass', 'bandpass']:
            raise ValueError('filter_type not valid.')
        
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive.")

        if a <= 0.0:
            raise ValueError("a(alpha) parameter must be strictly positive.")
            
        if b <= 0.0:
            raise ValueError("b(beta) parameter must be strictly positive.")

        self.filter_type = filter_type
        if filter_type == 'bandpass':
            self.n_samples = n_samples + 1
        else:
            self.n_samples = n_samples
        self.a = a
        self.b = b
        self.rng = rng
        self._rng = _get_rng(rng)

    def states(self, jam):

        cfactor = 20
        cutoff_list = cfactor*self._rng.beta(
            a=self.a, b=self.b, size=self.n_samples
        )

        for i, cutoff in enumerate(cutoff_list):
            cutoff = list(cutoff)
            if self.filter_type == 'bandpass':
                if i == len(cutoff_list)-1: break
                cutoff.append(cutoff[i+1])

            yield dict(
                        filter_type=self.filter_type,
                        cutoff=cutoff
                      )
