#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2020-06-17 by Jatin Khilnani <Jatin.Khilnani@nyu.edu>
"""Clipping (waveform/loudness distortion) transformations"""

import numpy as np

from ..base import BaseTransformer, _get_rng

__all__ = ["Clipping", "LinearClipping", "RandomClipping"]


class AbstractClipping(BaseTransformer):
    """Abstract base class for clipping

    This contains the deformation function 
    but does not manage state or parameters.
    """

    def __init__(self):
        BaseTransformer.__init__(self)

    @staticmethod
    def audio(mudabox, state):
        # Deform the audio
        mudabox._audio["y"] = np.clip(mudabox._audio["y"], \
                                      min(mudabox._audio["y"])*state["clip_limit"], \
                                      max(mudabox._audio["y"])*state["clip_limit"]
                                     )


class Clipping(AbstractClipping):
    """Static clipping beyond a fixed limit

    This transformation affects the following attributes:
    - Audio


    Attributes
    ----------
    clip_limit : float or list of floats, strictly (0,1)
        The amplitude fraction beyond which the waveform is clipped.

    Examples
    --------
    >>> D = muda.deformers.Clipping(clip_limit=0.75)
    >>> out_jams = list(D.transform(jam_in))

    See Also
    --------
    LinearClipping
    RandomClipping
    """

    def __init__(self, clip_limit=0.8):
        """Clipping"""
        AbstractClipping.__init__(self)

        self.clip_limit = np.atleast_1d(clip_limit).flatten()
        if np.any(self.clip_limit <= 0.0) or np.any(self.clip_limit >= 1.0):
            raise ValueError("clip_limit parameter domain is strictly (0,1).")
        self.clip_limit = self.clip_limit.tolist()

    def states(self, jam):
        for clip_limit in self.clip_limit:
            yield dict(clip_limit=clip_limit)


class LinearClipping(AbstractClipping):
    """Linearly spaced clipping.

    `n_samples` are generated with clipping spaced linearly
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
    Clipping
    RandomClipping
    """

    def __init__(self, n_samples=3, lower=0.4, upper=0.8):
        AbstractClipping.__init__(self)
        
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive.")
        
        if lower <= 0.0 or lower >= 1.0:
            raise ValueError("lower parameter domain is strictly (0,1).")

        if upper <= lower:
            raise ValueError("upper must be strictly larger than lower.")
            
        if upper >= 1.0:
            raise ValueError("upper parameter domain is strictly (0,1).")

        self.n_samples = n_samples
        self.lower = float(lower)
        self.upper = float(upper)

    def states(self, jam):
        clip_limits = np.linspace(
            self.lower, self.upper, num=self.n_samples, endpoint=True
        )

        for clip_limit in clip_limits:
            yield dict(clip_limit=clip_limit)


class RandomClipping(AbstractClipping):
    """Random clipping

    For each deformation, the clip_limit parameter is drawn from a
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
        clip_limit parameter is sampled.

    rng : None, int, or np.random.RandomState
        The random number generator state.

        If `None`, then `np.random` is used.

        If `int`, then `rng` becomes the seed for the random state.

    See Also
    --------
    Clipping
    LinearClipping
    """

    def __init__(self, n_samples=3, a=1.0, b=1.0, rng=None):

        AbstractClipping.__init__(self)
        
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive.")

        if a <= 0.0:
            raise ValueError("a(alpha) parameter must be strictly positive.")
            
        if b <= 0.0:
            raise ValueError("b(beta) parameter must be strictly positive.")

        self.n_samples = n_samples
        self.a = a
        self.b = b
        self.rng = rng
        self._rng = _get_rng(rng)

    def states(self, jam):
        clip_limits = self._rng.beta(
            a=self.a, b=self.b, size=self.n_samples
        )

        for clip_limit in clip_limits:
            yield dict(clip_limit=clip_limit)
