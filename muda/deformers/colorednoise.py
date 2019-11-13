#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Additive colored noise"""

import numpy as np
import librosa

from ..base import BaseTransformer, _get_rng

NOISE_TYPES = ["white", "pink", "brownian"]


def noise_generator(y, sr, color, rng):
    """generating noise given the type of color, length of
       the degrading audio clip and its sampling rate.

    Parameters
    ----------
    y : int > 0
        compute frame length of y, as the length of noise fragment
        to be generated.

    sr : int > 0
        The target sampling rate

    color : str
        keywords of desired noise color

    rng : np.random.RandomState
        The random state object

    Returns
    -------
    y : np.ndarray [shape=(n_samples,)]
        A fragment of noise clip that generated given the type of color
        and length.

    """
    n_frames = len(y)

    noise_white = rng.randn(n_frames)

    noise_fft = np.fft.rfft(noise_white)

    values = np.linspace(1, n_frames * 0.5 + 1, n_frames // 2 + 1)

    if color == "pink":
        colored_filter = values ** (-0.5)

    elif color == "brownian":
        colored_filter = values ** (-1)

    else:
        # default white
        colored_filter = np.linspace(1, n_frames / 2 + 1, n_frames // 2 + 1) ** 0

    noise_filtered = noise_fft * colored_filter

    return np.fft.irfft(noise_filtered)


class ColoredNoise(BaseTransformer):
    """Abstract base class for colored noise

    This contains several noise generator that generating different colored
    noise given the desired type and the length of clip data for degrading
    """

    def __init__(self, n_samples, color=None, weight_min=0.1, weight_max=0.5, rng=None):

        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive")

        if not 0 < weight_min < weight_max < 1.0:
            raise ValueError("weights must be in the range (0.0, 1.0)")

        BaseTransformer.__init__(self)

        self.n_samples = n_samples
        self.color = color
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.rng = rng
        self._rng = _get_rng(rng)

    def states(self, jam):
        for _ in range(self.n_samples):
            for type_name in self.color:
                if type_name not in NOISE_TYPES:
                    raise ValueError(
                        "Incorrect color type. Color parameter must from [white, pink, brownian] and be a list strictly"
                    )
                yield dict(
                    color=type_name,
                    weight=self._rng.uniform(
                        low=self.weight_min, high=self.weight_max, size=None
                    ),
                )

    def audio(self, mudabox, state):

        weight = state["weight"]
        color = state["color"]

        # Generating the noise data
        noise = noise_generator(
            mudabox._audio["y"], mudabox._audio["sr"], color, self._rng
        )

        # Normalize the data
        mudabox._audio["y"] = librosa.util.normalize(mudabox._audio["y"])
        noise = librosa.util.normalize(noise)

        mudabox._audio["y"] = (1.0 - weight) * mudabox._audio["y"] + weight * noise
