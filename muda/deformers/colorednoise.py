#path:../site-packages/muda/deformers/colorednoise.py

'''Additive colored noise'''

import numpy as np
import librosa
from numpy import inf

from ..base import BaseTransformer

NOISE_TYPES = ['white',
              'pink',
              'brownian']

def noise_generator(y, sr, color, seed):
    '''generating noise given the type of color, length of
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

    Returns
    -------
    y : np.ndarray [shape=(n_samples,)]
        A fragment of noise clip that generated given the type of color
        and length.

    '''
    n_frames = len(y)

    if seed:
        np.random.seed(True)
    noise_white = np.random.randn(n_frames)

    noise_fft = np.fft.rfft(noise_white)

    if color == 'pink':
        colored_filter = np.sqrt(np.linspace(1, n_frames/2 + 1, n_frames/2 + 1))**(-1)

    elif color == 'brownian':
        colored_filter = np.linspace(1, n_frames/2 + 1, n_frames/2 + 1)**(-1)

    else:
        colored_filter = np.linspace(1, n_frames/2 + 1, n_frames/2 + 1)**0 #default white

    noise_filtered = noise_fft * colored_filter

    return np.fft.irfft(noise_filtered)


class ColoredNoise(BaseTransformer):
    '''Abstract base class for colored noise

    This contains several noise generator that generating different colored
    noise given the desired type and the length of clip data for degrading
    '''

    def __init__(self, n_samples, color=None, weight_min=0.1, weight_max=0.5, seed = None ):

        if n_samples <= 0:
            raise ValueError('n_samples must be strictly positive')

        if not 0 < weight_min < weight_max < 1.0:
            raise ValueError('weights must be in the range (0.0, 1.0)')

        BaseTransformer.__init__(self)

        self.n_samples = n_samples
        self.color = color
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.seed = seed

    def states(self, jam):
        mudabox = jam.sandbox.muda
        for _ in range(self.n_samples):
            for type_name in self.color:
                if type_name not in NOISE_TYPES:
                    raise ValueError("Incorrect color type. Color parameter must from [white, pink, brownian] and be a list strictly")
                yield dict(colortype = type_name,
                           weight=np.random.uniform(low=self.weight_min,
                                                    high=self.weight_max,
                                                    size=None),
                            seed = self.seed)

    def audio(self, mudabox, state):

        weight = state['weight']
        colortype = state['colortype']
        seed = state['seed']

        #Generating the noise data
        noise = noise_generator(y = mudabox._audio['y'],
                                sr = mudabox._audio['sr'],
                                color = colortype,
                                seed = seed)

        # Normalize the data
        mudabox._audio['y'] = librosa.util.normalize(mudabox._audio['y'])
        noise = librosa.util.normalize(noise)

        mudabox._audio['y'] = ((1.0 - weight) * mudabox._audio['y'] +
                               weight * noise)
