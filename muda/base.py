#!/usr/bin/env python
'''Base module components.'''

import copy
from collections import OrderedDict
import six
import inspect

from sklearn.base import _pprint


class BaseTransformer(object):
    '''The base class for all transformation objects.
    This class implements a single transformation (history)
    and some various niceties.'''

    # This bit gleefully stolen from sklearn.base
    @classmethod
    def _get_param_names(cls):
        '''Get the list of parameter names for the object'''

        init = cls.__init__

        args, varargs = inspect.getargspec(init)[:2]

        if varargs is not None:
            raise RuntimeError('BaseTransformer objects cannot have varargs')

        args.pop(0)
        args.sort()
        return args

    def get_params(self, deep=True):
        '''Get the parameters for this object.  Returns as a dict.

        Parameters
        ----------
        deep : bool
            Recurse on nested objects

        Returns
        -------
        params : dict
            A dictionary containing all parameters for this object
        '''

        out = dict(__class__=self.__class__,
                   params=dict())

        for key in self._get_param_names():
            value = getattr(self, key, None)

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out['params'][key] = dict(__class__=value.__class__)
                out['params'][key].update((k, val) for k, val in deep_items)
            else:
                out['params'][key] = value

        return out

    def __repr__(self):
        '''Pretty-print this object'''

        class_name = self.__class__.__name__
        return '{:s}({:s})'.format(class_name,
                                   _pprint(self.get_params(deep=False)['params'],
                                           offset=len(class_name),),)

    def __init__(self):
        self.dispatch = OrderedDict()

    def states(self, jam):
        raise NotImplementedError

    def _register(self, pattern, function):
        self.dispatch[pattern] = function.__name__

    def _transform(self, jam, state):
        '''Apply the transformation to audio and annotations.

        The input jam is copied and modified, and returned
        contained in a list.

        Parameters
        ----------
        jam : jams.JAMS
            A single jam object to modify

        Returns
        -------
        jam_list : list
            A length-1 list containing `jam` after transformation

        See also
        --------
        core.load_jam_audio
        '''

        if not hasattr(jam.sandbox, 'muda'):
            raise RuntimeError('No muda state found in jams sandbox.')

        # We'll need a working copy of this object for modification purposes
        jam_w = copy.deepcopy(jam)

        # Push our reconstructor onto the history stack
        jam_w.sandbox.muda['history'].append({'transformer': self.__serialize__,
                                              'state': state})

        if hasattr(self, 'audio'):
            self.audio(jam_w.sandbox.muda, state)

        if hasattr(self, 'metadata'):
            self.metadata(jam_w.file_metadata, state)

        # Walk over the list of deformers
        for query, function_name in six.iteritems(self.dispatch):
            function = getattr(self, function_name)
            for matched_annotation in jam_w.search(namespace=query):
                function(matched_annotation, state)

        return jam_w

    def transform(self, jam):
        '''Iterative transformation generator

        Applies the deformation to an input jams object.

        This generates a sequence of deformed output JAMS.

        Parameters
        ----------
        jam : jams.JAMS
            The jam to transform

        Examples
        --------
        >>> for jam_out in deformer.transform(jam_in):
                process(jam_out)
        '''

        for state in self.states(jam):
            yield self._transform(jam, state)

    @property
    def __serialize__(self):
        '''Serializer'''

        data = self.get_params()
        data['__class__'] = data['__class__'].__name__
        return data


class Pipeline(object):
    '''Wrapper which allows multiple BaseDeformer objects to be chained together

    A given JAMS object will be transformed sequentially by
    each stage of the pipeline.

    The pipeline induces a graph over transformers

    Attributes
    ----------
    steps : argument array
        steps[i] is a tuple of `(name, Transformer)`

    Examples
    --------
    >>> P = muda.deformers.PitchShift(semitones=5)
    >>> T = muda.deformers.TimeStretch(speed=1.25)
    >>> Pipe = muda.Pipeline(steps=[('Pitch:maj3', P), ('Speed:1.25x', T)])
    >>> output_jams = list(Pipe.transform(jam_in))
    '''

    def __init__(self, steps=None):

        names, transformers = zip(*steps)

        if len(set(names)) != len(steps):
            raise ValueError("Names provided are not unique: "
                             " {}".format(names,))

        # shallow copy of steps
        self.steps = list(zip(names, transformers))

        for t in transformers:
            if not isinstance(t, BaseTransformer):
                raise TypeError('{:s} is not a BaseTransformer'.format(t))

    def get_params(self):
        '''Get the parameters for this object.  Returns as a dict.'''

        out = {}
        out['__class__'] = self.__class__
        out['params'] = dict(steps=[])

        for name, step in self.steps:
            out['params']['steps'].append([name, step.get_params(deep=True)])

        return out

    def __repr__(self):
        '''Pretty-print the object'''

        class_name = self.__class__.__name__
        return '{:s}({:s})'.format(class_name,
                                   _pprint(self.get_params(),
                                           offset=len(class_name),),)

    def __recursive_transform(self, jam, steps):
        '''A recursive transformation pipeline'''

        if len(steps) > 0:
            head_transformer = steps[0][1]
            for t_jam in head_transformer.transform(jam):
                for q in self.__recursive_transform(t_jam, steps[1:]):
                    yield q
        else:
            yield jam

    def transform(self, jam):
        '''Apply the sequence of transformations to a single jam object.

        Parameters
        ----------
        jam : jams.JAMS
            The jam object to transform

        See also
        --------
        BaseTransformer.transform
        '''

        for output in self.__recursive_transform(jam, self.steps):
            yield output
