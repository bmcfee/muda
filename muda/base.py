#!/usr/bin/env python
'''Base module components.'''

import copy

import inspect

from sklearn.base import _pprint

import six


class BaseTransformer(object):
    '''The base class for all transformation objects.
    This class implements a single transformation (history)
    and some various niceties.'''

    # This bit gleefully stolen from sklearn.base
    @classmethod
    def _get_param_names(cls):
        '''Get the list of parameter names for the object'''

        init = cls.__init__

        if init is object.__init__:
            return []

        args, varargs = inspect.getargspec(init)[:2]

        if varargs is not None:
            raise RuntimeError('varargs ist verboten')

        args.pop(0)
        args.sort()
        return args

    def get_params(self, deep=True):
        '''Get the parameters for this object.  Returns as a dict.'''

        out = dict()

        for key in self._get_param_names():
            value = getattr(self, key, None)

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def __repr__(self):
        '''Pretty-print this object'''

        class_name = self.__class__.__name__
        return '{:s}({:s})'.format(class_name,
                                   _pprint(self.get_params(deep=False),
                                           offset=len(class_name),),)

    def __init__(self):
        '''Base-class initialization'''
        self.dispatch = dict()

        # A cache for shared state among deformation objects
        self._state = dict()

    def transform(self, jam):
        '''Apply the transformation to audio and annotations.

        The input jam is copied and modified, and returned
        contained in a list.

        Parameters
        ----------
        jam : MUDA augmented pyjams.JAMS


        Returns
        -------
        jam_list : list
            A length-1 list containing `jam` after transformation

        See also
        --------
        muda.load_jam_audio
        '''

        if not hasattr(jam.sandbox, 'muda'):
            raise RuntimeError('No muda state found in jams sandbox.')

        # We'll need a working copy of this object for modification purposes
        jam_working = copy.deepcopy(jam)

        # Push our reconstructor onto the history stack
        jam_working.sandbox.muda['history'].append(self.__json__)

        if hasattr(self, 'audio'):
            self.audio(jam_working.sandbox.muda,
                       jam_working.file_metadata)

        # Walk over the list of deformers
        for query, function in six.iteritems(self.dispatch):
            for matched_annotation in jam_working.search(namespace=query):
                function(matched_annotation)

        return [jam_working]

    @property
    def __json__(self):
        '''Serializer'''

        return dict(name=self.__class__.__name__,
                    params=self.get_params())


class IterTransformer(BaseTransformer):
    '''Base class for stochastic or sequential transformations.
    If your transformation can generate multiple versions of a single input,
    (eg, by sampling multiple times), then this is for you.'''

    def __init__(self, n_samples):
        '''Iterative transformation objects can generate
        multiple outputs from each input.

        Parameters
        ----------
        n_samples : int or None
            Maximum number of samples to generate.
            If None, run indefinitely.

        '''

        BaseTransformer.__init__(self)

        self.n_samples = n_samples


    def transform(self, jam):
        '''Iterative transformations'''

        # Apply the transformation up to n_samples times
        i = 0
        while self.n_samples is None or i < self.n_samples:
            # Reset the state
            self._state = {}
            for jam_out in BaseTransformer.transform(self, jam):
                yield jam_out
                i += 1


class Pipeline(object):
    '''Wrapper which allows multiple transformers to be chained together'''

    def __init__(self, *steps):
        '''Parameters: one or more tuples of the form (name, TransformerObject).

        :example:
            >>> P = PitchShift(semitones=5)
            >>> T = TimeStretch(speed=1.25)
            >>> Pipe = Pipeline( ('Pitch:maj3', P), ('Speed:1.25x', T) )
            >>> output = Pipe.transform(data)
        '''

        self.named_steps = dict(steps)
        names, transformers = zip(*steps)

        if len(self.named_steps) != len(steps):
            raise ValueError("Names provided are not unique: "
                             " {:s}".format(names,))

        # shallow copy of steps
        self.steps = list(zip(names, transformers))

        for t in transformers:
            if not isinstance(t, BaseTransformer):
                raise TypeError('{:s} is not of type BaseTransformer')

    @property
    def __json__(self):
        '''Serialize the pipeline'''

        return dict(name=self.__class__.__name__,
                    params=[(name, t.__json__) for (name, t) in self.steps])

    def get_params(self):
        '''Get the parameters for this object.  Returns as a dict.'''

        out = self.named_steps.copy()
        for name, step in self.named_steps.iteritems():
            for key, value in step.get_params(deep=True).iteritems():
                out['{:s}__{:s}'.format(name, key)] = value
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
        '''Apply the sequence of transformations to a single jam object'''

        for output in self.__recursive_transform(jam, self.steps):
            yield output
