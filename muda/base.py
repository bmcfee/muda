#!/usr/bin/env python
'''Base module components.'''

# import copy

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

    def transform(self, jam):
        '''Apply the transformation to audio and annotations.'''

        if not hasattr(jam.sandbox, 'muda'):
            raise RuntimeError('No muda state found in jams sandbox.')

        # If we're iterable, local copies will have to be made
#         sandbox = copy.deepcopy(jam.sandbox)

        # Push repr(self) onto the history stack
        jam.sandbox.muda['history'].append(repr(self))

        if hasattr(self, 'audio'):
            y, sr = self.audio(jam.sandbox.muda['y'],
                               jam.sandbox.muda['sr'])
            jam.sandbox.muda['y'] = y
            jam.sandbox.muda['sr'] = sr

#         annotations = copy.deepcopy(jam.annotations)

        for query, function in six.iteritems(self.dispatch):
            for matched_annotation in jam.search(namespace=query):
                function(matched_annotation)

        return jam

        # Undo the damage of this deformation stage
#         jam.sandbox = sandbox
#         jam.annotations = annotations

    def count_deformers(self):
        '''Verify that at most 1 deformer is a generator'''

        n_generators = 0
        for function in six.itervalues(self.dispatch):
            if inspect.isgeneratorfunction(function):
                n_generators += 1

        if n_generators > 0:
            raise RuntimeError('At most no deformations can be generator.')

        return n_generators


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
        print self.named_steps
        names, transformers = zip(*steps)

        if len(self.named_steps) != len(steps):
            raise ValueError("Names provided are not unique: "
                             " {:s}".format(names,))

        # shallow copy of steps
        self.steps = list(zip(names, transformers))

        for t in transformers:
            if not hasattr(t, 'transform'):
                raise TypeError("All intermediate steps a the chain should "
                                "derive from the TransformMixin class"
                                " '{:s}' (type {:s}) doesn't implement "
                                " transform().".format(t, type(t)))

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

    def transform(self, payload):
        '''Apply the sequence of transformations'''

        output = payload.copy()

        for name, tx in self.steps:
            output = tx.transform(output)

        return output
