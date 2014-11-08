#!/usr/bin/env python
'''Base module components.'''

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

        if init is object.__init__:
            return []

        args, varargs, kw, default = inspect.getargspec(init)

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

    def transform(self, payload, top_level=True):
        '''Recursive transformation function.

        For each key in the payload object, we search for an appropriate
        transformation method, and apply it if found.

        If no transformer can be found, but the corresponding value is
        a dictionary, the method recurses.

        An additional 'history' field may be appended at the top level to
        track the series of modifications applied from the original.
        '''

        output = payload.copy()

        if top_level:
            output.setdefault('history', [])

        for key, value in output.iteritems():

            tx_func_name = '_{:s}'.format(key)

            if hasattr(self, tx_func_name):
                output[key] = getattr(self, tx_func_name)(value)

            elif isinstance(value, dict):
                output[key] = self.transform(value, top_level=False)

        return output

    def _history(self, payload):
        '''History transformation method'''
        # Push repr(self) onto payload
        return payload + [repr(self)]
