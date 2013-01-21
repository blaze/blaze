# -*- coding: utf-8 -*-

"""
Environment used for type reconstruction algorithms.

"""

from itertools import chain

class Env(object):
    """
    Type environment holds the associatition between bound variable and
    type value within the expression scope. Implemented as a ordered
    list of associative maps where lookup resolves to the first map
    containing the variable name.
    """

    def __init__(self, *evs):
        self.evs = [dict()] + list(evs)
        self.iev = self.evs[0]

    @classmethod
    def empty(self):
        """ Null environment """
        return Env()

    def lookup(self, key):
        for e in self.evs:
            if key in e:
                return e[key]
        raise KeyError, key

    def foldl(self, key):
        t = key
        while key in self:
            t = self[key]
        return t

    def collapse(self):
        iev = {}
        for e in self.evs:
            iev.update(e)
        return Env(iev)

    def has_key(self, key):
        for e in self.maps:
            if key in e:
                return True
        return False

    def update(self, other):
        if isinstance(other, Env):
            self.iev.update(other.iev)
        else:
            self.iev.update(other)

    def iterkeys(self):
        return self.__iter__()

    def __len__(self):
        return sum(len(e) for e in self.evs)

    def __getitem__(self, key):
        return self.lookup(key)

    def __setitem__(self, key, value):
        self.iev[key] = value

    def __contains__(self, key):
        for e in self.evs:
            if key in e:
                return True
        return False

    def __iter__(self):
        return chain(*[e.keys() for e in self.evs])

    def __repr__(self):
        return 'Env(' + repr(self.evs) + ')'
