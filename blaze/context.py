"""
Environment used for type reconstruction algorithms.
"""

from itertools import chain

class Env(object):

    def __init__(self, *evs):
        self.evs = [dict()] + list(evs)
        self.iev = self.evs[0]

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

    def __len__(self):
        return sum(len(e) for e in self.evs)

    def __getitem__(self, key):
        return self.lookup(key)

    def update(self, other):
        if isinstance(other, Env):
            self.iev.update(other.iev)
        else:
            self.iev.update(other)

    def __setitem__(self, key, value):
        self.iev[key] = value

    def __contains__(self, key):
        for e in self.evs:
            if key in e:
                return True
        return False

    def __iter__(self):
        return chain(*[e.keys() for e in self.evs])

    def iterkeys(self):
        return self.__iter__()

    def __repr__(self):
        return 'Env(' + repr(self.evs) + ')'
