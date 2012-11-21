from contextlib import contextmanager
from collections import MutableMapping

@contextmanager
def nobuiltins():
    """
    Don't clobber builtins
    """
    gls = globals()
    builtins = gls.pop('__builtins__')
    yield
    gls['__builtins__'] = builtins

class ReverseLookupDict(MutableMapping):
    def __init__(self, inits):
        self._map = {}
        self.update(inits)

    def __getitem__(self, key):
        return self._map.__getitem__(key)

    def __setitem__(self, key, val):
        self._map.__setitem__(key, val)
        self._map.__setitem__(val, key)

    def __delitem__(self, key):
        self._map.__delitem__(self[key])
        self._map.__delitem__(key)

    def __iter__(self):
        return self._map.__iter__()

    def __len__(self):
        return self._map
