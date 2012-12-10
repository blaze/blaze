from collections import Mapping, OrderedDict

class params(Mapping):
    """
    Container for parameters
    """
    __slots__ = ['cls', '__internal']

    def __init__(self, **kw):
        self.__internal = OrderedDict(kw)

        if not params.cls:
            params.cls = frozenset(dir(self))

    def __setattr__(self, key, value):
        if key == 'cls' or key == '__internal' or '_params' in key:
            super(params, self).__setattr__(key, value)
        else:
            self.__internal[key] = value
        return value

    def __getattr__(self, key):
        if key in self.cls:
            super(params, self).__getattr__(key)
        else:
            return self.__internal[key]

    def __getitem__(self, key):
        return self.__internal[key]

    def __contains__(self, key):
        return key in self.__internal

    def __len__(self):
        return len(self.__internal)

    def __iter__(self):
        return self.__internal.iterkeys()

    def iteritems(self):
        return self.__internal.items()

    def __repr__(self):
        return 'params({keys})'.format(
            keys=''.join('%s=%s, ' % (k,v) for k,v in self.__internal.items())
        )

    @classmethod
    def empty(cls, header):
        return cls(dct=dict(header))
