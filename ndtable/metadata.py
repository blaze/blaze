from collections import Mapping

# The set of possible facets that are specifiable in the
# metadata. Most structures will define a subset of these. This
# is the set over which we will do metadata transformations and
# unification when performing operations.

facets = frozenset([
    # ------------     # Numpy
    'TABLELIKE',
    'ARRAYLIKE',
    'ECLASS',          # Evaluation class
    # ------------     # Numpy
    'C_CONTIGUOUS',
    'F_CONTIGUOUS',
    'OWNDATA',
    'WRITEABLE',
    'ALIGNED',
    'UPDATEIFCOPY',
])

class metadata(Mapping):
    """
    Immutable container for metadata
    """
    __slots__ = ['cls', '__internal']

    def __init__(self, dct=None):
        self.__internal = dict()

        if not metadata.cls:
            metadata.cls = frozenset(dir(self))
        if dct:
            self.__internal.update(dct)

    def __setattr__(self, key, value):
        if key == 'cls' or key == '__internal' or '_metadata' in key:
            super(metadata, self).__setattr__(key, value)
        else:
            self.__internal[key] = value
        return value

    def __getattr__(self, key):
        if key in self.cls:
            super(metadata, self).__getattr__(key)
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

    def __add__(self, other):
        if not set(self.__internal) & set(other.__internal):
            return metadata(dct= dict(self.__internal.items() +
                other.__internal.items()))
        else:
            raise Exception("Union of overlapping metadata is not well-defined")

    def __repr__(self):
        return 'Meta({keys})'.format(
            keys=''.join('%s=%s, ' % (k,v) for k,v in self.__internal.items())
        )

    @classmethod
    def empty(cls):
        return cls(dct={})
