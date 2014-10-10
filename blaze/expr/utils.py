

class _slice(object):
    """ A hashable slice object

    >>> _slice(0, 10, None)
    0:10
    """
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __hash__(self):
        return hash((slice, self.start, self.stop, self.step))

    def __str__(self):
        s = ''
        if self.start is not None:
            s = s + str(self.start)
        s = s + ':'
        if self.stop is not None:
            s = s + str(self.stop)
        if self.step is not None:
            s = s + ':' + str(self.step)
        return s

    def __eq__(self, other):
        return (type(self), self.start, self.stop, self.step) == \
               (type(other), other.start, other.stop, other.step)

    __repr__ = __str__



def hashable_index(index):
    """ Convert slice-thing into something hashable

    >>> hashable_index(1)
    1

    >>> isinstance(hash(hashable_index((1, slice(10)))), int)
    True
    """
    if isinstance(index, tuple):
        return tuple(map(hashable_index, index))
    elif isinstance(index, slice):
        return _slice(index.start, index.stop, index.step)
    return index


def replace_slices(index):
    if isinstance(index, tuple):
        return tuple(map(replace_slices, index))
    elif isinstance(index, _slice):
        return slice(index.start, index.stop, index.step)
    return index


