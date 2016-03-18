from __future__ import absolute_import, division, print_function

from datashape import var


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

    def as_slice(self):
        return slice(self.start, self.stop, self.step)

    __repr__ = __str__


class hashable_list(tuple):
    def __str__(self):
        return str(list(self))


def hashable_index(index):
    """ Convert slice-thing into something hashable

    >>> hashable_index(1)
    1

    >>> isinstance(hash(hashable_index((1, slice(10)))), int)
    True
    """
    if type(index) is tuple:  # can't do isinstance due to hashable_list
        return tuple(map(hashable_index, index))
    elif isinstance(index, list):
        return hashable_list(index)
    elif isinstance(index, slice):
        return _slice(index.start, index.stop, index.step)
    return index


def replace_slices(index):
    """
    Takes input from Slice expression and returns either a list,
    slice object, or tuple.

    Examples
    -------
    >>> replace_slices([1, 2, 345, 12])
    [1, 2, 345, 12]
    >>> type(replace_slices(_slice(1, 5, None))) is slice
    True
    >>> type(replace_slices((2, 5))) is tuple
    True

    """
    if isinstance(index, hashable_list):
        return list(index)
    elif isinstance(index, _slice):
        return index.as_slice()
    elif isinstance(index, tuple):
        return tuple(map(replace_slices, index))
    return index


def maxvar(L):
    """

    >>> maxvar([1, 2, var])
    Var()

    >>> maxvar([1, 2, 3])
    3
    """
    if var in L:
        return var
    else:
        return max(L)


def maxshape(shapes):
    """

    >>> maxshape([(10, 1), (1, 10), ()])
    (10, 10)

    >>> maxshape([(4, 5), (5,)])
    (4, 5)
    """
    shapes = [shape for shape in shapes if shape]
    if not shapes:
        return ()
    ndim = max(map(len, shapes))
    shapes = [(1,) * (ndim - len(shape)) + shape for shape in shapes]
    for dims in zip(*shapes):
        if len(set(dims) - set([1])) >= 2:
            raise ValueError("Shapes don't align, %s" % str(dims))
    return tuple(map(maxvar, zip(*shapes)))
