from blaze.compatibility import _strtypes

def parse_index(ind, inside=False):
    """ Parse structured index into Pythonic form

    >>> parse_index([1, {'start': 0, 'stop': 10}])
    (1, slice(0, 10, None))

    See also:
        emit_index
    """
    if isinstance(ind, (int, _strtypes)):
        return ind
    if isinstance(ind, list):
        typ = list if inside else tuple
        return typ(parse_index(i, True) for i in ind)
    if isinstance(ind, dict):
        return slice(ind.get('start'), ind.get('stop'), ind.get('step'))
    raise ValueError('Do not know how to parse %s into an index' % str(ind))


def emit_index(ind):
    """ Emit Python index into structured form

    >>> emit_index((1, slice(0, 10, None))) #doctest: +SKIP
    [1, {'start': 0, 'stop': 10}]

    See also:
        parse_index
    """
    if isinstance(ind, (int, _strtypes)):
        return ind
    if isinstance(ind, (list, tuple)):
        return list(map(emit_index, ind))
    if isinstance(ind, slice):
        result = {'start': ind.start, 'stop': ind.stop, 'step': ind.step}
        if result['step'] is None:
            del result['step']
        return result
