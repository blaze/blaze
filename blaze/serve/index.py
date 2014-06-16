from blaze.compatibility import _strtypes

def parse_index(ind):
    if isinstance(ind, (int, _strtypes)):
        return ind
    if isinstance(ind, list):
        return tuple(map(parse_index, ind))
    if isinstance(ind, dict):
        return slice(ind.get('start'), ind.get('stop'), ind.get('step'))
    raise ValueError('Do not know how to parse %s into an index' % str(ind))
