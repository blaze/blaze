from coretypes import Fixed, Var, TypeVar, Record, \
    DataShape, CType

class CannotEmbed(Exception):
    def __init__(self, space, dim):
        self.space = space
        self.dim   = dim

    def __str__(self):
        return "Cannot embed space of values (%r) in (%r)" % (
            self.space, self.dim
        )

def describe(obj):
    # circular references...
    from ndtable.table import NDTable

    if isinstance(obj, DataShape):
        return obj

    elif isinstance(obj, list):
        return Fixed(len(obj))

    elif isinstance(obj, tuple):
        return Fixed(len(obj))

    elif isinstance(obj, NDTable):
        return obj.datashape

def can_embed(dim1, dim2):
    """
    Can we embed a ``obj`` inside of the space specified by the outer
    dimension ``dim``.
    """
    # We want explicit fallthrough
    if isinstance(dim1, Fixed):

        if isinstance(dim2, Fixed):
            if dim1 == dim2:
                return True
            else:
                return False

        if isinstance(dim2, Var):
            if dim2.lower < dim1.val < dim2.upper:
                return True
            else:
                return False

        if isinstance(dim2, TypeVar):
            return True

    if isinstance(dim1, Record):

        if isinstance(dim2, Record):
            # is superset
            return set(dim1.k) >= set(dim2.k)

    if isinstance(dim1, TypeVar):
        return True

    raise CannotEmbed(dim1, dim2)
