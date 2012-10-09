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

def can_embed(dim1, dim2):
    """
    Can we embed a ``dim1`` inside of the space specified by the outer
    dimension ``dim2``. This inspects the full chain, instead of
    unification which is pairwise.
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

    # TODO:
    if isinstance(dim1, CType):
        if isinstance(dim2, CType):
            return True

    raise CannotEmbed(dim1, dim2)
