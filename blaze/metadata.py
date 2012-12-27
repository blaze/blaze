from collections import Mapping, OrderedDict
from blaze.expr.utils import Symbol as S

#------------------------------------------------------------------------
# Symbols
#------------------------------------------------------------------------

# The set of possible facets that are specifiable in the
# metadata. Most structures will define a subset of these. This
# is the set over which we will do metadata transformations and
# unification when performing operations.

tablelike    = S('tablelike')
arraylike    = S('arraylike')
manifest     = S('manifest')
deferred     = S('deferred')
c_contigious = S('c_contigious')
f_contigious = S('f_contigious')
tablelike    = S('owndata')
writeable    = S('writeable')
aligned      = S('aligned')

update_if_copied = S('update_if_copied')

#------------------------------------------------------------------------
# Metadata Querying
#------------------------------------------------------------------------

def has_prop(indexable, prop):
    """
    Check whether the indexable object has the given property in
    its metadata.
    """
    return prop in indexable._metadata

def all_prop(args, prop):
    """
    Check whether the arguments to a Fun are all manifest
    array/table objects, needed to determine whether to construct
    a graph node or to
    """
    return all(has_prop(a, prop) for a in args)
