from collections import Mapping, OrderedDict
from utils import Symbol as S

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
