from __future__ import absolute_import

from collections import namedtuple

#------------------------------------------------------------------------
# Metadata Enum
#------------------------------------------------------------------------

CONTIGUOUS  = 1
C_ORDER     = 2
F_ORDER     = 3
STRIDED     = 4
DISTRIBUTED = 5
BUFFERED    = 6

#------------------------------------------------------------------------
# Invariants
#------------------------------------------------------------------------

preserves = namedtuple('Preserves', 'props')

#------------------------------------------------------------------------
# Rules
#------------------------------------------------------------------------

preservation_rules = {
    'add': preserves([ CONTIGUOUS ]),
}
