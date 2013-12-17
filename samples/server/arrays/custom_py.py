import blaze
from dynd import nd, ndt

# Available variables from the Blaze loader:
#  `catconf`   Catalog configuration object
#  `impdata`   Import data from the .array file
#  `catpath`   Catalog path of the array
#  `fspath`    Equivalent filesystem path of the array
#  `dshape`    The datashape expected
#
# The loaded array must be placed in `result`.

begin = impdata['begin']
end = impdata['end']
result = blaze.array(nd.range(begin, end))
