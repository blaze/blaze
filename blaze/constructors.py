from __future__ import absolute_import

# This are the constructors for the blaze array objects.  Having them
# as external functions allows to more flexibility and helps keeping
# the blaze array object compact, just showing the interface of the
# array itself.
#
# The blaze array __init__ method should be considered private and for
# advanced users only. It will provide the tools supporting the rest
# of the constructors, and will use low-level parameters, like
# ByteProviders, that an end user may not even need to know about.

from .concrete import NDArray


# note that this is rather naive. In fact, a proper way to implement
# the array from a numpy is creating a ByteProvider based on "data"
# and infer the indexer from the apropriate information in the numpy
# array.
def array(numpy_array_like):
    from numpy import array
    dd = SimpleDataDescriptor(NpByteProvider(array(numpy_array_like)))
    return NDArray(dd)

# for a temptative open function:
def open(uri):
    raise NotImplementedError
