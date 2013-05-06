from __future__ import absolute_import

from .datashape import dshape
from .byteprovider import bytefactory

#------------------------------------------------------------------------
# Data Descriptor  Abstract Base Class
#------------------------------------------------------------------------

class DataDescriptor(object):
    """DataDescriptors are the map between low-level references to Bytes
    and the Blaze Array.  It is basically an "unadorned" Array with no
    special methods and no math just mechanisms to get access to the
    data quickly and iterators.

    Whereas traditional data interfaces use iterators to programmatically
    retrieve data, Blaze preserves the ability to expose data in bulk form.

    There is either a single ByteProvider or a sequence of ByteProviders
    which map to a dimension in the data-shape (initially the outer-dimension)

    DataDescriptors can be the result of a "Program" in which case the Bytes
    Object is replaced with a Program Graph
    sequence of ByteProviders is replaced with a sequence of Program Graphs.

    The indexobject is like the meta-data in dynd holding the information
    necessary to map index domains into byte offsets
    """
    bytes = None
    dshape = None
    indexobj = None

    """
    def __init__(self, sources, dtype, indexfactory):
        self.bytes = [bytefactory(source) for source in sources]
        self.dshape = dshape(dtype)
        self.indexobj = indexfactory(bytes, dshape)
    """

# ByteProvider as Memory locations
class MemoryDescriptor(DataDescriptor):
    pass

# ByteProvider as a Deferred Graph
class DeferredDescriptor(DataDescriptor):
    pass

# ByteProviders are a Mixed Collection
class MixedDescriptor(DataDescriptor):
    self.bytes = []

# ByteProviders are Files
class FileDescriptor(DataDescriptor):
    self.bytes = []

# ByteProviders are remote arrays
class RemoteDescriptor(DataDescriptor):
    self.bytes = []
