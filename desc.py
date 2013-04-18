from datashape import dshape
from byteprovider import bytefactory

#------------------------------------------------------------------------
# Data Descriptor
#------------------------------------------------------------------------

class DataDescriptor(object):
    """ DataDescriptors are the map between low-level references to Bytes
    and the Blaze Array.  It is basically an "unadorned" NDArray with no special
    methods and no math just mechanisms to get access to the data quickly and
    iterators.

    Whereas traditional data interfaces use iterators to programmatically
    retrieve data, Blaze preserves the ability to expose data in bulk
    form.
    """

    def __init__(self, sources, dtype, indexfactory):
        self.bytes = [bytefactory(source) for source in sources]
        self.dshape = dshape(dtype)
        self.indexobj = indexfactory(bytes, dshape)

    def __getitem__(self, key):
        return self.indexobj[key]
