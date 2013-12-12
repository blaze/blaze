from .datadescriptor import (IDataDescriptor, BLZDataDescriptor,
        CatDataDescriptor, data_descriptor_from_ctypes,
        data_descriptor_from_cffi, DyNDDataDescriptor, DeferredDescriptor)
from .storage import open, drop, Storage
