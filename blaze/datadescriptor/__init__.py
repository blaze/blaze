from __future__ import absolute_import

from .data_descriptor import IDataDescriptor, Capabilities

from .blz_data_descriptor import BLZDataDescriptor
from .cat_data_descriptor import CatDataDescriptor
from .membuf_data_descriptor import (data_descriptor_from_ctypes,
                data_descriptor_from_cffi)
from .dynd_data_descriptor import DyNDDataDescriptor
from .blaze_func_descriptor import BlazeFuncDeprecatedDescriptor
from .deferred_descriptor import DeferredDescriptor

from .as_py import dd_as_py
