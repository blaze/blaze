from __future__ import absolute_import, division, print_function

from .data_descriptor import DDesc, copy

from .blz_data_descriptor import BLZ_DDesc
from ..optional_packages import tables_is_here
from .h5py_data_descriptor import H5PY_DDesc
if tables_is_here:
    from .pytables_data_descriptor import PyTables_DDesc
    HDF5_DDesc = PyTables_DDesc
else:
    HDF5_DDesc = H5PY_DDesc
from .cat_data_descriptor import Cat_DDesc
from .membuf_data_descriptor import (data_descriptor_from_ctypes,
                data_descriptor_from_cffi)
from .dynd_data_descriptor import DyND_DDesc
from .deferred_data_descriptor import Deferred_DDesc
from .stream_data_descriptor import Stream_DDesc
from .csv_data_descriptor import CSV_DDesc
from .json_data_descriptor import JSON_DDesc
from .sql_data_descriptor import SQL_DDesc
from .remote_data_descriptor import Remote_DDesc

from .as_py import ddesc_as_py
