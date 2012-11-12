"""
LLVM Representation of CArray objects.
"""

import llvm.core as lc
from numba.llvm_types import _pyobject_head, _pyobject_head_struct, \
    _sizeof_py_ssize_t, _numpy_array, _int32, _numpy_struct

_int          = _int32
npy_intp      = lc.Type.int(_sizeof_py_ssize_t * 8)
char          = lc.Type.int(8)
PyObject      = _pyobject_head_struct
PyArrayObject = _numpy_array
cython_vtab   = lc.Type.struct([])

_carray_struct = lc.Type.struct(_pyobject_head+\
      [
       cython_vtab,   # -- internal cython --
       _int,          # itemsize;
       _int,          # atomsize;
       _int,          # _chunksize;
       _int,          # _chunklen;
       _int,          # leftover;
       _int,          # nrowsinbuf;
       _int,          # _row;
       _int,          # sss_mode;
       _int,          # wheretrue_mode;
       _int,          # where_mode;
       npy_intp,      # startb;
       npy_intp,      # stopb;
       npy_intp,      # start;
       npy_intp,      # stop;
       npy_intp,      # step;
       npy_intp,      # nextelement;
       npy_intp,      # _nrow;
       npy_intp,      # nrowsread;
       npy_intp,      # _nbytes;
       npy_intp,      # _cbytes;
       npy_intp,      # nhits;
       npy_intp,      # limit;
       npy_intp,      # skip;
       npy_intp,      # expectedlen;
       char,          # *lastchunk;
       PyObject,      # *lastchunkarr;
       PyObject,      # *where_arr;
       PyObject,      # *arr1;
       PyObject,      # *_cparams;
       PyObject,      # *_dflt;
       PyObject,      # *_dtype;
       PyObject,      # *chunks;
       PyObject,      # *_rootdir;
       PyObject,      # *datadir;
       PyObject,      # *metadir;
       PyObject,      # *_mode;
       PyObject,      # *_attrs;
       PyArrayObject, # *iobuf;
       PyArrayObject, # *where_buf;
       _int,          # idxcache;
       PyArrayObject, # *blockcache;
       char,          # *datacache;
])

_carray = lc.Type.pointer(_carray_struct)
