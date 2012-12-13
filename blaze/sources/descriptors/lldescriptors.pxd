DEF MAX_NDIM = 8

cdef extern from "Python.h":
    ctypedef unsigned int Py_uintptr_t

#------------------------------------------------------------------------
# C-level Chunk or Tile Data Descriptor.
#------------------------------------------------------------------------
ctypedef public struct CTile:
    int ndim
    bint inplace
    void *data
    Py_ssize_t shape[MAX_NDIM]
    Py_ssize_t strides[MAX_NDIM]

ctypedef public struct IndexerMetaData:
    # TODO: make these structs
    # Borrowed reference to data source
    void *source
    # Borrowed reference to data shape
    void *datashape

#------------------------------------------------------------------------
# C-level Indexers on scalars or bulk data (tiles)
#------------------------------------------------------------------------

# global coordinates -> scalar
ctypedef public struct CIndexer:
    void (*index_read)(CIndexer *info, Py_ssize_t *indices, void *out)
    void (*index_write)(CIndexer *info, Py_ssize_t *indices, void *datum)
    IndexerMetaData meta

ctypedef void (*tile_read_t)(CTileIndexer *info, Py_ssize_t *indices,
                             CTile *out_tile)
ctypedef void (*tile_commit_t)(CTileIndexer *info, Py_ssize_t *indices,
                               CTile *in_tile)
# global coordinates -> ND tile
ctypedef public struct CTileIndexer:
    tile_read_t index
    tile_commit_t commit
    IndexerMetaData meta

#------------------------------------------------------------------------
# Python and Cython level exposures
#------------------------------------------------------------------------

cdef class TileIndexer(object):
    cdef CIndexer indexer
    cdef object datasource
    cdef object datashape

cdef class Tile(object):
    cdef CTile tile