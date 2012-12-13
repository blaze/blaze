from cpython cimport *

DEF MAX_NDIM = 8

cdef extern from "Python.h":
    ctypedef unsigned int Py_uintptr_t

#------------------------------------------------------------------------
# C-level Chunk or Tile Data Descriptors
#------------------------------------------------------------------------
ctypedef public struct CChunk:
    void *data
    Py_ssize_t size
    Py_ssize_t stride
    size_t chunk_index
    PyObject *obj # object to keep alive for the duration of the use
    void *extra   # miscellaneous data

ctypedef public struct CTile:
    int ndim
    void *data
    Py_ssize_t shape[MAX_NDIM]
    Py_ssize_t strides[MAX_NDIM]

ctypedef public struct IndexerMetaData:
    # TODO: make these structs
    # Borrowed reference to data source
    void *source
    # Borrowed reference to data shape
    void *datashape

cdef class Chunk(object):
    cdef CChunk chunk

cdef class Tile(object):
    cdef CTile tile

#------------------------------------------------------------------------
# C-level iterators and indexers on scalars or bulk data (tiles)
#------------------------------------------------------------------------

# Read chunks in sequence. After use, a written chunk must be committed, and a
# chunk that was only read must be disposed of.
ctypedef public struct CChunkIterator:
    void (*next)(CChunkIterator *info, CChunk *chunk)
    void (*commit)(CChunkIterator *info, CChunk *chunk)
    void (*dispose)(CChunkIterator *info, CChunk *chunk)

    IndexerMetaData meta
    size_t cur_chunk_idx

# global coordinates -> scalar
ctypedef public struct CIndexer:
    void (*index_read)(CIndexer *info, Py_ssize_t *indices, void *datum)
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
# Python and Cython level exposures of iterators and indexers
#------------------------------------------------------------------------

cdef class lldatadesc(object):
    cdef object data_obj
    cdef object datashape

cdef class ChunkIterator(lldatadesc):
    cdef CChunkIterator iterator

cdef class TileIndexer(lldatadesc):
    cdef CIndexer indexer
