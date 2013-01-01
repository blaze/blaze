#ifndef __PYX_HAVE__blaze__desc__lldescriptors
#define __PYX_HAVE__blaze__desc__lldescriptors

struct CChunk;
typedef struct CChunk CChunk;
struct CTile;
typedef struct CTile CTile;
struct IndexerMetaData;
typedef struct IndexerMetaData IndexerMetaData;
struct CChunkIterator;
typedef struct CChunkIterator CChunkIterator;
struct CIndexer;
typedef struct CIndexer CIndexer;
struct CTileIndexer;
typedef struct CTileIndexer CTileIndexer;

/* "blaze/desc/lldescriptors.pxd":11
 * # C-level Chunk or Tile Data Descriptors
 * #------------------------------------------------------------------------
 * ctypedef public struct CChunk:             # <<<<<<<<<<<<<<
 *     void *data
 *     Py_ssize_t size
 */
struct CChunk {
  void *data;
  Py_ssize_t size;
  Py_ssize_t stride;
  size_t chunk_index;
  PyObject *obj;
  void *extra;
};

/* "blaze/desc/lldescriptors.pxd":19
 *     void *extra   # miscellaneous data
 * 
 * ctypedef public struct CTile:             # <<<<<<<<<<<<<<
 *     int ndim
 *     void *data
 */
struct CTile {
  int ndim;
  void *data;
  Py_ssize_t shape[8];
  Py_ssize_t strides[8];
};

/* "blaze/desc/lldescriptors.pxd":25
 *     Py_ssize_t strides[MAX_NDIM]
 * 
 * ctypedef public struct IndexerMetaData:             # <<<<<<<<<<<<<<
 *     # TODO: make these structs
 *     # Borrowed reference to data source
 */
struct IndexerMetaData {
  void *source;
  void *datashape;
};

/* "blaze/desc/lldescriptors.pxd":44
 * # Read chunks in sequence. After use, a written chunk must be committed, and a
 * # chunk that was only read must be disposed of.
 * ctypedef public struct CChunkIterator:             # <<<<<<<<<<<<<<
 *     int (*next)(CChunkIterator *info, CChunk *chunk) except -1
 *     int (*commit)(CChunkIterator *info, CChunk *chunk) except -1
 */
struct CChunkIterator {
  int (*next)(CChunkIterator *, CChunk *);
  int (*commit)(CChunkIterator *, CChunk *);
  int (*dispose)(CChunkIterator *, CChunk *);
  IndexerMetaData meta;
  size_t cur_chunk_idx;
};

/* "blaze/desc/lldescriptors.pxd":53
 * 
 * # global coordinates -> scalar
 * ctypedef public struct CIndexer:             # <<<<<<<<<<<<<<
 *     int (*index_read)(CIndexer *info, Py_ssize_t *indices,
 *                       void *datum) except -1
 */
struct CIndexer {
  int (*index_read)(CIndexer *, Py_ssize_t *, void *);
  int (*index_write)(CIndexer *, Py_ssize_t *, void *);
  IndexerMetaData meta;
};

/* "blaze/desc/lldescriptors.pxd":64
 * 
 * # global coordinates -> ND tile
 * ctypedef public struct CTileIndexer:             # <<<<<<<<<<<<<<
 *     tile_read_t index
 *     tile_read_t commit
 */
struct CTileIndexer {
  __pyx_t_5blaze_4desc_13lldescriptors_tile_read_t index;
  __pyx_t_5blaze_4desc_13lldescriptors_tile_read_t commit;
  __pyx_t_5blaze_4desc_13lldescriptors_tile_read_t dispose;
  IndexerMetaData meta;
};

#ifndef __PYX_HAVE_API__blaze__desc__lldescriptors

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#endif /* !__PYX_HAVE_API__blaze__desc__lldescriptors */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initlldescriptors(void);
#else
PyMODINIT_FUNC PyInit_lldescriptors(void);
#endif

#endif /* !__PYX_HAVE__blaze__desc__lldescriptors */
