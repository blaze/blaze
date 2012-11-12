#ifndef __PYX_HAVE__carray__carrayExtension
#define __PYX_HAVE__carray__carrayExtension

struct carray;

/* "carray/carrayExtension.pyx":654
 * # XXX: Made this a public symbol so that we can work with it in C
 * # libraries and LLVM
 * cdef public class carray [type carraytype, object carray]:             # <<<<<<<<<<<<<<
 *   """
 *   carray(array, cparams=None, dtype=None, dflt=None, expectedlen=None, chunklen=None, rootdir=None, mode='a')
 */
struct carray {
  PyObject_HEAD
  struct __pyx_vtabstruct_6carray_15carrayExtension_carray *__pyx_vtab;
  int itemsize;
  int atomsize;
  int _chunksize;
  int _chunklen;
  int leftover;
  int nrowsinbuf;
  int _row;
  int sss_mode;
  int wheretrue_mode;
  int where_mode;
  npy_intp startb;
  npy_intp stopb;
  npy_intp start;
  npy_intp stop;
  npy_intp step;
  npy_intp nextelement;
  npy_intp _nrow;
  npy_intp nrowsread;
  npy_intp _nbytes;
  npy_intp _cbytes;
  npy_intp nhits;
  npy_intp limit;
  npy_intp skip;
  npy_intp expectedlen;
  char *lastchunk;
  PyObject *lastchunkarr;
  PyObject *where_arr;
  PyObject *arr1;
  PyObject *_cparams;
  PyObject *_dflt;
  PyObject *_dtype;
  PyObject *chunks;
  PyObject *_rootdir;
  PyObject *datadir;
  PyObject *metadir;
  PyObject *_mode;
  PyObject *_attrs;
  PyArrayObject *iobuf;
  PyArrayObject *where_buf;
  int idxcache;
  PyArrayObject *blockcache;
  char *datacache;
};

#ifndef __PYX_HAVE_API__carray__carrayExtension

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(PyTypeObject) carraytype;

#endif /* !__PYX_HAVE_API__carray__carrayExtension */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcarrayExtension(void);
#else
PyMODINIT_FUNC PyInit_carrayExtension(void);
#endif

#endif /* !__PYX_HAVE__carray__carrayExtension */
