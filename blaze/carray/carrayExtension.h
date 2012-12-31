#ifndef __PYX_HAVE__blaze__carray__carrayExtension
#define __PYX_HAVE__blaze__carray__carrayExtension

struct object;

/* "blaze/carray/carrayExtension.pxd":5
 * import_array()
 * 
 * cdef public class chunk [object object, type type]:             # <<<<<<<<<<<<<<
 *     cdef char typekind, isconstant
 *     cdef public int atomsize, itemsize, blocksize
 */
struct object {
  PyObject_HEAD
  struct __pyx_vtabstruct_5blaze_6carray_15carrayExtension_chunk *__pyx_vtab;
  char typekind;
  char isconstant;
  int atomsize;
  int itemsize;
  int blocksize;
  int nbytes;
  int cbytes;
  int cdbytes;
  int true_count;
  char *data;
  PyObject *atom;
  PyObject *constant;
  PyObject *dobject;
};

#ifndef __PYX_HAVE_API__blaze__carray__carrayExtension

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(PyTypeObject) type;

#endif /* !__PYX_HAVE_API__blaze__carray__carrayExtension */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcarrayExtension(void);
#else
PyMODINIT_FUNC PyInit_carrayExtension(void);
#endif

#endif /* !__PYX_HAVE__blaze__carray__carrayExtension */
