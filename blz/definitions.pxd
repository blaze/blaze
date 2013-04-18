########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

"""Here are some definitions for some C headers dependencies.

"""

import sys

# Standard C functions.
cdef extern from "stdlib.h":
  ctypedef long size_t
  ctypedef long uintptr_t
  void *malloc(size_t size)
  void *realloc(void *ptr, size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  char *strchr(char *s, int c)
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)
  void *memcpy(void *dest, void *src, size_t n)
  void *memset(void *s, int c, size_t n)

cdef extern from "time.h":
  ctypedef int time_t


#-----------------------------------------------------------------------------

# Some helper routines from the Python API
cdef extern from "Python.h":

  # special types
  ctypedef int Py_ssize_t

  # references
  void Py_INCREF(object)
  void Py_DECREF(object)

  # To release global interpreter lock (GIL) for threading
  void Py_BEGIN_ALLOW_THREADS()
  void Py_END_ALLOW_THREADS()

  # Functions for integers
  object PyInt_FromLong(long)
  long PyInt_AsLong(object)
  object PyLong_FromLongLong(long long)
  long long PyLong_AsLongLong(object)

  # Functions for floating points
  object PyFloat_FromDouble(double)

  # Functions for strings
  object PyString_FromString(char *)
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  size_t PyString_GET_SIZE(object string)

  # Functions for lists
  int PyList_Append(object list, object item)

  # Functions for tuples
  object PyTuple_New(int)
  int PyTuple_SetItem(object, int, object)
  object PyTuple_GetItem(object, int)
  int PyTuple_Size(object tuple)

  # Functions for dicts
  int PyDict_Contains(object p, object key)
  object PyDict_GetItem(object p, object key)

  # Functions for objects
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)
  long PyObject_Length(object o)
  int PyObject_Compare(object o1, object o2)
  int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)

  # Functions for buffers
  object PyBuffer_FromMemory(void *ptr, Py_ssize_t size)

  ctypedef unsigned int Py_uintptr_t


#-----------------------------------------------------------------------------

# API for NumPy objects
cdef extern from "numpy/arrayobject.h":

  # Platform independent types
  ctypedef int npy_intp
  ctypedef signed int npy_int8
  ctypedef unsigned int npy_uint8
  ctypedef signed int npy_int16
  ctypedef unsigned int npy_uint16
  ctypedef signed int npy_int32
  ctypedef unsigned int npy_uint32
  ctypedef signed long long npy_int64
  ctypedef unsigned long long npy_uint64
  ctypedef float npy_float32
  ctypedef double npy_float64

  cdef enum NPY_TYPES:
    NPY_BOOL
    NPY_BYTE
    NPY_UBYTE
    NPY_SHORT
    NPY_USHORT
    NPY_INT
    NPY_UINT
    NPY_LONG
    NPY_ULONG
    NPY_LONGLONG
    NPY_ULONGLONG
    NPY_FLOAT
    NPY_DOUBLE
    NPY_LONGDOUBLE
    NPY_CFLOAT
    NPY_CDOUBLE
    NPY_CLONGDOUBLE
    NPY_OBJECT
    NPY_STRING
    NPY_UNICODE
    NPY_VOID
    NPY_NTYPES
    NPY_NOTYPE

  # Platform independent types
  cdef enum:
    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128

  # Classes
  ctypedef extern class numpy.dtype [object PyArray_Descr]:
    cdef int type_num, elsize, alignment
    cdef char type, kind, byteorder, hasobject
    cdef object fields, typeobj

  ctypedef extern class numpy.ndarray [object PyArrayObject]:
    cdef char *data
    cdef int nd
    cdef npy_intp *dimensions
    cdef npy_intp *strides
    cdef object base
    cdef dtype descr
    cdef int flags

  # Functions
  object PyArray_GETITEM(object arr, void *itemptr)
  int PyArray_SETITEM(object arr, void *itemptr, object obj)
  dtype PyArray_DescrFromType(int type)
  object PyArray_Scalar(void *data, dtype descr, object base)

  # The NumPy initialization function
  void import_array()



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
