# For utils that just need to be written in C

cdef extern from "Python.h":
    int PyObject_AsWriteBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)
    object PyLong_FromVoidPtr(void *p)

def buffer_pointer(object obj):
    cdef void *buffer
    cdef Py_ssize_t buffer_len

    if PyObject_AsWriteBuffer(obj, &buffer, &buffer_len) < 0:
        raise Exception('Empty')

    return PyLong_FromVoidPtr(buffer), buffer_len
