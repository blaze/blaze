from definitions cimport ndarray

cdef public class chunk [object object, type type]:
    cdef char typekind, isconstant
    cdef public int atomsize, itemsize, blocksize
    cdef public int nbytes, cbytes, cdbytes
    cdef int true_count
    cdef char *data
    cdef object atom, constant, dobject

    cdef void _getitem(self, int start, int stop, char *dest)
    cdef compress_data(self, char *data, size_t itemsize, size_t nbytes, object cparams)
    cdef compress_arrdata(self, ndarray array, object cparams, object _memory)
