from __future__ import absolute_import

__all__ = ['IElementReader', 'IElementWriter',
           'IElementReadIter', 'IElementWriteIter',
           'IElementAppender', 'IDataDescriptor',
           'buffered_ptr_ctxmgr']

import abc
import ctypes
import contextlib
from blaze.error import StreamingDimensionError

@contextlib.contextmanager
def buffered_ptr_ctxmgr(ptr, buffer_flush):
    """
    A context manager to help implement the
    buffered_ptr method of IElementWriter.

    Parameters
    ----------
    ptr : integer
        The pointer to wrap.
    buffer : object
        Either None if the pointer is in the original array,
        and requires no buffering, or a callable which
        flushes the buffer.
    """
    yield ptr
    if buffer_flush:
        buffer_flush()

class IElementReader:
    """
    An interface for getting char* element pointers at fixed-size
    index tuples. Provides additional C and llvm function
    interfaces to use in a jitting context.

    >>> obj = blzarr.element_reader(3)
    >>> obj.read_single([i, j, k])
    <raw pointer value>
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        The dshape of elements returned by the
        element reader.
        """
        raise NotImplemented

    @abc.abstractproperty
    def nindex(self):
        """
        The number of indices the read_single() function
        requires. This is equal to the 'nindex'
        provided to the datadescriptor's
        element_reader() function.
        """
        raise NotImplemented

    @abc.abstractmethod
    def read_single(self, idx):
        """
        Returns a char* pointer to the element
        at the specified index. The value in
        'idx' must be a tuple of 'nindex' integers.
        """
        raise NotImplemented

    def read_single_into(self, idx, dst_ptr):
        """
        Reads a single element, placing into
        the memory location provided by dst_ptr.

        By default this is implemented in terms of
        read_single, but data descriptors which do
        processing can write directly into that pointer
        instead of allocating their own buffer.
        """
        src_ptr = self.read_single(idx)
        ctypes.memmove(dst_ptr, src_ptr, self.dshape.itemsize)

    def c_api(self):
        """
        Returns a tuple with a [CFFI or ctypes?] function pointer
        to a C get method, and a void* pointer to pass to
        the function's 'extra' parameter.

        Possible alternative: Returns a blaze kernel with a
        'get_element' function prototype. (This wraps destruction
        and the void* pointer in the blaze kernel low level interface)
        """
        raise NotImplemented

    def llvm_api(self, module):
        """
        Inserts a getter function into the llvm module, and
        returns it as a function object.
        """
        raise NotImplemented

class IElementWriter:
    """
    An interface for writing elements into the data at
    the specified index tuples. Provides additional C and
    llvm function interfaces to use in a jitting context.

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        The dshape of elements to be written to the
        element writer.
        """
        raise NotImplemented

    @abc.abstractproperty
    def nindex(self):
        """
        The number of indices the set() function
        requires. This is equal to the 'nindex'
        provided to the datadescriptor's
        get_element_interface() function.
        """
        raise NotImplemented

    @abc.abstractmethod
    def write_single(self, idx, ptr):
        """
        Writes a char* pointer to the element
        at the specified index. The value in
        'idx' must be a tuple of 'nindex' integers,
        and 'ptr' must be a pointer to an element
        with the element writer's dshape.
        """
        raise NotImplemented

    @abc.abstractmethod
    def buffered_ptr(self, idx):
        """
        Returns a context manager object which provides
        a pointer to a buffer (or a pointer to the final
        destination if the layout matches exactly), which
        is flushed when the 'with' context finishes.

        Example
        -------
        >>> with elw.buffered_ptr((3,5)) as dst_ptr:
                ctypes.memmove(dst_ptr, src_ptr, elw.dshape.c_itemsize)
        """
        raise NotImplemented

    def c_api(self):
        """
        Returns a tuple with a [CFFI or ctypes?] function pointer
        to a C set method, and a void* pointer to pass to
        the function's 'extra' parameter.

        Possible alternative: Returns a blaze kernel with a
        'set_element' function prototype. (This wraps destruction
        and the void* pointer in the blaze kernel low level interface)
        """
        raise NotImplemented

    def llvm_api(self, module):
        """
        Inserts a setter function into the llvm module, and
        returns it as a function object.
        """
        raise NotImplemented

class IElementAppender:
    """
    An interface for appending elements at the end of the
    data. Provides additional C and llvm function interfaces to use in
    a jitting context.

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        The dshape of elements to be appended to the
        element appender.
        """
        raise NotImplemented

    @abc.abstractmethod
    def append(self, ptr, nrows):
        """
        Appends a char* pointer to the element at the end of
        data. 'ptr' must be a pointer to an element with the element
        appender's dshape. 'nrows' is the number of values to appended
        in the outer dimension.
        """
        raise NotImplemented

    @abc.abstractmethod
    def close(self, ptr):
        """
        Flush whatever data remains in buffers.
        """
        raise NotImplemented

    def c_api(self):
        """
        Returns a tuple with a [CFFI or ctypes?] function pointer
        to a C set method, and a void* pointer to pass to
        the function's 'extra' parameter.

        Possible alternative: Returns a blaze kernel with a
        'append_element' function prototype. (This wraps destruction
        and the void* pointer in the blaze kernel low level interface)
        """
        raise NotImplemented

    def llvm_api(self, module):
        """
        Inserts a appender function into the llvm module, and
        returns it as a function object.
        """
        raise NotImplemented


class IElementReadIter:
    """
    In interface for iterating over the outermost dimension of
    a data descriptor for reading. It must return a char* pointer
    for each element along the dimension.

    If the dimension has a known size, it should be returned
    in the __len__ method. A streaming dimension does not
    have a size known ahead of time, and should not implement __len__.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        The dshape of elements returned by the
        element iterator.
        """
        raise NotImplemented

    def __iter__(self):
        return self

    def __len__(self):
        raise StreamingDimensionError('Cannot get the length of'
                        ' a streaming dimension')

    @abc.abstractmethod
    def __next__(self):
        raise NotImplemented

    def next(self):
        return self.__next__()

    def c_api(self):
        """
        Returns a tuple of objects providing the iteration as C data
        pointers and function pointers. This interface is similar to
        the iteration interface in NumPy's nditer C API
        (http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#simple-iteration-example).
        """
        raise NotImplemented

    def llvm_api(self, module):
        """
        An LLVM api for the iterator, to be designed as we hook
        up LLVM here.
        """
        raise NotImplemented

class IElementWriteIter:
    """
    An interface for iterating over the outermost dimension of
    a data descriptor for writing. Each iteration returns a
    pointer to a write buffer into which an element can be written.

    The iterator may be using buffering, flushing the buffer
    on some iterations, and in that case it must flush everything
    before it raises a StopIteration.

    If the dimension has a known size, it should be returned
    in the __len__ method. A streaming dimension does not
    have a size known ahead of time, and should not implement __len__.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        The dshape of elements returned by the
        element iterator.
        """
        raise NotImplemented

    def __iter__(self):
        return self

    def __len__(self):
        raise StreamingDimensionError('Cannot get the length of'
                        ' a streaming dimension')

    @abc.abstractmethod
    def __next__(self):
        raise NotImplemented

    @abc.abstractmethod
    def close(self):
        """
        This method should flush any buffers that are still
        outstanding in the writing process.
        """
        raise NotImplemented

    def next(self):
        return self.__next__()

    def c_api(self):
        """
        Returns a tuple of objects providing the iteration as C data
        pointers and function pointers. This interface is similar to
        the iteration interface in NumPy's nditer C API
        (http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#simple-iteration-example).
        """
        raise NotImplemented

    def llvm_api(self, module):
        """
        An LLVM api for the iterator, to be designed as we hook
        up LLVM here.
        """
        raise NotImplemented

class IDataDescriptor:
    """
    The Blaze data descriptor is an interface which exposes
    data to Blaze. The data descriptor doesn't implement math
    or any other kind of functions, its sole purpose is providing
    single and multi-dimensional data to Blaze via a data shape,
    and the indexing/iteration interfaces.

    Indexing and python iteration must always return data descriptors,
    this is the python interface to the data. A summary of the
    data access patterns for a data descriptor dd, in the initial
    0.1 blaze are:

     - descriptor integer indexing
            child_dd = dd[i, j, k]
     - descriptor outer/leading dimension iteration
            for child_dd in dd: do_something(child_dd)
     - element integer indexing
            ixr = dd.get_element_interface(3)
            # Python access
            rawptr = ixr.get([i, j, k])
            # C access
            cffi_fn, cffi_voidptr = ixr.c_getter()
            # LLVM access
            llvm_fn = ixr.llvm_getter(llvm_module)
     - element outer/leading dimension iteration
            itr = dd.element_iter_interface()
            # Python access
            for rawptr in itr: do_something(rawptr)
            # C access
            cffi_fn, ...(TBD) = itr.c_iter()
            # LLVM access
            (TBD) = itr.llvm_iter()

    The descriptor-based indexing methods operate only through the
    Python interface, while the element-based methods allow C
    and LLVM versions that can be integrated into a JIT or
    C ABI runtime.

    Presently, the elements returned by the element interfaces must
    be C-contiguous, aligned, and in native byte order. The data
    descriptor may make a temporary copy, owned by the interface
    object, to achieve this.
    """
    __metaclass__ = abc.ABCMeta

    deferred = False

    @property
    def persistent(self):
        return False

    @abc.abstractproperty
    def dshape(self):
        """
        Returns the datashape for the data behind this datadescriptor.
        Every data descriptor implementation must provide a dshape.
        """
        raise NotImplemented

    @abc.abstractproperty
    def writable(self):
        """
        Returns True if the data is writable,
        False otherwise.
        """
        raise NotImplemented

    @abc.abstractproperty
    def immutable(self):
        """
        Returns True if the data is immutable,
        False otherwise.
        """
        raise NotImplemented

    def appendable(self):
        """
        Returns True if the data is appendable,
        False otherwise.
        """
        raise NotImplemented

    def __len__(self):
        """
        The default implementation of __len__ is for the
        behavior of a streaming dimension, where the size
        of the dimension isn't known ahead of time.

        If a data descriptor knows its dimension size,
        it should implement __len__, and provide the size
        as an integer.
        """
        raise StreamingDimensionError('Cannot get the length of'
                        ' a streaming dimension')

    @abc.abstractmethod
    def __iter__(self):
        """
        This returns an iterator/generator which iterates over
        the outermost/leading dimension of the data. If the
        dimension is not also a stream, __len__ should also
        be implemented. The iterator must return data
        descriptors.
        """
        raise NotImplemented

    @abc.abstractmethod
    def __getitem__(self, key):
        """
        This does integer/slice indexing, producing another
        data descriptor.
        """
        raise NotImplemented

    @abc.abstractmethod
    def element_reader(self, nindex):
        """
        This returns an object which implements the
        IElementReader interface for the specified number
        of indices. The returned object can also
        expose a C-level function to get an element.
        """
        raise NotImplemented

    def element_writer(self, nindex):
        """
        This returns an object which implements the
        IElementWriter interface for the specified number
        of indices. The returned object can also
        expose a C-level function to set an element.
        """
        raise TypeError('data descriptor %r is read only'
                        % type(self))

    def element_appender(self):
        """
        This returns an object which implements the IElementAppender
        interface. The returned object can also expose a C-level
        function to append an element.
        """
        raise TypeError('data descriptor %r does not support appends'
                        % type(self))

    # TODO: When/If this becomes needed
    #@abc.abstractmethod
    #def element_mutator(self, nindex):
    #    """
    #    This returns an object which implements the
    #    ISetElement interface for the specified number
    #    of indices. The returned object can also
    #    expose a C-level function to set an element.
    #    """
    #    raise NotImplemented

    @abc.abstractmethod
    def element_read_iter(self):
        """
        This returns an iterator with the IElementReadIter
        interface which iterates over
        the leftmost dimension of the data for reading,
        returning a char* at a time. The returned object can also
        expose a C-level chunked iterator interface, similar
        to NumPy nditer.
        """
        raise NotImplemented

    def element_write_iter(self):
        """
        This returns an context manager for an iterator with
        the IElementWriteIter interface which iterates over
        the leftmost dimension of the data for writing,
        returning a char* at a time. The returned object can also
        expose a C-level chunked iterator interface, similar
        to NumPy nditer.

        # Example implementation
        return contextlib.closing(_MyElementWriteIter(self))
        """
        raise TypeError('data descriptor %r is read only'
                        % type(self))

    # TODO: When/If this becomes needed
    #@abc.abstractmethod
    #def element_mutate_iter(self):
    #    """
    #    This returns an iterator which iterates over
    #    the leftmost dimension of the data for mutation (read+write),
    #    returning a char* at a time. The returned object can also
    #    expose a C-level chunked iterator interface, similar
    #    to NumPy nditer.
    #    """
    #    raise NotImplemented

