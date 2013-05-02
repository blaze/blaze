from __future__ import absolute_import

import abc
from blaze.error import StreamingDimensionError
from ..cgen.utils import letters

_stream_of_uniques = letters()

class IGetElement:
    """
    An interface for getting char* element pointers at fixed-size
    index tuples. Provides additional C and llvm function
    interfaces to use in a jitting context.

    >>> obj = blzarr.get_element_interface(3)
    >>> obj.get([i, j, k])
    CTypes/CFFIobj("char *", 0x...)
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def nindex(self):
        """
        The number of indices the get() function
        requires. This is equal to the 'nindex'
        provided to the datadescriptor's
        get_element_interface() function.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get(self, idx):
        """
        Returns a char* pointer to the element
        at the specified index. The value in
        'idx' must be a tuple of 'nindex' integers.
        """
        raise NotImplemented

    def c_getter(self):
        """
        Returns a tuple with a [CFFI or ctypes?] function pointer
        to a C get method, and a void* pointer to pass to
        the function's 'extra' parameter.
        
        Possible alternative: Returns a blaze kernel with a
        'get_element' function prototype. (This wraps destruction
        and the void* pointer in the blaze kernel low level interface)
        """
        raise NotImplemented

    def llvm_getter(self, module):
        """
        Inserts a getter function into the llvm module, and
        returns it as a function object.
        """
        raise NotImplemented

class IElementIter:
    """
    In interface for iterating over the outermost dimension of a data descriptor.
    It must return a char* pointer for each element along the dimension.
    If the dimension has a known size, it should be returned in the __len__
    method. A streaming dimension does not have a size known ahead of time,
    and should not implement __len__.
    
    
    """
    __metaclass__ = abc.ABCMeta

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

    def c_iter(self):
        """
        Returns a tuple of objects providing the iteration as C data pointers and
        function pointers. This interface is similar to the iteration interface
        in NumPy's nditer C API
        (http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#simple-iteration-example).
        """
        raise NotImplemented


class DataDescriptor:
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
            cffi_fn, ...(TBD) = ixr.c_iter()
            # LLVM access
            (TBD) = ixr.llvm_iter()

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
    _unique_name = ''

    @abc.abstractproperty
    def dshape(self):
        """
        Returns the datashape for the data behind this datadescriptor.
        """
        raise NotImplemented

    #@abc.abstractproperty
    def unique_name(self):
        """
        Returns a unique name (in this process space)
        """
        if not self._unique_name:
            self._unique_name = next(_stream_of_uniques)
        return self._unique_name

    def __len__(self):
        """
        The default implementation of __len__ is for the
        behavior of a streaming dimension, where the size
        of the dimension isn't known ahead of time.
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
    def get_element_interface(self, nindex):
        """
        This returns an object which implements the
        IGetElement interface for the specified number
        of indices. The returned object can also
        expose a C-level function to get an element.
        """
        raise NotImplemented

    @abc.abstractmethod
    def element_iter_interface(self):
        """
        This returns an iterator which iterates over
        the leftmost dimension of the data, returning
        a char* at a time. The returned object can also
        expose a C-level chunked iterator interface, similar
        to NumPy nditer.
        """
        raise NotImplemented
