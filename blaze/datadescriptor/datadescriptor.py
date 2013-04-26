import abc

class IGetDescriptor:
    """
    An interface for getting DataDescriptor objects at fixed-size
    index tuples.
    
    >>> obj = blzarr.get_descriptor_interface(3)
    >>> obj.get([i, j, k])
    BLZGetDescriptorObject(...)
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def nindex(self):
        """
        The number of indices the get() function
        requires. This is equal to the 'nindex'
        provided to the datadescriptor's
        get_descriptor_interface() function.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get(self, idx):
        """
        Returns a DataDescriptor to the subarray
        at the specified index. The value in
        'idx' must be a tuple of 'nindex' integers.
        """
        raise NotImplemented

class IDescriptorIter:
    """
    In interface for iterating over the outermost dimension of a data descriptor.
    It must return a data descriptor for each subarray along the dimension.
    If the dimension has a known size, it should be returned in the __len__
    method. A streaming dimension does not have a size known ahead of time,
    and should not implement __len__.
    """
    __metaclass__ = abc.ABCMeta

    def __iter__(self):
        return self

    def __len__(self):
        # TODO: raise StreamingDimensionError("Cannot get the length of a streaming dimension")
        raise NotImplemented

    @abc.abstractmethod
    def __next__(self):
        raise NotImplemented

    def next(self):
        return self.__next__()

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

            @abc.abstractproperty

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
        # TODO: raise StreamingDimensionError("Cannot get the length of a streaming dimension")
        raise NotImplemented

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
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        Returns the datashape for the data behind this datadescriptor.
        """
        raise NotImplemented
    
    @abc.abstractmethod
    def get_descriptor_interface(self, nindex):
        """
        This returns an object which implements the
        IGetDescriptor interface for the specified number
        of indices. This only operates at the Python
        level presently, there is no C-level component
        """
        raise NotImplemented

    @abc.abstractmethod
    def descriptor_iter_interface(self):
        """
        This returns an iterator which iterates over
        the leftmost dimension of the data, returning
        a DataDescriptor at a time. This only operates
        at the Python level presently, there is no C-level component.
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
