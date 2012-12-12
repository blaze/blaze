import abc

class Layout(object):
    """
    Layout's build the Index objects neccessary to perform
    arbitrary getitem/getslice operations given a data layout of
    byte providers.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def change_coordinates(self, indexer):
        raise NotImplementedError

    @abc.abstractproperty
    def desc(self):
        """ String description of the Layout instance with the
        parameters passed to the constructor """
        raise NotImplementedError

    @abc.abstractproperty
    def wraparound(self):
        """ Allow negative indexing """
        return True

    @abc.abstractproperty
    def boundscheck(self):
        """
        If set to False, layout will to assume that indexing operations
        will not cause any IndexErrors to be raised
        """
        return False

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Layout:
            if any("change_coordinates" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
